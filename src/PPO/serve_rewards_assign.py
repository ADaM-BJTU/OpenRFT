import json
import argparse
import re

import torch
import uvicorn
import numpy as np
from dataclasses import dataclass
from functools import partial
from enum import Enum
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from transformers import AutoTokenizer

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

from skywork_o1_prm_inference.prm_model import PRM_MODEL
from skywork_o1_prm_inference.io_utils import (
    prepare_input,
    prepare_batch_input_for_model,
    derive_step_rewards
)
logger = init_logger(__name__)

import re


def extract_final_answer(text):
    pattern = r"\\boxed(?:\{\\text\{([A-Da-d])\}\}|\{([A-Da-d])\})"
    matches = re.findall(pattern, text)
    answers = [match[0] or match[1] for match in matches]
    return answers[-1] if answers else "None"

def extract_all_answers(text):
    pattern = r"\\boxed(?:\{\\text\{([A-Da-d])\}\}|\{([A-Da-d])\})"
    matches = re.findall(pattern, text)
    answers = [match[0] or match[1] for match in matches]
    return answers

def check_response(gold_answer, response):
    correctness = []
    extract_options  = []
    gold_response = ""
    if isinstance(response, str):
        response = [response]
    for res in response:
        answer = extract_final_answer(res)
        extract_options.append(answer)
    for i, option in enumerate(extract_options):
        if option == gold_answer and gold_response == "":
            gold_response = response[i]
        correctness.append(option == gold_answer)
    return gold_response, correctness, extract_options
    
def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)
    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)
    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


class RewardModelProxy:
    def __init__(self, args):
        self.reward_model = get_llm_for_sequence_regression(
            args.reward_pretrain,
            "reward",
            normalize_reward=args.normalize_reward,
            use_flash_attention_2=args.flash_attn,
            bf16=args.bf16,
            load_in_4bit=args.load_in_4bit,
            value_head_prefix=args.value_head_prefix,
            device_map="auto",
        )
        self.reward_model.eval()

        self.tokenizer = get_tokenizer(
            args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        )
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.alpha = 0.5

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
                + self.tokenizer.eos_token
            )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                inputs = self.tokenize_fn(
                    queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                )
                r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                r = r.tolist()
                scores.extend(r)
        return scores

    def tokenize_fn(self, texts, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}


CHAT_TEMPLATE = {
    "Llama": {
        "user_turn": r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
        "one_turn": r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>.*?<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>",
        "query_role": "user",
        "response_role": "assistant",
        "end_of_turn": "<|eot_id|>",
        "pad_token": "<|end_of_text|>"
    },
    "Qwen": {
        "user_turn": r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
        "one_turn": r"<\|im_start\|>user\n(.*?)<\|im_end\|>.*?<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
        "query_role": "user",
        "response_role": "assistant",
        "end_of_turn": "<|im_end|>",
        "pad_token": "<|endoftext|>"
    }
}

@dataclass
class RewardStrategy(Enum):
    MEAN = "mean"
    MIN = "min"
    LAST = "last"


class SkyworkO1PRM(RewardModelProxy):
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.reward_pretrain, trust_remote_code=True)
        self.reward_model = PRM_MODEL.from_pretrained(
            args.reward_pretrain,
            _attn_implementation="flash_attention_2" if args.flash_attn else "eager",
            torch_dtype=torch.bfloat16 if args.bf16 else "auto",
            device_map="auto",
        ).eval()
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        self.reward_assign_strategy = RewardStrategy(args.reward_assign_strategy)
        self.alpha = 0.7
        self.question_answer_mapping = None

    def set_question_answer_mapping(self, question_answer_mapping):
        self.question_answer_mapping = question_answer_mapping

    def assign_rewards(self, step_rewards: List[List[int]], strategy: RewardStrategy) -> List[int]:
        if strategy is RewardStrategy.MEAN:
            outcome_rewards = [np.mean(rewards) for rewards in step_rewards]
        elif strategy is RewardStrategy.MIN:
            outcome_rewards = [np.min(rewards) for rewards in step_rewards]
        elif strategy is RewardStrategy.LAST:
            outcome_rewards = [rewards[-1] for rewards in step_rewards]
        else:
            raise ValueError("Invalid reward strategy")
        return outcome_rewards

    def compute_outcome_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        outcome_scores = []
        for i in range(0, len(queries), batch_size):
            # TODO: Replace this with a more robust way to get the ground truth answer
            # This is a very lazy way of extracting the ground truth answer; you could replace it with a more robust method.
            
            for j in range(i, min(len(queries), i + batch_size)):
                gold_answer='sorry, I do not know the answer'
                for key in self.question_answer_mapping:
                    q, optionA = key.split("<SEP>")
                    max_l = min(len(q)-3, 200)
                    if q[5:max_l].replace(' ','').replace('\n','') in queries[j].split('[End of examples]', 1)[-1].replace(' ','').replace('\n',''):
                        gold_answer = self.question_answer_mapping[key]
                        
                        if optionA.strip().replace(' ','').replace('\n','') in queries[j].split('[End of examples]', 1)[-1].replace(' ','').replace('\n',''):
                            gold_answer = self.question_answer_mapping[key]
                            break
                logger.info(f"gold_answer: {gold_answer}")
                gold_response, correctness, extract_options = check_response(gold_answer, queries[j])
                if correctness[0]:
                    outcome_scores.append(1.0)
                elif extract_options[0]:
                    outcome_scores.append(0.1)
                elif len(gold_answer)> 5:
                    outcome_scores.append(0.5)
                else:
                    outcome_scores.append(0.0)

        return outcome_scores

    def compute_process_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        chat_style = None
        for template, template_style in CHAT_TEMPLATE.items():
            user_style = template_style["user_turn"]
            user_pattern = re.compile(user_style, re.DOTALL)
            user_matches = user_pattern.findall(queries[0])
            if user_matches:
                chat_style = template
                break

        if chat_style is None:
            logger.error(f"queries[0] not found chat template: {queries[0]}")
            raise ValueError("Chat style not found in the query")
        logger.info(f"chat_style's model name: {chat_style}")    
        
        for i in range(len(queries)):
            queries[i] = (
                strip_sequence(queries[i], pad_token=CHAT_TEMPLATE[chat_style]["pad_token"], eos_token=CHAT_TEMPLATE[chat_style]["end_of_turn"])
                + CHAT_TEMPLATE[chat_style]["end_of_turn"]
            )
        logger.info(f"queries[0]: {queries[0]}")
        
        format_samples = []
        for raw_text in queries:
            chat_pattern = re.compile(CHAT_TEMPLATE[chat_style]["one_turn"], re.DOTALL)
            try:
                query, response = chat_pattern.findall(raw_text)[0]
            except Exception as e:
                logger.error(f"error query: {raw_text}")
                raise ValueError(f"Error in chat pattern: {e}")
            
            format_samples.append((query, response))

        processed_data = [prepare_input(sample[0], sample[1], tokenizer=self.tokenizer, step_token="\n\n") for sample in format_samples]

        scores = []
        device = self.reward_model.v_head.summary.weight.device
        with torch.no_grad():
            for i in range(0, len(processed_data), batch_size):
                input_ids, steps, reward_flags = zip(*processed_data[i: i+batch_size])
                input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, self.tokenizer.pad_token_id)
                _, _, rewards = self.reward_model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    return_probs=True
                )
                step_rewards = derive_step_rewards(rewards, reward_flags)
                # outcome_rewards = self.assign_rewards(step_rewards, strategy=self.reward_assign_strategy)
                aggregated_rewards = self.assign_rewards(step_rewards, strategy=self.reward_assign_strategy)

                scores.extend(aggregated_rewards)
        return scores

    def get_outcome_reward(self, queries):

        return self.compute_outcome_reward(queries)

    def get_process_reward(self, queries):

        return self.compute_process_reward(queries)

    def get_combined_reward(self, queries):
        outcome_scores = self.compute_outcome_reward(queries)
        process_scores = self.compute_process_reward(queries)
        assert len(outcome_scores) == len(process_scores)

        combined_scores = [self.alpha * outcome_scores[i] + (1 - self.alpha) * process_scores[i] for i in range(len(outcome_scores))]
        return combined_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

    # Process-Supervised Reward Model
    parser.add_argument("--reward_assign_strategy", choices=["mean", "min", "last"], default="mean")

    parser.add_argument("--port", type=int, default=5000, help="Port number for the server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")

    # Performance
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--dataset", type=str, default=None, help="Dataset for the reward model")
    
    args = parser.parse_args()

    question_answer_mapping = {}
    for item in json.load(open(args.dataset)):
        question_answer_mapping[item["question"]+"<SEP>"+item["choices"]["text"][0].strip()]= item["answerKey"]

    reward_model = SkyworkO1PRM(args)
    reward_model.set_question_answer_mapping(question_answer_mapping)
    print(reward_model.reward_model)
    app = FastAPI()

    @app.post("/get_outcome_reward")
    async def get_outcome_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_outcome_reward(queries=queries)
        torch.cuda.empty_cache()
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    @app.post("/get_process_reward")
    async def get_process_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_process_reward(queries=queries)
        torch.cuda.empty_cache()
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_combined_reward(queries=queries)
        torch.cuda.empty_cache()
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
