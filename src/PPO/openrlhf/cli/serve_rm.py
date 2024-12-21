import argparse
import re
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.utils import get_tokenizer
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

import re


def extract_final_answer(text):
    
    pattern = r"\\boxed(?:\{\\text\{([A-Da-d])\}\}|\{([A-Da-d])\})"
    
    matches = re.findall(pattern, text)
    
    # if answers is None:
    #     print(text)
    answers = [match[0] or match[1] for match in matches]
    # print(answers)
    
    return answers[-1] if answers else "None"

def extract_all_answers(text):
    # 定义正则表达式匹配 \boxed{选项} 或 \boxed{\text{选项}}，包括大写和小写的 a-d
    pattern = r"\\boxed(?:\{\\text\{([A-Da-d])\}\}|\{([A-Da-d])\})"
    # 使用 re.findall 获取所有匹配的选项
    matches = re.findall(pattern, text)
    # 提取非空的捕获组
    answers = [match[0] or match[1] for match in matches]
    return answers


def check_response(gold_answer, response):
    correctness = []
    extract_options  = []
    gold_response = ""
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
        # self.reward_model = get_llm_for_sequence_regression(
        #     args.reward_pretrain,
        #     "reward",
        #     normalize_reward=args.normalize_reward,
        #     use_flash_attention_2=args.flash_attn,
        #     bf16=args.bf16,
        #     load_in_4bit=args.load_in_4bit,
        #     value_head_prefix=args.value_head_prefix,
        #     device_map="auto",
        # )
        # self.reward_model.eval()

        # self.tokenizer = get_tokenizer(
        #     args.reward_pretrain, self.reward_model, "left", None, use_fast=not args.disable_fast_tokenizer
        # )
        self.max_length = args.max_len
        self.batch_size = args.batch_size
        
        self.alpha = 0.5

    def get_reward(self, queries):
        if self.batch_size is None:
            batch_size = len(queries)
        else:
            batch_size = self.batch_size

        # remove pad_token
        # for i in range(len(queries)):
        #     queries[i] = (
        #         strip_sequence(queries[i], self.tokenizer.pad_token, self.tokenizer.eos_token)
        #         + self.tokenizer.eos_token
        #     )
        logger.info(f"queries[0]: {queries[0]}")

        scores = []
        outcome_rewards = []
        # batch
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                
                for j in range(i, min(len(queries), i + batch_size)):
                    gold_answer=None
                    for key in self.question_answer_mapping:
                        # print(key)
                        # print(queries[j])
                        key, optionA = key.split("<SEP>")
                        if key[5:-5] in queries[j]:
                            if optionA.strip() in queries[j]:
                                gold_answer = self.question_answer_mapping[key]
                                break
                    assert gold_answer is not None
                    logger.info(f"gold_answer: {gold_answer}")
                    gold_response, correctness, extract_options = check_response(gold_answer, queries[j])
                    if correctness[0]:
                        outcome_rewards.append(1.0)
                    elif len(extract_options) == 1:
                        outcome_rewards.append(0.1)
                    else:
                        outcome_rewards.append(0.0)
                    
                # inputs = self.tokenize_fn(
                #     queries[i : min(len(queries), i + batch_size)], device=self.reward_model.device
                # )
                # r = self.reward_model(inputs["input_ids"], inputs["attention_mask"])
                # r = r.tolist()
                
                # scores.extend(r)
                
        # 然后对二者线性加权求和
        # scores = [self.alpha * scores[i] + (1 - self.alpha) * outcome_rewards[i] for i in range(len(scores))]
        # return scores
        return outcome_rewards

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

    def set_question_answer_mapping(self, question_answer_mapping):
        self.question_answer_mapping = question_answer_mapping

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Reward Model
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--max_len", type=int, default="2048")

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


    
    # server
    reward_model = RewardModelProxy(args)
    # "choices": {
    #         "text": [
    #             "Fe in Al",
    #             "Fe in Ge",
    #             "Fe in As",
    #             "Fe in Ir"
    #         ],
    #         "label": [
    #             "A",
    #             "B",
    #             "C",
    #             "D"
    #         ]
    question_answer_mapping = {}
    for item in json.load(open(args.dataset)):
        question_answer_mapping[item["question"]+"<SEP>"+item["choices"]["text"][0].strip()]= item["answerKey"]
    print(question_answer_mapping)
    reward_model.set_question_answer_mapping(question_answer_mapping)
    
    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        queries = data.get("query")
        rewards = reward_model.get_reward(queries)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
