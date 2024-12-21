import os
import json

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from openai import OpenAI
import json
import concurrent.futures
import time
from tqdm import tqdm


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8122/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

from multiprocessing import Pool
from tqdm import tqdm

def request_completion(messages, model, max_tokens=1048, n=1, temperature=0.6):
    completion = client.chat.completions.create(model=model,
                                       messages=messages,
                                       max_tokens=max_tokens,
                                       n=n,
                                       temperature=temperature,
                                       )
    texts = []
    for choice in completion.choices:
        texts.append(choice.message.content)
    return texts

def request_completion_async(args):
    prompt, model, max_tokens, n, temperature = args
    return request_completion(prompt, model, max_tokens, n, temperature)

def process_prompts(prompts, data, writer,model_name):
    args = [(prompt, model_name, 2048, 3, 0.6) for prompt in prompts]
    
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(request_completion_async, args), total=len(prompts)))

    for idx, result in enumerate(results):
        item = data[idx]
        item["model_answers"] = result
        writer.write(item)
        
        
def main():
    tasks = ["retrosynthesis", "perovskite_stability_prediction", "molecule_structure_prediction", "material_calculation", 
             "high_school_physics_calculation","GB1_ftness_prediction", "diffusion_rate_analysis", "chemical_calculation"]
    from ArgumentParser import ArgumentParser
    
    parser = ArgumentParser()
    
    parser.add_argument("--model_name", type=str, required=True, help="The model name to use for inference")
    parser.add_argument("--task_name", type=str, required=True, help="The task name to use for inference", choices=tasks)
    parser.add_argument("--results_dir", type=str, required=True, help="The directory to save the results")
    args = parser.parse_args()
    
    if os.path.exists(args.results_dir) == False:
        os.makedirs(args.results_dir)
    file = os.path.join('test_data', '{}.json'.format(args.task_name))
        
    data =json.load(open(file))
    
    
    prompts = []
    for item in data:
        
        gold_answer = item["answerKey"]
        if gold_answer not in ["A", "B", "C", "D", "a", "b", "c", "d"]:
            continue
        prompt = 'Given a question and four options, please select the right answer. Your answer should be \"A\", \"B\", \"C\" or \"D\". Please solve the problem through step-by-step reasoning and please give the final answer in the last line in the following format: The correct answer is: \\boxed{{Your answer}}\n\n{question}\n{options}'
        question_content = item["question"]
        option_content = "\n"
        for option, text in zip(item["choices"]["label"], item["choices"]["text"]):
            option_content += f"{option}. {text}\n"
        actual_question = prompt.format(question=question_content, options=option_content)

        request_massage = [
                {"role": "user", "content": actual_question}
            ]

        prompts.append(request_massage)
    

        import jsonlines 
        
        
    writer = jsonlines.open(os.path.join(args.results_dir, '{}.jsonl'.format(args.task_name)), mode='w')

    process_prompts(prompts, data, writer, model_name)
    writer.close()
    print("Done")

if __name__ == "__main__":

    

    main()