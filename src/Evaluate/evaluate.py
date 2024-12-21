import re
import jsonlines


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



if __name__ == "__main__":
    from ArgumentParser import ArgumentParser
    
    parser = ArgumentParser()
    
    parser.add_argument("--file", type=str, required=True, help="The file to evaluate")
    
    args = parser.parse_args()
    
    data = jsonlines.open(args.file)
    
    
    

    correctness = []
    data = list(data)
    print(len(data))

    for item in data:
        gold_answer = item["answerKey"]
        if gold_answer not in ["A", "B", "C", "D", "a", "b", "c", "d"]:
            continue
        gold_response, correct, extract_options = check_response(gold_answer, item["model_answers"])
        correctness.append(sum(correct)/len(correct))
        
        
        # print(item)
        
    print('Accuracy:', sum(correctness)/len(correctness))