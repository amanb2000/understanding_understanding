# Function to generate a list of minute math addition problems
# E.g. ['1 + 2 =', '0 + 5 =', '6 + 3 =', '7 + 8 =', '2 + 4 =']

import random
import anthropic

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

import pdb

MODEL_NAME = "claude-3-5-sonnet-20240620"
HF_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def generate_minute_math(num_unique_problems=100, min_problem=0, min_answer=15, max_answer=25):
    problems = []
    answers = []
    for i in range(num_unique_problems):
        ans = i % (max_answer - min_answer + 1) + min_answer
        num1 = random.randint(min_problem, ans+1)

        num2 = ans - num1

        # Randomly decide to switch num1 and num2
        if random.randint(0, 1) == 1:
            temp = num1
            num1 = num2
            num2 = temp
        problem = f"{num1} + {num2} ="
        problems.append(problem)
        answer = f"{num1 + num2}"
        answers.append(answer)


    return problems, answers


def generate_minute_math_with_tokenizer(tokenizer=None, num_unique_problems=100, min_problem=0, min_answer=15, max_answer=25):
    # Create tokenizer
    if tokenizer is None:
        # throw error
        assert False, "Tokenizer is None"

    # Generate problems
    problems, answers = generate_minute_math(num_unique_problems=num_unique_problems, min_problem=min_problem, min_answer=min_answer, max_answer=max_answer)

    # Tokenize problems
    if type(tokenizer) == transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
        print("Recognizied HF tokenizer...")
        tokenized_problems = tokenizer(problems)['input_ids']
        tokenized_answers = tokenizer(answers, add_special_tokens=False)['input_ids']
    else: 
        print("Recognized anthropic tokenizer...")
        tokenized_problems = tokenizer.encode_batch(problems)
        tokenized_answers = tokenizer.encode_batch(answers)

    # Remove any problems that are not the mode number of tokens
    # First find mode number of tokens
    num_tokens_questions = [len(problem) for problem in tokenized_problems]
    mode_num_tokens_questions = max(set(num_tokens_questions), key=num_tokens_questions.count)
    num_tokens_answers = [len(answer) for answer in tokenized_answers]
    mode_num_tokens_answers = max(set(num_tokens_answers), key=num_tokens_answers.count)

    tokenized_problems_clean = []
    tokenized_answers_clean = []
    for i in range(len(tokenized_problems)):
        if len(tokenized_problems[i]) != mode_num_tokens_questions or len(tokenized_answers[i]) != mode_num_tokens_answers:
            continue
        else:
            tokenized_problems_clean.append(tokenized_problems[i])
            tokenized_answers_clean.append(tokenized_answers[i])
    if type(tokenizer) != transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
        tokenized_problems_clean_ids = [problem.ids for problem in tokenized_problems_clean]
        tokenized_answers_clean_ids = [answer.ids for answer in tokenized_answers_clean]
    else: 
        tokenized_problems_clean_ids = [problem for problem in tokenized_problems_clean]
        tokenized_answers_clean_ids = [answer for answer in tokenized_answers_clean]


    # Convert back into strings
    if type(tokenizer) == transformers.tokenization_utils_fast.PreTrainedTokenizerFast:
        problems_back_decoded = tokenizer.batch_decode(tokenized_problems_clean_ids)
        answers_back_decoded = tokenizer.batch_decode(tokenized_answers_clean_ids)
    else: 
        problems_back_decoded = tokenizer.decode_batch(tokenized_problems_clean_ids)
        answers_back_decoded = tokenizer.decode_batch(tokenized_answers_clean_ids)

    # return problems_back_decoded, answers_back_decoded
    return tokenized_problems_clean, tokenized_answers_clean

if __name__ == "__main__":
    # Generate problems
    print("Generating problems with no tokenizer...")
    problems, answers = generate_minute_math(num_unique_problems=100, min_problem=0, min_answer=15, max_answer=25)
    print("Done!")
    print("\tGenerated ", len(problems), " problems")
    print("\tGenerated ", len(answers), "answers")

    # Generate problems with tokenizer
    tokenizer = anthropic.Anthropic.get_tokenizer(MODEL_NAME)
    print("Generating problems with anthropic tokenizer...")
    problems, answers = generate_minute_math_with_tokenizer(tokenizer=tokenizer, num_unique_problems=100, min_problem=0, min_answer=15, max_answer=25)
    print("Done!")
    print("\tGenerated ", len(problems), " problems")
    print("\tGenerated ", len(answers), "answers")

    # Generate problems with HF tokenizer
    print("\nGenerating problems with HF tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    problems, answers = generate_minute_math_with_tokenizer(tokenizer=tokenizer, num_unique_problems=100, min_problem=0, min_answer=15, max_answer=25)
    print("Done!")
    print("\tGenerated ", len(problems), " problems")
    print("\tGenerated ", len(answers), "answers")
    print("Problems: ", problems)
    print("Answers: ", answers)