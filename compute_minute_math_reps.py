"""
This script generates a minute math dataset with `generate_minute_math.py` 
and stores the internal representation of a model along with the dataset to 
disk in a given output directory.
"""
import argparse 
# args: output directory
# args: model name (default=meta-llama/Meta-Llama-3-8B-Instruct)
# args: num_unique_problems (default=100)
# args: min_answer=15
# args: max_answer=25
# args: min_problem=0 (minimum value for num1, num2=ans-num1)

parser = argparse.ArgumentParser(description="Generate minute math dataset and store internal representation of model")
parser.add_argument("--output_dir", type=str, help="Directory to store dataset and model representation")
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
parser.add_argument("--num_unique_problems", type=int, default=100, help="Number of unique problems")
parser.add_argument("--min_answer", type=int, default=15, help="Minimum answer")
parser.add_argument("--max_answer", type=int, default=25, help="Maximum answer")
parser.add_argument("--min_problem", type=int, default=0, help="Minimum value for num1, num2=ans-num1")
args = parser.parse_args()

import os
# if output dir exists, warn and ask for user input
if os.path.exists(args.output_dir):
    print(f"Output directory {args.output_dir} already exists. Overwrite? (y/n)")
    response = input()
    if response.lower() != "y":
        print("Exiting...")
        exit(0)
else:
    print(f"Making output dir {args.output_dir}...")
    os.makedirs(args.output_dir)
    print("Done!\n")

# save args.json to output dir
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from generate_minute_math import generate_minute_math_with_tokenizer
from tqdm import tqdm 
import pdb

print("Saving args.json...")
with open(os.path.join(args.output_dir, "args.json"), "w") as f:
    json.dump(vars(args), f)
print("Done!\n")




# Load tokenizer 
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
print("Done!\n")

# Generate minute math dataset
print("Generating minute math dataset...")
problems, answers = generate_minute_math_with_tokenizer(tokenizer=tokenizer, num_unique_problems=args.num_unique_problems, min_problem=args.min_problem, min_answer=args.min_answer, max_answer=args.max_answer)
print("Done!\n")


# Save dataset to disk
print("Saving dataset to disk...")
np.save(os.path.join(args.output_dir, "problems.npy"), problems)
np.save(os.path.join(args.output_dir, "answers.npy"), answers)
print("Done!\n")


# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
print("Done!\n")


# Get internal representation and logits for each problem
print("Getting internal representation and logits for each problem...")
internal_reps = []
logits = []

cnt = 0
for problem in tqdm(problems):
    input_ids = torch.tensor(problem).unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        internal_reps.append(outputs['past_key_values'])
        logits.append(outputs['logits'].cpu())
print("Done!\n")

# Save internal representations and logits to disk
print("Saving internal representations and logits to disk...")
# internal_reps = torch.cat(internal_reps, dim=0)
logits = torch.cat(logits, dim=0)
torch.save(internal_reps, os.path.join(args.output_dir, "internal_reps.pt"))
torch.save(logits, os.path.join(args.output_dir, "logits.pt"))
print("Done!\n")
print("Exiting...")
exit(0)


