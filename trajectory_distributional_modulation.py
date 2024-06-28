import os
import argparse
import httpx
import asyncio
from tqdm import tqdm
import random
import anthropic
import json
import torch
import numpy as np
import torch.nn.functional as F

import pdb

MODEL_NAME = "claude-3-5-sonnet-20240620"


# Receive command line arguments

def get_args():
    parser = argparse.ArgumentParser(description='Trajectory Distributional Modulation')

    parser.add_argument('--anthr-key', type=str, default='anthropic_key.txt', help='Anthropic key file. Defaults to anthropic_key.txt')
    parser.add_argument('--minf-ce-url', type=str, default='http://conway.languagegame.io/colossus/ce_loss', help='URL for the minference CE loss endpoint. Defaults to http://localhost:4444/ce_loss')
    parser.add_argument('--x0-traj-dataset', type=str, default='datasets/webgazer_AQ_20240613.jsonl',help='Imposed state & trajectory dataset file. Should be .jsonl. Default is datasets/webgazer_AQ_20240613.jsonl')
    parser.add_argument('--markdown-sys-prompt', type=str, default='prompts/default_mixing_prompt.md', help='Markdown system prompt. Defaults to prompts/default_mixing_prompt.md')
    parser.add_argument('--num-alt-u', type=int, default=8, help='Number of alternative control prompts to generate each iteration. Defaults to 8')
    parser.add_argument('--num-fit-u', type=int, default=2, help='Number of control prompts that pass the fitness criterion for each iteration. Defaults to 2')
    parser.add_argument('--u-max-char', type=int, default=100, help='Maximum number of characters in a control prompt. Defaults to 100')
    parser.add_argument('--output_dir', type=str, default='output.jsonl', help='Output directory. Defaults to output.jsonl')
    parser.add_argument('--temp', type=float, default=0.4, help='Temperature for sampling u_i. Defaults to 0.4')
    parser.add_argument('--minf-timeout', type=int, default=300, help='Timeout for the minference CE loss endpoint. Defaults to 300 seconds')
    parser.add_argument('--num-iters', type=int, default=10, help='Number of iterations to generate new u_i. Defaults to 10')
    parser.add_argument('--out_dir', type=str, default='results/test00', help='Output directory for U and E. Defaults to results/test00')

    return parser.parse_args()

async def generate_u_i(client, base_sys_prompt, max_tokens, temp):
    """ Generate a brand new u_i from the LLM. 

    args: 
        client: Anthropic client (non-async)
        base_sys_prompt: System prompt for generating brand new u_i
        max_tokens: Maximum number of tokens for the LLM
        temperature: Sampling temperature for the LLM

    returns: 
        message: Generated u_i string
    """
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=temp,
        system=base_sys_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please generate a new prompt. The prompt should be less than " + str(max_u_len - 20) + " characters long."
                    }
                ]
            },
        ]
    ).content[0].text
    return message

async def create_llm_calls(client, U, base_sys_prompt, max_tokens, temp, num_alt_u):
    """ 
    Given a list of existing prompts U, generate num_alt_u
    alternative control prompts.

    Works fine for empty U.

    args: 
        client: anthropic client 
        base_sys_prompt: base system prompt as string for new u_i
        max_tokens: maximum number of tokens for the LLM
        temp: temperature for sampling u_i
        num_alt_u: number of alternative control prompts to generate
    """
    tasks = []
    for i in tqdm(range(num_alt_u)):
        # Select two random u's from U (or empty token lists if U is empty) and insert them into the {} within the base system prompt
        u_i_parents = []
        # Select two random parents
        if len(U) > 1:
            u_i_parents = random.sample(U, 2)
        elif len(U) == 1:
            u_i_parents = [U[0], '']
        else:
            u_i_parents = ['', '']

        # print(u_i_parents)
        sys_prompt = base_sys_prompt % (u_i_parents[0], u_i_parents[1])
        
        tasks.append(generate_u_i(client, sys_prompt, max_tokens, temp))
    prompts = await asyncio.gather(*tasks)
    return prompts

async def get_ce_loss(context_string, corpus_string, u_i, question, url, timeout=300):
    minf_client = httpx.AsyncClient(timeout = timeout)
    ce_loss_json = await minf_client.post(
        url,
        json={
            "context_string": context_string,
            "corpus_string": corpus_string
        }
    )
    return {'u_i': u_i, 'question': question, 'answer': corpus_string, 'ce_loss': ce_loss_json.json()['loss']}
    #return ce_loss_json.json()

async def create_ce_loss_tasks(x0_traj_list, u_i_list, url, timeout=300):
    tasks = []
    for x0_traj in x0_traj_list:
        for u_i in u_i_list:
            question = x0_traj[0]
            context_string = u_i + " " + question
            corpus_string = x0_traj[1]
            tasks.append(get_ce_loss(context_string, corpus_string, u_i, question, url, timeout=timeout))
    ce_loss = await asyncio.gather(*tasks)
    return ce_loss

def get_embeds(qa_dataset_path:str, 
               u_i_list:list[str], 
               minf_ce_url:str, 
               timeout=300):
    """
    Given a dataset of questions and answers `qa_dataset_path`, and a list of
    control prompts `u_i_list`, get the embeddings for the questions and
    answers.

    args: 
        qa_dataset_path: path to the dataset of questions and answers
        u_i_list: list of control prompts as strings
        minf_ce_url: URL for the minference CE loss endpoint

    returns: 
        embeds: tensor of shape [num_u_i, num_x0, num_answers]
            where the order of u is given by u_i_list
        ce_losses: list of dictionaries with keys: u_i, question, answer,
            ce_loss
    """
    x0_traj_dataset = open(qa_dataset_path, 'r')
    # read in all lines 
    x0_traj_dataset = x0_traj_dataset.readlines()
    ce_losses = []

    # make empty tensor of shape [num_u_i, num_x0_questions, num_answers]
    # then populate it with the ce_losses_json
    embeds = torch.zeros(len(u_i_list), len(x0_traj_dataset), len(json.loads(x0_traj_dataset[0])['answers']))

    qnum=0
    for line in tqdm(x0_traj_dataset):
        x0_traj = json.loads(line) # single question json

        # list of [question_i, answer_ij] for j in [num_answers]
        x0_traj_list = [[x0_traj['question'], ans] for ans in x0_traj['answers']]


        ce_losses_json = asyncio.run(create_ce_loss_tasks(x0_traj_list, u_i_list, minf_ce_url, timeout=timeout))
        ce_losses.append(ce_losses_json)

        # ce_losses_json is a list of dicts with keys ['u_i', 'question', 'answer', 'ce_loss']

        for ce_loss in ce_losses_json:
            # get the u_i index
            u_idx = u_i_list.index(ce_loss['u_i'])
            # get the x0 question index
            x0_idx = qnum
            # get the answer index
            ans_idx = x0_traj['answers'].index(ce_loss['answer'])
            # populate the tensor
            embeds[u_idx, x0_idx, ans_idx] = ce_loss['ce_loss']

        qnum+=1

    return embeds, ce_losses 


    # todo: convert ce_losses list of jsons to tensor
    # using the ordering of u_i_list and x0_traj_dataset

    # ce_losses[i] is a list of dictionaries with keys: u_i, question, answer, ce_loss

    # for i, ce_loss in enumerate(ce_losses):
    #     for j, ce_loss_json in enumerate(ce_loss):
    #         # get u_idx based on ce_loss_json['u_i']
    #         u_idx = 



    return ce_losses


if __name__ == '__main__':
    args = get_args()
    print(args)

    # Initialize U -- a list of control prompts as strings.
    U = []

    # Initialize U' (U prime) -- list of potential new control prompts as strings.
    U_prime = []

    base_sys_prompt = open(args.markdown_sys_prompt).read()
    print("\nBase system prompt: ",base_sys_prompt)

    anthr_key_file = open(args.anthr_key, 'r')
    anthr_key = anthr_key_file.read()

    # Make call to claude to get a new u_i
    client = anthropic.Anthropic(
        api_key=anthr_key,
    )
    max_u_len = args.u_max_char
    max_tokens = max_u_len // 4

    # Generate initial u' list from U = []
    print(f"Making calls to generate first {args.num_alt_u} u_i's with Claude...")
    u_i_list = asyncio.run(create_llm_calls(client, U, base_sys_prompt, max_tokens, args.temp, args.num_alt_u))
    print("Done!\n")
    # remove duplicates
    u_i_list = list(set(u_i_list))


    # get the ce losses over the answers
    # E has shape [num_u, num_questions, num_answers]
    print(f"Getting embeddings (probability over answers) for the initial u_i's (length {len(u_i_list)})")
    E, ce_losses = get_embeds(args.x0_traj_dataset, u_i_list, args.minf_ce_url, timeout=args.minf_timeout)
    print("Done!\n")

    U = U + u_i_list

    # iterate through number of iterations
    for i in range(args.num_iters):
        # Generate new u' list from U
        print(f"[ITER {i}] Making calls to generate {args.num_alt_u} u_i's with Claude...")
        u_i_list = asyncio.run(create_llm_calls(client, U, base_sys_prompt, max_tokens, args.temp, args.num_alt_u))
        # remove duplicates
        u_i_list = list(set(u_i_list))

        # get the ce losses over the answers for the newly spawned u_i_list
        print(f"[ITER {i}] Getting embeddings (probability over answers) for the new u_i's (length {len(u_i_list)})")
        embeds_i, ce_losses = get_embeds(args.x0_traj_dataset, u_i_list, args.minf_ce_url, timeout=args.minf_timeout)
        flat_Ep_normd = F.normalize(embeds_i.reshape(embeds_i.shape[0], -1), p=2, dim=1)

        flat_E_normd = F.normalize(E.reshape(E.shape[0], -1), p=2, dim=1)

        cosine_sims = flat_E_normd @ flat_Ep_normd.T

        # take max similarity over dimension 0 of cosine_sims
        max_sims = torch.max(cosine_sims, dim=0).values

        print(f"[ITER {i}] Max cosine sims: ", max_sims)

        # get argsort of max_sims
        sorted_sims = torch.argsort(max_sims, descending=False)

        # new u idx 
        new_u_idx = sorted_sims[:args.num_fit_u]

        for idx in new_u_idx:
            print(f"\tNew u_i: {u_i_list[idx]}")
            # add to U
            U.append(u_i_list[idx])
            # add to E
            E = torch.cat([E, embeds_i[idx].unsqueeze(0)], dim=0)

    # save U, E to disk in args.out_dir as npz
    # make out_dir if it does not exist 
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # save U, E to disk
    U_path = os.path.join(args.out_dir, 'U.npy')
    E_path = os.path.join(args.out_dir, 'E.npy')
    print("Saving U to ", U_path)
    np.save(U_path, U)
    print("Done!\n")
    print("Saving E to ", E_path)
    np.save(E_path, E)
    print("Done!\n")

    # save args.json to disk
    args_path = os.path.join(args.out_dir, 'args.json')
    print("Saving args to ", args_path)
    with open(args_path, 'w') as f:
        json.dump(vars(args), f)
    print("Done!\n")

    print("Goodbye, have a nice day.")

