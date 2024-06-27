
import argparse
import httpx
import asyncio
from tqdm import tqdm
import random
import anthropic
import json

MODEL_NAME = "claude-3-5-sonnet-20240620"


# Receive command line arguments

def get_args():
    parser = argparse.ArgumentParser(description='Trajectory Distributional Modulation')

    parser.add_argument('--anthr-key', type=str, default='anthropic_key.txt', help='Anthropic key file. Defaults to anthropic_key.txt')
    parser.add_argument('--minf-ce-url', type=str, default='http://conway.languagegame.io/colossus/ce_loss', help='URL for the minference CE loss endpoint. Defaults to http://localhost:4444/ce_loss')
    parser.add_argument('--x0-traj-dataset', type=str, default='datasets/webgazer_AQ_20240613.jsonl',help='Imposed state & trajectory dataset file. Should be .jsonl. Default is datasets/webgazer_AQ_20240613.jsonl')
    parser.add_argument('--markdown-sys-prompt', type=str, default='prompts/default_mixing_prompt.md', help='Markdown system prompt. Defaults to prompts/default_mixing_prompt.md')
    parser.add_argument('--num-alt-u', type=int, default=8, help='Number of alternative control prompts to generate each iteration. Defaults to 16')
    parser.add_argument('--u-max-char', type=int, default=100, help='Maximum number of characters in a control prompt. Defaults to 100')
    parser.add_argument('--output_dir', type=str, default='output.jsonl', help='Output directory. Defaults to output.jsonl')
    parser.add_argument('--temp', type=float, default=0.4, help='Temperature for sampling u_i. Defaults to 0.4')

    return parser.parse_args()

async def generate_u_i(client, base_sys_prompt, max_tokens, temp):
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

async def create_llm_calls(client, base_sys_prompt, max_tokens, temp, num_alt_u):
    tasks = []
    for i in range(num_alt_u):
        # Select two random u's from U (or empty token lists if U is empty) and insert them into the {} within the base system prompt
        u_i_parents = []
        # Select two random parents
        if len(U) > 1:
            u_i_parents = random.sample(U, 2)
        elif len(U) == 1:
            u_i_parents = [U[0], '']
        else:
            u_i_parents = ['', '']

        print(u_i_parents)
        sys_prompt = base_sys_prompt % (u_i_parents[0], u_i_parents[1])
        
        tasks.append(generate_u_i(client, sys_prompt, max_tokens, temp))
    prompts = await asyncio.gather(*tasks)
    return prompts

async def get_ce_loss(context_string, corpus_string, u_i, question, url):
    minf_client = httpx.AsyncClient()
    ce_loss_json = await minf_client.post(
        url,
        json={
            "context_string": context_string,
            "corpus_string": corpus_string
        }
    )
    return {'u_i': u_i, 'question': question, 'answer': corpus_string, 'ce_loss': ce_loss_json.json()['loss']}
    #return ce_loss_json.json()

async def create_ce_loss_tasks(x0_traj_list, u_i_list, url):
    tasks = []
    for x0_traj in x0_traj_list:
        for u_i in u_i_list:
            question = x0_traj[0]
            context_string = u_i + " " + question
            corpus_string = x0_traj[1]
            tasks.append(get_ce_loss(context_string, corpus_string, u_i, question, url))
    ce_loss = await asyncio.gather(*tasks)
    return ce_loss

if __name__ == '__main__':
    args = get_args()
    print(args)

    # Initialize U
    U = []

    # Initialize U' (U prime)
    U_prime = []

    base_sys_prompt = open(args.markdown_sys_prompt).read()
    print(base_sys_prompt)

    anthr_key_file = open(args.anthr_key, 'r')
    anthr_key = anthr_key_file.read()

    # Make call to claude to get a new u_i
    client = anthropic.Anthropic(
        api_key=anthr_key,
    )
    max_u_len = args.u_max_char
    max_tokens = max_u_len // 4

    u_i_list = asyncio.run(create_llm_calls(client, base_sys_prompt, max_tokens, args.temp, args.num_alt_u))

    #print(prompts)

    x0_traj_dataset = open(args.x0_traj_dataset, 'r')

    for line in x0_traj_dataset:
        x0_traj = json.loads(line)

        x0_traj_list = [[x0_traj['question'], ans]for ans in x0_traj['answers']]

        # Make call to minference to get CE loss
        ce_losses_json = asyncio.run(create_ce_loss_tasks(x0_traj_list, u_i_list, args.minf_ce_url))

        for ce_loss in ce_losses_json:
            print(ce_loss)
            print('\n')
