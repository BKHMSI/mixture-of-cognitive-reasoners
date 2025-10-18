import os
import ast
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from dotenv import load_dotenv

import sys
import yaml
import argparse
import torch.nn.functional as F

from models.micro_olmo import MiCRoOLMo
from models.micro_llama import MiCRoLlama
from models.moe_llama import LlamaMoE

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from generate import build_model, generate_continuation

load_dotenv()

TOMI_DIR_PATH = os.environ["TOMI_DIR_PATH"]

# instruction = "The following multiple choice questions are based on the following story. The questions are related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer from the options provided by printing it as is without any modifications."
# instruction_zeroshot = "The following multiple choice questions is based on the following story. The question is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer from the options provided by printing it as is without any modifications."
context_template = "Story: {context}\nQuestion: {question}\nOptions:{cands}"

instruction = "Read the story below and answer the multiple-choice question that follows. Provide a brief explanation for your answer, then select the most appropriate option from the choices given."

fewshot_examples = [
    # {
    #     "role": "user",
    #     "content": "Story: Jack entered the laundry. Logan entered the study. Nathan entered the laundry. The persimmon is in the storage room. Jack moved the persimmon to the table. Nathan exited the laundry. Logan exited the study. Jack exited the laundry. Nathan entered the study.\nQuestion: Where will Jack look for the persimmon?\nOptions: \n- storage room\n- table\nThink step by step.",
    # },
    # {
    #     "role": "assistant",
    #     "content": "Jack will look for the persimmon in the table because he is the one who moved it there from the storage room. Since he directly observed and performed the action, he knows the persimmon's current location. No events occurred that would alter his belief about where he placed it, so he would still expect it to be in the table. The answer is table.",
    # },
    # {
    #     "role": "user",
    #     "content": "Story: Ella entered the dining room. Carter entered the dining room. The lemon is in the bag. Ella exited the dining room. Carter moved the lemon to the fridge. Ella loves the carrot.\nQuestion: Where will Ella look for the lemon?\nOptions: \n- fridge\n- bag\nThink step by step."
    # },
    # {
    #     "role": "assistant",
    #     "content": "Ella will look for the lemon in the bag because that's where it was originally placed, and she left the dining room before Carter moved it to the fridge. Since Ella did not witness the move and there's no indication that she received any new information, her belief about the lemon's location remains unchanged. She still thinks it's where she last saw or knew it to be. The answer is bag.",
    # },
    [
        {
            "role": "user",
            "content": "Story: Maya entered the kitchen. Jonah entered the kitchen. The keys are in the drawer. Maya exited the kitchen. Jonah moved the keys to the coat pocket. Maya prefers tea to coffee.\nQuestion: Where will Maya look for the keys?\nOptions:\n- coat pocket\n- drawer\nThink step by step."
        },
        {
            "role": "assistant",
            "content": "Maya will look for the keys in the drawer because that's where they were when she was last in the kitchen, and she left before Jonah moved them to the coat pocket. She has no new information to update her belief. The answer is drawer.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Story: Nina entered the study. Leo entered the study. The book is on the desk. Leo moved the book to the shelf. Nina watched Leo move the book. Nina stretched her arms.\nQuestion: Where will Nina look for the book?\nOptions:\n- shelf\n- desk\nThink step by step."
        },
        {
            "role": "assistant",
            "content": "Nina will look for the book on the shelf because she directly observed Leo move it there, so her belief matches the current reality. The answer is shelf.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Story: Zoe entered the office. Max entered the office. The notebook is in the cabinet. Zoe exited the office. Max moved the notebook to the backpack. Zoe hummed a tune.\nQuestion: Where will Zoe look for the notebook?\nOptions:\n- backpack\n- cabinet\nThink step by step."
        },
        {
            "role": "assistant",
            "content": "Zoe will look for the notebook in the cabinet because she left before Max moved it to the backpack, so her belief remains that it's still in the cabinet. The answer is cabinet.",
        },
    ],
    [
        {
            "role": "user",
            "content": "Story: Ava entered the living room. Liam entered the living room. The remote is on the coffee table. Liam moved the remote to the TV stand. Ava watched Liam move the remote. Ava checked the time.\nQuestion: Where will Ava look for the remote?\nOptions: \n- TV stand\n- coffee table\nThink step by step."
        },
        {
            "role": "assistant",
            "content": "Ava will look for the remote on the TV stand because she saw Liam move it there, so her belief matches the current state of the world. The answer is TV stand.",
        },
    ],
    [
        {
            "role": "user",
            "content": ""
        }
    ]
]

fewshot_examples_logprobs = [
    {
        "role": "user",
        "content": "Story: Jack entered the laundry. Logan entered the study. Nathan entered the laundry. The persimmon is in the storage room. Jack moved the persimmon to the table. Nathan exited the laundry. Logan exited the study. Jack exited the laundry. Nathan entered the study.\nQuestion: Where will Jack look for the persimmon?\nOptions: \n- storage room\n- table",
    },
    {
        "role": "assistant",
        "content": "Answer: table",
    },
    {
        "role": "user",
        "content": "Story: Ella entered the dining room. Carter entered the dining room. The lemon is in the bag. Ella exited the dining room. Carter moved the lemon to the fridge. Ella loves the carrot.\nQuestion: Where will Ella look for the lemon?\nOptions: \n- fridge\n- bag"
    },
    {
        "role": "assistant",
        "content": "Answer: bag",
    }
]

def eval_logprobs(model, tokenizer, text, options):
    model.eval()
    tokenizer.padding_side = "left"
    device = model.device
    
    scores = []
    with torch.no_grad():
        # Tokens up to (and including) the assistant prefix, with empty content.
        # This lets us find where the assistant content starts.
        prefix_ids = tokenizer.apply_chat_template(
            fewshot_examples_logprobs + [{"role": "user", "content": text}],
            tokenize=True,                
            add_generation_prompt=True,   
            return_tensors="pt"
        ).to(device)
        start = prefix_ids.shape[1]  # index where assistant content begins

        for opt in options:
            full_ids = tokenizer.apply_chat_template(
                fewshot_examples_logprobs + 
                [{"role": "user", "content": text},
                 {"role": "assistant", "content": opt}],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            ).to(device)                           # (1, L)

            logits = model(input_ids=full_ids).logits          # (1, L, V)
            logprobs = torch.nn.functional.log_softmax(
                logits[:, :-1, :], dim=-1                       # normalize over vocab
            )                                                   # (1, L-1, V)

            tgt = full_ids[:, 1:]                               # next tokens (1, L-1)
            tok_logprobs = logprobs.gather(
                dim=-1, index=tgt.unsqueeze(-1)
            ).squeeze(-1)                                       # (1, L-1)

            # Sum ONLY over assistant content tokens.
            # The first assistant content token at position `start`

            option_lp = tok_logprobs[:, start:].sum().item()
            scores.append(option_lp)

    scores = np.array(scores)
    return options[int(scores.argmax())]

def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def subsample(df, n):
    
    df_tb = df[df["falseTrueBelief"] == True]
    df_fb = df[df["falseTrueBelief"] == False]

    df_sampled = pd.concat([df_tb.sample(n=n//2), df_fb.sample(n=n//2)])
    return df_sampled

def create_fewshot(df, n):
    df_fewshot = subsample(df, n)

    df_fewshot["fewshot"] = df_fewshot.apply(lambda row: context_template.format(
        context=row["story"],
        question=row["question"],
        cands='\n- ' + '\n- '.join(list(row["cands"])),
        answer=row["answer"]
    ), axis=1)

    return '\n'.join(df_fewshot["fewshot"].values)

def clean_pred(text: str) -> str:
    import re
    text = text.lower()
    regexs = [
        "the answer is\s+\[?([^.\]]+)\]?\.?",
        "answer:\s+\[?([^.\]]+)\]?\.?",
        "the final answer is\s+([^.\]]+)\]?\.?",
    ]
    
    for regex in regexs:
        match = re.search(regex, text)
        if match:
            return match.group(1)
    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_llama.yml", help='path of config file')
    parser.add_argument('--ablate',  type=str,
                        default="none", help='expert to ablate')  
    parser.add_argument('--logprobs', action='store_true', help="Use logprobs mode")
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    use_logprobs = args.logprobs

    use_cache = True
    model, tokenizer = build_model(config, args, use_cache=use_cache)

    model_name = config["run-title"]
    save_path = f"outputs/{model_name}/generations/tomi_4shot.json"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    path = f"{TOMI_DIR_PATH}/tomi_dataset.csv"
    df = pd.read_csv(path)

    df = df[["story", "question", "answer", "cands", "factVsMind", "qOrder", "falseTrueBelief"]]
    df["cands"] = df["cands"].apply(ast.literal_eval)

    batch_size = 16
    first_order = True
    fewshot = True

    df_mind = df[df["factVsMind"]=="mind"]
    df_mind = df_mind[df_mind["qOrder"] == "first_order"] if first_order else df_mind[df_mind["qOrder"] == "second_order"]
    # df_mind = df_mind[df_mind["falseTrueBelief"] == False]
    df_mind = df_mind.reset_index()

    predictions = []

    np.random.seed(42)

    stories = []
    for i, row in tqdm(df_mind.iterrows(), total=len(df_mind)):
        context = context_template.format(
            context=row["story"],
            question=row["question"],
            cands='\n- ' + '\n- '.join(list(row["cands"])),
            answer=''
        ).strip()

        if not use_logprobs:
            context += "\nThink step by step."

        stories.append(context)

    accuracy = []
    invalid = 0
    print(f">> Number of stories: {len(stories)}")

    for batch_idx in tqdm(range(0, len(stories), batch_size)):
        batch = stories[batch_idx:batch_idx+batch_size]

        if use_logprobs:
            for sample_idx, story in enumerate(batch):
                row = df_mind.iloc[batch_idx+sample_idx]
                prediction = eval_logprobs(model, 
                    tokenizer, 
                    story, 
                    [f"Answer: {cand}" for cand in row["cands"]]
                )
                prediction = prediction.replace("Answer:", "").strip()
                answer = row["answer"]
                accuracy.append(prediction == answer)
                predictions += [{
                    "story": row["story"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "generation": None,
                    "cands": row["cands"],
                    "falseTrueBelief": str(row["falseTrueBelief"]),
                    "qOrder": row["qOrder"],
                }]

        else:
            if fewshot:
                chat_batch = []
                for story in batch:
                    sample = deepcopy(fewshot_examples)
                    fewshot_samples = sample[:-1]
                    np.random.shuffle(fewshot_samples)
                    sample = fewshot_samples[0]+fewshot_samples[1]+fewshot_samples[2]+fewshot_samples[3]+sample[-1]
                    
                    sample[-1]["content"] = story
                    chat_batch.append(sample)
            else:
                chat_batch = [[{"role": "user", "content": story}] for story in batch]

            decoded_text = generate_continuation(
                model=model,
                tokenizer=tokenizer,
                prompts=chat_batch,
                max_tokens=128,
                use_cache=use_cache,
                return_routing_weights=False,
            )
            
            for j, text in enumerate(decoded_text):
                raw_gen = text.replace(batch[j],'').strip()
                prediction = clean_pred(raw_gen)

                row = df_mind.iloc[batch_idx+j]

                accuracy.append(prediction == row["answer"])

                if prediction not in row["cands"]:
                    invalid += 1

                predictions += [{
                    "story": row["story"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "prediction": prediction,
                    "generation": raw_gen,
                    "cands": row["cands"],
                    "falseTrueBelief": str(row["falseTrueBelief"]),
                    "qOrder": row["qOrder"],
                }]

        write_json(predictions, save_path)

    print(f"Accuracy: {np.mean(accuracy)}")
    print(f"Invalid predictions: {invalid} / {len(accuracy)}")

    print(f"Saved predictions to {save_path}")

