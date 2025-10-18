import os
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn.functional as F

import random
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from generate import build_model, generate_continuation
from benchmarks.empathy import fewshot_samples

from glob import glob

from dotenv import load_dotenv

load_dotenv()

LM_EVAL_PATH = os.environ["LM_EVAL_PATH"]

def save_pickle(data, path):
    with open(path, "wb") as f:
        pkl.dump(data, f)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

def results_path(model_name, task):
    results_filename = {
        "micro-moe-llama-2": {
            "gsm8k": "results_2025-09-02T05-24-46.684112.json",
            "minerva_math": "results_2025-09-02T06-04-11.321038.json",
            "mmlu": "results_2025-09-02T07-05-28.315208.json",
        },
        "llama-moe-top1-tuluv3-plus-experts-1": {
            "gsm8k": "results_2025-08-19T16-19-12.859757.json",
            "minerva_math": "results_2025-08-19T17-13-57.471382.json",
            "mmlu": "results_2025-08-19T18-59-07.875715.json",
        },
        "micro-smollm2-moe-1.7b-1": {
            "gsm8k": "results_2025-09-10T04-19-57.059297.json",
        },
        "micro-smollm2-1.7b-2": {
            "gsm8k": "results_2025-08-26T11-56-04.028234.json",
            "minerva_math": "results_2025-08-26T14-47-30.353503.json",
            "mmlu": "results_2025-08-26T18-42-27.271879.json",
        },
        "micro-smollm2-moe-135m-1": {
            "gsm8k": "results_2025-09-06T17-09-58.605053.json",
            "minerva_math": "results_2025-09-06T18-14-46.893370.json",
            "mmlu": "results_2025-09-06T20-13-34.937732.json",
        },
        "micro-smollm2-moe-360m-1": {
            "gsm8k": "results_2025-09-06T12-05-13.004751.json",
            "minerva_math": "results_2025-09-06T13-17-54.038753.json",
            "mmlu": "results_2025-09-06T14-42-59.010586.json",
        },
        "micro-llama-3b-1": {
            "gsm8k": "results_2025-09-17T20-57-15.964718.json",
            "minerva_math": "results_2025-09-17T23-06-12.515112.json",
            "mmlu": "results_2025-09-18T02-36-34.958706.json",
        },
        "llama-mxtr-1b-base-top1-tuluv3-15": {
            "gsm8k": "results_2025-05-03T06-09-49.562135.json",
            "minerva_math": "results_2025-05-03T07-24-58.117863.json",
            "mmlu": "results_2025-05-04T06-58-49.571417.json",
        },
        "olmo-mxtr-1b-base-top1-tuluv3-3": {
            "gsm8k": "results_2025-05-09T22-01-21.643038.json",
            "minerva_math": "results_2025-05-10T00-00-35.338053.json",
            "mmlu": "results_2025-05-10T03-24-12.953434.json",
        }
    }[model_name][task]

    results_filename_part = results_filename.replace("results_", "").replace(".json", "")
    samples_paths = glob(f"{LM_EVAL_PATH}/results/{model_name}/samples_{task}_*_{results_filename_part}.jsonl")
    return samples_paths

def read_samples(model_name, task):
    samples_paths = results_path(model_name, task)
    sentences = []
    for samples_path in samples_paths:
        samples = read_jsonl(samples_path)
        visited = set()
        for sample in samples:
            doc_id = sample["doc_id"]
            if doc_id in visited:
                continue 
            visited.add(doc_id)
            sentences += [[sample['arguments']['gen_args_0']['arg_0'], sample['resps'][0][0]]]
    return sentences 

def aggregate_routing_weights(routing_weights):
    experts = ["Logic", "Social", "World", "Language"]
    expert_token_model = np.zeros((len(experts)), dtype=int)
    expert_layer_token = np.zeros((routing_weights.shape[0], len(experts)), dtype=int)
    num_layers = routing_weights.shape[0]

    for layer_idx in range(num_layers):
        for token_idx in range(len(routing_weights[layer_idx])):
            expert_idx = routing_weights[layer_idx][token_idx].argmax()
            if layer_idx >= 2 and layer_idx < num_layers - 2:
                expert_token_model[expert_idx] += 1
            expert_layer_token[layer_idx][expert_idx] += 1
    return expert_token_model, expert_layer_token

def get_routing_weights(model, tokenizer, prompts, apply_chat_template=True):
    """
    Get routing weights for the given prompts using the model.
    Args:
        model: The MiCRoLlama or MiCRoOLMo model.
        tokenizer: The tokenizer for the model.
        prompts: A string or list of dictionaries containing the prompts.
    Returns:
        routing_weights: A list of routing weights for each layer.
    """

    tokenizer.padding_side = "left"
    if apply_chat_template:
        if isinstance(prompts, str):
            prompts = [{"role": "user", "content": prompts}]

        inputs = tokenizer.apply_chat_template([
            prompt for prompt in prompts
        ], return_tensors="pt", padding=True).to('cuda')

        input_without_response = tokenizer.apply_chat_template([
                prompt[:-1] for prompt in prompts
            ], return_tensors="pt", padding=True,
        ).to('cuda')
    else:
        inputs = tokenizer(prompts[0] + prompts[1], return_tensors="pt", padding=True).input_ids.to('cuda')
        input_without_response = tokenizer(prompts[0], return_tensors="pt", padding=True).input_ids.to('cuda')

    attention_mask = torch.ones_like(inputs)
    attention_mask[inputs == tokenizer.pad_token_id] = 0

    model_output = model(input_ids=inputs, attention_mask=attention_mask)

    routing_weights = model_output.routing_weights   
    routing_weights = np.stack([F.softmax(rw, dim=-1).detach().float().cpu().numpy() for rw in routing_weights], axis=0).squeeze()

    offset = len(input_without_response[0])-1
    routing_weights = routing_weights[:, offset:-1]

    return routing_weights
   
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_llama.yml", help='path of config file')
    parser.add_argument('--task',  type=str,
                        default="empathy", help='task to plot')
    parser.add_argument('--ablate',  type=str,
                        default="none", help='expert to ablate')
    
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    task = args.task
    model_name = config["run-title"]
    model, tokenizer = build_model(config, args, use_cache=True)
    max_sentences = 1000

    if task in ["empathy"]:
        path = f"outputs/{model_name}/generations/{task}.json"
        sentences = read_json(path)
        apply_chat_template = True
    elif task in ["gsm8k", "minerva_math", "mmlu"]:
        apply_chat_template = False
        sentences = read_samples(model_name, task)
    elif "gpt5" in task:
        subtask = task.split("-")[-1]
        apply_chat_template = True
        path = f"data/{subtask}.jsonl"
        sentences = read_jsonl(path)
    elif task == "sample":
        apply_chat_template = True
        # prompt = "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nThink step by step and then finish your answer with \"The answer is X\" where X is the final answer."
        prompt = "Solve the following equation: 3x + 4 = 10."
        # prompt = "Ahmed and Sarah are playing a game. Sarah loses the game and feels sad. Ahmed notices that Sarah is quiet and looking down.\n\nQuestion: What should Ahmed do next?"
        # prompt = "What is the capital of the country that is west of Egypt? Think step by step."
        # prompt = "Sally and Anne are in a room together. Sally places her chocolate bar inside a blue box and then leaves the room. While she is gone, Anne moves the chocolate bar from the blue box to a red box. Sally did not see this happen.\nQuestion: When Sally returns, where does she think the chocolate bar is? Let's think step by step."
        task = "Math Example"  # "Empathy", "Math", "World"
        sentences = [{
            "prompt": prompt,
            "response": "To solve the equation 3x + 4 = 10, we'll follow these steps:\n\n1. Subtract 4 from both sides of the equation:\n3x + 4 - 4 = 10 - 4\n3x = 6\n\n2. Divide both sides by 3:\n(3x) / 3 = 6 / 3\nx = 2\n\nSo, the solution to the equation is x = 2.\n\nLet's check our solution by substituting x = 2 back into the original equation:\n3(2) + 4 = 10\n6 + 4 = 10\n10 = 10, which checks out.\n\nTherefore, the solution is x = 2.\n\nLet's check our solution by substituting x = 2 back into the original equation:\n3(2) + 4 = 10\n6 + 4 = 10\n10 = 10, which checks out.\n\nTherefore, the solution is x = 2.\n\nLet's check our solution by substituting x = 2 back into the original equation:\n3(2) + 4 = 10\n6 + 4 = 10\n10 = 10, which checks out.\n\nTherefore, the solution is x = 2.\n\nLet's check our solution by substituting x = 2 back into the original equation:\n3(2) + 4 = 10\n6 + 4 = 10\n10 = 10, which checks out.\n\nTherefore, the solution is x = 2.\n\nLet's check our solution by substituting x = 2 back into the original equation:\n3(2) + 4",
        }]
    
    all_token_routing = []
    plot_token_routing = []
    plot_layer_token_routing = []

    np.random.seed(42)
    random.seed(42) 
    if len(sentences) > max_sentences: 
        sentences = random.sample(sentences, max_sentences)

    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):

        if task == "empathy":
            prompt = [
                fewshot_samples + 
                [
                    {"role": "user", "content": sentence["prompt"]},
                    {"role": "assistant", "content": sentence["response"]},
                ]
            ]
        elif "gpt5" in task:
            prompt = [
                [
                    {"role": "user", "content": sentence["user"]},
                    {"role": "assistant", "content": sentence["assistant"]},
                ]
            ]
        elif task in ["gsm8k", "minerva_math", "mmlu"]:
            prompt = sentence
        else:
            prompt = [
                [
                    {"role": "user", "content": sentence["prompt"]},
                    {"role": "assistant", "content": sentence["response"]},
                ]
            ]

        routing_weights = get_routing_weights(
            model=model, 
            tokenizer=tokenizer,
            prompts=prompt,
            apply_chat_template=apply_chat_template
        )

        if config["top-k-experts"] == 1:
            token_routing, layer_token_routing = aggregate_routing_weights(routing_weights.copy())
        else:
            token_routing = routing_weights.copy()

        all_token_routing.append(np.roll(routing_weights, shift=1, axis=-1))
        plot_token_routing.append(np.roll(token_routing, shift=1))
        plot_layer_token_routing.append(np.roll(layer_token_routing, shift=1, axis=-1))
        if idx >= max_sentences:
            break

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plot_layer_token_routing = np.array(plot_layer_token_routing)

    plot_data = []
    layer_plot_data = []
    if "micro" in config["model"]:
        experts = ["Language", "Logic", "Social", "World"]
    else:
        experts = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
    
    if config["top-k-experts"] == 1:
        for expert_idx in range(len(experts)):
            for sample_idx in range(len(plot_token_routing)):
                percentage = plot_token_routing[sample_idx][expert_idx] / sum(plot_token_routing[sample_idx]) * 100
                plot_data += [{
                    "expert": experts[expert_idx],
                    "percentage": percentage,
                    "expert_idx": expert_idx,
                }]

        num_samples, num_layers, num_experts = plot_layer_token_routing.shape
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                for sample_idx in range(num_samples):
                    if sum(plot_layer_token_routing[sample_idx][layer_idx]) == 0:
                        percentage = 0
                    else:
                        percentage = plot_layer_token_routing[sample_idx][layer_idx][expert_idx] / sum(plot_layer_token_routing[sample_idx][layer_idx]) * 100
                    
                    layer_plot_data += [{
                        "layer": layer_idx,
                        "expert": experts[expert_idx],
                        "percentage": percentage,
                        "expert_idx": expert_idx,
                    }]
        
    else:
        for sample_idx in range(len(plot_token_routing)):
            # mean over layers then mean over sequence
            percentage = plot_token_routing[sample_idx].mean(axis=0).mean(axis=0) * 100
            for expert_idx in range(len(experts)):
                plot_data += [{
                    "expert": experts[expert_idx],
                    "percentage": percentage[expert_idx],
                    "expert_idx": expert_idx,
                }]

    plot_data = pd.DataFrame(plot_data)
    plot_data["expert"] = pd.Categorical(plot_data["expert"], categories=experts, ordered=True)

    g = sns.barplot(data=plot_data, 
        x="percentage", 
        hue="expert", 
        palette=["#63bb8e", "#97D077", "#4285F4", "#FFAB40", "#A64D79"][1:], 
        orient="h",
        hue_order=experts,
        dodge=True, 
    )

    # put labels on top of barplots
    for i in range(len(experts)):
        bar_x = g.patches[0]
        bar_y = g.patches[i]
        plt.text(x=bar_x.get_x() + bar_x.get_width() / 2, y=bar_y.get_y() + bar_y.get_height() / 2, s=experts[i], va='center', color='black', fontsize=25, fontweight='bold', ha='left')

    sns.despine()

    g.legend_.remove()

    plt.xlim(0, 100)
    plt.title(f"{task}", fontdict={'fontsize': 20, 'fontweight': 'bold'})
    plt.ylabel("")
    plt.xlabel("Percentage of Tokens (%)")
  
    plt.tight_layout()
    
    dirpath = f"outputs/{model_name}"
    if not os.path.exists(f"{dirpath}/figures"):
        os.makedirs(f"{dirpath}/figures")

    task = task.replace(" ", "-").lower()
    plt.savefig(f"{dirpath}/figures/token_routing_{task}.png", bbox_inches='tight')

    plt.close()
    plt.clf()
    plt.cla()

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    layer_plot_data = pd.DataFrame(layer_plot_data)
    layer_plot_data["expert"] = pd.Categorical(layer_plot_data["expert"], categories=experts, ordered=True)
    layer_plot_data["layer"] = (layer_plot_data["layer"] + 1).astype(str) 
    g = sns.lineplot(data=layer_plot_data, 
        x="layer", 
        y="percentage", 
        hue="expert", 
        palette=["#63bb8e", "#97D077", "#4285F4", "#FFAB40", "#A64D79"][1:], 
        hue_order=experts,
        marker="o",
        markersize=10,
        linewidth=2,
    )

    sns.despine()
    g.legend_.remove()
    plt.xlabel("Layer Number")
    # plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.ylabel("Percentage of Tokens (%)")
    plt.tight_layout()
    plt.title(f"{task}", fontdict={'fontsize': 15, 'fontweight': 'bold'})
    plt.savefig(f"{dirpath}/figures/layer_token_routing_{task}.png", bbox_inches='tight')

    if not os.path.exists(f"{dirpath}/routing_weights"):
        os.makedirs(f"{dirpath}/routing_weights")

    save_pickle(all_token_routing, f"{dirpath}/routing_weights/{task}.pkl")



