import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from generate import build_model

def aggregate_routing_weights(routing_weights):
    experts = ["Logic", "Social", "World", "Language"]
    expert_token_model = np.zeros((len(experts)), dtype=int)
    for layer_idx in range(len(routing_weights)):
        for token_idx in range(len(routing_weights[layer_idx][0])):
            expert_idx = routing_weights[layer_idx][0][token_idx].argmax()
            expert_token_model[expert_idx] += 1
    return expert_token_model

def get_routing_weights(model, tokenizer, prompts):
    """
    Get routing weights for the given prompts using the model.
    Args:
        model: The MiCRoLlama or MiCRoOLMo model.
        tokenizer: The tokenizer for the model.
        prompts: A string or list of dictionaries containing the prompts.
    Returns:
        routing_weights: A list of routing weights for each layer.
    """

    if isinstance(prompts, str):
        prompts = [{"role": "user", "content": prompts}]

    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template([
        prompt for prompt in prompts
    ], return_tensors="pt", padding=True, add_generation_prompt=True).to('cuda')

    attention_mask = torch.ones_like(inputs)
    attention_mask[inputs == tokenizer.pad_token_id] = 0

    model_output = model(input_ids=inputs, attention_mask=attention_mask)

    routing_weights = model_output.routing_weights        
    routing_weights = [F.softmax(rw, dim=-1).detach().float().cpu().numpy() for rw in routing_weights]

    return routing_weights
   

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_llama.yml", help='path of config file')
    parser.add_argument('--ablate',  type=str,
                        default="none", help='expert to ablate')
    
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    datapath = "data/baseline200.csv"
    data = pd.read_csv(datapath, header=0)

    model, tokenizer = build_model(config, args, use_cache=True)

    sentences = data["sentence"].tolist()

    all_token_routing = []
    for sentence in tqdm(sentences):
        routing_weights = get_routing_weights(model, tokenizer, sentence)
        token_routing = aggregate_routing_weights(routing_weights)
        all_token_routing.append(np.roll(token_routing, shift=1))

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    plot_data = []
    experts = ["Language", "Logic", "Social", "World"]

    for expert_idx in range(len(experts)):
        for sample_idx in range(len(all_token_routing)):
            percentage = all_token_routing[sample_idx][expert_idx] / sum(all_token_routing[sample_idx]) * 100
            plot_data += [{
                "expert": experts[expert_idx],
                "percentage": percentage,
                "expert_idx": expert_idx,
            }]

    plot_data = pd.DataFrame(plot_data)
    plot_data["expert"] = pd.Categorical(plot_data["expert"], categories=experts, ordered=True)

    sns.barplot(data=plot_data, 
        x="percentage", 
        hue="expert", 
        palette=["#63bb8e", "#97D077", "#4285F4", "#FFAB40", "#A64D79"][1:], 
        orient="h",
        hue_order=experts,
        dodge=True, 
    )

    sns.despine()
    plt.title(f"Driving/Suppressing Baseline Data")
    plt.ylabel("")
    plt.xlabel("Percentage of Tokens (%)")
  
    plt.tight_layout()
    plt.legend(title="Expert")
    plt.savefig("outputs/token_routing_baseline.png", bbox_inches='tight')


        



