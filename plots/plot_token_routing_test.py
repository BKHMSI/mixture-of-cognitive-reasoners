from collections import OrderedDict

import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from generate import build_model

def _get_layer(module, layer_name: str) -> torch.nn.Module:
    SUBMODULE_SEPARATOR = '.'
    for part in layer_name.split(SUBMODULE_SEPARATOR):
        module = module._modules.get(part)
        assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
    return module

def _register_hook(layer: torch.nn.Module, key: str, target_dict: dict) -> torch.utils.hooks.RemovableHandle:
    # instantiate parameters to function defaults; otherwise they would change on next function call
    def hook_function(_layer: torch.nn.Module, _input, output: torch.Tensor, key=key):
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output

    hook = layer.register_forward_hook(hook_function)
    return hook

def _setup_hooks(model, layer_names):
    """ set up the hooks for recording internal neural activity from the model (aka layer activations) """
    hooks = []
    layer_representations = OrderedDict()
    for layer_idx, layer_name in enumerate(layer_names):
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, key=layer_name, target_dict=layer_representations)
        hooks.append(hook)

    return hooks, layer_representations

def dump_pickle(obj, filename):
    """
    Dump an object to a pickle file.
    Args:
        obj: The object to be dumped.
        filename: The name of the file to dump the object into.
    """
    with open(filename, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def aggregate_routing_weights(routing_weights):
    experts = ["Logic", "Social", "World", "Language"]
    expert_token_model = np.zeros((len(experts)), dtype=int)
    for layer_idx in range(len(routing_weights)):
        for token_idx in range(len(routing_weights[layer_idx][0])):
            expert_idx = routing_weights[layer_idx][token_idx].argmax()
            expert_token_model[expert_idx] += 1
    return expert_token_model

def get_routing_weights(model, tokenizer, prompts, layer_names):
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
    ], return_tensors="pt", padding=True).to('cuda')

    input_without_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        return_tensors="pt",
        padding=True,
    ).to('cuda')

    decoded_word = tokenizer.decode(inputs[0][len(input_without_prompt[0])-1:])
    assert len(decoded_word.split()) == 6

    attention_mask = torch.ones_like(inputs)
    attention_mask[inputs == tokenizer.pad_token_id] = 0

    hooks, layer_representations = _setup_hooks(model, layer_names)

    model_output = model(input_ids=inputs, attention_mask=attention_mask)

    for hook in hooks:
        hook.remove()

    routing_weights = model_output.routing_weights        
    routing_weights = np.concatenate([F.softmax(rw, dim=-1).detach().float().cpu().numpy() for rw in routing_weights])

    offset = len(input_without_prompt[0])-1
    routing_weights = routing_weights[:, offset:-1]
    layer_representations = {k: v[:, offset:-1] for k, v in layer_representations.items()}
    layer_representations = {k: v.detach().float().cpu().numpy() for k, v in layer_representations.items()}
    layer_representations = {
        k.replace("experts.0", "logic").replace("experts.1", "social").replace("experts.2", "world").replace("experts.3", "language"): v
        for k, v in layer_representations.items()
    }

    return routing_weights, layer_representations
   

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
    layer_names = [
        f"layers.{layer_num}.experts.{expert_num}"
        for layer_num in range(16)
        for expert_num in range(4)
    ]

    model_embeddings = []
    all_token_routing = []
    plot_token_routing = []
    for sentence in tqdm(sentences):
        routing_weights, model_sentence_embeddings = get_routing_weights(model, tokenizer, sentence, layer_names)
        token_routing = aggregate_routing_weights(routing_weights.copy())
        
        all_token_routing.append(np.roll(routing_weights, shift=1, axis=-1))
        plot_token_routing.append(np.roll(token_routing, shift=1))

        model_embeddings.append(model_sentence_embeddings)

    dump_pickle(all_token_routing, "outputs/token_routing_baseline.pkl")
    dump_pickle(model_embeddings, "outputs/model_embeddings_baseline.pkl")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

    plot_data = []
    experts = ["Language", "Logic", "Social", "World"]

    for expert_idx in range(len(experts)):
        for sample_idx in range(len(plot_token_routing)):
            percentage = plot_token_routing[sample_idx][expert_idx] / sum(plot_token_routing[sample_idx]) * 100
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


        



