from collections import OrderedDict

import sys
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

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

def extract_expert_embeddings(model, tokenizer, prompts, layer_names):
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

    offset = len(input_without_prompt[0])-1
    layer_representations = {k: v[:, offset:-1] for k, v in layer_representations.items()}
    layer_representations = {k: v.detach().float().cpu().numpy() for k, v in layer_representations.items()}

    return np.stack(tuple(layer_representations.values())).squeeze()
   

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

    experts = ["logic", "social", "world", "language"]

    ablations = [
        ["social", "world", "language"], # logic
        ["logic", "world", "language"], # social
        ["logic", "social", "language"], # world
        ["logic", "social", "world"], # language
    ]

    for expert_idx, expert_ablations in enumerate(ablations):

        args.ablate = ",".join(expert_ablations)
        model, tokenizer = build_model(config, args, use_cache=True)

        sentences = data["sentence"].tolist()
        layer_names = [
            f"layers.{layer_num}.experts.{expert_idx}"
            for layer_num in range(16)
        ]

        model_embeddings = []
        for sentence in tqdm(sentences):
            model_sentence_embeddings = extract_expert_embeddings(model, tokenizer, sentence, layer_names)
            
            model_embeddings.append(model_sentence_embeddings)

        dump_pickle(model_embeddings, f"outputs/{experts[expert_idx]}_model_embeddings_baseline_sentences.pkl")

    
