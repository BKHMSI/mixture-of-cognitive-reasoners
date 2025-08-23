import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from dotenv import load_dotenv

from models.micro_llama import MiCRoLlama
from models.micro_olmo import MiCRoOLMo
from models.moe_llama import LlamaMoE
from utils.generate_html import generate_html

from transformers import AutoTokenizer, AutoConfig

load_dotenv()

def aggregate_routing_weights(routing_weights, tokenizer):
    all_token_map = []
    experts = ["Logic", "Social", "World", "Language"]
    expert_token_layer = np.zeros((len(routing_weights), len(routing_weights[0][0])), dtype=int)
    expert_token_model = np.zeros((len(experts)), dtype=int)
    for layer_idx in range(len(routing_weights)):
        token_map = []
        for token_idx in range(len(routing_weights[layer_idx][0])):
            decoded_token = tokenizer.decode(token_ids[0, token_idx].unsqueeze(0))
            expert_idx = routing_weights[layer_idx][0][token_idx].argmax()
            token_map.append((decoded_token, expert_idx))
            expert_token_layer[layer_idx][token_idx] = expert_idx
            expert_token_model[expert_idx] += 1
            
        all_token_map.append(token_map)

    mv_per_token = np.apply_along_axis(lambda x: np.bincount(x, minlength=4).argmax(), axis=0, arr=expert_token_layer)
    token_map = []
    for token_idx in range(len(mv_per_token)):
        decoded_token = tokenizer.decode(token_ids[0, token_idx].unsqueeze(0))
        token_map.append((decoded_token, mv_per_token[token_idx]))
    all_token_map.append(token_map)
    
    return all_token_map, expert_token_model

def generate_continuation(model, 
    tokenizer, 
    prompts, 
    max_tokens=1024,
    use_cache=True, 
    return_routing_weights=True
):

    if isinstance(prompts, str):
        prompts = [{"role": "user", "content": prompts}]

    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template([
        prompt for prompt in prompts
    ], return_tensors="pt", padding=True, add_generation_prompt=True).to('cuda')

    attention_mask = torch.ones_like(inputs)
    attention_mask[inputs == tokenizer.pad_token_id] = 0

    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask, 
        max_new_tokens=max_tokens,
        use_cache=use_cache,
        stop_strings=["</s>","<|eot_id|>", "<|im_start|>user"],
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0,
        top_p=1.0,
        do_sample=False,
    )
    
    if return_routing_weights:
        attention_mask = torch.ones_like(outputs)
        attention_mask[outputs == tokenizer.pad_token_id] = 0
        model_output = model(input_ids=outputs, attention_mask=attention_mask)
        torch.cuda.empty_cache()

        routing_weights = model_output.routing_weights        
        routing_weights = np.concatenate([
            F.softmax(rw, dim=-1)[:, inputs.shape[1]:].detach().float().cpu().numpy() 
            for rw in routing_weights
        ])
        
    else:
        routing_weights = None

    inputs_text = tokenizer.batch_decode(inputs, skip_special_tokens=False)

    generations = []
    for i, output in enumerate(outputs):
        decoded_output = tokenizer.decode(output, skip_special_tokens=False)
        decoded_output = decoded_output.replace(inputs_text[i], "")
        decoded_output = decoded_output.replace(tokenizer.pad_token, "").strip()
        decoded_output = decoded_output.replace("<|end_of_text|>", "").strip()
        decoded_output = decoded_output.replace("<|endoftext|>", "").strip()
        decoded_output = decoded_output.replace("<|eot_id|>", "").strip()
        decoded_output = decoded_output.replace("\n<|im_start|>user", "").strip()
        generations.append(decoded_output)

    gen_token_ids = outputs[:, inputs.shape[1]:]
    return (generations, routing_weights) if return_routing_weights else generations


def model_path(model_name):
    return {
        "llama-moe": ("ckpts/llama-moe-top1-tuluv3-1/checkpoint-29355", LlamaMoE),
        "micro-llama": ("bkhmsi/micro-llama", MiCRoLlama),
        "micro-llama-dpo": ("ckpts/llama-mxtr-1b-base-top1-tuluv3-15-dpo-4/checkpoint-36850", MiCRoLlama),
        "micro-olmo": ("bkhmsi/micro-olmo", MiCRoOLMo),
        "micro-smollm2-135m": ("ckpts/micro-smollm2-135m-2/stage-3/checkpoint-29355", MiCRoLlama),
        # "micro-smollm2-360m": ("ckpts/micro-smollm2-360m-1/stage-3/checkpoint-29355", MiCRoLlama),
        "micro-smollm2-360m": ("ckpts/micro-smollm2-360m-v2-1/stage-2/checkpoint-196", MiCRoLlama),
        "micro-smollm2-1.7b": ("ckpts/micro-smollm2-1.7b-2/stage-2/checkpoint-196", MiCRoLlama),
    }[model_name]

def build_model(config, args, use_cache=True):
    model_config = AutoConfig.from_pretrained(config["base-model"])
    model_config.config_path = f"configs/{args.config}"

    model_config.torch_dtype = torch.bfloat16
    model_config.use_bfloat16 = True
    model_config._attn_implementation = "flash_attention_2" #"flash_attention_2"
    model_config.use_cache = use_cache
    model_config.ablate = args.ablate.split(",")

    path, model_class = model_path(config["model"])

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    tokenizer.padding_side = "left"

    if "llama" in config["model"]:
        tokenizer.pad_token_id = 128004
    if "olmo" in config["model"]:
        tokenizer.pad_token_id = 100277
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<|assistant|>']})
    elif "smollm2" in config["model"]:
        tokenizer.pad_token_id = 2
    else:
        tokenizer.pad_token_id = 128004

    if "olmo" in config["model"]:
        model_config.vocab_size = len(tokenizer)
  
    model = model_class.from_pretrained(path, config=model_config, low_cpu_mem_usage=True)

    model.to(f'cuda')
    model = model.bfloat16()
    model.eval()
    return model, tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_llama.yml", help='path of config file')
    parser.add_argument('--prompt',  type=str,
                        default=None, help='input prompt')
    parser.add_argument('--ablate',  type=str,
                        default="none", help='expert to ablate')
    
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    use_cache = True
    model, tokenizer = build_model(config, args, use_cache=use_cache)

    prompt = args.prompt if args.prompt != "" and args.prompt is not None else "What is the Mixture of Experts (MoE) model?"
    # prompt = "Solve the following equation: 3x + 4 = 10."
    prompt = "Ahmed and Sarah are playing a game. Sarah loses the game and feels sad. Ahmed notices that Sarah is quiet and looking down.\n\nQuestion: What should Ahmed do next?"
    # prompt = "What is the capital of the country that is west of Egypt? Think step by step."
    prompt = "Sally and Anne are in a room together. Sally places her chocolate bar inside a blue box and then leaves the room. While she is gone, Anne moves the chocolate bar from the blue box to a red box. When Sally returns,\nQuestion: Where does Sally think the chocolate bar is? Let's think step by step."
    # prompt = "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nThink step by step and then finish your answer with \"The answer is X\" where X is the final answer."

    chat_prompt = [{'role': 'user', 'content': prompt}]

    print(chat_prompt[-1]["content"])
    print("=="*50)

    generation, routing_weights = generate_continuation(model, tokenizer, chat_prompt, max_tokens=150, use_cache=use_cache)
    print(generation)

    # token_map = aggregate_routing_weights(routing_weights, tokenizer)
    # generate_html(prompt, token_map)

