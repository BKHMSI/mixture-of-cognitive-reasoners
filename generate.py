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
from models.micro_moe_llama import MiCRoLlamaMoE
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

def write_txt(filename, text):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(text)

def model_path(model_name):
    return {
        # MiCRo-Llama
        "micro-llama-1b": ("bkhmsi/micro-llama-1b", MiCRoLlama),
        "micro-llama-3b": ("bkhmsi/micro-llama-3b", MiCRoLlama),
        "micro-llama-1b-dpo": ("bkhmsi/micro-llama-1b-dpo", MiCRoLlama),

        # MiCRo-MoE-Llama
        "micro-moe-llama-1b": ("bkhmsi/micro-moe-llama-1b", MiCRoLlamaMoE),
        
        # MiCRo-OLMo
        "micro-olmo": ("bkhmsi/micro-olmo-1b", MiCRoOLMo),

        # MiCRo-SmolLM2
        "micro-smollm2-135m": ("bkhmsi/micro-smollm2-135m", MiCRoLlama),
        "micro-smollm2-360m": ("bkhmsi/micro-smollm2-360m", MiCRoLlama),

        # MiCRo-MoE-SmolLM2
        "micro-moe-smollm2-135m": ("bkhmsi/micro-moe-smollm2-135m", MiCRoLlamaMoE),
        "micro-moe-smollm2-360m": ("bkhmsi/micro-moe-smollm2-360m", MiCRoLlamaMoE),

        # Training Checkpoints
        "micro-llama-ckpt-0": ("/ckpts/llama-mxtr-1b-base-top1-tuluv3-15/stage-2/checkpoint-194", MiCRoLlama),
        "micro-llama-ckpt-1": ("/ckpts/llama-mxtr-1b-base-top1-tuluv3-15/stage-3/checkpoint-7339", MiCRoLlama),
        "micro-llama-ckpt-2": ("/ckpts/llama-mxtr-1b-base-top1-tuluv3-15/stage-3/checkpoint-14678", MiCRoLlama),
        "micro-llama-ckpt-3": ("/ckpts/llama-mxtr-1b-base-top1-tuluv3-15/stage-3/checkpoint-22017", MiCRoLlama),
        "micro-llama-ckpt-4": ("/ckpts/llama-mxtr-1b-base-top1-tuluv3-15/stage-3/checkpoint-29354", MiCRoLlama),

        # Other
        "llama-moe": ("ckpts/llama-moe-top1-tuluv3-plus-experts-1/checkpoint-29550", LlamaMoE),
        "llama-mob": ("ckpts/llama-mob-top1-tuluv3-plus-experts-2/checkpoint-29549", MiCRoLlama),        
        "micro-moe-llama-3b": ("ckpts/micro-moe-llama-3b-1/stage-2/checkpoint-196", MiCRoLlamaMoE),
        "micro-moe-llama-top4": ("ckpts/micro-moe-llama-top4-1/stage-3/checkpoint-29354", MiCRoLlamaMoE),
        "micro-smollm2-1.7b": ("ckpts/micro-smollm2-1.7b-2/stage-3/checkpoint-29355", MiCRoLlama),
        "smollm2-mob-1.7b": ("ckpts/smollm2-mob-1.7b-1/checkpoint-29550", MiCRoLlama),
        "micro-moe-smollm2-1.7b": ("ckpts/micro-smollm2-moe-1.7b-1/stage-2/checkpoint-196", MiCRoLlamaMoE),
    }[model_name]

def build_model(config, args, use_cache=True):
    model_config = AutoConfig.from_pretrained(config["base-model"])
    model_config.config_path = f"../mixture-of-reasoners/configs/{args.config}"

    model_config.torch_dtype = torch.bfloat16
    model_config.use_bfloat16 = True
    model_config._attn_implementation = "flash_attention_2"
    model_config.use_cache = use_cache
    model_config.ablate = args.ablate.split(",")
    print(f"> Ablating experts: {model_config.ablate}")

    if config["top-k-experts"] != 1:
        config["model"] += f"-top{config['top-k-experts']}"

    path, model_class = model_path(config["model"])
    print(f"> Loading model from {path}")

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
    parser.add_argument('--output_file',  type=str,
                        default='data/output.txt', help='output file to save generations')
    
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    use_cache = True
    model, tokenizer = build_model(config, args, use_cache=use_cache)

    prompt = args.prompt if args.prompt != "" and args.prompt is not None else "What is the Mixture of Experts (MoE) model?"

    chat_prompt = [{'role': 'user', 'content': prompt}]

    print(chat_prompt[-1]["content"])
    print("=="*50)

    generation, routing_weights = generate_continuation(model, tokenizer, chat_prompt, max_tokens=512, use_cache=use_cache)
    print(generation[0])

    write_txt(args.output_file, generation[0])

