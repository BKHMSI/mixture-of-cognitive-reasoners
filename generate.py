import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from dotenv import load_dotenv

from models.micro_llama import MiCRoLlama
from models.micro_olmo import MiCRoOLMo

from transformers import AutoTokenizer, AutoConfig

load_dotenv()
DIR_PATH = os.getenv("CKPT_PATH", './ckpts')

def generate_continuation(model, tokenizer, prompts, max_tokens=1024, use_cache=True, return_routing_weights=True):

    if isinstance(prompts, str):
        prompts = [prompts]

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
        stop_strings=["</s>","<|eot_id|>"],
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )
    
    
    if return_routing_weights:
        attention_mask = torch.ones_like(outputs)
        attention_mask[outputs == tokenizer.pad_token_id] = 0
        model_output = model(input_ids=outputs, attention_mask=attention_mask)
        torch.cuda.empty_cache()

        routing_weights = model_output.routing_weights        
        routing_weights = [F.softmax(rw, dim=-1)[:, inputs.shape[1]:].detach().float().cpu().numpy() for rw in routing_weights]
        # routing_weights = [F.softmax(rw, dim=-1).detach().float().cpu().numpy() for rw in routing_weights]
        loss_indices = None
    else:
        routing_weights = None
        loss_indices = None

    inputs_text = tokenizer.batch_decode(inputs, skip_special_tokens=False)

    generations = []
    for i, output in enumerate(outputs):
        decoded_output = tokenizer.decode(output, skip_special_tokens=False)
        decoded_output = decoded_output.replace(inputs_text[i], "")
        decoded_output = decoded_output.replace(tokenizer.pad_token, "").strip()
        decoded_output = decoded_output.replace("<|end_of_text|>", "").strip()
        decoded_output = decoded_output.replace("<|endoftext|>", "").strip()
        decoded_output = decoded_output.replace("<|eot_id|>", "").strip()
        generations.append(decoded_output)

    gen_token_ids = outputs[:, inputs.shape[1]:]
    return generations, gen_token_ids, routing_weights, loss_indices

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_llama.yml", help='path of config file')
    
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    model_config = AutoConfig.from_pretrained(config["base-model"])
    model_config.config_path = f"configs/{args.config}"

    use_cache = True
    model_config.torch_dtype = torch.bfloat16
    model_config.use_bfloat16 = True
    model_config._attn_implementation = "flash_attention_2"
    model_config.use_cache = use_cache
    model_config.ablate = []

    if "olmo" in config["model"]:
        model_name = "olmo-mxtr-1b-base-top1-tuluv3-3"
    else:
        model_name = "llama-mxtr-1b-base-top1-tuluv3-15"
        # model_name = "llama-mxtr-1b-base-top1-tuluv3-random-labels-5"

    path  = f"{DIR_PATH}/{model_name}/stage-3/checkpoint-29354"

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    tokenizer.padding_side = "left"
    if "llama" in config["model"]:
        tokenizer.pad_token_id = 128004
    if "olmo" in config["model"]:
        tokenizer.pad_token_id = 100277
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<|assistant|>']})
    else:
        tokenizer.pad_token_id = 128004

    if "olmo" in config["model"]:
        print(">> Loading Olmo2 model")
        model_config.vocab_size = len(tokenizer)
        model = MiCRoOLMo.from_pretrained(path, config=model_config, low_cpu_mem_usage=True)
    else:
        print(">> Loading Llama model")
        model = MiCRoLlama.from_pretrained(path, config=model_config, low_cpu_mem_usage=True)

    model.to(f'cuda')
    model = model.bfloat16()
    model.eval()

    prompt = "What is the Mixture of Experts (MoE) model?"

    chat_prompt = [{'role': 'user', 'content': prompt}]

    print(chat_prompt[-1]["content"])
    print("=="*50)
    
    generation, token_ids, routing_weights, _ = generate_continuation(model, tokenizer, chat_prompt, max_tokens=384, use_cache=use_cache)
    print(generation[0])
    