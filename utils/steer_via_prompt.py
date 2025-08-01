import os
import yaml
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from dotenv import load_dotenv

from models.micro_llama import MiCRoLlama
from models.micro_olmo import MiCRoOLMo
from utils.generate_html import generate_html

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

load_dotenv()
CKPT_PATH = os.getenv("CKPT_PATH", "ckpts")

SYSTEM_PROMPT = {
    "social": (
        "You are a social reasoning expert. You excel at understanding human emotions, social cues, relationships, and motivations. "
        "You can interpret people's feelings, predict behaviors in social contexts, and offer empathetic insights."
    ),
    
    "world": (
        "You are an expert in real-world knowledge and common sense. You bring broad understanding of everyday scenarios, background knowledge, history, geography, and how the world works. "
        "You're great at contextualizing facts and making sense of real-life events."
    ),
    
    "language": (
        "You are a language expert. You have a deep command of grammar, vocabulary, semantics, and style. "
        "You write fluently, paraphrase with precision, and maintain coherence and clarity in all kinds of text."
    ),
    
    "logic": (
        "You are a logical reasoning expert. You're skilled at solving complex problems, analyzing patterns, doing math, and following strict logical rules. "
        "You think clearly, systematically, and accurately."
    ),

    "social-logic": (
        "You are an expert in both social reasoning and logical problem-solving. You can understand emotional dynamics and interpersonal behavior while also applying structured analysis and logic to resolve conflicts or explain outcomes."
    ),

    "world-logic": (
        "You combine deep real-world knowledge with logical precision. You can reason about historical events, interpret data, and draw conclusions based on both facts and logical analysis."
    ),

    "language-logic": (
        "You blend strong language skills with logical reasoning. You can write clearly and fluently while also solving puzzles, interpreting patterns, or clarifying ambiguous statements with precision."
    ),

    "social-world": (
        "You're an expert in both social understanding and world knowledge. You can reason about people’s emotions and intentions in real-world contexts, using cultural and historical awareness to interpret complex situations."
    ),

    "social-language": (
        "You are an expert in both social intelligence and language use. You interpret subtle emotional and interpersonal cues and express them clearly and naturally through well-crafted language."
    ),

    "world-language": (
        "You are an expert in world knowledge and language. You can explain complex real-world events and ideas in fluent, accessible, and well-structured text."
    ),

    "social-world-language": (
        "You integrate social reasoning, real-world understanding, and advanced language skills. "
        "You can explain emotionally complex, socially situated, and factually grounded scenarios with clarity and nuance."
    ),

    "social-world-logic": (
        "You are an expert in social reasoning, real-world knowledge, and logical thinking. "
        "You understand how people behave in realistic contexts and can analyze situations with clear, structured reasoning—though you focus less on expressive language."
    ),

    "world-language-logic": (
        "You combine strong real-world knowledge, fluent language use, and logical analysis. "
        "You can explain events clearly, reason through facts, and write with precision—without focusing on emotional or social interpretation."
    ),

    "social-language-logic": (
        "You are skilled in understanding emotions, communicating fluently, and solving problems with logic. "
        "You excel at interpreting people’s intentions, expressing ideas clearly, and thinking systematically—without emphasizing broader world knowledge or factual context."
    ),

    "social-world-language-logic": (
        "You are a comprehensive expert in social reasoning, world knowledge, language, and logic. You interpret emotions, contextualize facts, write fluently, and reason systematically—offering thoughtful and well-rounded responses to complex situations."
    ),
}


def generate_continuation(model, 
    tokenizer, 
    prompts, 
    max_tokens=1024,
    use_cache=True, 
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
        stop_strings=["</s>","<|eot_id|>"],
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )
    
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
    return generations

def build_model(config, args, use_cache=True):
    model_config = AutoConfig.from_pretrained(config["base-model"])
    model_config.config_path = f"configs/{args.config}"

    model_config.torch_dtype = torch.bfloat16
    model_config.use_bfloat16 = True
    model_config._attn_implementation = "flash_attention_2"
    model_config.use_cache = use_cache

    if hasattr(args, "ablate") and args.ablate is not None:
        model_config.ablate = args.ablate.split(",")

    if "baseline" in config["model"]:
        path = os.path.join(CKPT_PATH, config["resume-path"])
    elif "olmo" in config["model"]:
        path = "bkhmsi/micro-olmo"
    else:
        path = "bkhmsi/micro-llama"

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    tokenizer.padding_side = "left"

    if "llama" in config["model"]:
        tokenizer.pad_token_id = 128004
    if "olmo" in config["model"]:
        tokenizer.pad_token_id = 100277
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<|assistant|>']})
    else:
        tokenizer.pad_token_id = 128004

    if "baseline" in config["model"]:
        print(f">> Loading Baseline model: {config['resume-path']}")
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            config=model_config, 
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
    elif "olmo" in config["model"]:
        print(">> Loading Olmo2 model")
        model_config.vocab_size = len(tokenizer)
        model = MiCRoOLMo.from_pretrained(path, config=model_config, low_cpu_mem_usage=True)
    else:
        print(">> Loading Llama model")
        model = MiCRoLlama.from_pretrained(path, config=model_config, low_cpu_mem_usage=True)

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

    model, tokenizer = build_model(config, args, use_cache=True)

    prompt = args.prompt if args.prompt != "" and args.prompt is not None else "What is the Mixture of Experts (MoE) model?"
    # prompt = "Solve the following equation 2x+8=-2?
    # prompt = "Ahmed and Sarah are playing a game. Sarah loses the game and feels sad. Ahmed notices that Sarah is quiet and looking down.\n\nQuestion: What should Ahmed do next?"
    # prompt = "What is the capital of Egypt?"
    # prompt = "Sally and Anne are in a room together. Sally places her chocolate bar inside a blue box and then leaves the room. While she is gone, Anne moves the chocolate bar from the blue box to a red box. When Sally returns,\nQuestion: Where does Sally think the chocolate bar is? Let's think step by step."

    chat_prompt = [{'role': 'user', 'content': prompt}]

    print(chat_prompt[-1]["content"])
    print("=="*50)
    
    generation = generate_continuation(model, tokenizer, chat_prompt, max_tokens=384, use_cache=True)
    print(generation[0])
    