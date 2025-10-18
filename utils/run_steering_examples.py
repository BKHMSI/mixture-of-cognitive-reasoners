import os
import json

def read_jsonl(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    data = read_jsonl('data/steering_prompts.jsonl')
    
    ablations = ["social,world", "logic,world", "logic,social", "none"]
    config_path = "config_micro_llama.yml"
    for idx, row in enumerate(data):
        prompt = row["user"]
        for ablation in ablations:
            savepath = f"data/steering/micro_llama_1b_steering_examples_{idx}_{ablation.replace(',', '_')}.txt"
            command = f"python generate.py --config {config_path} --prompt '{prompt}' --ablate {ablation} --output_file {savepath}"
            os.system(command)
