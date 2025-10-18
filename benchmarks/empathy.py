import os
import yaml
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from dotenv import load_dotenv

from generate import build_model, generate_continuation

load_dotenv()

fewshot_samples = [
    {
        'role': 'user',
        'content': "For the sentence: \"When I read about the floods in Bangladesh, I was shocked by the scale of destruction. I can't imagine losing my home like that. It's heartbreaking to see families wading through water with nothing but the clothes on their backs. No one deserves to suffer like that.\", is it expressing empathy?",
    },
    {
        'role': 'assistant',
        'content': "The speaker is expressing concern and emotional distress for people suffering due to a natural disaster. They acknowledge the pain of others and show a desire for their well-being. This indicates an empathetic response. The answer is Yes."
    },
    {
        'role': 'user',
        'content': "For the sentence: \"Some people just don't manage their time well, and that's why they end up failing. I mean, everyone has the same 24 hours. If you can't figure it out, maybe you don't deserve success.\", is it expressing empathy?",
    },
    {
        'role': 'assistant',
        'content': "The speaker is judgmental and lacks understanding of others' circumstances. There is no attempt to relate to others' struggles or express care, which shows a lack of empathy. The answer is No."

    },
    {
        'role': 'user',
        'content': "For the sentence: \"I saw a story about a young girl walking miles every day just to get clean water for her family. It made me realize how lucky I am, and I felt a deep sadness knowing people still go through that. I wish there were more ways we could help.\", is it expressing empathy?"
    },
    {
        'role': 'assistant',
        'content': "The speaker reflects on someone else's hardship with sadness and a desire to help. They are emotionally moved and express concern, which are key indicators of empathy. The answer is Yes."
    }
]

def read_dataset(path, task):
    with open(f"{path}/{task}/test_text.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().splitlines()]

    with open(f"{path}/{task}/test_labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() if task[-4:]!='span' else eval(line) for line in f.read().splitlines()]
    
    return Dataset.from_dict({"text": lines, "label": labels})

def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def data_iterator(data, batch_size = 64):
    n_batches = np.ceil(len(data) / batch_size)
    for idx in range(n_batches):
        x = data[idx *batch_size:(idx+1) * batch_size]
        yield x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_llama.yml", help='path of config file')
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    use_cache = True
    model_name = config["run-title"]
    args.ablate = "none"
    save_path = f"outputs/{model_name}/generations/empathy.json"
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    model, tokenizer = build_model(config, args, use_cache=use_cache)

    task = "empathy#empathy_bin"
    batch_size = 32
    max_new_tokens = 128

    dataset_dir_path = os.getenv("BENCHMARKS_DATA_PATH") + "/SOCKET/experiments/zeroshot"
    ppt_df = pd.read_csv(f'{dataset_dir_path}/socket_prompts.csv')

    task_info = ppt_df[ppt_df['task']==task]

    dataset_dir_path = os.getenv("BENCHMARKS_DATA_PATH") + "/SOCKET/SOCKET"
    dataset = read_dataset(f"{dataset_dir_path}/SOCKET_DATA", task)
    
    ppt_template = task_info['question'].item()
    task_labels = ['not empathy', 'empathy']

    prompts = []
    for text in dataset['text']:
        prompts.append(ppt_template.replace("{text}", text))
    dataset = dataset.add_column('prompt', prompts)
    
    d_labels = [it.replace('-', ' ').lower() for it in task_labels]
    labels = task_info['options']
    label2id = {l: i for i, l in enumerate(labels)}

    # SOCKET Standard
    dataset = dataset[:1000]
    data_iter = data_iterator(dataset['prompt'], batch_size)
    num_batches = int(np.ceil(len(dataset['prompt']) / batch_size))

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        predictions = []

    start_batch = int(np.ceil(len(predictions) / batch_size))
    
    for batch_idx in tqdm(range(start_batch, num_batches)):
        batch = dataset['prompt'][batch_idx * batch_size:(batch_idx + 1) * batch_size]
        chat_batch = [fewshot_samples + [{'role': 'user', 'content': it}] for it in batch]
        outputs = generate_continuation(
            model=model,
            tokenizer=tokenizer,
            prompts=chat_batch,
            max_tokens=max_new_tokens,
            use_cache=use_cache,
            return_routing_weights=False
        )

        for idx, output in enumerate(outputs):
            predictions += [{
                "prompt": batch[idx],
                "response": output
            }]

        write_json(save_path, predictions)

    
