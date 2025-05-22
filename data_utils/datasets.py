import os
import json
import torch
import datasets
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv

from datasets import load_from_disk
from torch.utils.data import Dataset

load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", None)

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
def read_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data

class Tuluv3SftMixture(Dataset):
    def __init__(self):
        data_path = "tulu-3-sft-mixture-full"
        if DATA_PATH is not None:
            data_path = os.path.join(DATA_PATH, data_path)
        self.hf_dataset = load_from_disk(data_path)["train"]

class ExpertsDataset(Dataset):
    def __init__(self, config):
        self.random_labels = config.get("random-labels", False)

        if "Llama" in config["tokenizer"]:
            self.seq_seperator = "<|reserved_special_token_242|>"
        elif "Qwen" in config["tokenizer"]:
            self.seq_seperator = "<|file_sep|>"
        elif "OLMo" in config["tokenizer"]:
            self.seq_seperator = "<|extra_id_1|>"

        if self.random_labels:
            print(">> Using random labels for routing weights")

        self.domains = ["MD", "ToM", "DMN", "LN"]

        dirpath = Path(__file__).parent.parent

        paths = [
            "generations/openai_pseudo_labels_v2_cleaned.json",
            "generations/openai_bigtom_pseudo_labels_v3_cleaned.json",
        ]

        paths = [os.path.join(dirpath, path) for path in paths]
        
        self.datasets = []
        remove_datasets = []

        for path in paths:
            self.datasets.extend(read_json(path))

        print(f"> Loaded {len(self.datasets)} samples")

        self.filter_datasets(remove_datasets)
        self.build_dataset()

    def filter_datasets(self, remove_datasets):
        filtered_datasets = []
        existing_datasets = set()
        for dataset in self.datasets:
            if dataset["dataset"] not in remove_datasets:
                filtered_datasets.append(dataset)
                existing_datasets.add(dataset["dataset"])
            
        self.datasets = filtered_datasets
        print(f"> Existing datasets: {existing_datasets}")
        print(f"> Filtered samples: {len(self.datasets)}")

    def build_dataset(self):
        data = {"messages": [], "routing_weights": []}
        for index in tqdm(range(len(self.datasets))):
            user_prompt = self.datasets[index]["prompt"]
            if isinstance(user_prompt, list):
                system_prompt = user_prompt[0]["content"]
                user_prompt = system_prompt + '\n\n' + user_prompt[1]["content"]
                
            assistant_response: str = self.datasets[index]["assistant"]

            continuation = self.datasets[index]["continuation"]

            item_routing_weights = []
            for j, (indices, expert) in enumerate(continuation.items()):
                _, end_idx = indices.split("-")
                end_idx = int(end_idx) + j * len(self.seq_seperator)

                assistant_response = assistant_response[:end_idx] + self.seq_seperator + assistant_response[end_idx:]

                if not self.random_labels:
                    item_routing_weights.append(torch.eye(len(self.domains))[self.domains.index(expert)])
                else:
                    item_routing_weights.append(torch.eye(len(self.domains))[torch.randint(0, len(self.domains), (1,)).item()])

            data["messages"].append([
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ])

            data["routing_weights"].append(item_routing_weights)

        self.hf_dataset = datasets.Dataset.from_dict(data)

    def __len__(self):
        return len(self.hf_dataset)