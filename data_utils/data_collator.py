import json
import torch

from transformers import DataCollatorForLanguageModeling
from typing import Union, List, Any, Dict, Optional
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from tqdm import tqdm

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
    
def read_json(path):
    with open(path, 'r') as fin:
        data = json.load(fin)
    return data

class DataCollatorForCompletionLM(DataCollatorForLanguageModeling):
    
    def __init__(
        self,
        response_template: Optional[Union[str, List[int]]] = None,
        seq_seperator_id: Optional[int] = None,
        model_name: Optional[str] = None,
        random_router_labels: bool = False,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        if response_template is None or seq_seperator_id is None:
            if model_name and "qwen2" in model_name:
                # Default response template for Qwen2
                print(f">> Using Qwen2 response template")
                response_template = "<|im_start|>assistant"
                self.seq_seperator_id = 151664
            elif model_name and "olmo" in model_name:
                # Default response template for Olmo2
                print(f">> Using Olmo2 response template")
                response_template = "\n<|assistant|>\n"
                if len(self.tokenizer.encode(response_template, add_special_tokens=False)) > 1:
                    print(f">> WARNING: Response template is not a single token")
                    self.tokenizer.add_special_tokens({'additional_special_tokens': [response_template]})

                self.seq_seperator_id = 100266
            else:
                # Default response template for Llama-3
                print(f">> Using Llama3 response template")
                response_template = "<|start_header_id|>assistant<|end_header_id|>"
                self.seq_seperator_id = 128250
        else:
            self.seq_seperator_id = seq_seperator_id

        self.model_name = model_name
        self.random_router_labels = random_router_labels
        if self.random_router_labels:
            print(">> Using random labels for routing weights")

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        self.ignore_index = ignore_index
        
        
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        # Make first 0 token of attention mask to 1
        labels = input_ids.clone()

        # Where attention mask is 0, set the label to -100
        labels[attention_mask == 0] = self.ignore_index

        routing_weights_batch = []
        for i, seq in enumerate(input_ids):
            start_idx = 0
            for idx in torch.where(seq == self.response_token_ids[0])[0]:
                if seq[idx:idx+len(self.response_token_ids)].tolist() == self.response_token_ids:
                    start_idx = idx + len(self.response_token_ids)
                    break

            if start_idx == 0:
                print(f">> WARNING: No response token found in sequence {i}")
                
            labels[i, :start_idx] = self.ignore_index
            if "routing_weights" in batch and "baseline" not in self.model_name:
                routing_weights_idx = 0

                if self.random_router_labels:
                    routing_weights_seq = list(torch.eye(4)[torch.randint(4, size=(start_idx,))].long())
                else:
                    routing_weights_seq = [torch.tensor([0,0,0,1])] * start_idx

                for token_id in seq[start_idx:]:
                    if token_id == self.seq_seperator_id and routing_weights_idx+1 < len(batch["routing_weights"][i]):
                        routing_weights_idx += 1
                    elif routing_weights_idx+1 < len(batch["routing_weights"][i]):
                        if self.random_router_labels:
                            rand_label = torch.eye(4)[torch.randint(0, 4, (1,)).item()].long()
                            routing_weights_seq.append(rand_label)
                        else:
                            routing_weights_seq.append(batch["routing_weights"][i][routing_weights_idx].long().flatten())
                    else:
                        if self.random_router_labels:
                            rand_label = torch.eye(4)[torch.randint(0, 4, (1,)).item()].long()
                            routing_weights_seq.append(rand_label)
                        else:
                            routing_weights_seq.append(torch.tensor([0,0,0,1]))

                routing_weights_batch.append(torch.stack(routing_weights_seq))

        batch_size = input_ids.shape[0]
        labels = labels[input_ids != self.seq_seperator_id].reshape(batch_size, -1)
        attention_mask = attention_mask[input_ids != self.seq_seperator_id].reshape(batch_size, -1)
        input_ids = input_ids[input_ids != self.seq_seperator_id].reshape(batch_size, -1)

        if "routing_weights" in batch:
            routing_weights_batch = torch.stack(routing_weights_batch)[:, :-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "routing_weights": routing_weights_batch if "routing_weights" in batch else None,
            "labels": labels
        }