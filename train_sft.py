import os
import wandb
import yaml
import torch
import random
import argparse
import deepspeed
import numpy as np
import multiprocessing

from glob import glob
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import TrainerCallback
from transformers import set_seed as hf_set_seed
from trl import SFTConfig, SFTTrainer

from data_utils.data_collator import DataCollatorForCompletionLM
from data_utils.train_datasets import Tuluv3SftMixture, Tuluv3SftPlusExperts, ExpertsDataset, MeditronSFT

from models.micro_llama import MiCRoLlama
from models.micro_moe_llama import MiCRoLlamaMoE
from models.micro_olmo import MiCRoOLMo
from models.moe_llama import LlamaMoE
from models.micro_moe_olmo import MiCRoOlmoMoE

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.serialization.add_safe_globals([deepspeed.runtime.fp16.loss_scaler.LossScaler])
torch.serialization.add_safe_globals([deepspeed.runtime.zero.config.ZeroStageEnum])

_orig_load = torch.load
def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _load

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Also set transformers' RNGs
    hf_set_seed(seed)

class ZeroFillUnusedGradsCallback(TrainerCallback):
    def on_substep_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # print(f">>> Filling zero grads for unused parameters")
        for p in model.parameters():
            if p.requires_grad and p.grad is None:
                p.grad = torch.zeros_like(p.data, device=p.device, dtype=p.dtype)

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', True)

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config.yml", help='path of config file')
    parser.add_argument('--debug',  action='store_true',
                        help='Force debug')
    parser.add_argument('--wandb',  action='store_true',
                        help='Use WANDB')
    parser.add_argument('--cuda', type=int, default=None,
                        help='cuda device number')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()

    set_seed(seed=args.seed)

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
    with open(args.config, 'r', encoding="utf-8") as file:
        config_raw = file.read()
        config = yaml.load(config_raw, Loader=yaml.FullLoader)

    config["debug"] = args.debug 
    config["wandb"] = args.wandb if not args.debug else False

    print(">> Config: ", config)

    run_title = config["run-title"]
    save_path = config["save-path"]
    config["model"] = config.get("model", "mxtr-reasoners")

    print(">> Process: ", os.environ.get('LOCAL_RANK',-1))

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    tokenizer.padding_side = "right"
    num_new_tokens = 0

    vocab_size = len(tokenizer)
    if config["model"] == "llama-baseline":
        model_class = AutoModelForCausalLM
        tokenizer.pad_token_id = 128004
    elif config["model"] == "olmo-baseline":
        model_class = AutoModelForCausalLM
        tokenizer.pad_token_id = 100277
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<|assistant|>']})
        print(">> Adding <|assistant|> token")
    elif "smollm2-baseline" in config["model"]:
        print(">> Using SmolLM2 model baseline")
        model_class = AutoModelForCausalLM
        tokenizer.pad_token_id = 2
    elif "micro-llama" in config["model"]:
        print(">> Using MiCRo-Llama")
        model_class = MiCRoLlama
        tokenizer.pad_token_id = 128004
    elif "micro-moe-llama" in config["model"]:
        print(">> Using MiCRo-MoE-Llama")
        model_class = MiCRoLlamaMoE
        tokenizer.pad_token_id = 128004
    elif config["model"] == "micro-olmo-moe":
        print(">> Using MiCRo-OLMo-MoE")
        model_class = MiCRoOlmoMoE
        tokenizer.pad_token_id = 100277
        print(">> Adding <|assistant|> token")
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<|assistant|>']})
    elif "micro-smollm2-moe" in config["model"]:
        print(">> Using MiCRo-SmolLM2-MoE")
        model_class = MiCRoLlamaMoE
        tokenizer.pad_token_id = 2
    elif "smollm2-moe" in config["model"]:
        print(">> Using SmolLM2-MoE")
        model_class = LlamaMoE
        tokenizer.pad_token_id = 2
    elif "micro-smollm2" in config["model"]:
        print(">> Using MiCRo-SmolLM2")
        model_class = MiCRoLlama
        tokenizer.pad_token_id = 2
    elif "smollm2-mob" in config["model"]:
        print(">> Using SmolLM2-MoB")
        model_class = MiCRoLlama
        tokenizer.pad_token_id = 2
    elif "llama-mob" in config["model"]:
        print(">> Using Llama MoB")
        model_class = MiCRoLlama
        tokenizer.pad_token_id = 128004
    elif config["model"] == "llama-moe":
        print(">> Using Llama MoE")
        model_class = LlamaMoE
        tokenizer.pad_token_id = 128004
    elif config["model"] == "micro-olmo":
        print(">> Using MiCRo-OLMo")
        model_class = MiCRoOLMo
        tokenizer.pad_token_id = 100277
        print(">> Adding <|assistant|> token")
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<|assistant|>']})
    
    print(f">> Vocab size: {vocab_size} -> {len(tokenizer)}")

    model_config = AutoConfig.from_pretrained(config["base-model"])
    model_config.config_path = args.config
    model_config.ablate = []

    if config["resume"]:
        if "olmo" in config["model"]:
            model_config.vocab_size = len(tokenizer)
        print(f">> Resuming from {config['resume-path']}")
        model = model_class.from_pretrained(config["resume-path"], config=model_config)
        num_new_tokens = 0
    else:
        if "baseline" in config["model"]:
            model = model_class.from_pretrained(config["base-model"], config=model_config)
        else:
            model = model_class(model_config)
            model.load_pretrained(config["base-model"])
        if num_new_tokens > 0:
            print(">> Resizing embedding table")
            model.resize_token_embeddings(len(tokenizer))
            assert model.get_input_embeddings().weight.shape == model.get_output_embeddings().weight.shape

    print(model)

    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"> # Trainable Parameters: {num_params:,}")

    if config["dataset"] == "tuluv3":
        train_dataset = Tuluv3SftMixture(config)
        valid_dataset = None
        eval_strategy = "no"
        load_best_model_at_end = False
    elif config["dataset"] == "tuluv3-plus-experts":
        train_dataset = Tuluv3SftPlusExperts(config)
        valid_dataset = None
        eval_strategy = "no"
        load_best_model_at_end = False
    elif config["dataset"] == "medical-sft":
        train_dataset = MeditronSFT(config)
        valid_dataset = None
        eval_strategy = "no"
        load_best_model_at_end = False
    elif config["dataset"] == "experts":
        train_dataset = ExpertsDataset(config)
        valid_dataset = None
        eval_strategy = "no"
        load_best_model_at_end = False

    if WANDB_API_KEY is not None and config["wandb"]:
        report_to = "wandb"
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="mixture-of-cog-reasoners", name=run_title, config=config)
    else:
        report_to = "none"
        print(">> WANDB is not enabled")

    if config.get("gradient-checkpointing", False):
        print(f"> Enabling Gradient Checkpointing!")
        gradient_checkpointing = True 
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        gradient_checkpointing = False

    if "stage-3" not in run_title:
        save_strategy = "epoch"
        save_steps = 1
    else:
        save_strategy = "steps"
        save_steps = config.get("save-steps", 0.1)

    training_args = SFTConfig(
        output_dir=save_path,
        eval_strategy=eval_strategy,
        eval_steps=0.1,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=1,
        load_best_model_at_end=load_best_model_at_end,
        dataloader_num_workers=8 if not config["debug"] else 0,
        learning_rate=config["learning-rate"],
        per_device_train_batch_size=config["batch-size"],
        per_device_eval_batch_size=config["batch-size"],
        gradient_accumulation_steps=config["gradient-accumulation-steps"],
        gradient_checkpointing=gradient_checkpointing,
        num_train_epochs=config["num-epochs"],
        weight_decay=0.01,
        report_to=report_to,
        bf16=True,
        ddp_find_unused_parameters=True,
        dataloader_drop_last=False,
        dataloader_pin_memory=False,
        group_by_length=False,
        lr_scheduler_type=config["lr-scheduler"],
        warmup_ratio=config["warmup-ratio"],
        max_seq_length=config["max-length"],
        remove_unused_columns=True,
        save_safetensors=True,
        ddp_broadcast_buffers=False, 
        torch_compile=False,
    )
    
    resume_from_ckpt = len(glob(os.path.join(save_path, "checkpoint-*"))) > 0

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.hf_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCompletionLM(tokenizer=tokenizer, model_name=config["model"], random_router_labels=config["random-labels"]),
        callbacks=[ZeroFillUnusedGradsCallback()],
    )
    
    trainer.train(resume_from_checkpoint=resume_from_ckpt)
