import os
import wandb
import yaml
import argparse

from dotenv import load_dotenv
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoConfig

from data_utils.datasets import Tulu2p5DPO
from glob import glob

from models.micro_llama import MiCRoLlama
from models.micro_olmo import MiCRoOLMo

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config',  type=str,
                        default="config_micro_olmo.yml", help='path of config file')
    parser.add_argument('--debug',  action='store_true',
                        help='Force debug')
    parser.add_argument('--wandb',  action='store_true',
                        help='Use WANDB')
    parser.add_argument('--cuda', type=int, default=None,
                        help='cuda device number')
    args = parser.parse_args()

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
    elif config["model"] == "olmo2-baseline":
        model_class = AutoModelForCausalLM
        tokenizer.pad_token_id = 100277
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['\n<|assistant|>\n']})
        print(">> Adding <|assistant|> token")
    elif config["model"] == "micro-llama":
        print(">> Using LlamaModelMXTR")
        model_class = MiCRoLlama
        tokenizer.pad_token_id = 128004
    elif config["model"] == "micro-olmo":
        print(">> Using Olmo2ModelMXTR")
        model_class = MiCRoOLMo
        tokenizer.pad_token_id = 100277
        print(">> Adding <|assistant|> token")
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['\n<|assistant|>\n']})
    
    print(f">> Vocab size: {vocab_size} -> {len(tokenizer)}")

    model_config = AutoConfig.from_pretrained(config["base-model"])
    model_config.config_path = args.config
    model_config.ablate = []

    if "olmo" in config["model"]:
        model_config.vocab_size = len(tokenizer)
    
    print(f">> Resuming from {config['resume-path']}")
    model = model_class.from_pretrained(config["resume-path"], config=model_config)

    train_dataset = Tulu2p5DPO(config)

    print(model)

    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"> # Trainable Parameters: {num_params:,}")

    if WANDB_API_KEY is not None and config["wandb"]:
        report_to = "wandb"
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project="mixture-of-cog-reasoners", name=run_title, config=config)
    else:
        report_to = "none"
        print(">> WANDB is not enabled")

    training_args = DPOConfig(
        output_dir=save_path, 
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=50,
        load_best_model_at_end=False,
        dataloader_num_workers=8,
        learning_rate=config["learning-rate"],
        per_device_train_batch_size=config["batch-size"],
        per_device_eval_batch_size=config["batch-size"],
        gradient_accumulation_steps=config["gradient-accumulation-steps"],
        gradient_checkpointing=False,
        num_train_epochs=config["num-epochs"],
        weight_decay=0.01,
        report_to=report_to,
        bf16=True,
        ddp_find_unused_parameters=False,
        lr_scheduler_type=config["lr-scheduler"],
        warmup_ratio=config["warmup-ratio"],
        remove_unused_columns=True,
    )

    trainer = DPOTrainer(model=model, 
        args=training_args, 
        processing_class=tokenizer, 
        train_dataset=train_dataset.hf_dataset
    )

    trainer.train()