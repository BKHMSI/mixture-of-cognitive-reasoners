import os
import wandb
import yaml
import argparse
import multiprocessing

from glob import glob
from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from data_utils.data_collator import DataCollatorForCompletionLM
from data_utils.train_datasets import Tuluv3SftMixture, ExpertsDataset, MeditronSFT

from models.micro_llama import MiCRoLlama
from models.micro_olmo import MiCRoOLMo
from models.moe_llama import LlamaMoE

load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    elif config["model"] == "olmo-baseline":
        model_class = AutoModelForCausalLM
        tokenizer.pad_token_id = 100277
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['\n<|assistant|>\n']})
        print(">> Adding <|assistant|> token")
    elif config["model"] == "micro-llama":
        print(">> Using MiCRo-Llama")
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
        num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['\n<|assistant|>\n']})
    
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
        if num_new_tokens > 0 and "stage-1" in run_title:
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

    if "stage-3" not in run_title:
        save_strategy = "epoch"
        save_steps = 1
    else:
        save_strategy = "steps"
        save_steps = config.get("save-steps", 0.25)

    training_args = SFTConfig(
        output_dir=save_path,
        eval_strategy=eval_strategy,
        eval_steps=0.1,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=50,
        load_best_model_at_end=load_best_model_at_end,
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
        max_seq_length=config["max-length"],
        remove_unused_columns=True,
    )
    
    resume_from_ckpt = len(glob(os.path.join(save_path, "checkpoint-*"))) > 0

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.hf_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCompletionLM(tokenizer=tokenizer, model_name=config["model"], random_router_labels=config["random-labels"]),
    )
    
    trainer.train(resume_from_checkpoint=resume_from_ckpt)
