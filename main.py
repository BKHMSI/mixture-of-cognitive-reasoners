import os
import yaml
import torch
import argparse

from glob import glob
from copy import deepcopy

if __name__ == "__main__":

    num_gpus = torch.cuda.device_count()

    parser = argparse.ArgumentParser(description='Paramaters')
    parser.add_argument('-c', '--config', type=str, default="config.yml", help='path of config file')
    parser.add_argument('--start-stage', type=int, default=1, help='start stage')
    parser.add_argument("--debug", action="store_true", help="Force debug")
    parser.add_argument("--stage-1-only", action="store_true", help="Train stage-1 only")
    parser.add_argument("--stage-2-only", action="store_true", help="Train stage-2 only")
    parser.add_argument("--dpo", action="store_true", help="Train using DPO")
    args = parser.parse_args()

    with open(f"configs/{args.config}", 'r', encoding="utf-8") as file:
        config_raw = file.read()
        base_config = yaml.load(config_raw, Loader=yaml.FullLoader)

    run_title = base_config["run-title"]
    exp_path = os.path.join(base_config["save-path"], base_config["run-title"])

    if not os.path.isdir(exp_path): 
        os.mkdir(exp_path)

    accelerate_config_file = "accelerate_config.yml"

    if args.dpo:
        config_path = os.path.join(exp_path, "config.yml")
        base_config["save-path"] = exp_path
        with open(config_path, 'w', encoding="utf-8") as fout:
            fout.write(yaml.dump(base_config))

        if not args.debug:
            command = f"bash scripts/train.sh {config_path} {num_gpus} {accelerate_config_file} dpo"
        else:
            command = f"python train_dpo.py --debug -c {config_path}"
        
        os.system(command)
    else:

        for stage_i in range(args.start_stage,4):
            stage_config = deepcopy(base_config)
            stage_str = f"stage-{stage_i}-medical-sft-2"
            if not os.path.isdir(os.path.join(exp_path, stage_str)): 
                os.mkdir(os.path.join(exp_path, stage_str))

            if "baseline" in base_config["model"] and stage_i == 2:
                continue 
            
            stage_config["run-title"] += f"_{stage_str}"
            if stage_i in [1,2]:
                stage_config["dataset"] = "experts"
                stage_config["num-epochs"] = base_config.get("stage-2-epochs", 2)
                stage_config["top-k-experts"] = base_config.get("stage-2-top-k-experts", 2)
            else:
                stage_config["dataset"] = "medical-sft" #tuluv3"
                stage_config["num-epochs"] = base_config["num-epochs"]
                stage_config["top-k-experts"] = 1

            if stage_i in [2,3] and not stage_config["resume"]:
                stage_config["resume"] = True
                if "baseline" in base_config["model"]:
                    checkpoints = glob(os.path.join(exp_path, f"stage-1", "checkpoint-*"))
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
                else:
                    checkpoints = glob(os.path.join(exp_path, f"stage-{stage_i-1}", "checkpoint-*"))
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))

                stage_config["resume-path"] = checkpoints[-1]

            if stage_i == 1:
                stage_config["trainable"] = ["reasoners"]
            elif stage_i == 2:
                stage_config["trainable"] = ["experts-router"]
            else:
                stage_config["trainable"] = ["model", "reasoners", "experts-router"]
            
            if stage_i == 1 and "baseline" not in base_config["run-title"]:
                stage_config["use-router"] = False 
            else:
                stage_config["use-router"] = True

            if stage_i == 3:
                stage_config["loss"] = "all"

            stage_config["save-path"] = os.path.join(exp_path, stage_str)

            config_path = os.path.join(exp_path, stage_str, "config.yml")
            with open(config_path, 'w', encoding="utf-8") as fout:
                fout.write(yaml.dump(stage_config))

            if not args.debug:
                command = f"bash scripts/train.sh {config_path} {num_gpus} {accelerate_config_file} sft"
            else:
                command = f"python train_sft.py --debug -c {config_path}"

            os.system(command)