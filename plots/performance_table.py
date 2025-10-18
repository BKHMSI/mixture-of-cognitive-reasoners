import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob

from plots.plot_performance import read_results, read_scores, read_json, task_metric_filter

from dotenv import load_dotenv

load_dotenv()

LM_EVAL_PATH = os.environ["LM_EVAL_PATH"]

def to_model_class(model_name):
    if "micro" in model_name or "mxtr" in model_name:
        return "MiCRo"
    elif "mob" in model_name:
        return "MoB"
    elif "moe" in model_name:
        return "MoE"
    else:
        return "Dense"

def read_lm_eval(model_name, task):
    data = read_results(model_name, task)
    metric, filter_ = task_metric_filter(task)
    task_name = list(data["results"].keys())[0]
    score = data["results"][task_name][f"{metric},{filter_}"]
    stderr = data["results"][task_name][f"{metric}_stderr,{filter_}"]
    return score, stderr

def find_lm_eval(model_name, task):
    paths = glob(f"{LM_EVAL_PATH}/results/{model_name}/*.json")
    for path in paths:
        data = read_json(path)
        task_name = list(data["results"].keys())[0]
        if task_name == "mmlu_continuation":
            continue 
        ablate = data["config"]["model_args"].split(",")[3].split("=")[1]
        if task_name.startswith(task) and (ablate == "none" or ablate == ""):
            metric, filter_ = task_metric_filter(task)
            score = data["results"][task_name][f"{metric},{filter_}"]
            stderr = data["results"][task_name][f"{metric}_stderr,{filter_}"]
            return score, stderr
    return None, None



if __name__ == "__main__":

    task_map = {
        "gsm8k": "GSM8K",
        "minerva_math": "Minerva Math",
        "mmlu": "MMLU",
        "bbh": "BBH",
        "arc_easy": "ARC Easy",
        "arc_challenge": "ARC Challenge",
        "hellaswag": "HellaSwag",
        "piqa": "PIQA",
    }

    tasks = [
        "gsm8k",
        "minerva_math",
        "mmlu",
        "bbh",
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
    ]

    cot_tasks = tasks[:4]

    base_models = [
        "SmollM2-135M",
        "SmollM2-360M",
        "Llama-3.2-1B",
        "Llama-3.2-3B",
    ]

    # models = [
    #     ("micro-smollm2-135m-2", "smollm2-mob-135m-1", "smollm2-135m-baseline-dense-1"),
    #     ("micro-smollm2-360m-1", "smollm2-mob-360m-1", "smollm2-360m-baseline-dense-v2-1"),
    #     ("llama-mxtr-1b-base-top1-tuluv3-15", "llama-mob-top1-tuluv3-plus-experts-2", "llama-dense-1"),
    #     ("micro-llama-3b-1", "llama-mob-3b-1", "llama-dense-3b-1"),
    # ]


    models = [
        ("micro-smollm2-moe-135m-1", "smollm2-moe-135m-1", "smollm2-135m-baseline-dense-1"),
        ("micro-smollm2-moe-360m-1", "smollm2-moe-360m-1", "smollm2-360m-baseline-dense-v2-1"),
        ("micro-moe-llama-2", "llama-moe-top1-tuluv3-plus-experts-1", "llama-dense-1"),
    ]


    plot_data = []
    for idx, model_list in enumerate(models):
        for model_name in model_list:
            for task in tasks:
                if task in cot_tasks[2:] and "llama" in model_name:
                    acc, stderr = read_scores(model_name, task)
                elif task in cot_tasks[:2] and "llama" in model_name:
                    acc, stderr = read_lm_eval(model_name, task)
                else:
                    acc, stderr = find_lm_eval(model_name, task)

                model_class = to_model_class(model_name)
                plot_data.append({
                    "Base Model": base_models[idx],
                    "Model": model_class,
                    "model_name": model_name,
                    "task": task_map[task],
                    "accuracy": acc*100,
                    "stderr": stderr*100
                })

    df = pd.DataFrame(plot_data)
    df = df.drop("model_name", axis=1)

    tasks = [task_map[t] for t in tasks]

    # Keep the MultiIndex (Base Model, Model)
    acc_wide = df.pivot_table(
        index=["Base Model", "Model"],
        columns="task",
        values="accuracy",
        aggfunc="mean",
    )
    se_wide = df.pivot_table(
        index=["Base Model", "Model"],
        columns="task",
        values="stderr",
        aggfunc="mean",
    )

    column_order = tasks
    acc_wide = acc_wide[column_order]
    se_wide = se_wide[column_order]


    # Format accuracy ± stderr
    def fmt_cell(a, s):
        if pd.isna(a):
            return ""
        return f"{a:.1f} $\\pm$ {s:.1f}"

    formatted = acc_wide.copy()
    for col in acc_wide.columns:
        formatted[col] = [
            fmt_cell(a, s) for a, s in zip(acc_wide[col].to_numpy(), se_wide[col].to_numpy())
        ]

    # Bold all column names
    formatted.columns = [f"\\textbf{{{col}}}" for col in formatted.columns]

    # Also bold the MultiIndex names if you want
    formatted.index = formatted.index.set_names(
        [f"\\textbf{{{name}}}" if name else name for name in formatted.index.names]
    )

    # Now export with multirow=True
    latex_table = formatted.to_latex(
        multirow=True,   # <-- this triggers correct \multirow output
        escape=False,    # allow \textbf and \pm
        caption="Accuracy (\\%) ± stderr across tasks for each model class.",
        label="tab:results"
    )

    with open("outputs/performance_table.tex", "w") as f:
        f.write(latex_table)
