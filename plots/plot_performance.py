import os
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

import scipy.stats as stats
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

from dotenv import load_dotenv


load_dotenv()

LM_EVAL_PATH = os.environ["LM_EVAL_PATH"]

model_names_map = {
    "micro-moe-llama-2": "MiCRo-Llama-MoE",
    "llama-moe-top1-tuluv3-plus-experts-1": "Llama-MoE",
    "llama-baseline-1b-base-tuluv3-1": "Llama-Dense",
    "llama-dense-1": "Llama-Dense",
    "llama-mxtr-1b-base-top1-tuluv3-15": "MiCRo-Llama-MoB",
    "llama-mxtr-1b-base-top1-tuluv3-15-social": "MiCRo-Llama-MoB",
    "llama-mxtr-1b-base-top1-tuluv3-15-best-ablation": "MiCRo-Llama-MoB",
    "llama-mob-top1-tuluv3-plus-experts-2": "Llama-MoB",
    "llama-mob-top1-tuluv3-plus-experts-2-best-ablation": "Llama-MoB",
    "micro-smollm2-moe-135m-1": "MiCRo-SmolLM2-135M-MoE",
    "micro-smollm2-135m-2": "MiCRo-SmolLM2-135M-MoB",
    "smollm2-mob-135m-1": "SmolLM2-135M-MoB",
    "smollm2-moe-135m-1": "SmolLM2-135M-MoE",
    "micro-smollm2-moe-360m-1": "MiCRo-SmolLM2-360M-MoE",
    "micro-smollm2-360m-1": "MiCRo-SmolLM2-360M-MoB",
    "micro-smollm2-360m-v2-1": "MiCRo-SmolLM2-360M-MoB",
    "smollm2-mob-360m-1": "SmolLM2-360M-MoB",
    "smollm2-moe-360m-1": "SmolLM2-360M-MoE",
    "micro-smollm2-1.7b-2": "MiCRo-SmolLM2-1.7B-MoB",
    "micro-smollm2-1.7b-2-social": "MiCRo-SmolLM2-1.7B-MoB",
    "smollm2-mob-1.7b-1": "SmolLM2-1.7B-MoB",
    "smollm2-1.7b-dense-1": "SmolLM2-1.7B-Dense",
    "llama-dense-3b-1": "Llama-3.2-3B-Dense",
    "micro-llama-3b-1": "MiCRo-Llama-3B-Dense",
    "micro-llama-3b-1-social": "MiCRo-Llama-3B-Dense",
    "micro-llama-3b-1-best-ablation": "MiCRo-Llama-3B-Dense",
    "llama-mob-3b-1": "Llama-3.2-3B-MoB",
    "llama-mob-3b-1-best-ablation": "Llama-3.2-3B-MoB",
}

task_name_map = {
    "gsm8k": "GSM8K",
    "minerva_math": "Minerva-Math",
    "mmlu": "MMLU",
    "bbh": "BBH",
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "piqa": "PIQA",
    "Average": "Average",
}

task_name_count_map = {
    "GSM8K": 1319,
    "Minerva-Math": 5000,
    "MMLU": 14042,
    "BBH": 6511,
    "ARC-Challenge": 1172,
}

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def read_scores(model_name, task):
    return {
        "micro-moe-llama-2": {
            "mmlu": (0.3035180174, 0.0038233308),
            "bbh": (0.3006, 0.0052),
        },
        "llama-moe-top1-tuluv3-plus-experts-1": {
            "mmlu": (0.2568010255, 0.0036431781),
            "bbh": (0.2858, 0.0051),
        },
        "llama-baseline-1b-base-tuluv3-1": {
            "mmlu": (0.2838626976, 0.0037525114),
            "bbh": (0.3030, 0.0051),
        },
        "llama-mxtr-1b-base-top1-tuluv3-15": {
            "mmlu": (0.3119925936, 0.0038523935),
            "bbh": (0.2984, 0.0052),
        },
        "llama-mxtr-1b-base-top1-tuluv3-15-social": {
            "mmlu": (0.3153396952, 0.0038667526),
            "bbh": (0.3059, 0.0053),
        },
        "llama-mxtr-1b-base-top1-tuluv3-15-best-ablation": {
            "mmlu": (0.3352798747, 0.0039314489),
            "bbh": (0.3277530333, 0.0053083578),
        },
         "llama-mob-top1-tuluv3-plus-experts-2": {
            "mmlu": (0.2706879362, 0.0037042326),
            "bbh": (0.2742, 0.0050),
        },
        "llama-mob-top1-tuluv3-plus-experts-2-best-ablation": {
            "mmlu": (0.31562, 0.00387),
            "bbh": (0.3037935801, 0.0052330903),
        },

        "micro-smollm2-moe-135m-1":{
            "mmlu": (0.2215496368038741, 0.0034914137489268035),
            "bbh": (0.24512363692213177, 0.004845741753675477),
        },
        "micro-smollm2-135m-2":{
            "mmlu": (0.22482552342971088,  0.003511843222954989),
            "bbh": (0.2544923974811857, 0.004959231611146648),
        },
        "smollm2-mob-135m-1":{
            "mmlu": (0.21912832929782083, 0.0034738445938305856),
            "bbh": (0.2349869451697128, 0.004810702753742449),
        },
        "smollm2-moe-135m-1": {
            "mmlu": (0.22261786070360348, 0.003489423088337277),
            "bbh": (0.24573798187682383, 0.004902436966361801),
        },
        "llama-dense-3b-1": {
            "mmlu": (0.4862555192, 0.0040350879),
            "bbh": (0.4414, 0.0057),
        },
        "llama-dense-1": {
            "mmlu": (0.2974647486, 0.0037830828),
            "bbh": (0.3036, 0.0052),
        },

        "llama-mob-3b-1": {
            "mmlu": (0.4522147842, 0.0040331582),
            "bbh": (0.422362, 0.005511),
        },

        "llama-mob-3b-1-best-ablation": {
            "mmlu": (0.4821962683, 0.0040404400),
            "bbh": (0.4529258178, 0.0055690734)
        },

        "micro-llama-3b-1": {
            "mmlu": (0.4539239425, 0.0040239122),
            "bbh": (0.419751, 0.005612),
        },

        "micro-llama-3b-1-social": {
            "mmlu": (0.4568437545, 0.0040140974),
            "bbh": (0.421748, 0.005603),
        },

        "micro-llama-3b-1-best-ablation": {
            "mmlu": (0.4872525281, 0.0040696721),
            "bbh": (0.4497005068, 0.0056338338),
        },

        "micro-moe-llama-2": {
            "mmlu": (0.3035180174, 0.0038233308),
            "bbh": (0.3006, 0.0052),
        }
    }[model_name][task]

def read_results(model_name, task):
    results_filename = {
        "micro-moe-llama-2": {
            "gsm8k": "results_2025-09-02T05-24-46.684112.json",
            "minerva_math": "results_2025-09-02T06-04-11.321038.json",
            "mmlu": "results_2025-09-02T07-05-28.315208.json",
            "hellaswag": "results_2025-09-02T05-06-27.771924.json",
            "arc_easy": "results_2025-09-02T05-07-48.753409.json",
            "arc_challenge": "results_2025-09-02T05-08-36.381481.json",
            "piqa": "results_2025-09-02T05-14-48.535839.json",
        },
        "llama-moe-top1-tuluv3-plus-experts-1": {
            "gsm8k": "results_2025-08-19T16-19-12.859757.json",
            "minerva_math": "results_2025-08-19T17-13-57.471382.json",
            "mmlu": "results_2025-08-19T18-59-07.875715.json",
            "hellaswag": "results_2025-08-25T14-43-50.430000.json",
            "arc_easy": "results_2025-08-25T14-44-54.405403.json",
            "arc_challenge": "results_2025-08-25T14-45-34.676513.json",
            "piqa": "results_2025-08-25T14-51-12.246566.json",
        },
        "llama-baseline-1b-base-tuluv3-1": {
            "gsm8k": "results_2025-05-11T00-42-41.235038.json",
            "minerva_math": "results_2025-05-11T01-03-36.919345.json",
            "hellaswag": "results_2025-08-20T14-22-07.106195.json",
            "arc_easy": "results_2025-08-20T14-30-12.701035.json",
            "arc_challenge": "results_2025-08-20T14-13-44.924708.json",
            "piqa": "results_2025-08-20T14-18-05.712421.json",
        },
        "llama-dense-1": {
            "gsm8k": "results_2025-09-05T05-32-47.081797.json",
            "minerva_math": "results_2025-09-05T05-56-56.908324.json",
            "hellaswag": "",
            "arc_easy": "",
            "arc_challenge": "results_2025-09-05T04-47-13.094360.json",
            "piqa": "",
        },
        "llama-mxtr-1b-base-top1-tuluv3-15": {
            "gsm8k": "results_2025-05-03T06-09-49.562135.json",
            "minerva_math": "results_2025-05-03T07-24-58.117863.json",
            "hellaswag": "results_2025-08-20T15-25-25.658508.json",
            "arc_easy": "results_2025-08-20T15-26-41.994263.json",
            "arc_challenge": "results_2025-08-20T15-27-36.472999.json",
            "piqa": "results_2025-08-20T14-19-00.606873.json",
        },
        "llama-mxtr-1b-base-top1-tuluv3-15-social": {
            "gsm8k": "results_2025-05-03T06-01-53.001207.json",
            "minerva_math": "results_2025-05-03T07-27-45.567074.json",
            "arc_challenge": "results_2025-09-18T12-12-01.681308.json",
        },

        "llama-mxtr-1b-base-top1-tuluv3-15-best-ablation": {
            "gsm8k": "results_2025-05-03T06-01-53.001207.json",
            "minerva_math": "results_2025-05-03T07-27-45.567074.json",
            "arc_challenge": "results_2025-09-18T12-12-01.681308.json",
        },

        "llama-mob-top1-tuluv3-plus-experts-2": {
            "gsm8k": "results_2025-08-29T06-13-55.625288.json",
            "minerva_math": "results_2025-08-29T07-51-56.991752.json",
            "hellaswag": "results_2025-08-29T05-28-10.697404.json",
            "arc_easy": "results_2025-08-29T05-30-01.343187.json",
            "arc_challenge": "results_2025-08-29T05-31-13.799867.json",
            "piqa": "results_2025-08-29T05-42-13.738696.json",
        },

        "llama-mob-top1-tuluv3-plus-experts-2-best-ablation": {
            "gsm8k": "results_2025-08-29T06-13-55.625288.json",
            "minerva_math": "results_2025-09-21T15-26-54.839749.json",
        },

        "micro-smollm2-135m-2": {
            "gsm8k": "results_2025-08-20T13-37-54.172455.json",
            "minerva_math": "results_2025-08-20T15-48-53.495465.json",
        },
        "micro-smollm2-moe-135m-1": {
            "gsm8k": "results_2025-09-06T17-09-58.605053.json",
            "minerva_math": "results_2025-09-06T18-14-46.893370.json"
        },
        "smollm2-mob-135m-1": {
            "gsm8k": "results_2025-09-12T19-06-47.074806.json",
            "minerva_math": "results_2025-09-12T20-58-06.761143.json",
        },
        "smollm2-moe-135m-1": {
            "gsm8k": "results_2025-09-07T09-57-39.024713.json",
            "minerva_math": "results_2025-09-07T11-13-44.901656.json"
        },

        "micro-smollm2-360m-1": {
            "gsm8k": "results_2025-08-21T13-05-35.916841.json",
            "minerva_math": "results_2025-08-21T15-32-07.824398.json",
            "mmlu": "results_2025-08-21T18-05-28.242349.json",
            "bbh": "results_2025-08-21T21-35-00.379994.json",
        },

        "micro-smollm2-moe-360m-1": {
            "gsm8k": "results_2025-09-06T12-05-13.004751.json",
            "minerva_math": "results_2025-09-06T13-17-54.038753.json",
            "mmlu": "results_2025-09-06T14-42-59.010586.json",
            "bbh": "results_2025-09-06T16-58-59.301233.json",
        },

        "micro-smollm2-360m-v2-1": {
            "gsm8k": "results_2025-08-23T12-51-24.565135.json",
            "minerva_math": "results_2025-08-23T14-55-45.515610.json",
            "mmlu": "results_2025-08-23T23-27-09.716607.json",
            "bbh": "results_2025-08-24T10-25-32.994882.json",
        },

        "smollm2-mob-360m-1": {
            "gsm8k": "results_2025-09-12T21-00-51.959514.json",
            "minerva_math": "results_2025-09-12T23-18-58.469971.json",
            "mmlu": "results_2025-09-13T01-44-31.170441.json",
            "bbh": "results_2025-09-13T06-04-57.726283.json",
        },

        "smollm2-moe-360m-1": {
            "gsm8k": "results_2025-09-07T10-01-51.390155.json",
            "minerva_math": "results_2025-09-07T11-33-51.662726.json",
            "mmlu": "results_2025-09-07T12-57-10.000587.json",
            "bbh": "results_2025-09-07T14-26-57.951007.json",
        },

        "micro-smollm2-1.7b-2": {
            "gsm8k": "results_2025-08-26T11-56-04.028234.json",
            "minerva_math": "results_2025-08-26T14-47-30.353503.json",
            "mmlu": "results_2025-08-26T18-42-27.271879.json",
            "bbh": "results_2025-08-26T23-01-23.529584.json",
        },

        "micro-smollm2-1.7b-2-social": {
            "gsm8k": "results_2025-09-18T13-59-32.792443.json",
            "minerva_math": "results_2025-09-18T16-19-33.644921.json",
            "mmlu": "results_2025-09-18T20-00-34.929480.json",
            "bbh": "results_2025-09-18T23-30-57.449057.json",
        },

        "smollm2-mob-1.7b-1": {
            "gsm8k": "results_2025-09-14T14-13-21.482218.json",
            "minerva_math": "results_2025-09-14T16-41-14.701274.json",
            "mmlu": "results_2025-09-14T21-00-56.160603.json",
            "bbh": "results_2025-09-15T01-11-49.020182.json",
        },

        "smollm2-1.7b-dense-1": {
            "gsm8k": "results_2025-09-19T12-55-34.316058.json",
            "minerva_math": "results_2025-09-19T13-25-05.043830.json",
            "mmlu": "results_2025-09-19T14-07-35.405055.json",
            "bbh": "results_2025-09-19T14-42-11.451486.json"
        },

        "llama-dense-3b-1": {
            "gsm8k": "results_2025-09-16T13-18-56.313432.json",
            "minerva_math": "results_2025-09-16T13-59-02.065429.json",
        },

        "llama-mob-3b-1": {
            "gsm8k": "results_2025-09-19T14-05-43.108783.json",
            "minerva_math": "results_2025-09-19T16-39-59.582870.json",
        },

        "micro-llama-3b-1": {
            "gsm8k": "results_2025-09-17T20-57-15.964718.json",
            "minerva_math": "results_2025-09-17T23-06-12.515112.json",
        },

        "micro-llama-3b-1-social": {
            "gsm8k": "results_2025-09-17T21-49-38.177924.json",
            "minerva_math": "results_2025-09-17T23-31-36.429775.json",
        },

        "micro-llama-3b-1-best-ablation": {
            "gsm8k": "results_2025-09-17T21-49-38.177924.json",
            "minerva_math": "results_2025-09-17T23-06-12.515112.json",
        },

        "llama-mob-3b-1-best-ablation": {
            "gsm8k": "results_2025-09-21T14-36-17.104600.json",
            "minerva_math": "results_2025-09-19T16-39-59.582870.json",
        },

        "micro-moe-llama-2": {
            "gsm8k": "results_2025-09-02T05-24-46.684112.json",
            "minerva_math": "results_2025-09-02T06-04-11.321038.json",

        }
    }[model_name][task]

    if "-social" in model_name:
        model_name = model_name.replace("-social", "")
    elif "best-ablation" in model_name:
        model_name = model_name.replace("-best-ablation", "")
    path = f"{LM_EVAL_PATH}/results/{model_name}/{results_filename}"
    return read_json(path)

def task_metric_filter(task):
    if task in ["gsm8k"]:
        return ("exact_match", "flexible-extract")
    elif task in ["minerva_math"]:
        return ("exact_match", "none")
    elif task in ["mmlu", "bbh"]:
        return ("exact_match", "get-answer")
    elif task in ["hellaswag", "arc_challenge", "piqa"]:
        return ("acc_norm", "none")
    elif task in ["arc_easy"]:
        return ("acc", "none")
    else:
        raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":

    model_to_plot = "Llama-3.2-3B" # {"SmolLM2-135M", "SmolLM2-360M", "Llama-3.2-1B", "SmolLM2-1.7B"}
    
    tasks = ["gsm8k", "minerva_math", "mmlu", "bbh"] 
    if model_to_plot == "Llama-3.2-1B":
        # Llama Models
        model_names = [
            "llama-dense-1",
            "llama-mxtr-1b-base-top1-tuluv3-15-best-ablation",
            "llama-mxtr-1b-base-top1-tuluv3-15",
            "llama-mob-top1-tuluv3-plus-experts-2",
            "llama-mob-top1-tuluv3-plus-experts-2-best-ablation",
        ]
    elif model_to_plot == "Llama-3.2-3B":
        model_names = [
            "llama-dense-3b-1",
            "micro-llama-3b-1-best-ablation",
            "micro-llama-3b-1",
            "llama-mob-3b-1",
            "llama-mob-3b-1-best-ablation",
        ]
    elif model_to_plot == "SmolLM2-135M":
        # SmolLM2-135m Models
        model_names = [
            "micro-smollm2-135m-2",
            "micro-smollm2-moe-135m-1",
            "smollm2-mob-135m-1",
            "smollm2-moe-135m-1",
        ]
    elif model_to_plot == "SmolLM2-360M":
        # SmolLM2-360m Models
        model_names = [
            "micro-smollm2-360m-v2-1",
            "micro-smollm2-moe-360m-1",
            "smollm2-mob-360m-1",
            "smollm2-moe-360m-1",
        ]
    elif model_to_plot == "SmolLM2-1.7B":
        # SmolLM2-1.7B Models
        model_names = [
            "smollm2-1.7b-dense-1",
            "micro-smollm2-1.7b-2-social",
            "micro-smollm2-1.7b-2",
            "smollm2-mob-1.7b-1",
        ]

    plot_data = []
    for model_name in model_names:
        for task in tasks:
            if task in ["mmlu", "bbh"] and model_to_plot in ["SmolLM2-135M", "Llama-3.2-1B", "Llama-3.2-3B"]:
                score, stderr = read_scores(model_name, task)
            else:
                data = read_results(model_name, task)
                metric, filter_ = task_metric_filter(task)
                task_name = list(data["results"].keys())[0]
                score = data["results"][task_name][f"{metric},{filter_}"]
                stderr = data["results"][task_name][f"{metric}_stderr,{filter_}"]

            model_type = "MoE" if "moe" in model_name else "Dense" if "dense" in model_name else "MoB"
            if "-social" in model_name or "best-ablation" in model_name:
                model_type += " (Ablation)"
            is_micro = "micro" in model_name or "mxtr" in model_name

            if is_micro:
                model_type = f"MiCRo-{model_type}"

            plot_data.append({
                "model": model_names_map[model_name],
                "task": task_name_map[task],
                "model_type": model_type,
                "MiCRo": "Yes" if is_micro else "No",
                "score": score*100,
                "stderr": stderr*100,
            })

    df = pd.DataFrame(plot_data)

    tasks += ["Average"]

    avg_df = df.groupby(["model", "model_type"])[["score", "stderr"]].mean().reset_index()
    avg_df["task"] = "Average"
    df = pd.concat([df,avg_df]).reset_index()

    print(df)

    # plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font_scale=1.5)

    df = df.copy()
    df["dummy"] = "All"  # single x-category, all separation is by hue (legend)

    order_hue = ["MiCRo-MoB (Ablation)", "MiCRo-MoB", "MoB (Ablation)", "MoB"]

    g = sns.catplot(
        kind="bar",
        data=df,
        col="task",
        x="dummy",                # one x category only
        y="score",
        hue="model_type",         # groups go to legend
        hue_order=order_hue,
        errorbar=None,            # we'll draw error bars ourselves
        palette=["#80CBC4", "#80CBC4", "#CE93D8", "#CE93D8"],
        height=5,
        aspect=0.7,
        legend_out=True,
        sharey=False,
        col_wrap=11,
    )

    # Column titles = task names
    for ax, task in zip(g.axes.flat, tasks):
        ax.set_title(task_name_map[task])

    # 2) Per-facet y-lims, labels, and hide x tick labels (since we use legend)
    # 3) Use a uniform headroom function per facet (independent of model_to_plot)
    for ax in g.axes.flat:
        task_name = ax.get_title()
        task_data = df[df["task"] == task_name]
        min_score = float(task_data["score"].min())
        max_score = float(task_data["score"].max())
        span = max(max_score - min_score, 1.0)  # avoid zero-span
        pad_top = 0.02 * span                   # fixed 2% headroom for brackets
        pad_bot = 0.20 * span if model_name == "Llama-3.2-1B" else 1 * span
        ax.set_ylim(min_score - pad_bot, max_score + pad_top)
        ax.set_ylabel("Score (%)")
        ax.set_xlabel(task_name)
        ax.set_xticklabels([])

        # Dense baseline (same as yours)
        if "Dense" in task_data["model_type"].values:
            dense_score = task_data[task_data["model_type"] == "Dense"]["score"].values[0]
            ax.axhline(dense_score, color="darkslategray", linestyle="--")
            yoffset = 0.2 * span
            ax.text(0.25, dense_score - yoffset, f"Dense: {dense_score:.1f}%",
                    color="darkslategray", ha='center', va='bottom', fontsize=10, fontweight="bold", backgroundcolor='white')

    # (Optional) turn off constrained layout if your matplotlib rc enables it
    # g.fig.subplots_adjust(wspace=0.25, hspace=0.35) 

    g.fig.set_constrained_layout(False)

    for ax in g.axes.flat:
        task_name = ax.get_title()
        task_data = df[df["task"] == task_name]

        # stderr lookup by model_type, using the rows actually shown in this facet
        stderr_map = {
            row["model_type"]: float(row["stderr"])
            for _, row in task_data.drop_duplicates(subset=["model_type"]).iterrows()
        }

        # Find BarContainers only (one per hue, in hue order)
        containers = [c for c in ax.containers if isinstance(c, mpl.container.BarContainer)]

        # Align containers with your hue_order explicitly
        # (Seaborn’s order usually matches, but we enforce it)
        if len(containers) != len(order_hue):
            # Fallback: sort by the x of their first patch (still robust when there’s one x bin)
            containers = sorted(containers, key=lambda c: c.patches[0].get_x())

        for i, (mt, container) in enumerate(zip(order_hue, containers)):
            if not container.patches:  # safety
                continue
            bar = container.patches[0]   # one x-category => one patch per hue
            x_center = bar.get_x() + bar.get_width()/2
            y = bar.get_height()

            if i % 2 == 0 and "ablation" in order_hue[i].lower():
                bar.set_hatch("//")

            serr = stderr_map.get(mt, np.nan)
            if np.isnan(serr):
                continue

            ax.errorbar(x_center, y, yerr=serr, capsize=5, fmt="none", color="black")

 
    # 4) Significance brackets with statannotations using precomputed p-values
    #    We’ll compute Welch t-tests from mean±SD and N via ttest_ind_from_stats.
    from itertools import combinations
    from scipy.stats import ttest_ind_from_stats
    import numpy as np

    def pairs_for_hue(dummy_value="All"):
        """Pairs format required by Annotator when using hue: ((x, hue1), (x, hue2))."""
        return [((dummy_value, a), (dummy_value, b)) for a, b in combinations(order_hue, 2)]

    def pvalues_for_task(task_name):
        """Compute p-values for the three pairwise comparisons using SD (not SEM)."""
        task_data = df[df["task"] == task_name].drop_duplicates(subset=["model_type"])
        # Extract mean and stderr, then convert to stdev using provided Ns
        # Use your map; Average uses the mean N across datasets as you had
        if task_name != "Average":
            N = task_name_count_map[task_name]
        else:
            N = int(round(sum(task_name_count_map.values()) / len(task_name_count_map)))
        stats_map = {}
        for mt in order_hue:
            row = task_data[task_data["model_type"] == mt]
            if len(row) == 0:
                continue
            mean = float(row["score"].iloc[0])
            sem  = float(row["stderr"].iloc[0])
            sd   = sem * np.sqrt(N)
            stats_map[mt] = (mean, sd, N)

        # build p-values in the same order as pairs()
        pvals = []
        for a, b in combinations(order_hue, 2):
            if a not in stats_map or b not in stats_map:
                pvals.append(np.nan)
                continue
            mean1, sd1, n1 = stats_map[a]
            mean2, sd2, n2 = stats_map[b]
            # Welch's t-test from summary stats
            tstat, pval = ttest_ind_from_stats(mean1, sd1, n1, mean2, sd2, n2, equal_var=False)
            pvals.append(pval)
        return pvals

    # Apply per-facet
    for ax in g.axes.flat:
        task_name = ax.get_title()
        # Prepare data slice for Annotator (must match the facet)
        task_df = df[df["task"] == task_name]
        prs = pairs_for_hue("All")
        pvals = pvalues_for_task(task_name)

        annot = Annotator(
            ax, prs,
            data=task_df, x="dummy", y="score",
            hue="model_type", hue_order=order_hue
        )
        annot.configure(
            test=None,                  # we provide p-values
            text_format="star",
            show_test_name=False,
            comparisons_correction=None,
            line_width=1,
            fontsize=14,
            loc="inside",
            hide_non_significant=False, # <<< show "ns" instead of dropping
            line_offset=0.03,           # spacing from the bar tops
            line_height=0.02,           # vertical distance between stacked brackets
            text_offset=0.005            # a bit above each bracket
        )
        annot.set_pvalues(pvals)
        annot.annotate()

        for txt in ax.texts:
            t = txt.get_text()
            if t and set(t) == {"*"}:        # "*", "**", "***", etc.
                txt.set_color("red")
            elif t.lower() == "ns":          # optional: dim non-sig labels
                txt.set_color("0.4")

        # (B) Ensure enough headroom after annotations:
        ymin, ymax = ax.get_ylim()
        if model_to_plot == "Llama-3.2-1B":
            ax.set_ylim(ymin, ymax * 1.05)  # crude but effective margin at the top
        elif model_to_plot == "SmolLM2-1.7B":
            ax.set_ylim(ymin, ymax)  # crude but effective margin at the top

    # 5) Legend title and optional hatch for first legend handle (to match your style)
    leg = g._legend
    if leg is not None:
        leg.set_title("Architecture")
        # Optional: hatch first legend handle
        for i, handle in enumerate(leg.legend_handles):
            if hasattr(handle, "set_hatch") and "ablation" in order_hue[i].lower():
                handle.set_hatch("//")
                print("Set hatch for legend handle:", handle)

    for ax, task in zip(g.axes.flat, tasks):
        ax.set_title("")    

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    if "Dense" in df["model_type"].values:
        from matplotlib.lines import Line2D
        handles += [Line2D([0],[0], linestyle="--", color="darkslategray")]
        labels  += ["Dense"]

    if g._legend is not None:
        g._legend.remove()

    # g.fig.legend(handles, labels, title="Architecture",
    #             loc="center right", ncol=len(labels))
    
    plt.tight_layout()
    plt.savefig(f"outputs/performance_comparison_{model_to_plot}.png", dpi=300)

