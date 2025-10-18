import os
import json 
import yaml
import numpy as np
from glob import glob
from prettytable import PrettyTable

import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dotenv import load_dotenv

load_dotenv()

LM_EVAL_PATH = os.environ["LM_EVAL_PATH"]

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_plot_title(task):
    if task == "gsm8k_cot_zeroshot":
        return "GSM8K"
    elif task == "minerva_math":
        return "MATH"
    else:
        return task
    
    
if __name__ == "__main__":

    model_name = "llama-mxtr-1b-base-top1-tuluv3-15"
    # model_name = "micro-smollm2-1.7b-2"
    model_name = "micro-llama-3b-1"
    # model_name = "llama-mob-3b-1"
    # model_name = "llama-mob-top1-tuluv3-plus-experts-2"
    model_name = "micro-smollm2-360m-1"
    # model_name = "micro-smollm2-135m-2"
    model_name = "olmo-mxtr-1b-base-top1-tuluv3-3"

    dirpath = f"{LM_EVAL_PATH}/results/{model_name}"
    paths = glob(f"{dirpath}/*.json")
    paths = sorted(paths)

    prettytable = PrettyTable()
    prettytable.field_names = ["Task", "Ablation", "Score", "Std", "N", "Path"]

    plot_data = []

    task_to_plot = "gsm8k_cot_zeroshot" # {gsm8k_cot_zeroshot, minerva_math, mmlu_flan_cot_fewshot, bbh_cot_fewshot}
    ablation_results = []

    for path in paths:
        results = read_json(path)

        task = list(results["results"].keys())[0]
        ablate: str = results["config"]["model_args"].split(",")[3].split("=")[-1]
        config_name = results["config"]["model_args"].split(",")[2].split("=")[-1]
        top_k  = "top-2" if "top2" in config_name else ("top-4" if "top4" in config_name else ("top-3" if "top3" in config_name else "top-1"))
        limit = results["config"]["limit"]

        if task != task_to_plot:
            continue 

        if limit is not None or top_k != "top-1":
            continue
        if task == "gsm8k_cot_zeroshot":
            score = results["results"][task]["exact_match,flexible-extract"]
            stderr = results["results"][task]["exact_match_stderr,flexible-extract"]
            num_samples = results["n-samples"][task]["effective"]
        elif task == "minerva_math":
            score = results["results"][task]["exact_match,none"]
            stderr = results["results"][task]["exact_match_stderr,none"]
            num_samples = 0
            for subtask in results["n-samples"]:
                num_samples += results["n-samples"][subtask]["effective"]
        elif task == "bbh_cot_fewshot" or task == "mmlu_flan_cot_fewshot":
            score = results["results"][task]["exact_match,get-answer"]
            stderr = results["results"][task]["exact_match_stderr,get-answer"]
            num_samples = 0
            for subtask in results["n-samples"]:
                num_samples += results["n-samples"][subtask]["effective"]
        else:
            score = results["results"][task]["acc,none"]

        if ablate == "none" or ablate.strip() == "":
            baseline = (score, stderr, num_samples)
        else:
            ablation_results += [(ablate, score, stderr, num_samples)]

        prettytable.add_row([task, ablate, score, stderr, num_samples, os.path.basename(path)])

    print(prettytable)

    for result in ablation_results:
        ablate, score, stderr, num_samples = result
        plot_data.append({
            "task": task_to_plot,
            "ablation": ablate.capitalize(),
            "score": (score - baseline[0]) * 100,
            "stderr": np.sqrt((stderr**2) + (baseline[1]**2)) * 100,
        })

    hue_order = ["Language", "Logic", "Social", "World"]
    color_palette = ["#97D077", "#4285F4", "#FFAB40", "#A64D79"]

    sns.set_theme(style="whitegrid", font_scale=2, context="paper")

    df = pd.DataFrame(plot_data)
    df = df[df["ablation"].isin(hue_order)]
    df["ablation"] = pd.Categorical(df["ablation"], categories=hue_order, ordered=True)
    df = df.sort_values(by="ablation")
    
    # Build the single-facet bar plot
    g = sns.catplot(
        data=df,
        kind="bar",
        y="score",
        hue="ablation",
        hue_order=hue_order,
        palette=color_palette,
        height=5,      # overall subplot height (inches)
        aspect=1,    # width = height * aspect
        legend=False,  # we'll handle legend/text ourselves
        errorbar=None, # we add error bars manually
    )


    # g = sns.catplot(data=df, 
    #     kind="bar",
    #     col="task",
    #     y="score", 
    #     hue="ablation", 
    #     hue_order=hue_order, 
    #     palette=color_palette,
    #     height=5,
    #     aspect=0.7,
    #     legend=False,
    #     errorbar=None,
    #     sharey=False,
    # )
    ax = g.ax  # the single Axes in this FacetGrid

    # Optional title & despine
    ax.set_title("")              # or set to your title
    sns.despine(ax=ax, left=False, bottom=False)

    # --- Manual error bars aligned with hue_order ---
    # Build a lookup: ablation level -> (mean, stderr)
    stat_map = (
        df.set_index("ablation")
        .loc[hue_order][["score", "stderr"]]   # ensure sorted by hue_order
        .to_dict(orient="index")
    )

    # Find one BarContainer per hue (in draw order)
    containers = [c for c in ax.containers if isinstance(c, mpl.container.BarContainer)]

    # If needed, enforce alignment with hue_order by sorting containers by the x of their first patch
    if len(containers) != len(hue_order):
        containers = sorted(containers, key=lambda c: c.patches[0].get_x())

    # Add error bars at the center of each hue bar
    for level, container in zip(hue_order, containers):
        # single x-bin => one patch
        bar = container.patches[0]
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        serr = stat_map[level]["stderr"]
        ax.errorbar(x, y, yerr=serr, fmt="none", capsize=5, linewidth=1, color="black")

    # --- Bold baseline text centered below axes ---
    score_text = f"No Ablation = {baseline[0]*100:.2f}%"
    ax.text(
        0.5, -0.06, score_text,   # adjust -0.06 if you need more/less margin
        ha="center", va="top",
        transform=ax.transAxes,
        fontsize=15, fontweight="bold"
    )

    g.ax.set_ylabel("Δ Exact Match (%)")
    plt.tight_layout()
    plt.savefig(f"outputs/ablations_task={task_to_plot}_model={model_name}.png")

