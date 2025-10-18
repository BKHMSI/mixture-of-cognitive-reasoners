import os
import yaml
import json
import torch
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

import seaborn as sns
import matplotlib.pyplot as plt

cogbench_results = {
    "Llama-Dense-1B": {
        "performance": 0.304,
        "behavior": 0.367, 
    },
    "Llama-MoE-1B": {
        "performance": 0.312,
        "behavior": 0.401,
    },
    "MiCRo-Llama-MoE-1B":{
        "performance": 0.280,
        "behavior": 0.574,
    },

    "SmolLM2-Dense-360M": {
        "performance": 0.235,
        "behavior": 0.258, 
    },
    "SmolLM2-MoE-360M": {
        "performance": 0.261,
        "behavior": 0.194,
    },
    "MiCRo-SmolLM2-MoE-360M":{
        "performance": 0.270,
        "behavior": 0.244,
    },

    "SmolLM2-Dense-135M": {
        "performance": 0.247,
        "behavior": 0.351, 
    },
    "SmolLM2-MoE-135M": {
        "performance": 0.333,
        "behavior": 0.271,
    },
    "MiCRo-SmolLM2-MoE-135M":{
        "performance": 0.334,
        "behavior": 0.316,
    }
}

if __name__ == "__main__":

    save_path = "outputs/cogbench_results.png"

    sns.set_theme(style="whitegrid", font_scale=1.5)

    df = pd.DataFrame(cogbench_results).T.reset_index().rename(columns={"index": "model"})
    df["score"] = (df["performance"]+df["behavior"])/2
    df["model_type"] = df["model"].apply(lambda x: "Dense" if "Dense" in x else "MiCRo-MoE" if "MiCRo" in x else "MoE")
    df["model_class"] = df["model"].apply(lambda x: "Llama-1B" if "Llama" in x else "SmolLM2-135M" if "135M" in x else "SmolLM2-360M")

    x_order = ["SmolLM2-135M", "SmolLM2-360M", "Llama-1B"]
    df["model_class"] = pd.Categorical(df["model_class"], categories=x_order, ordered=True)

    sns.lineplot(
        data=df,
        x="model_class",
        y="score",
        hue="model_type",
        marker="o",
        markersize=10,
        linewidth=2.5,
        hue_order=["MoE", "MiCRo-MoE"],
    )

    sns.despine()
    plt.ylim(0.2, 0.5)
    plt.ylabel("CogBench Score")
    plt.xlabel("Model Class")
    plt.legend(title="Model Type", loc="upper left")
    plt.title("Human Alignment (CogBench)")
    plt.tight_layout()
    plt.savefig(save_path)


