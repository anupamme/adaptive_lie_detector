#!/usr/bin/env python3
"""
plot_feature_collapse.py — Feature collapse figure for NeurIPS paper (W8).

Grouped bar chart showing |d| for each of the 5 features, with instructed (filled)
vs equalized (hatched) bars, faceted by model. Visually demonstrates the collapse
from instructed to equalized condition.

Outputs: output/figures/feature_collapse.pdf

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/plot_feature_collapse.py
"""

import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "results")
OUT_DIR = os.path.join(BASE, "output", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
FEAT_SHORT = ["Consist.", "Specif.", "Defens.", "Confid.", "Elab."]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return [r for r in data["results"] if r.get("status") != "error"]


def per_trial_feature_means(results):
    rows, labels = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        means = [np.mean([t[f] for t in traj if t.get(f) is not None] or [0]) for f in FEATURES]
        rows.append(means)
        labels.append(1 if r["ground_truth"] == "lying" else 0)
    return np.array(rows), np.array(labels)


def cohens_d_abs(X, y):
    """Compute |d| for each feature."""
    ds = []
    for j in range(X.shape[1]):
        g0 = X[y == 0, j]
        g1 = X[y == 1, j]
        n0, n1 = len(g0), len(g1)
        m0, m1 = np.mean(g0), np.mean(g1)
        s0, s1 = np.std(g0, ddof=1), np.std(g1, ddof=1)
        pooled = np.sqrt(((n0 - 1) * s0**2 + (n1 - 1) * s1**2) / (n0 + n1 - 2))
        d = abs((m0 - m1) / pooled) if pooled > 0 else 0
        ds.append(d)
    return ds


# ---------------------------------------------------------------------------
# Model definitions: (display_name, instructed_file, equalized_file)
# ---------------------------------------------------------------------------
MODELS = [
    ("Llama 3.2 3B",
     "ollama_eval_llama3_2_3b_latest.json",
     "ollama_eval_llama3_2_3b_prompt_equalized_latest.json"),
    ("Mistral 7B",
     "ollama_eval_mistral_7b_latest.json",
     "ollama_eval_mistral_7b_prompt_equalized_latest.json"),
    ("Llama 3.3 70B",
     "bedrock_eval_llama70b_latest.json",
     "bedrock_eval_llama70b_prompt_equalized_latest.json"),
]


def main():
    n_models = len(MODELS)
    n_feats = len(FEATURES)

    fig, axes = plt.subplots(1, n_models, figsize=(12, 3.5), sharey=True)

    for ax_idx, (model_name, instr_file, equal_file) in enumerate(MODELS):
        ax = axes[ax_idx]
        instr_path = os.path.join(DATA_DIR, instr_file)
        equal_path = os.path.join(DATA_DIR, equal_file)

        # Instructed
        if os.path.exists(instr_path):
            results_i = load_results(instr_path)
            Xi, yi = per_trial_feature_means(results_i)
            d_instr = cohens_d_abs(Xi, yi)
        else:
            d_instr = [0] * n_feats
            print(f"Warning: {instr_file} not found")

        # Equalized
        if os.path.exists(equal_path):
            results_e = load_results(equal_path)
            Xe, ye = per_trial_feature_means(results_e)
            d_equal = cohens_d_abs(Xe, ye)
        else:
            d_equal = [0] * n_feats
            print(f"Warning: {equal_file} not found")

        x = np.arange(n_feats)
        width = 0.35

        bars_i = ax.bar(x - width/2, d_instr, width,
                        label="Instructed" if ax_idx == 0 else "",
                        color="#4C72B0", edgecolor="white", linewidth=0.5)
        bars_e = ax.bar(x + width/2, d_equal, width,
                        label="Equalized" if ax_idx == 0 else "",
                        color="#DD8452", edgecolor="white", linewidth=0.5,
                        hatch="///")

        ax.set_title(model_name, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(FEAT_SHORT, fontsize=7.5, rotation=30, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0:
            ax.set_ylabel("Cohen's |d|", fontsize=10)

        # Print values
        print(f"\n{model_name}:")
        for f, di, de in zip(FEATURES, d_instr, d_equal):
            print(f"  {f:<16} instr={di:.2f}  equal={de:.2f}  collapse={di-de:.2f}")

    fig.legend(["Instructed", "Equalized"], loc="upper right",
               fontsize=9, framealpha=0.9, ncol=2,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("Feature Separability Collapse: Instructed vs. Equalized",
                 fontsize=11, fontweight="bold", y=1.02)

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "feature_collapse.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    out_png = os.path.join(OUT_DIR, "feature_collapse.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
