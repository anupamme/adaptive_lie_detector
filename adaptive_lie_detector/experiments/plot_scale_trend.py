#!/usr/bin/env python3
"""
plot_scale_trend.py — Scale trend figure for NeurIPS paper (W8).

Bar chart of prompt-equalized LOO accuracy vs model scale, with bootstrap CIs
as error bars. Includes hedging baseline as comparison. Horizontal line at 50%.

Outputs: output/figures/scale_trend.pdf

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/plot_scale_trend.py [--include-8b]
"""

import json
import os
import sys
import warnings
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import re

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "results")
OUT_DIR = os.path.join(BASE, "output", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

# Hedging patterns (same as hedging_baseline.py / analyze_70b_equalized.py)
HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly",
]
REFUSAL_PATTERNS = [
    r"\bI cannot\b", r"\bI can't\b", r"\bI must correct\b",
    r"\bI need to correct\b", r"\bI should clarify\b",
    r"\bActually,?\s+(?:water|the|this|that)\b",
    r"\bThat(?:'s| is) not (?:correct|accurate|true|right)\b",
    r"\bThis (?:claim|statement|information) is (?:incorrect|inaccurate|false|wrong)\b",
    r"\bincorrect\b", r"\binaccurate\b",
    r"\bI (?:would|must) (?:like to |)(?:point out|note|clarify|correct)\b",
    r"\bnot (?:entirely |quite |)(?:accurate|correct|true|right)\b",
    r"\bcontrary to\b", r"\bmisconception\b",
]
CONFIDENCE_WORDS = [
    "certainly", "definitely", "absolutely", "clearly",
    "of course", "without doubt", "undoubtedly",
]


# ---------------------------------------------------------------------------
# Dataset configs: (label, param_count_billions, data_file)
# ---------------------------------------------------------------------------
# Full dataset list in paper ordering (by scale/capability)
ALL_DATASETS = [
    ("Llama 3.2\n3B",  3,  "ollama_eval_llama3_2_3b_prompt_equalized_latest.json"),
    ("Llama 3.1\n8B",  8,  "bedrock_eval_llama8b_prompt_equalized_latest.json"),
    ("Mistral\n7B",    7,  "ollama_eval_mistral_7b_prompt_equalized_latest.json"),
    ("Qwen 2.5\n7B",   7,  "ollama_eval_qwen2_5_7b_prompt_equalized_latest.json"),
    ("Qwen 2.5\n14B", 14,  "ollama_eval_qwen2_5_14b_prompt_equalized_latest.json"),
    ("Llama 3.3\n70B", 70,  "bedrock_eval_llama70b_prompt_equalized_latest.json"),
    ("Claude\nHaiku",  80,  "bedrock_eval_haiku_prompt_equalized_latest.json"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def extract_hedging_features(conversation):
    assistant_texts = [msg["content"] for msg in conversation if msg.get("role") == "assistant"]
    if not assistant_texts:
        return None
    all_text = " ".join(assistant_texts).lower()
    total_words = len(all_text.split())
    hedge_count = sum(all_text.count(hw) for hw in HEDGE_WORDS)
    refusal_count = sum(1 for pat in REFUSAL_PATTERNS
                        if re.search(pat, " ".join(assistant_texts), re.IGNORECASE))
    lengths = [len(t.split()) for t in assistant_texts]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths) if len(lengths) > 1 else 0
    confidence_count = sum(all_text.count(cw) for cw in CONFIDENCE_WORDS)
    question_count = sum(t.count("?") for t in assistant_texts)
    return [hedge_count, hedge_count / max(total_words, 1) * 100,
            refusal_count, avg_length, std_length, confidence_count, question_count]


def loo_accuracy(X, y):
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    acc = np.mean(preds == y)
    return acc, preds


def bootstrap_ci(preds, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    correct = (preds == y).astype(int)
    boot_accs = [np.mean(correct[rng.integers(0, len(y), size=len(y))]) for _ in range(n_boot)]
    return np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)


def compute_stats(data_path):
    """Returns (loo_acc, ci_lo, ci_hi, hedge_acc, hedge_ci_lo, hedge_ci_hi, n)."""
    results = load_results(data_path)
    X, y = per_trial_feature_means(results)
    acc, preds = loo_accuracy(X, y)
    ci_lo, ci_hi = bootstrap_ci(preds, y)

    # Hedging baseline
    hedge_rows, hedge_labels = [], []
    for r in results:
        conv = r.get("conversation", [])
        if not conv:
            continue
        feats = extract_hedging_features(conv)
        if feats is None:
            continue
        hedge_rows.append(feats)
        hedge_labels.append(1 if r["ground_truth"] == "lying" else 0)

    X_h = np.array(hedge_rows)
    y_h = np.array(hedge_labels)
    h_acc, h_preds = loo_accuracy(X_h, y_h)
    h_ci_lo, h_ci_hi = bootstrap_ci(h_preds, y_h)

    return acc, ci_lo, ci_hi, h_acc, h_ci_lo, h_ci_hi, len(y)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-8b", action="store_true",
                        help="(legacy flag, 8B now included by default)")
    args = parser.parse_args()

    datasets = ALL_DATASETS

    labels = []
    scales = []
    loo_accs = []
    loo_ci_lo = []
    loo_ci_hi = []
    hedge_accs = []
    hedge_ci_lo_list = []
    hedge_ci_hi_list = []

    for label, scale, fname in datasets:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"Skipping {label}: {fname} not found")
            continue
        print(f"Computing {label.replace(chr(10), ' ')} ({scale}B) ...", flush=True)
        acc, ci_lo, ci_hi, h_acc, h_ci_lo, h_ci_hi, n = compute_stats(path)
        print(f"  LOO: {acc:.1%} [{ci_lo:.1%}, {ci_hi:.1%}], Hedge: {h_acc:.1%}, n={n}")

        labels.append(label)
        scales.append(scale)
        loo_accs.append(acc * 100)
        loo_ci_lo.append(ci_lo * 100)
        loo_ci_hi.append(ci_hi * 100)
        hedge_accs.append(h_acc * 100)
        hedge_ci_lo_list.append(h_ci_lo * 100)
        hedge_ci_hi_list.append(h_ci_hi * 100)

    loo_accs = np.array(loo_accs)
    loo_ci_lo = np.array(loo_ci_lo)
    loo_ci_hi = np.array(loo_ci_hi)
    hedge_accs = np.array(hedge_accs)
    hedge_ci_lo_arr = np.array(hedge_ci_lo_list)
    hedge_ci_hi_arr = np.array(hedge_ci_hi_list)

    # Error bars (asymmetric)
    loo_err_lo = loo_accs - loo_ci_lo
    loo_err_hi = loo_ci_hi - loo_accs
    hedge_err_lo = hedge_accs - hedge_ci_lo_arr
    hedge_err_hi = hedge_ci_hi_arr - hedge_accs

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, loo_accs, width,
                   yerr=[loo_err_lo, loo_err_hi],
                   label="ADAGE LOO", color="#4C72B0", capsize=4, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, hedge_accs, width,
                   yerr=[hedge_err_lo, hedge_err_hi],
                   label="Hedging regex", color="#DD8452", capsize=4, edgecolor="white", linewidth=0.5)

    # Chance line
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Chance (50%)")

    ax.set_ylabel("LOO Accuracy (%)", fontsize=11)
    ax.set_xlabel("Target Model", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(40, 100)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Prompt-Equalized Detection Accuracy by Model Scale", fontsize=11, fontweight="bold")

    # Value labels on bars
    for bar, val in zip(bars1, loo_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars2, hedge_accs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "scale_trend.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Also save PNG for quick preview
    out_png = os.path.join(OUT_DIR, "scale_trend.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
