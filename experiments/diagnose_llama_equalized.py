#!/usr/bin/env python3
"""
diagnose_llama_equalized.py — EXP-E

Diagnoses why the LLM pipeline underperforms the hedging-word regex baseline
by 14pp on the Llama 3.2 3B equalized condition (64% vs 78% LOO).

Hypothesis: under neutral prompting, Llama 3B correctly *rejects* false claims
with refusal/correction language. The LLM extractor assigns these rejections
high consistency and specificity — feature values indistinguishable from
truthful responses. The regex baseline avoids this by directly detecting
correction markers regardless of epistemic state.

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python3 experiments/diagnose_llama_equalized.py
"""

import json
import os
import re
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")

LLAMA_EQ = os.path.join(DATA, "ollama_eval_llama3_2_3b_prompt_equalized_latest.json")
MISTRAL_EQ = os.path.join(DATA, "ollama_eval_mistral_7b_prompt_equalized_latest.json")

LLM_FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

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
    r"\bActually\b", r"\bactually\b",
]

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly"
]


def cohen_d(a, b):
    """Compute Cohen's d between two groups."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def loo_accuracy_single(X, y):
    """LOO accuracy for a 1-D feature array."""
    X2d = X.reshape(-1, 1)
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X2d):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X2d[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X2d[test_idx])[0]
    return np.mean(preds == y)


def load_llm_features(path):
    with open(path) as f:
        data = json.load(f)
    rows, labels = [], []
    for r in data.get("results", []):
        traj = r.get("feature_trajectory", [])
        if not traj or r.get("status") == "error":
            continue
        means = [np.mean([t[f] for t in traj if t.get(f) is not None]) for f in LLM_FEATURES]
        rows.append(means)
        labels.append(1 if r.get("ground_truth") == "lying" else 0)
    return np.array(rows), np.array(labels)


def load_refusal_features(path):
    """Extract refusal count and hedge count from conversation texts."""
    with open(path) as f:
        data = json.load(f)
    refusal_counts, hedge_counts, labels = [], [], []
    for r in data.get("results", []):
        conv = r.get("conversation", [])
        if not conv or r.get("status") == "error":
            continue
        texts = [msg["content"] for msg in conv if msg.get("role") == "assistant"]
        if not texts:
            continue
        joined = " ".join(texts)
        refusal_ct = sum(1 for pat in REFUSAL_PATTERNS if re.search(pat, joined, re.IGNORECASE))
        all_lower = joined.lower()
        hedge_ct = sum(all_lower.count(hw) for hw in HEDGE_WORDS)
        refusal_counts.append(refusal_ct)
        hedge_counts.append(hedge_ct)
        labels.append(1 if r.get("ground_truth") == "lying" else 0)
    return np.array(refusal_counts), np.array(hedge_counts), np.array(labels)


def analyze_refusal_rates(path, label):
    """Show % of lying vs truthful trials that contain ≥1 refusal marker."""
    with open(path) as f:
        data = json.load(f)
    truth_refusals, lie_refusals = [], []
    for r in data.get("results", []):
        conv = r.get("conversation", [])
        if not conv or r.get("status") == "error":
            continue
        texts = [msg["content"] for msg in conv if msg.get("role") == "assistant"]
        if not texts:
            continue
        joined = " ".join(texts)
        has_refusal = any(re.search(pat, joined, re.IGNORECASE) for pat in REFUSAL_PATTERNS)
        if r.get("ground_truth") == "lying":
            lie_refusals.append(has_refusal)
        else:
            truth_refusals.append(has_refusal)
    print(f"\n  Refusal marker presence ({label}):")
    print(f"    Lying  trials: {sum(lie_refusals)}/{len(lie_refusals)} = {100*np.mean(lie_refusals):.1f}%")
    print(f"    Truthful trials: {sum(truth_refusals)}/{len(truth_refusals)} = {100*np.mean(truth_refusals):.1f}%")
    return np.array(lie_refusals), np.array(truth_refusals)


def main():
    print("=" * 70)
    print("EXP-E: DIAGNOSING LLAMA EQUALIZED LLM PIPELINE UNDERPERFORMANCE")
    print("=" * 70)

    for name, path in [("Llama 3.2 3B (equalized)", LLAMA_EQ),
                        ("Mistral 7B (equalized)", MISTRAL_EQ)]:
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: {path} not found")
            continue

        print(f"\n{'─' * 70}")
        print(f"{name}")
        print(f"{'─' * 70}")

        # LLM feature Cohen's d
        X_llm, y = load_llm_features(path)
        print(f"\n  LLM feature Cohen's |d| (truthful vs lying):")
        print(f"  {'Feature':<18} {'d':>8}  {'mean_truth':>12}  {'mean_lie':>10}")
        for j, feat in enumerate(LLM_FEATURES):
            truth_vals = X_llm[y == 0, j]
            lie_vals   = X_llm[y == 1, j]
            d = cohen_d(truth_vals, lie_vals)
            print(f"  {feat:<18} {d:+8.3f}  {np.mean(truth_vals):12.3f}  {np.mean(lie_vals):10.3f}")

        # Refusal / hedge features
        refusal_ct, hedge_ct, y2 = load_refusal_features(path)

        # Cohen's d for refusal count
        d_refusal = cohen_d(refusal_ct[y2 == 0], refusal_ct[y2 == 1])
        d_hedge   = cohen_d(hedge_ct[y2 == 0], hedge_ct[y2 == 1])
        print(f"\n  Regex feature Cohen's |d|:")
        print(f"  {'refusal_count':<18} {d_refusal:+8.3f}  mean_truth={np.mean(refusal_ct[y2==0]):.2f}  mean_lie={np.mean(refusal_ct[y2==1]):.2f}")
        print(f"  {'hedge_count':<18} {d_hedge:+8.3f}  mean_truth={np.mean(hedge_ct[y2==0]):.2f}  mean_lie={np.mean(hedge_ct[y2==1]):.2f}")

        # Refusal rate breakdown
        analyze_refusal_rates(path, name)

        # Single-feature LOO: refusal count only
        if len(np.unique(y2)) > 1 and len(refusal_ct) > 5:
            ref_loo = loo_accuracy_single(refusal_ct.astype(float), y2)
            hedge_loo = loo_accuracy_single(hedge_ct.astype(float), y2)
            print(f"\n  Single-feature LOO accuracy:")
            print(f"    refusal_count alone: {ref_loo:.1%}")
            print(f"    hedge_count alone:   {hedge_loo:.1%}")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
