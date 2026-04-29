#!/usr/bin/env python3
"""
analyze_pacchiardi_replication.py

Analyze EXP-K results: LOO recalibration, hedging baseline, and comparison
with EXP-1 (related questions) equalized results on the same claims.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    python experiments/analyze_pacchiardi_replication.py
"""

import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")

# Import hedging baseline functions
from hedging_baseline import extract_text_features, load_dataset, load_llm_features, loo_accuracy


def bootstrap_ci(preds, labels, n_boot=10000, seed=42):
    """Compute bootstrap 95% CI for accuracy."""
    rng = np.random.RandomState(seed)
    correct = (preds == labels).astype(int)
    n = len(correct)
    accs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        accs.append(np.mean(correct[idx]))
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def permutation_test(X, y, n_perms=1000, seed=42):
    """Permutation test: p-value for LOO accuracy under null."""
    real_acc, _ = loo_accuracy(X, y)
    rng = np.random.RandomState(seed)
    null_accs = []
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        acc, _ = loo_accuracy(X, y_perm)
        null_accs.append(acc)
    p_value = np.mean(np.array(null_accs) >= real_acc)
    return p_value, np.mean(null_accs), np.std(null_accs)


def refusal_only_loo(path):
    """LOO with refusal count as the only feature."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    rows, labels = [], []
    for r in results:
        conv = r.get("conversation", [])
        if not conv or r.get("status") == "error":
            continue
        feats = extract_text_features(conv)
        if feats is None:
            continue
        rows.append([feats["refusal_count"]])
        labels.append(1 if r.get("ground_truth") == "lying" else 0)
    X = np.array(rows)
    y = np.array(labels)
    acc, preds = loo_accuracy(X, y)
    return acc, preds, y


def main():
    # Paths
    unrelated_path = os.path.join(DATA, "ollama_eval_mistral_7b_pacchiardi_unrelated_latest.json")
    related_path = os.path.join(DATA, "ollama_eval_mistral_7b_prompt_equalized_latest.json")

    if not os.path.exists(unrelated_path):
        print(f"ERROR: EXP-K data not found at {unrelated_path}")
        print(f"Run: python experiments/run_pacchiardi_replication.py --model mistral:7b --n_samples 100 --resume")
        sys.exit(1)

    print("=" * 70)
    print("EXP-K ANALYSIS: PACCHIARDI UNRELATED-QUESTION REPLICATION")
    print("=" * 70)

    # --- EXP-K (unrelated questions) ---
    print("\n--- EXP-K: Unrelated Questions (Mistral 7B, equalized) ---")
    X_text_unrel, y_text_unrel = load_dataset(unrelated_path)
    X_llm_unrel, y_llm_unrel = load_llm_features(unrelated_path)

    print(f"  N samples: {len(y_text_unrel)} (text), {len(y_llm_unrel)} (LLM)")

    # LLM pipeline LOO
    llm_acc_unrel, llm_preds_unrel = loo_accuracy(X_llm_unrel, y_llm_unrel)
    llm_ci = bootstrap_ci(llm_preds_unrel, y_llm_unrel)
    print(f"\n  LLM pipeline LOO:     {llm_acc_unrel:.1%}  CI [{llm_ci[0]:.1%}, {llm_ci[1]:.1%}]")

    # Hedging baseline LOO
    text_acc_unrel, text_preds_unrel = loo_accuracy(X_text_unrel, y_text_unrel)
    text_ci = bootstrap_ci(text_preds_unrel, y_text_unrel)
    print(f"  Hedging baseline LOO: {text_acc_unrel:.1%}  CI [{text_ci[0]:.1%}, {text_ci[1]:.1%}]")

    # Refusal-count-only LOO
    ref_acc_unrel, ref_preds_unrel, ref_y_unrel = refusal_only_loo(unrelated_path)
    ref_ci = bootstrap_ci(ref_preds_unrel, ref_y_unrel)
    print(f"  Refusal-only LOO:     {ref_acc_unrel:.1%}  CI [{ref_ci[0]:.1%}, {ref_ci[1]:.1%}]")

    # Permutation test on LLM features
    p_val, null_mean, null_std = permutation_test(X_llm_unrel, y_llm_unrel, n_perms=1000)
    print(f"  Permutation p-value:  {p_val:.4f} (null: {null_mean:.1%} ± {null_std:.1%})")

    # --- EXP-1 comparison (related questions) ---
    if os.path.exists(related_path):
        print("\n--- EXP-1: Related Questions (Mistral 7B, equalized) ---")
        X_text_rel, y_text_rel = load_dataset(related_path)
        X_llm_rel, y_llm_rel = load_llm_features(related_path)

        llm_acc_rel, llm_preds_rel = loo_accuracy(X_llm_rel, y_llm_rel)
        llm_ci_rel = bootstrap_ci(llm_preds_rel, y_llm_rel)
        print(f"  LLM pipeline LOO:     {llm_acc_rel:.1%}  CI [{llm_ci_rel[0]:.1%}, {llm_ci_rel[1]:.1%}]")

        text_acc_rel, text_preds_rel = loo_accuracy(X_text_rel, y_text_rel)
        text_ci_rel = bootstrap_ci(text_preds_rel, y_text_rel)
        print(f"  Hedging baseline LOO: {text_acc_rel:.1%}  CI [{text_ci_rel[0]:.1%}, {text_ci_rel[1]:.1%}]")

        ref_acc_rel, ref_preds_rel, ref_y_rel = refusal_only_loo(related_path)
        ref_ci_rel = bootstrap_ci(ref_preds_rel, ref_y_rel)
        print(f"  Refusal-only LOO:     {ref_acc_rel:.1%}  CI [{ref_ci_rel[0]:.1%}, {ref_ci_rel[1]:.1%}]")

        # --- Comparison ---
        print(f"\n{'=' * 70}")
        print("COMPARISON: RELATED vs. UNRELATED QUESTIONS")
        print(f"{'=' * 70}")
        print(f"  {'Metric':<30} {'Related':>12} {'Unrelated':>12} {'Delta':>10}")
        print(f"  {'-' * 65}")
        print(f"  {'LLM pipeline LOO':<30} {llm_acc_rel:>12.1%} {llm_acc_unrel:>12.1%} "
              f"{(llm_acc_unrel - llm_acc_rel) * 100:>+10.1f}pp")
        print(f"  {'Hedging baseline LOO':<30} {text_acc_rel:>12.1%} {text_acc_unrel:>12.1%} "
              f"{(text_acc_unrel - text_acc_rel) * 100:>+10.1f}pp")
        print(f"  {'Refusal-only LOO':<30} {ref_acc_rel:>12.1%} {ref_acc_unrel:>12.1%} "
              f"{(ref_acc_unrel - ref_acc_rel) * 100:>+10.1f}pp")

        # Feature-level comparison
        FEAT_NAMES = ["hedge_ct", "hedge_rate", "refusal_ct", "avg_len",
                      "std_len", "confidence_ct", "question_ct"]
        print(f"\n  Text Feature Cohen's |d| comparison:")
        print(f"  {'Feature':<15} {'Related |d|':>12} {'Unrelated |d|':>14}")
        for j, fn in enumerate(FEAT_NAMES):
            if len(y_text_rel) > 0 and len(y_text_unrel) > 0:
                # Related
                t_mean_r = np.mean(X_text_rel[y_text_rel == 0, j])
                l_mean_r = np.mean(X_text_rel[y_text_rel == 1, j])
                t_std_r = np.std(X_text_rel[y_text_rel == 0, j])
                l_std_r = np.std(X_text_rel[y_text_rel == 1, j])
                pooled_r = np.sqrt((t_std_r**2 + l_std_r**2) / 2)
                d_rel = abs(t_mean_r - l_mean_r) / pooled_r if pooled_r > 0 else 0

                # Unrelated
                t_mean_u = np.mean(X_text_unrel[y_text_unrel == 0, j])
                l_mean_u = np.mean(X_text_unrel[y_text_unrel == 1, j])
                t_std_u = np.std(X_text_unrel[y_text_unrel == 0, j])
                l_std_u = np.std(X_text_unrel[y_text_unrel == 1, j])
                pooled_u = np.sqrt((t_std_u**2 + l_std_u**2) / 2)
                d_unrel = abs(t_mean_u - l_mean_u) / pooled_u if pooled_u > 0 else 0

                print(f"  {fn:<15} {d_rel:>12.2f} {d_unrel:>14.2f}")

    # --- LLM feature-level analysis ---
    print(f"\n{'=' * 70}")
    print("LLM FEATURE COHEN'S |d| (EXP-K: Unrelated Questions)")
    print(f"{'=' * 70}")
    LLM_FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
    with open(unrelated_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    feat_arrays = {f: ([], []) for f in LLM_FEATURES}
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj or r.get("status") == "error":
            continue
        is_lying = r.get("ground_truth") == "lying"
        for f in LLM_FEATURES:
            vals = [t[f] for t in traj if t.get(f) is not None]
            if vals:
                feat_arrays[f][int(is_lying)].append(np.mean(vals))

    print(f"  {'Feature':<20} {'Truth mean':>12} {'Lie mean':>12} {'|d|':>8}")
    for f in LLM_FEATURES:
        t_vals = np.array(feat_arrays[f][0])
        l_vals = np.array(feat_arrays[f][1])
        if len(t_vals) > 0 and len(l_vals) > 0:
            t_mean, l_mean = np.mean(t_vals), np.mean(l_vals)
            pooled = np.sqrt((np.std(t_vals)**2 + np.std(l_vals)**2) / 2)
            d = abs(t_mean - l_mean) / pooled if pooled > 0 else 0
            print(f"  {f:<20} {t_mean:>12.2f} {l_mean:>12.2f} {d:>8.2f}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
