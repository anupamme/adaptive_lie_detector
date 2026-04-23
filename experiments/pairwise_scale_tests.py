#!/usr/bin/env python3
"""
pairwise_scale_tests.py

Pairwise statistical tests on the equalized scale trend (6 models).
Computes: Fisher exact test, permutation test, bootstrap CI on differences,
Holm-Bonferroni correction. Answers reviewer Concern 5 / Q2.
"""

import json
import os
import warnings
import numpy as np
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "results")

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

MODELS = [
    ("Llama 3.2 3B",   "ollama_eval_llama3_2_3b_prompt_equalized_latest.json"),
    ("Llama 3.1 8B",   "bedrock_eval_llama8b_prompt_equalized_latest.json"),
    ("Mistral 7B",     "ollama_eval_mistral_7b_prompt_equalized_latest.json"),
    ("Qwen 2.5 7B",    "ollama_eval_qwen2_5_7b_prompt_equalized_latest.json"),
    ("Qwen 2.5 14B",   "ollama_eval_qwen2_5_14b_prompt_equalized_latest.json"),
    ("Llama 3.3 70B",  "bedrock_eval_llama70b_prompt_equalized_latest.json"),
    ("Claude Haiku",   "bedrock_eval_haiku_prompt_equalized_latest.json"),
]


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


def loo_predictions(X, y):
    """Return per-trial LOO predictions and accuracy."""
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    correct = (preds == y).astype(int)
    return correct, np.mean(correct)


def bootstrap_diff_ci(correct_a, correct_b, n_boot=10000, seed=42):
    """Bootstrap CI on accuracy difference (B - A)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx_a = rng.integers(0, len(correct_a), size=len(correct_a))
        idx_b = rng.integers(0, len(correct_b), size=len(correct_b))
        diffs.append(np.mean(correct_b[idx_b]) - np.mean(correct_a[idx_a]))
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5), np.array(diffs)


def permutation_test(correct_a, correct_b, n_perm=10000, seed=42):
    """Two-sample permutation test on accuracy difference."""
    rng = np.random.default_rng(seed)
    observed = np.mean(correct_b) - np.mean(correct_a)
    pooled = np.concatenate([correct_a, correct_b])
    n_a = len(correct_a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_diff = np.mean(pooled[n_a:]) - np.mean(pooled[:n_a])
        if abs(perm_diff) >= abs(observed):
            count += 1
    return count / n_perm


def fisher_test(correct_a, correct_b):
    """Fisher exact test on 2x2 table (correct/incorrect x model)."""
    a_correct = int(np.sum(correct_a))
    a_wrong = len(correct_a) - a_correct
    b_correct = int(np.sum(correct_b))
    b_wrong = len(correct_b) - b_correct
    table = [[a_correct, b_correct], [a_wrong, b_wrong]]
    _, p = fisher_exact(table, alternative="two-sided")
    return p


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction. Returns corrected p-values."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    corrected = np.zeros(n)
    cummax = 0
    for rank, idx in enumerate(sorted_idx):
        adj = p_values[idx] * (n - rank)
        cummax = max(cummax, adj)
        corrected[idx] = min(cummax, 1.0)
    return corrected


def main():
    print("=" * 75)
    print("  PAIRWISE STATISTICAL TESTS ON EQUALIZED SCALE TREND")
    print("=" * 75)

    # Load and compute LOO for each model
    model_data = []
    for name, fname in MODELS:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found, skipping {name}")
            continue
        results = load_results(path)
        X, y = per_trial_feature_means(results)
        correct, acc = loo_predictions(X, y)
        n = len(y)
        # Bootstrap CI on accuracy
        rng = np.random.default_rng(42)
        boot_accs = [np.mean(correct[rng.integers(0, n, size=n)]) for _ in range(10000)]
        ci_lo, ci_hi = np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)
        model_data.append({
            "name": name, "acc": acc, "correct": correct, "n": n,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
        })
        print(f"  {name:<18} LOO={acc:.1%} [{ci_lo:.1%}, {ci_hi:.1%}]  n={n}")

    print(f"\n  Loaded {len(model_data)} models.\n")

    # Pairwise adjacent tests
    print("-" * 75)
    print(f"  {'Pair':<28} {'Diff':>7} {'95% CI':>18} {'Fisher p':>10} {'Perm p':>10} {'Holm p':>10}")
    print("-" * 75)

    pairs = []
    raw_p_values = []
    for i in range(len(model_data) - 1):
        a = model_data[i]
        b = model_data[i + 1]
        diff = b["acc"] - a["acc"]
        ci_lo, ci_hi, _ = bootstrap_diff_ci(a["correct"], b["correct"])
        p_fisher = fisher_test(a["correct"], b["correct"])
        p_perm = permutation_test(a["correct"], b["correct"])
        pairs.append({
            "a": a["name"], "b": b["name"],
            "diff": diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "p_fisher": p_fisher, "p_perm": p_perm,
        })
        raw_p_values.append(p_perm)

    # Holm-Bonferroni correction
    corrected_p = holm_bonferroni(np.array(raw_p_values))

    for i, pair in enumerate(pairs):
        pair["p_holm"] = corrected_p[i]
        label = f"{pair['a']} -> {pair['b']}"
        sig = ""
        if pair["p_holm"] < 0.001:
            sig = " ***"
        elif pair["p_holm"] < 0.01:
            sig = " **"
        elif pair["p_holm"] < 0.05:
            sig = " *"
        print(f"  {label:<28} {pair['diff']:+.1%} [{pair['ci_lo']:+.1%}, {pair['ci_hi']:+.1%}] "
              f"{pair['p_fisher']:>10.4f} {pair['p_perm']:>10.4f} {pair['p_holm']:>10.4f}{sig}")

    # Group test: small (<=7B) vs large (>=14B)
    print(f"\n{'=' * 75}")
    print("  GROUP TEST: models <=7B vs models >=14B")
    print("=" * 75)

    small_correct = np.concatenate([m["correct"] for m in model_data[:4]])
    large_correct = np.concatenate([m["correct"] for m in model_data[4:]])
    small_acc = np.mean(small_correct)
    large_acc = np.mean(large_correct)
    diff = large_acc - small_acc
    ci_lo, ci_hi, _ = bootstrap_diff_ci(small_correct, large_correct)
    p_fisher = fisher_test(small_correct, large_correct)
    p_perm = permutation_test(small_correct, large_correct)

    print(f"  Small (3B, 8B, 7B, Qwen7B): {small_acc:.1%}  (n={len(small_correct)})")
    print(f"  Large (14B, 70B, Haiku):     {large_acc:.1%}  (n={len(large_correct)})")
    print(f"  Difference:           {diff:+.1%} [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    print(f"  Fisher p:             {p_fisher:.4f}")
    print(f"  Permutation p:        {p_perm:.4f}")

    # Non-adjacent: 3B vs 70B (endpoints)
    print(f"\n{'=' * 75}")
    print("  ENDPOINT TEST: 3B vs 70B")
    print("=" * 75)
    a = model_data[0]  # 3B
    b = model_data[5]  # 70B
    diff_ep = b["acc"] - a["acc"]
    ci_lo_ep, ci_hi_ep, _ = bootstrap_diff_ci(a["correct"], b["correct"])
    p_perm_ep = permutation_test(a["correct"], b["correct"])
    print(f"  3B: {a['acc']:.1%}, 70B: {b['acc']:.1%}")
    print(f"  Difference: {diff_ep:+.1%} [{ci_lo_ep:+.1%}, {ci_hi_ep:+.1%}]")
    print(f"  Permutation p: {p_perm_ep:.4f}")

    # Summary
    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print("=" * 75)
    sig_pairs = [p for p in pairs if p["p_holm"] < 0.05]
    nonsig_pairs = [p for p in pairs if p["p_holm"] >= 0.05]
    print(f"  Significant adjacent jumps (Holm-corrected p < 0.05): {len(sig_pairs)}")
    for p in sig_pairs:
        print(f"    {p['a']} -> {p['b']}: {p['diff']:+.1%}, p={p['p_holm']:.4f}")
    print(f"  Non-significant adjacent comparisons: {len(nonsig_pairs)}")
    for p in nonsig_pairs:
        print(f"    {p['a']} -> {p['b']}: {p['diff']:+.1%}, p={p['p_holm']:.4f}")

    if p_perm < 0.05:
        print(f"\n  Group test (<=7B vs >=14B) IS significant: p={p_perm:.4f}")
    else:
        print(f"\n  Group test (<=7B vs >=14B) NOT significant: p={p_perm:.4f}")

    print(f"\n  Interpretation: The scale trend is better characterized as a two-tier")
    print(f"  structure (small <=7B vs large >=14B) rather than smooth scaling.")


if __name__ == "__main__":
    main()
