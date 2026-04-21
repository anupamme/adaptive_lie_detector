#!/usr/bin/env python3
"""
held_out_evaluation.py — EXP-B

Addresses reviewer concern that LOO on ~100 samples with 5 features overfits.

Two analyses:
  B1: 50/50 stratified held-out split (100 random seeds, mean ± SD)
  B2: Permutation test (1000 label shuffles, null distribution of LOO accuracy)

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python3 experiments/held_out_evaluation.py
"""

import json
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

DATASETS = {
    "Llama 3.2 3B": os.path.join(DATA, "ollama_eval_llama3_2_3b_latest.json"),
    "Mistral 7B": os.path.join(DATA, "ollama_eval_mistral_7b_latest.json"),
    "Claude Haiku": os.path.join(DATA, "claude_haiku_n100_latest.json"),
}


def load_features(path):
    """Load per-trial mean features and labels."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])

    rows, labels = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj or r.get("status") == "error":
            continue
        means = []
        for feat in FEATURES:
            vals = [t[feat] for t in traj if t.get(feat) is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
        rows.append(means)
        labels.append(1 if r.get("ground_truth") == "lying" else 0)

    return np.array(rows), np.array(labels)


def loo_accuracy(X, y):
    """Standard LOO accuracy."""
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    return np.mean(preds == y), preds


def held_out_split(X, y, n_splits=100):
    """B1: 50/50 stratified split, repeated n_splits times."""
    accs = []
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        acc = pipe.score(X[test_idx], y[test_idx])
        accs.append(acc)
    return np.array(accs)


def permutation_test(X, y, n_perms=1000):
    """B2: Shuffle labels, run LOO, build null distribution."""
    rng = np.random.default_rng(42)
    null_accs = []
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        acc, _ = loo_accuracy(X, y_perm)
        null_accs.append(acc)
        if (i + 1) % 100 == 0:
            print(f"    permutation {i+1}/{n_perms}...")
    return np.array(null_accs)


def main():
    print("=" * 70)
    print("EXP-B: HELD-OUT EVALUATION + PERMUTATION TEST")
    print("=" * 70)

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: file not found")
            continue

        X, y = load_features(path)
        print(f"\n{'─' * 70}")
        print(f"{name}: n={len(y)} (truthful={sum(y==0)}, lying={sum(y==1)})")
        print(f"{'─' * 70}")

        # LOO baseline
        loo_acc, _ = loo_accuracy(X, y)
        print(f"\n  LOO accuracy: {loo_acc:.1%}")

        # B1: Held-out split
        print(f"\n  B1: 50/50 stratified held-out (100 splits)...")
        ho_accs = held_out_split(X, y, n_splits=100)
        print(f"      Mean: {np.mean(ho_accs):.1%} ± {np.std(ho_accs):.1%}")
        print(f"      Range: [{np.min(ho_accs):.1%}, {np.max(ho_accs):.1%}]")
        print(f"      95% CI: [{np.percentile(ho_accs, 2.5):.1%}, {np.percentile(ho_accs, 97.5):.1%}]")

        # B2: Permutation test
        print(f"\n  B2: Permutation test (1000 shuffles, LOO on each)...")
        null_accs = permutation_test(X, y, n_perms=1000)
        p_val = np.mean(null_accs >= loo_acc)
        print(f"      Null distribution: mean={np.mean(null_accs):.1%}, SD={np.std(null_accs):.1%}")
        print(f"      95th percentile of null: {np.percentile(null_accs, 95):.1%}")
        print(f"      99th percentile of null: {np.percentile(null_accs, 99):.1%}")
        print(f"      Real LOO: {loo_acc:.1%}")
        print(f"      p-value: {p_val:.4f}" + (" ***" if p_val < 0.001 else " **" if p_val < 0.01 else " *" if p_val < 0.05 else " (ns)"))

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
