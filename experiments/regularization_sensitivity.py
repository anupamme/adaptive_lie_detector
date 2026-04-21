#!/usr/bin/env python3
"""
Regularization sensitivity analysis for LOO recalibration.

Tests whether LOO accuracy is sensitive to the LogisticRegression C parameter
(inverse of regularization strength). Addresses reviewer minor concern about
hyperparameter selection within LOO loop.

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python3 experiments/regularization_sensitivity.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

FEATURE_KEYS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

# Test both Llama and Mistral
LLAMA_PATH = "data/results/ollama_eval_llama3_2_3b_latest.json"
MISTRAL_PATH = "data/results/ollama_eval_mistral_7b_latest.json"


def load_features_and_labels(path):
    """Load features and labels from saved results."""
    with open(path) as f:
        data = json.load(f)
    X, y = [], []
    for r in data["results"]:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        feat_vals = {k: [] for k in FEATURE_KEYS}
        for step in traj:
            for k in FEATURE_KEYS:
                v = step.get(k)
                if v is not None:
                    feat_vals[k].append(float(v))
        if any(len(feat_vals[k]) == 0 for k in FEATURE_KEYS):
            continue
        vec = [np.mean(feat_vals[k]) for k in FEATURE_KEYS]
        label = 1 if r["ground_truth"] == "lying" else 0
        X.append(vec)
        y.append(label)
    return np.array(X), np.array(y)


def run_loo_with_C(X, y, C=1.0):
    """Run LOO with specified regularization strength."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        preds[test_idx] = clf.predict(X_test_s)

    truthful_mask = y == 0
    lying_mask = y == 1
    overall = accuracy_score(y, preds)
    truthful_acc = accuracy_score(y[truthful_mask], preds[truthful_mask])
    lying_acc = accuracy_score(y[lying_mask], preds[lying_mask])

    return overall, truthful_acc, lying_acc


def sweep_regularization(X, y, model_name):
    """Sweep over C values and report accuracy."""
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    print(f"\n{'=' * 70}")
    print(f"REGULARIZATION SENSITIVITY: {model_name}")
    print(f"{'=' * 70}")
    print(f"  n={len(y)} ({(y==0).sum()} truthful, {(y==1).sum()} lying)")
    print(f"\n  {'C':>8} {'Overall':>10} {'Truthful':>10} {'Lying':>10}")
    print(f"  {'-' * 40}")

    for C in C_values:
        overall, truth, lie = run_loo_with_C(X, y, C=C)
        print(f"  {C:>8.2f} {overall:>10.1%} {truth:>10.1%} {lie:>10.1%}")

    print(f"\n  Interpretation:")
    print(f"  - C = inverse regularization strength (higher C = less regularization)")
    print(f"  - If accuracy varies <2pp across C, model is robust to hyperparameter choice")
    print(f"  - Default C=1.0 is reasonable for this sample size")


def main():
    print("=" * 70)
    print("REGULARIZATION SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nTesting LOO recalibration accuracy across regularization strengths.")
    print("Addresses reviewer concern about hyperparameter selection within LOO loop.")

    # Llama
    if os.path.exists(LLAMA_PATH):
        X_llama, y_llama = load_features_and_labels(LLAMA_PATH)
        sweep_regularization(X_llama, y_llama, "Llama 3.2 3B")
    else:
        print(f"\nSkipping Llama (file not found: {LLAMA_PATH})")

    # Mistral
    if os.path.exists(MISTRAL_PATH):
        X_mistral, y_mistral = load_features_and_labels(MISTRAL_PATH)
        sweep_regularization(X_mistral, y_mistral, "Mistral 7B")
    else:
        print(f"\nSkipping Mistral (file not found: {MISTRAL_PATH})")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("If accuracy is stable across C values (±2pp), the LOO results are robust")
    print("to regularization strength choice. Default C=1.0 is standard practice for")
    print("sample sizes in the n=50-100 range.")


if __name__ == "__main__":
    main()
