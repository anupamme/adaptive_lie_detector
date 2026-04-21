#!/usr/bin/env python3
"""
recalibrate_on_real_llm.py

Tests whether a classifier trained on real LLM data (instead of mock data) can
recover the truthful accuracy that collapses when using the mock-trained classifier.

Experiment design:
  - Load n=98 Llama 3.2 3B trials from ollama_eval_llama3_2_3b_latest.json
  - Extract per-trial feature vectors (average across questions)
  - Run 5-fold stratified CV and LOO for 5-feature and 3-feature models
  - Compare against mock-trained baseline (61.2% overall, 26.5% truthful on n=98)

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python experiments/recalibrate_on_real_llm.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ALL_FEATURE_KEYS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
# 3-feature reduced model: drop confidence and elaboration (|d|≈0.2 on real LLM)
REDUCED_FEATURE_KEYS = ["consistency", "specificity", "defensiveness"]

DATA_PATH = "data/results/ollama_eval_llama3_2_3b_latest.json"

# Mock-trained baseline results on n=98 Llama 3B trials
MOCK_TRAINED_BASELINE = {
    "overall_accuracy": 0.6122,
    "truthful_accuracy": 0.2653,
    "lying_accuracy": 0.9592,
    "precision": 0.566,
    "recall": 0.959,
    "f1": 0.712,
}


def load_features_and_labels(path: str, feature_keys=None):
    """Load feature vectors (averaged across Q/A turns) and ground-truth labels."""
    if feature_keys is None:
        feature_keys = ALL_FEATURE_KEYS
    with open(path) as f:
        data = json.load(f)

    X, y, claims = [], [], []
    for r in data["results"]:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue

        feat_vals = {k: [] for k in feature_keys}
        for step in traj:
            for k in feature_keys:
                v = step.get(k)
                if v is not None:
                    feat_vals[k].append(float(v))

        if any(len(feat_vals[k]) == 0 for k in feature_keys):
            continue

        vec = [np.mean(feat_vals[k]) for k in feature_keys]
        label = 1 if r["ground_truth"] == "lying" else 0
        X.append(vec)
        y.append(label)
        claims.append(r["claim"][:50])

    return np.array(X), np.array(y), claims


def compute_metrics(y_true, y_pred):
    n = len(y_true)
    truthful_mask = y_true == 0
    lying_mask = y_true == 1
    return {
        "overall_accuracy": accuracy_score(y_true, y_pred),
        "truthful_accuracy": accuracy_score(y_true[truthful_mask], y_pred[truthful_mask]) if truthful_mask.any() else 0,
        "lying_accuracy": accuracy_score(y_true[lying_mask], y_pred[lying_mask]) if lying_mask.any() else 0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n": n,
    }


def run_kfold_cv(X, y, n_splits=5, random_state=42, verbose=True):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics = []
    all_preds = np.zeros_like(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)

        all_preds[test_idx] = preds
        m = compute_metrics(y_test, preds)
        fold_metrics.append(m)
        if verbose:
            print(f"    Fold {fold+1}: acc={m['overall_accuracy']:.1%}  "
                  f"truth={m['truthful_accuracy']:.1%}  "
                  f"lie={m['lying_accuracy']:.1%}  "
                  f"F1={m['f1']:.3f}")

    agg = {}
    for key in ["overall_accuracy", "truthful_accuracy", "lying_accuracy", "precision", "recall", "f1"]:
        vals = [m[key] for m in fold_metrics]
        agg[f"{key}_mean"] = np.mean(vals)
        agg[f"{key}_std"] = np.std(vals)

    pooled = compute_metrics(y, all_preds)
    agg["pooled"] = pooled
    return agg


def run_loo(X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        preds[test_idx] = clf.predict(X_test_s)

    return compute_metrics(y, preds)


def print_full_comparison(kfold_5feat, loo_5feat, kfold_3feat, loo_3feat):
    b = MOCK_TRAINED_BASELINE
    p5 = kfold_5feat["pooled"]
    l5 = loo_5feat
    p3 = kfold_3feat["pooled"]
    l3 = loo_3feat

    print(f"\n{'=' * 90}")
    print("RECALIBRATION RESULTS: Real LLM Data (Llama 3.2 3B, n=98)")
    print(f"{'=' * 90}")
    hdr = f"  {'Metric':<22} {'Mock-trained':>13} {'5-feat CV':>10} {'5-feat LOO':>11} {'3-feat CV':>10} {'3-feat LOO':>11}"
    print(hdr)
    print(f"  {'-' * 79}")
    rows = [
        ("Overall accuracy",  "overall_accuracy"),
        ("Truthful accuracy", "truthful_accuracy"),
        ("Lying accuracy",    "lying_accuracy"),
        ("Precision",         "precision"),
        ("Recall",            "recall"),
        ("F1",                "f1"),
    ]
    for label, key in rows:
        bv = b[key]
        fmt = ".1%" if "accuracy" in key else ".3f"
        print(f"  {label:<22} {bv:>13{fmt}} {p5[key]:>10{fmt}} {l5[key]:>11{fmt}} {p3[key]:>10{fmt}} {l3[key]:>11{fmt}}")

    print(f"\n  5-fold CV (mean ± std) — 5-feature:")
    for key in ["overall_accuracy", "truthful_accuracy", "lying_accuracy", "f1"]:
        label = key.replace("_", " ").capitalize()
        mean = kfold_5feat[f"{key}_mean"]
        std = kfold_5feat[f"{key}_std"]
        print(f"    {label:<24} {mean:.1%} ± {std:.1%}")

    print(f"\n  5-fold CV (mean ± std) — 3-feature:")
    for key in ["overall_accuracy", "truthful_accuracy", "lying_accuracy", "f1"]:
        label = key.replace("_", " ").capitalize()
        mean = kfold_3feat[f"{key}_mean"]
        std = kfold_3feat[f"{key}_std"]
        print(f"    {label:<24} {mean:.1%} ± {std:.1%}")


def main():
    print(f"Loading trial data from: {DATA_PATH}")

    # --- 5-feature model ---
    X5, y, claims = load_features_and_labels(DATA_PATH, feature_keys=ALL_FEATURE_KEYS)
    n_truthful = (y == 0).sum()
    n_lying = (y == 1).sum()
    print(f"Loaded {len(X5)} trials ({n_truthful} truthful, {n_lying} lying)")
    print()

    print("Feature statistics (5-feature, truth vs lie):")
    for i, k in enumerate(ALL_FEATURE_KEYS):
        truth_vals = X5[y == 0, i]
        lie_vals = X5[y == 1, i]
        pooled_std = np.sqrt((np.std(truth_vals)**2 + np.std(lie_vals)**2) / 2)
        d = abs(np.mean(truth_vals) - np.mean(lie_vals)) / pooled_std if pooled_std > 0 else 0
        direction = "truth>lie" if np.mean(truth_vals) > np.mean(lie_vals) else "lie>truth"
        print(f"  {k:<14} truth={np.mean(truth_vals):.2f}±{np.std(truth_vals):.2f}  "
              f"lie={np.mean(lie_vals):.2f}±{np.std(lie_vals):.2f}  "
              f"|d|={d:.2f}  ({direction})")

    print(f"\n5-fold stratified CV — 5-feature (seed=42):")
    kfold_5feat = run_kfold_cv(X5, y, verbose=True)

    print(f"\nLeave-one-out CV — 5-feature:")
    loo_5feat = run_loo(X5, y)
    print(f"  acc={loo_5feat['overall_accuracy']:.1%}  "
          f"truth={loo_5feat['truthful_accuracy']:.1%}  "
          f"lie={loo_5feat['lying_accuracy']:.1%}  "
          f"F1={loo_5feat['f1']:.3f}")

    # --- 3-feature reduced model ---
    X3, _, _ = load_features_and_labels(DATA_PATH, feature_keys=REDUCED_FEATURE_KEYS)
    print(f"\n{'='*60}")
    print(f"3-feature reduced model: {REDUCED_FEATURE_KEYS}")
    print(f"(dropping confidence and elaboration, |d|≈0.2 on real LLM)")
    print(f"{'='*60}")

    print(f"\n5-fold stratified CV — 3-feature (seed=42):")
    kfold_3feat = run_kfold_cv(X3, y, verbose=True)

    print(f"\nLeave-one-out CV — 3-feature:")
    loo_3feat = run_loo(X3, y)
    print(f"  acc={loo_3feat['overall_accuracy']:.1%}  "
          f"truth={loo_3feat['truthful_accuracy']:.1%}  "
          f"lie={loo_3feat['lying_accuracy']:.1%}  "
          f"F1={loo_3feat['f1']:.3f}")

    print_full_comparison(kfold_5feat, loo_5feat, kfold_3feat, loo_3feat)


if __name__ == "__main__":
    main()
