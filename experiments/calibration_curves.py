#!/usr/bin/env python3
"""
Calibration analysis for recalibrated classifiers.

Computes Expected Calibration Error (ECE) and reliability statistics
for LOO-recalibrated classifiers. Addresses reviewer Q3: calibration curves.

Calibration measures whether predicted probabilities match actual outcomes:
- Well-calibrated: P(lying)=0.7 → 70% of such predictions are actually lying
- Poorly calibrated: High confidence but low accuracy (or vice versa)

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python3 experiments/calibration_curves.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

FEATURE_KEYS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

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


def run_loo_with_probs(X, y):
    """Run LOO and return predicted probabilities."""
    loo = LeaveOneOut()
    probs = np.zeros(len(y), dtype=float)
    preds = np.zeros_like(y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        probs[test_idx] = clf.predict_proba(X_test_s)[0, 1]  # P(lying)
        preds[test_idx] = clf.predict(X_test_s)

    return probs, preds


def compute_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE) and reliability statistics.

    ECE = weighted average of |predicted_prob - actual_fraction| across bins.
    Lower ECE = better calibration (0 = perfect).
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_counts = []
    bin_accuracies = []
    bin_confidences = []

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Last bin includes right edge
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])

        count = mask.sum()
        if count > 0:
            accuracy = y_true[mask].mean()
            confidence = y_prob[mask].mean()
        else:
            accuracy = 0.0
            confidence = bin_centers[i]

        bin_counts.append(count)
        bin_accuracies.append(accuracy)
        bin_confidences.append(confidence)

    # Expected Calibration Error
    total = len(y_true)
    ece = sum(
        (count / total) * abs(acc - conf)
        for count, acc, conf in zip(bin_counts, bin_accuracies, bin_confidences)
        if count > 0
    )

    return ece, bin_centers, bin_counts, bin_accuracies, bin_confidences


def print_calibration_table(model_name, y_true, y_prob, y_pred):
    """Print calibration analysis."""
    ece, bin_centers, bin_counts, bin_accs, bin_confs = compute_calibration_metrics(y_true, y_prob)

    accuracy = (y_true == y_pred).mean()

    print(f"\n{'=' * 70}")
    print(f"CALIBRATION ANALYSIS: {model_name}")
    print(f"{'=' * 70}")
    print(f"  n={len(y_true)} | Overall accuracy: {accuracy:.1%} | ECE: {ece:.3f}")
    print(f"\n  Reliability Diagram (10 bins):")
    print(f"  {'Bin':>3} {'Range':>15} {'Count':>7} {'Accuracy':>10} {'Avg Prob':>10} {'Error':>8}")
    print(f"  {'-' * 65}")

    for i, (center, count, acc, conf) in enumerate(zip(bin_centers, bin_counts, bin_accs, bin_confs)):
        if count > 0:
            error = abs(acc - conf)
            print(f"  {i+1:>3} {center-0.05:.2f}-{center+0.05:.2f}    {count:>7} {acc:>10.1%} {conf:>10.3f} {error:>8.3f}")

    print(f"\n  Interpretation:")
    print(f"  - ECE < 0.05: Well-calibrated")
    print(f"  - ECE 0.05-0.10: Moderate miscalibration")
    print(f"  - ECE > 0.10: Poorly calibrated")
    print(f"  - Accuracy ≈ Avg Prob in each bin → good calibration")


def main():
    print("=" * 70)
    print("CALIBRATION ANALYSIS FOR RECALIBRATED CLASSIFIERS")
    print("=" * 70)
    print("\nComputes Expected Calibration Error (ECE) for LOO-recalibrated classifiers.")
    print("Well-calibrated: predicted probabilities match actual outcomes.")

    # Llama
    if os.path.exists(LLAMA_PATH):
        print(f"\nLoading Llama data from {LLAMA_PATH}...")
        X_llama, y_llama = load_features_and_labels(LLAMA_PATH)
        print(f"Running LOO with probability extraction...")
        probs_llama, preds_llama = run_loo_with_probs(X_llama, y_llama)
        print_calibration_table("Llama 3.2 3B", y_llama, probs_llama, preds_llama)
    else:
        print(f"\nSkipping Llama (file not found: {LLAMA_PATH})")

    # Mistral
    if os.path.exists(MISTRAL_PATH):
        print(f"\nLoading Mistral data from {MISTRAL_PATH}...")
        X_mistral, y_mistral = load_features_and_labels(MISTRAL_PATH)
        print(f"Running LOO with probability extraction...")
        probs_mistral, preds_mistral = run_loo_with_probs(X_mistral, y_mistral)
        print_calibration_table("Mistral 7B", y_mistral, probs_mistral, preds_mistral)
    else:
        print(f"\nSkipping Mistral (file not found: {MISTRAL_PATH})")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("Compare ECE with mock classifier (paper reports confidence >0.99 for all")
    print("predictions). If recalibrated ECE < 0.10, the classifier is reasonably")
    print("well-calibrated on real LLM data, unlike the mock-trained baseline.")


if __name__ == "__main__":
    main()
