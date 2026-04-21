#!/usr/bin/env python3
"""Recalibration on Mistral 7B data (standalone, n=100) with LOO margin analysis."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

FEATURE_KEYS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
DATA_PATH = "data/results/ollama_eval_mistral_7b_latest.json"

def load_features_and_labels(path):
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

def run_loo(X, y):
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    probs = np.zeros(len(y), dtype=float)  # P(deceptive)
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        preds[test_idx] = clf.predict(X_test_s)
        probs[test_idx] = clf.predict_proba(X_test_s)[0, 1]  # P(lying)

    truthful_mask = y == 0
    lying_mask = y == 1
    overall = accuracy_score(y, preds)
    truthful_acc = accuracy_score(y[truthful_mask], preds[truthful_mask])
    lying_acc = accuracy_score(y[lying_mask], preds[lying_mask])
    f1 = f1_score(y, preds, zero_division=0)
    return overall, truthful_acc, lying_acc, f1, probs

X, y = load_features_and_labels(DATA_PATH)
print(f"Loaded {len(X)} Mistral 7B trials ({(y==0).sum()} truthful, {(y==1).sum()} lying)")
print("\nMock-trained baseline (from experiment output):")
print(f"  Overall: 86.0%, Truthful: 80.0%, Lying: 92.0%")

print("\nRunning LOO recalibration on Mistral 7B data...")
overall, truth, lie, f1_score_val, probs = run_loo(X, y)
print(f"  Overall: {overall:.1%}, Truthful: {truth:.1%}, Lying: {lie:.1%}, F1: {f1_score_val:.3f}")

# Decision margin analysis
margins = np.abs(probs - 0.5)
truthful_mask = y == 0
lying_mask = y == 1

print("\n" + "=" * 60)
print("LOO DECISION MARGIN ANALYSIS")
print("=" * 60)
print(f"  margin = |P(deceptive) - 0.5|")
print(f"\n  {'':15} {'Min':>6} {'Mean':>7} {'Median':>7} {'Std':>6} {'Max':>6}")
print(f"  {'-'*50}")
for label, mask in [("All trials", np.ones(len(y), dtype=bool)),
                     ("Truthful", truthful_mask),
                     ("Lying", lying_mask)]:
    m = margins[mask]
    print(f"  {label:15} {m.min():>6.3f} {m.mean():>7.3f} {np.median(m):>7.3f} {m.std():>6.3f} {m.max():>6.3f}")

print(f"\n  Margin distribution:")
for threshold in [0.1, 0.2, 0.3, 0.4]:
    n_below = (margins < threshold).sum()
    print(f"    margin < {threshold}: {n_below}/{len(margins)} trials")

# Hardest trials
sorted_idx = np.argsort(margins)
print(f"\n  5 hardest trials (smallest margins):")
print(f"  {'Trial':>5} {'Truth':>8} {'P(dec)':>7} {'Margin':>7} {'Correct':>8}")
for i in sorted_idx[:5]:
    correct = "yes" if (probs[i] >= 0.5) == (y[i] == 1) else "NO"
    truth_label = "lying" if y[i] == 1 else "truthful"
    print(f"  {i:>5} {truth_label:>8} {probs[i]:>7.3f} {margins[i]:>7.3f} {correct:>8}")
