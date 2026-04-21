#!/usr/bin/env python3
"""
threshold_sweep_real_llm.py

Post-hoc analysis of the adaptive stopping threshold on real LLM data (n=98 Llama 3.2 3B).

For each trial, we have feature_trajectory (per-turn feature scores). We simulate the
classifier confidence at each Q/A turn using a LOO-trained logistic regression, then
test different stopping thresholds τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9} with min_questions=2.

Also analyzes misclassified trials: at which question did stopping occur, and would
asking more questions have changed the outcome?

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python experiments/threshold_sweep_real_llm.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_KEYS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
DATA_PATH = "data/results/ollama_eval_llama3_2_3b_latest.json"
MIN_QUESTIONS = 2


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    trials = []
    for r in data["results"]:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue

        # Build per-turn cumulative-average feature vectors
        # turn_features[t] = average of features over turns 0..t
        turn_features = []
        running = {k: [] for k in FEATURE_KEYS}
        for step in traj:
            for k in FEATURE_KEYS:
                v = step.get(k)
                if v is not None:
                    running[k].append(float(v))
            if all(len(running[k]) > 0 for k in FEATURE_KEYS):
                vec = [np.mean(running[k]) for k in FEATURE_KEYS]
                turn_features.append(vec)

        if len(turn_features) < MIN_QUESTIONS:
            continue

        label = 1 if r["ground_truth"] == "lying" else 0
        trials.append({
            "claim": r.get("claim", ""),
            "label": label,
            "turn_features": [np.array(v) for v in turn_features],
            "n_turns": len(turn_features),
        })
    return trials


def simulate_threshold(trials, tau, scalers_and_clfs, min_q=MIN_QUESTIONS):
    """
    For each trial, simulate stopping at the first turn t >= min_q where
    confidence >= tau. Uses per-trial LOO classifier (scalers_and_clfs[i]).

    Returns list of dicts with prediction, stopping_q, correct.
    """
    results = []
    for i, trial in enumerate(trials):
        scaler, clf = scalers_and_clfs[i]
        label = trial["label"]
        turns = trial["turn_features"]
        n_turns = len(turns)

        pred = None
        stopping_q = n_turns  # default: exhaust all questions

        for t in range(n_turns):
            feat = turns[t].reshape(1, -1)
            feat_s = scaler.transform(feat)
            prob = clf.predict_proba(feat_s)[0]
            # confidence = |P(lying) - 0.5| * 2, ranges [0,1]
            conf = abs(prob[1] - 0.5) * 2

            if t + 1 >= min_q and conf >= tau:
                pred = clf.predict(feat_s)[0]
                stopping_q = t + 1
                break

        if pred is None:
            # Exhausted questions — use final prediction
            feat = turns[-1].reshape(1, -1)
            feat_s = scaler.transform(feat)
            pred = clf.predict(feat_s)[0]
            stopping_q = n_turns

        results.append({
            "label": label,
            "pred": pred,
            "stopping_q": stopping_q,
            "correct": int(pred == label),
        })
    return results


def compute_summary(results, label_name=""):
    y_true = np.array([r["label"] for r in results])
    y_pred = np.array([r["pred"] for r in results])
    avg_q = np.mean([r["stopping_q"] for r in results])

    overall = np.mean(y_pred == y_true)
    truth_mask = y_true == 0
    lie_mask = y_true == 1
    truthful_acc = np.mean(y_pred[truth_mask] == y_true[truth_mask]) if truth_mask.any() else 0
    lying_acc = np.mean(y_pred[lie_mask] == y_true[lie_mask]) if lie_mask.any() else 0
    return {
        "overall": overall,
        "truthful": truthful_acc,
        "lying": lying_acc,
        "avg_q": avg_q,
    }


def train_loo_classifiers(trials):
    """Train one LOO classifier per trial (leave trial i out, train on all others)."""
    X_all = np.array([t["turn_features"][-1] for t in trials])  # final-turn features
    y_all = np.array([t["label"] for t in trials])

    scalers_and_clfs = []
    for i in range(len(trials)):
        train_idx = [j for j in range(len(trials)) if j != i]
        X_train = X_all[train_idx]
        y_train = y_all[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_s, y_train)
        scalers_and_clfs.append((scaler, clf))
    return scalers_and_clfs


def analyze_misclassifications(trials, scalers_and_clfs, tau=0.8):
    """For misclassified trials, report stopping Q and whether more questions would help."""
    results = simulate_threshold(trials, tau, scalers_and_clfs)
    misclassified = [(i, r) for i, r in enumerate(results) if not r["correct"]]

    print(f"\nMisclassified trials (τ={tau}, n={len(misclassified)}):")
    print(f"  {'#':<4} {'GT':<8} {'Pred':<8} {'Stop Q':<8} {'Max Q':<8} {'Claim'}")
    print(f"  {'-'*70}")
    for i, r in misclassified:
        trial = trials[i]
        gt_str = "truthful" if r["label"] == 0 else "lying"
        pred_str = "truthful" if r["pred"] == 0 else "lying"
        print(f"  {i:<4} {gt_str:<8} {pred_str:<8} {r['stopping_q']:<8} {trial['n_turns']:<8} {trial['claim'][:40]}")

    # Would more questions help? Simulate exhausting all questions
    improved = 0
    for i, r in misclassified:
        trial = trials[i]
        max_q = trial["n_turns"]
        if r["stopping_q"] < max_q:
            # Simulate not stopping until max Q
            scaler, clf = scalers_and_clfs[i]
            feat = trial["turn_features"][-1].reshape(1, -1)
            feat_s = scaler.transform(feat)
            final_pred = clf.predict(feat_s)[0]
            if final_pred == r["label"]:
                improved += 1
                print(f"    Trial {i}: WOULD IMPROVE with more questions (stops at Q{r['stopping_q']}, max={max_q})")

    if improved == 0:
        print(f"\n  No misclassified trials would improve by asking more questions.")
    else:
        print(f"\n  {improved}/{len(misclassified)} misclassified trials would improve with exhaustive questioning.")


def main():
    print(f"Loading data from {DATA_PATH}")
    trials = load_data(DATA_PATH)
    print(f"Loaded {len(trials)} trials with feature trajectories")
    print(f"  Truthful: {sum(1 for t in trials if t['label']==0)}, Lying: {sum(1 for t in trials if t['label']==1)}")
    print(f"  Avg turns per trial: {np.mean([t['n_turns'] for t in trials]):.1f}")
    print()

    print("Training LOO classifiers (one per trial)...")
    scalers_and_clfs = train_loo_classifiers(trials)
    print("Done.")

    taus = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n{'='*72}")
    print("REAL-LLM THRESHOLD SWEEP (LOO recalibrated classifier, min_questions=2)")
    print(f"{'='*72}")
    print(f"  {'τ':<6} {'Avg Q':<8} {'Overall':>9} {'Truthful':>10} {'Lying':>8}")
    print(f"  {'-'*50}")

    for tau in taus:
        results = simulate_threshold(trials, tau, scalers_and_clfs)
        s = compute_summary(results)
        print(f"  {tau:<6.1f} {s['avg_q']:<8.2f} {s['overall']:>9.1%} {s['truthful']:>10.1%} {s['lying']:>8.1%}")

    # Misclassification analysis at default τ=0.8
    analyze_misclassifications(trials, scalers_and_clfs, tau=0.8)

    print()


if __name__ == "__main__":
    main()
