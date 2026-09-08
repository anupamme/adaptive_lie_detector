#!/usr/bin/env python3
"""Comprehensive analysis of Qwen 2.5 7B equalized experiment results.

Computes: LOO accuracy, bootstrap CIs, 5-fold CV, feature Cohen's d,
hedging regex baseline, K=1 LOO, and permutation test.
"""

import json
import os
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")
RESULT_FILE = os.path.join(DATA_DIR, "ollama_eval_qwen2_5_7b_prompt_equalized_latest.json")


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def per_trial_feature_means(results):
    """Extract per-trial mean features and labels."""
    X, y = [], []
    for trial in results:
        features = trial.get("features") or trial.get("feature_trajectory", [])
        if not features:
            continue
        if isinstance(features[0], dict):
            keys = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
            feat_matrix = [[f.get(k, 5.0) for k in keys] for f in features]
        else:
            feat_matrix = features
        mean_feat = np.mean(feat_matrix, axis=0)
        X.append(mean_feat)
        gt = trial.get("ground_truth", trial.get("is_lying", False))
        if isinstance(gt, str):
            gt = gt.lower() in ("lying", "true", "deceptive", "1")
        y.append(int(gt))
    return np.array(X), np.array(y)


def loo_accuracy(X, y):
    """LOO-recalibrated accuracy."""
    loo = LeaveOneOut()
    correct = []
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        correct.append(int(pred[0] == y[test_idx][0]))
    correct = np.array(correct)
    return correct, correct.mean()


def bootstrap_ci(correct, n_boot=10000, seed=42):
    """Bootstrap 95% CI on accuracy."""
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_boot):
        idx = rng.choice(len(correct), len(correct), replace=True)
        accs.append(correct[idx].mean())
    return np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def kfold_cv(X, y, k=5, n_repeats=20):
    """Repeated k-fold CV."""
    accs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42 + rep)
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))
        accs.append(np.mean(fold_accs))
    return np.mean(accs), np.std(accs)


def cohens_d(X, y):
    """Cohen's d for each feature."""
    keys = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
    ds = {}
    for i, k in enumerate(keys):
        truth = X[y == 0, i]
        lie = X[y == 1, i]
        pooled_std = np.sqrt(((len(truth)-1)*truth.std()**2 + (len(lie)-1)*lie.std()**2) / (len(truth)+len(lie)-2))
        if pooled_std > 0:
            d = (truth.mean() - lie.mean()) / pooled_std
        else:
            d = 0
        ds[k] = d
    return ds


def k1_loo(X, y):
    """LOO accuracy using only K=1 features (first question only)."""
    # X already has per-trial means; for K=1 we need first-question features
    # We'll return None if we can't extract K=1 features from raw data
    return None


def hedging_baseline(results):
    """Hedging-word regex baseline LOO accuracy."""
    HEDGE_WORDS = re.compile(r'\b(maybe|perhaps|possibly|i think|not sure|i believe|might|could be|it seems|unclear)\b', re.I)
    REFUSAL_WORDS = re.compile(r'\b(i cannot|i can\'t|actually|that\'s not correct|that is not correct|incorrect|i must correct|not accurate|i should note)\b', re.I)
    CONFIDENCE_WORDS = re.compile(r'\b(certainly|definitely|absolutely|clearly|obviously|of course|without doubt|undoubtedly)\b', re.I)

    X_hedge, y_hedge = [], []
    for trial in results:
        responses = []
        transcript = trial.get("transcript", [])
        for turn in transcript:
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                responses.append(turn.get("content", ""))
            elif isinstance(turn, dict) and "response" in turn:
                responses.append(turn["response"])

        if not responses:
            # Try to extract from other fields
            conv = trial.get("conversation", [])
            for turn in conv:
                if isinstance(turn, dict) and turn.get("role") == "assistant":
                    responses.append(turn.get("content", ""))

        if not responses:
            continue

        all_text = " ".join(responses)
        words = all_text.split()
        n_words = max(len(words), 1)

        hedge_count = len(HEDGE_WORDS.findall(all_text))
        hedge_rate = hedge_count / n_words
        refusal_count = len(REFUSAL_WORDS.findall(all_text))
        avg_len = np.mean([len(r.split()) for r in responses]) if responses else 0
        len_sd = np.std([len(r.split()) for r in responses]) if len(responses) > 1 else 0
        conf_count = len(CONFIDENCE_WORDS.findall(all_text))
        q_count = all_text.count("?")

        X_hedge.append([hedge_count, hedge_rate, refusal_count, avg_len, len_sd, conf_count, q_count])

        gt = trial.get("ground_truth", trial.get("is_lying", False))
        if isinstance(gt, str):
            gt = gt.lower() in ("lying", "true", "deceptive", "1")
        y_hedge.append(int(gt))

    if not X_hedge:
        return None, None

    X_h = np.array(X_hedge)
    y_h = np.array(y_hedge)

    loo = LeaveOneOut()
    correct = []
    for train_idx, test_idx in loo.split(X_h):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_h[train_idx], y_h[train_idx])
        pred = clf.predict(X_h[test_idx])
        correct.append(int(pred[0] == y_h[test_idx][0]))
    return np.array(correct), np.mean(correct)


def permutation_test(X, y, n_perms=1000, seed=42):
    """Permutation test: shuffle labels, compute LOO each time."""
    rng = np.random.default_rng(seed)
    null_accs = []
    for _ in range(n_perms):
        y_perm = rng.permutation(y)
        _, acc = loo_accuracy(X, y_perm)
        null_accs.append(acc)
    return np.array(null_accs)


def main():
    print("=" * 60)
    print("ANALYSIS: Qwen 2.5 7B Prompt-Equalized")
    print("=" * 60)

    results = load_results(RESULT_FILE)
    X, y = per_trial_feature_means(results)
    n = len(y)
    n_truth = (y == 0).sum()
    n_lie = (y == 1).sum()

    print(f"\nSample: n={n} ({n_truth} truthful, {n_lie} deceptive)")

    # LOO accuracy
    correct, acc = loo_accuracy(X, y)
    truth_correct = correct[y == 0]
    lie_correct = correct[y == 1]
    truth_acc = truth_correct.mean()
    lie_acc = lie_correct.mean()

    print(f"\n--- LOO Recalibrated ---")
    print(f"  Overall:   {acc:.1%} ({int(correct.sum())}/{n})")
    print(f"  Truthful:  {truth_acc:.1%} ({int(truth_correct.sum())}/{n_truth})")
    print(f"  Lying:     {lie_acc:.1%} ({int(lie_correct.sum())}/{n_lie})")

    # F1
    tp = lie_correct.sum()
    fn = n_lie - tp
    fp = n_truth - truth_correct.sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"  F1:        {f1:.3f}")

    # Bootstrap CI
    lo, hi = bootstrap_ci(correct)
    print(f"  Bootstrap 95% CI: [{lo:.1%}, {hi:.1%}]")

    # 5-fold CV
    cv_mean, cv_std = kfold_cv(X, y)
    print(f"  5-fold CV (20 repeats): {cv_mean:.1%} ± {cv_std:.1%}")

    # Feature Cohen's d
    ds = cohens_d(X, y)
    print(f"\n--- Feature Separability (Cohen's d) ---")
    print(f"  (positive d = truth > lie)")
    for k, d in sorted(ds.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "truth > lie" if d > 0 else "lie > truth"
        print(f"  {k:<15} |d| = {abs(d):.2f}  ({direction})")

    # Feature means
    print(f"\n--- Feature Means ---")
    keys = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
    for i, k in enumerate(keys):
        t_mean = X[y == 0, i].mean()
        l_mean = X[y == 1, i].mean()
        print(f"  {k:<15} truth={t_mean:.2f}  lie={l_mean:.2f}")

    # Hedging baseline
    h_correct, h_acc = hedging_baseline(results)
    if h_acc is not None:
        print(f"\n--- Hedging Regex Baseline ---")
        print(f"  LOO accuracy: {h_acc:.1%}")
        h_lo, h_hi = bootstrap_ci(h_correct)
        print(f"  Bootstrap 95% CI: [{h_lo:.1%}, {h_hi:.1%}]")
    else:
        print(f"\n--- Hedging Regex Baseline ---")
        print(f"  Could not extract response text for hedging analysis")

    # Permutation test
    print(f"\n--- Permutation Test (1000 shuffles) ---")
    null_accs = permutation_test(X, y)
    p_val = (null_accs >= acc).mean()
    print(f"  Null mean: {null_accs.mean():.1%}")
    print(f"  Null 99th percentile: {np.percentile(null_accs, 99):.1%}")
    print(f"  Real LOO: {acc:.1%}")
    print(f"  p-value: {p_val:.4f}")

    # Avg questions
    avg_q = np.mean([t.get("questions_asked", t.get("n_questions", 2)) for t in results])
    print(f"\n--- Interrogation ---")
    print(f"  Avg questions: {avg_q:.1f}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY FOR PAPER")
    print(f"{'=' * 60}")
    print(f"  LOO: {acc:.1%} [{lo:.1%}, {hi:.1%}]")
    print(f"  Truthful: {truth_acc:.1%}, Lying: {lie_acc:.1%}, F1: {f1:.3f}")
    print(f"  5-fold CV: {cv_mean:.1%} ± {cv_std:.1%}")
    print(f"  Top feature: {max(ds.items(), key=lambda x: abs(x[1]))[0]} |d|={max(abs(v) for v in ds.values()):.2f}")
    if h_acc is not None:
        print(f"  Hedging regex: {h_acc:.1%}")
    print(f"  Permutation p: {p_val:.4f}")
    print(f"  Avg questions: {avg_q:.1f}")


if __name__ == "__main__":
    main()
