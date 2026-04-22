#!/usr/bin/env python3
"""
Quick analysis of 8B equalized and instructed-matched 70B experiments.
Computes LOO, bootstrap CIs, feature Cohen's d, hedging baseline.
"""

import json
import os
import re
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "results")

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly",
]
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
]
CONFIDENCE_WORDS = [
    "certainly", "definitely", "absolutely", "clearly",
    "of course", "without doubt", "undoubtedly",
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


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled > 0 else 0


def loo_evaluate(X, y):
    preds = np.zeros(len(y), dtype=int)
    probs = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
        probs[test_idx[0]] = pipe.predict_proba(X[test_idx])[0, 1]
    return np.mean(preds == y), preds, probs


def bootstrap_ci(preds, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    correct = (preds == y).astype(int)
    boot_accs = [np.mean(correct[rng.integers(0, len(y), size=len(y))]) for _ in range(n_boot)]
    return np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)


def extract_hedging_features(conversation):
    assistant_texts = [msg["content"] for msg in conversation if msg.get("role") == "assistant"]
    if not assistant_texts:
        return None
    all_text = " ".join(assistant_texts).lower()
    total_words = len(all_text.split())
    hedge_count = sum(all_text.count(hw) for hw in HEDGE_WORDS)
    refusal_count = sum(1 for pat in REFUSAL_PATTERNS
                        if re.search(pat, " ".join(assistant_texts), re.IGNORECASE))
    lengths = [len(t.split()) for t in assistant_texts]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths) if len(lengths) > 1 else 0
    confidence_count = sum(all_text.count(cw) for cw in CONFIDENCE_WORDS)
    question_count = sum(t.count("?") for t in assistant_texts)
    return [hedge_count, hedge_count / max(total_words, 1) * 100,
            refusal_count, avg_length, std_length, confidence_count, question_count]


def analyze_dataset(name, path):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    results = load_results(path)
    X, y = per_trial_feature_means(results)
    n = len(y)
    truthful_mask = y == 0
    lying_mask = y == 1
    n_t, n_l = truthful_mask.sum(), lying_mask.sum()
    print(f"  n={n} (truthful={n_t}, lying={n_l})")

    # Feature Cohen's d
    print(f"\n  {'Feature':<16} {'Truth mean':>11} {'Lying mean':>11} {'|d|':>8}  Direction")
    print("  " + "-" * 65)
    for j, feat in enumerate(FEATURES):
        t_vals = X[truthful_mask, j]
        l_vals = X[lying_mask, j]
        d = cohens_d(t_vals, l_vals)
        direction = "truth > lying" if d > 0 else "lying > truth"
        print(f"  {feat:<16} {np.mean(t_vals):11.3f} {np.mean(l_vals):11.3f} {abs(d):8.3f}  {direction}")

    # LOO
    acc, preds, probs = loo_evaluate(X, y)
    t_acc = np.mean(preds[truthful_mask] == y[truthful_mask])
    l_acc = np.mean(preds[lying_mask] == y[lying_mask])
    f1 = f1_score(y, preds)
    ci_lo, ci_hi = bootstrap_ci(preds, y)

    print(f"\n  LOO Accuracy:   {acc:.1%} (truthful={t_acc:.1%}, lying={l_acc:.1%})")
    print(f"  LOO F1:         {f1:.3f}")
    print(f"  Bootstrap CI:   [{ci_lo:.1%}, {ci_hi:.1%}]")

    # 5-fold CV
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"  5-fold CV:      {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")

    # Hedging baseline
    hedge_rows, hedge_labels = [], []
    for r in results:
        conv = r.get("conversation", [])
        if not conv:
            continue
        feats = extract_hedging_features(conv)
        if feats is None:
            continue
        hedge_rows.append(feats)
        hedge_labels.append(1 if r["ground_truth"] == "lying" else 0)

    X_h = np.array(hedge_rows)
    y_h = np.array(hedge_labels)
    h_acc, h_preds, _ = loo_evaluate(X_h, y_h)
    h_ci_lo, h_ci_hi = bootstrap_ci(h_preds, y_h)
    t_h = y_h == 0
    l_h = y_h == 1
    h_t_acc = np.mean(h_preds[t_h] == y_h[t_h])
    h_l_acc = np.mean(h_preds[l_h] == y_h[l_h])

    print(f"\n  Hedging LOO:    {h_acc:.1%} (truthful={h_t_acc:.1%}, lying={h_l_acc:.1%})")
    print(f"  Hedging CI:     [{h_ci_lo:.1%}, {h_ci_hi:.1%}]")
    print(f"  LLM vs Hedge:   {acc:.1%} vs {h_acc:.1%} (delta={acc-h_acc:+.1%})")

    # Fixed K=1
    rows_k1, labels_k1 = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        t1 = traj[:1]
        means = [np.mean([t[f] for t in t1 if t.get(f) is not None] or [0]) for f in FEATURES]
        rows_k1.append(means)
        labels_k1.append(1 if r["ground_truth"] == "lying" else 0)
    X_k1, y_k1 = np.array(rows_k1), np.array(labels_k1)
    k1_acc, _, _ = loo_evaluate(X_k1, y_k1)
    print(f"  K=1 LOO:        {k1_acc:.1%}")

    # Category breakdown (for equalized data with known category ranges)
    CATEGORY_RANGES = {
        "Scientific":  (0, 30),
        "Historical":  (30, 50),
        "Geographic":  (50, 70),
        "Technology":  (70, 86),
        "Cultural":    (86, 100),
    }
    valid_results = [r for r in results if r.get("feature_trajectory")]
    if len(valid_results) == n and n == 100:
        print(f"\n  Category breakdown (LOO):")
        for cat, (lo, hi) in CATEGORY_RANGES.items():
            cat_correct = sum(1 for i in range(lo, min(hi, n)) if preds[i] == y[i])
            cat_total = min(hi, n) - lo
            if cat_total > 0:
                print(f"    {cat:<14} {cat_correct}/{cat_total} = {cat_correct/cat_total:.1%}")

    return acc, ci_lo, ci_hi, h_acc


def main():
    print("="*70)
    print("  ANALYSIS OF NEW ROUND-4 EXPERIMENTS")
    print("="*70)

    # 8B Equalized
    path_8b = os.path.join(DATA_DIR, "bedrock_eval_llama8b_prompt_equalized_latest.json")
    if os.path.exists(path_8b):
        acc_8b, ci_lo_8b, ci_hi_8b, h_acc_8b = analyze_dataset(
            "Llama 3.1 8B — Prompt-Equalized", path_8b)
    else:
        print("8B equalized data not found!")
        acc_8b = None

    # Instructed-Matched 70B
    path_im = os.path.join(DATA_DIR, "bedrock_eval_llama70b_instructed_matched_latest.json")
    if os.path.exists(path_im):
        acc_im, ci_lo_im, ci_hi_im, h_acc_im = analyze_dataset(
            "Llama 3.3 70B — Instructed-Matched", path_im)
    else:
        print("Instructed-matched data not found!")
        acc_im = None

    # Summary: Scale trend
    print(f"\n{'='*70}")
    print("  SCALE TREND SUMMARY (Equalized LOO)")
    print("="*70)
    print(f"  {'Model':<20} {'LOO':>8} {'CI':>20}")
    print(f"  {'-'*50}")
    print(f"  {'Llama 3.2 3B':<20} {'64.0%':>8} {'[50%, 76%]':>20}")
    if acc_8b is not None:
        print(f"  {'Llama 3.1 8B':<20} {acc_8b:>7.1%} {'[' + f'{ci_lo_8b:.0%}, {ci_hi_8b:.0%}' + ']':>20}")
    print(f"  {'Mistral 7B':<20} {'71.0%':>8} {'[62%, 80%]':>20}")
    print(f"  {'Llama 3.3 70B':<20} {'83.9%':>8} {'[76%, 91%]':>20}")

    if acc_8b is not None:
        trend = [64.0, acc_8b * 100, 71.0, 83.9]
        # Note: 8B is between 3B and 7B in scale. Check if sorted by scale produces monotonic.
        # Scale order: 3B, 8B, 7B, 70B. But 8B > 7B in params, so...
        # Actually Llama 8B and Mistral 7B: 8B > 7B, but they're different families.
        # The real question is whether the accuracy at 8B is consistent with the trend.
        is_between = trend[0] <= trend[1] <= trend[2]
        print(f"\n  8B accuracy between 3B and 7B: {'YES' if is_between else 'NO'}")
        print(f"  8B result: {acc_8b:.1%}")

    # Decomposition for instructed-matched
    if acc_im is not None:
        print(f"\n{'='*70}")
        print("  GAP DECOMPOSITION (Llama 70B)")
        print("="*70)
        print(f"  Equalized LOO:          83.9%")
        print(f"  Instructed-matched LOO: {acc_im:.1%}")
        print(f"  Instructed-original LOO: 93.0%")
        print(f"\n  Claim-type contribution:     {93.0 - acc_im*100:+.1f} pp  (original - matched)")
        print(f"  Instruction-following:       {acc_im*100 - 83.9:+.1f} pp  (matched - equalized)")
        print(f"  Total gap:                   {93.0 - 83.9:+.1f} pp  (original - equalized)")


if __name__ == "__main__":
    main()
