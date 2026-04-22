#!/usr/bin/env python3
"""
analyze_70b_results.py — Comprehensive analysis of Llama 70B experiment results.

Computes:
  1. Feature Cohen's d (truthful vs lying)
  2. LOO recalibration + 5-fold CV
  3. Fixed-K ablation (K=1..8)
  4. Hedging regex baseline (LOO)
  5. Knowledge precheck summary
  6. Error analysis (wrong predictions)

Usage:
    source /Users/mediratta/code/interpret/adaptive_lie_detector/.venv/bin/activate
    python3 experiments/analyze_70b_results.py
"""

import json
import os
import re
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data", "results", "bedrock_eval_llama70b_latest.json")

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

# ---------------------------------------------------------------------------
# Hedging / refusal patterns (from hedging_baseline.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    results = [r for r in data["results"] if r.get("status") != "error"]
    return results


def per_trial_feature_means(results):
    """Return (X, y) where X is n_trials x 5 averaged features, y is binary labels."""
    rows, labels = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        means = []
        for f in FEATURES:
            vals = [t[f] for t in traj if t.get(f) is not None]
            means.append(np.mean(vals) if vals else 0.0)
        rows.append(means)
        labels.append(1 if r["ground_truth"] == "lying" else 0)
    return np.array(rows), np.array(labels)


def per_trial_feature_means_first_k(results, k):
    """Return (X, y) using only the first K questions' features."""
    rows, labels = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        traj_k = traj[:k]
        means = []
        for f in FEATURES:
            vals = [t[f] for t in traj_k if t.get(f) is not None]
            means.append(np.mean(vals) if vals else 0.0)
        rows.append(means)
        labels.append(1 if r["ground_truth"] == "lying" else 0)
    return np.array(rows), np.array(labels)


def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def loo_evaluate(X, y):
    """LOO cross-validation. Returns (accuracy, preds)."""
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    acc = np.mean(preds == y)
    return acc, preds


def extract_hedging_features(conversation):
    """Extract 7 regex-based text features from conversation (same as hedging_baseline.py)."""
    assistant_texts = [msg["content"] for msg in conversation if msg.get("role") == "assistant"]
    if not assistant_texts:
        return None

    all_text = " ".join(assistant_texts).lower()
    total_words = len(all_text.split())

    hedge_count = sum(all_text.count(hw) for hw in HEDGE_WORDS)
    refusal_count = sum(
        1 for pat in REFUSAL_PATTERNS
        if re.search(pat, " ".join(assistant_texts), re.IGNORECASE)
    )
    lengths = [len(t.split()) for t in assistant_texts]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths) if len(lengths) > 1 else 0
    confidence_count = sum(all_text.count(cw) for cw in CONFIDENCE_WORDS)
    question_count = sum(t.count("?") for t in assistant_texts)

    return [
        hedge_count,
        hedge_count / max(total_words, 1) * 100,  # hedge_rate
        refusal_count,
        avg_length,
        std_length,
        confidence_count,
        question_count,
    ]


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    results = load_data()
    n = len(results)
    print("=" * 75)
    print(f"  COMPREHENSIVE ANALYSIS — Llama 70B  (n={n} trials)")
    print("=" * 75)

    X, y = per_trial_feature_means(results)
    truthful_mask = y == 0
    lying_mask = y == 1
    n_t = truthful_mask.sum()
    n_l = lying_mask.sum()
    print(f"\n  Completed trials: {n}  (truthful={n_t}, lying={n_l})")

    # -----------------------------------------------------------------------
    # 1. Feature Cohen's d
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  1. FEATURE COHEN'S d  (truthful vs lying, per-trial averages)")
    print("=" * 75)
    print(f"\n  {'Feature':<16} {'Truth mean':>11} {'Lying mean':>11} {'|d|':>8}  Direction")
    print("  " + "-" * 65)
    for j, feat in enumerate(FEATURES):
        t_vals = X[truthful_mask, j]
        l_vals = X[lying_mask, j]
        d = cohens_d(t_vals, l_vals)
        direction = "truth > lying" if d > 0 else "lying > truth"
        print(f"  {feat:<16} {np.mean(t_vals):11.3f} {np.mean(l_vals):11.3f} {abs(d):8.3f}  {direction}")
        # Also show std
    print()
    print(f"  {'Feature':<16} {'Truth std':>11} {'Lying std':>11}")
    print("  " + "-" * 40)
    for j, feat in enumerate(FEATURES):
        t_vals = X[truthful_mask, j]
        l_vals = X[lying_mask, j]
        print(f"  {feat:<16} {np.std(t_vals, ddof=1):11.3f} {np.std(l_vals, ddof=1):11.3f}")

    # -----------------------------------------------------------------------
    # 2. LOO recalibration + 5-fold CV
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  2. LOO RECALIBRATION  (LogisticRegression on 5 averaged features)")
    print("=" * 75)

    acc, preds = loo_evaluate(X, y)
    t_acc = np.mean(preds[truthful_mask] == y[truthful_mask])
    l_acc = np.mean(preds[lying_mask] == y[lying_mask])
    f1 = f1_score(y, preds)
    print(f"\n  LOO overall accuracy:   {acc:.1%}  ({int(acc * n)}/{n})")
    print(f"  LOO truthful accuracy:  {t_acc:.1%}  ({int(t_acc * n_t)}/{n_t})")
    print(f"  LOO lying accuracy:     {l_acc:.1%}  ({int(l_acc * n_l)}/{n_l})")
    print(f"  LOO F1 (lying class):   {f1:.3f}")
    print()
    print("  Full classification report (LOO):")
    report = classification_report(y, preds, target_names=["truthful", "lying"], digits=3)
    for line in report.split("\n"):
        print(f"    {line}")

    # 5-fold stratified CV
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"\n  5-fold stratified CV:   {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")
    print(f"    Fold scores: {', '.join(f'{s:.1%}' for s in cv_scores)}")

    # -----------------------------------------------------------------------
    # 3. Fixed-K ablation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  3. FIXED-K ABLATION  (LOO accuracy using first K questions)")
    print("=" * 75)

    # Determine max K from data
    max_k = max(len(r.get("feature_trajectory", [])) for r in results)
    print(f"\n  Max trajectory length: {max_k}")
    print(f"\n  {'K':>3}  {'LOO Acc':>8}  {'n usable':>9}  {'truthful':>9}  {'lying':>9}")
    print("  " + "-" * 50)
    for k in range(1, min(max_k + 1, 9)):
        Xk, yk = per_trial_feature_means_first_k(results, k)
        if len(yk) < 5:
            print(f"  {k:>3}  {'(skip)':>8}  {len(yk):>9}")
            continue
        acc_k, preds_k = loo_evaluate(Xk, yk)
        t_mask_k = yk == 0
        l_mask_k = yk == 1
        t_acc_k = np.mean(preds_k[t_mask_k] == yk[t_mask_k]) if t_mask_k.sum() > 0 else 0
        l_acc_k = np.mean(preds_k[l_mask_k] == yk[l_mask_k]) if l_mask_k.sum() > 0 else 0
        print(f"  {k:>3}  {acc_k:>8.1%}  {len(yk):>9}  {t_acc_k:>9.1%}  {l_acc_k:>9.1%}")

    # -----------------------------------------------------------------------
    # 4. Hedging regex baseline
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  4. HEDGING REGEX BASELINE  (7 text features, LOO)")
    print("=" * 75)

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

    X_hedge = np.array(hedge_rows)
    y_hedge = np.array(hedge_labels)

    HEDGE_FEAT_NAMES = [
        "hedge_count", "hedge_rate", "refusal_count",
        "avg_resp_len", "std_resp_len", "confidence_count", "question_count",
    ]

    t_mask_h = y_hedge == 0
    l_mask_h = y_hedge == 1

    print(f"\n  Usable trials: {len(y_hedge)}  (truthful={t_mask_h.sum()}, lying={l_mask_h.sum()})")
    print(f"\n  {'Feature':<16} {'Truth mean':>11} {'Lying mean':>11} {'Diff':>10}")
    print("  " + "-" * 55)
    for j, fn in enumerate(HEDGE_FEAT_NAMES):
        tm = np.mean(X_hedge[t_mask_h, j])
        lm = np.mean(X_hedge[l_mask_h, j])
        print(f"  {fn:<16} {tm:11.2f} {lm:11.2f} {lm - tm:+10.2f}")

    hedge_acc, hedge_preds = loo_evaluate(X_hedge, y_hedge)
    h_t_acc = np.mean(hedge_preds[t_mask_h] == y_hedge[t_mask_h])
    h_l_acc = np.mean(hedge_preds[l_mask_h] == y_hedge[l_mask_h])
    h_f1 = f1_score(y_hedge, hedge_preds)
    print(f"\n  Hedging baseline LOO accuracy:  {hedge_acc:.1%}")
    print(f"    Truthful accuracy:            {h_t_acc:.1%}")
    print(f"    Lying accuracy:               {h_l_acc:.1%}")
    print(f"    F1 (lying class):             {h_f1:.3f}")
    print(f"\n  Comparison: LLM features LOO = {acc:.1%},  Hedging LOO = {hedge_acc:.1%},  Delta = {acc - hedge_acc:+.1%}")

    # -----------------------------------------------------------------------
    # 5. Knowledge precheck summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  5. KNOWLEDGE PRECHECK SUMMARY")
    print("=" * 75)

    precheck_table = {"truthful": {"True": 0, "False": 0, "None": 0},
                      "lying": {"True": 0, "False": 0, "None": 0}}
    for r in results:
        gt = r["ground_truth"]
        mk = r.get("knowledge_precheck", {}).get("model_knows")
        key = str(mk)
        if key not in precheck_table[gt]:
            key = "None"
        precheck_table[gt][key] += 1

    print(f"\n  {'':>12} {'knows=True':>12} {'knows=False':>12} {'knows=None':>12}")
    print("  " + "-" * 50)
    for gt in ["truthful", "lying"]:
        row = precheck_table[gt]
        print(f"  {gt:>12} {row['True']:>12} {row['False']:>12} {row['None']:>12}")
    total_knows = precheck_table["truthful"]["True"] + precheck_table["lying"]["True"]
    total_not = precheck_table["truthful"]["False"] + precheck_table["lying"]["False"]
    total_none = precheck_table["truthful"]["None"] + precheck_table["lying"]["None"]
    print(f"  {'total':>12} {total_knows:>12} {total_not:>12} {total_none:>12}")

    # Accuracy conditioned on knowledge
    knows_correct = sum(1 for r in results
                        if r.get("knowledge_precheck", {}).get("model_knows") is True and r["correct"])
    knows_total = sum(1 for r in results
                      if r.get("knowledge_precheck", {}).get("model_knows") is True)
    notknows_correct = sum(1 for r in results
                           if r.get("knowledge_precheck", {}).get("model_knows") is False and r["correct"])
    notknows_total = sum(1 for r in results
                         if r.get("knowledge_precheck", {}).get("model_knows") is False)
    if knows_total > 0:
        print(f"\n  Pipeline accuracy when model_knows=True:  {knows_correct}/{knows_total} = {knows_correct/knows_total:.1%}")
    if notknows_total > 0:
        print(f"  Pipeline accuracy when model_knows=False: {notknows_correct}/{notknows_total} = {notknows_correct/notknows_total:.1%}")

    # -----------------------------------------------------------------------
    # 6. Error analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  6. ERROR ANALYSIS  (all wrong predictions)")
    print("=" * 75)

    wrong = [r for r in results if not r["correct"]]
    print(f"\n  Total wrong: {len(wrong)} / {n}  ({len(wrong)/n:.1%})")

    if wrong:
        print(f"\n  {'#':>3}  {'Ground Truth':<13} {'Prediction':<12} {'Conf':>6} {'Qs':>3}  {'model_knows':>12}  Claim")
        print("  " + "-" * 110)
        for i, r in enumerate(wrong, 1):
            mk = r.get("knowledge_precheck", {}).get("model_knows")
            claim = r["claim"]
            if len(claim) > 55:
                claim = claim[:52] + "..."
            print(f"  {i:>3}  {r['ground_truth']:<13} {r['prediction']:<12} {r['confidence']:>6.3f} {r['questions_asked']:>3}  {str(mk):>12}  {claim}")

        # Breakdown
        wrong_t = [r for r in wrong if r["ground_truth"] == "truthful"]
        wrong_l = [r for r in wrong if r["ground_truth"] == "lying"]
        print(f"\n  Wrong on truthful claims (false positives): {len(wrong_t)}")
        print(f"  Wrong on lying claims (false negatives):    {len(wrong_l)}")

        # Confidence distribution for wrong predictions
        wrong_confs = [r["confidence"] for r in wrong]
        print(f"\n  Confidence on wrong predictions: mean={np.mean(wrong_confs):.3f}, "
              f"std={np.std(wrong_confs):.3f}, min={min(wrong_confs):.3f}, max={max(wrong_confs):.3f}")
        right_confs = [r["confidence"] for r in results if r["correct"]]
        print(f"  Confidence on right predictions: mean={np.mean(right_confs):.3f}, "
              f"std={np.std(right_confs):.3f}, min={min(right_confs):.3f}, max={max(right_confs):.3f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    pipeline_acc = sum(1 for r in results if r["correct"]) / n
    print(f"\n  Raw pipeline accuracy:          {pipeline_acc:.1%}")
    print(f"  LOO recalibration accuracy:     {acc:.1%}")
    print(f"  Hedging regex baseline (LOO):   {hedge_acc:.1%}")
    print(f"  5-fold CV (mean +/- std):       {cv_scores.mean():.1%} +/- {cv_scores.std():.1%}")

    # Best K
    best_k_acc = 0
    best_k = 1
    for k in range(1, min(max_k + 1, 9)):
        Xk, yk = per_trial_feature_means_first_k(results, k)
        if len(yk) < 5:
            continue
        a, _ = loo_evaluate(Xk, yk)
        if a > best_k_acc:
            best_k_acc = a
            best_k = k
    print(f"  Best fixed-K LOO:               K={best_k} -> {best_k_acc:.1%}")
    print()


if __name__ == "__main__":
    main()
