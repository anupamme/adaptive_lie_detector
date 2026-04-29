#!/usr/bin/env python3
"""
analyze_refusal_only.py — EXP-J: Refusal-count-only LOO across all equalized models.

Tests whether a single regex feature (refusal/correction count) can match the
full LLM-extracted pipeline on equalized data, addressing the reviewer's question
about whether findings hold without LLM extraction.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/analyze_refusal_only.py
"""

import json
import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data", "results")

# --- Feature definitions (from hedging_baseline.py) ---

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

LLM_FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly"
]

# All equalized datasets
DATASETS = {
    "Llama 3.2 3B (equalized)":   os.path.join(DATA_DIR, "ollama_eval_llama3_2_3b_prompt_equalized_latest.json"),
    "Llama 3.1 8B (equalized)":   os.path.join(DATA_DIR, "bedrock_eval_llama8b_prompt_equalized_latest.json"),
    "Mistral 7B (equalized)":     os.path.join(DATA_DIR, "ollama_eval_mistral_7b_prompt_equalized_latest.json"),
    "Qwen 2.5 7B (equalized)":    os.path.join(DATA_DIR, "ollama_eval_qwen2_5_7b_prompt_equalized_latest.json"),
    "Qwen 2.5 14B (equalized)":   os.path.join(DATA_DIR, "ollama_eval_qwen2_5_14b_prompt_equalized_latest.json"),
    "Llama 3.3 70B (equalized)":  os.path.join(DATA_DIR, "bedrock_eval_llama70b_prompt_equalized_latest.json"),
    "Claude Haiku 4.5 (equalized)": os.path.join(DATA_DIR, "bedrock_eval_haiku_prompt_equalized_latest.json"),
}


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def loo_accuracy_single(X, y):
    """LOO accuracy for a 1-D feature array."""
    X2d = X.reshape(-1, 1)
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X2d):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X2d[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X2d[test_idx])[0]
    return np.mean(preds == y), preds


def loo_accuracy_multi(X, y):
    """LOO accuracy for multi-dimensional features."""
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    return np.mean(preds == y), preds


def extract_refusal_count(conversation):
    """Extract refusal/correction marker count from a conversation."""
    assistant_texts = [msg["content"] for msg in conversation
                       if msg.get("role") == "assistant"]
    if not assistant_texts:
        return None
    all_text = " ".join(assistant_texts)
    refusal_count = sum(1 for pat in REFUSAL_PATTERNS
                        if re.search(pat, all_text, re.IGNORECASE))
    return refusal_count


def extract_all_text_features(conversation):
    """Extract all 7 hedging features from conversation."""
    assistant_texts = [msg["content"] for msg in conversation
                       if msg.get("role") == "assistant"]
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
    confidence_words = ["certainly", "definitely", "absolutely", "clearly",
                        "of course", "without doubt", "undoubtedly"]
    confidence_count = sum(all_text.count(cw) for cw in confidence_words)
    question_count = sum(t.count("?") for t in assistant_texts)
    return [hedge_count, hedge_count / max(total_words, 1) * 100,
            refusal_count, avg_length, std_length, confidence_count, question_count]


def load_data(path):
    """Load dataset, return refusal counts, hedging features, LLM features, and labels."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])

    refusal_counts, hedging_rows, llm_rows, labels = [], [], [], []

    for r in results:
        conv = r.get("conversation", [])
        traj = r.get("feature_trajectory", [])
        if not conv or r.get("status") == "error":
            continue

        rc = extract_refusal_count(conv)
        hf = extract_all_text_features(conv)
        if rc is None or hf is None:
            continue

        refusal_counts.append(rc)
        hedging_rows.append(hf)
        labels.append(1 if r.get("ground_truth") == "lying" else 0)

        if traj:
            means = [np.mean([t[f] for t in traj if t.get(f) is not None]) for f in LLM_FEATURES]
            llm_rows.append(means)

    return (np.array(refusal_counts, dtype=float),
            np.array(hedging_rows),
            np.array(llm_rows) if llm_rows else None,
            np.array(labels))


def bootstrap_ci(preds, y, n_boot=1000, seed=42):
    """Bootstrap 95% CI on accuracy."""
    rng = np.random.RandomState(seed)
    accs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        accs.append(np.mean(preds[idx] == y[idx]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return lo, hi


def main():
    print("=" * 70)
    print("EXP-J: REFUSAL-COUNT-ONLY LOO ACROSS ALL EQUALIZED MODELS")
    print("=" * 70)

    summary = []

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: not found")
            continue

        refusal_cts, hedging_X, llm_X, y = load_data(path)
        n = len(y)
        n_truth = np.sum(y == 0)
        n_lie = np.sum(y == 1)

        print(f"\n{'─' * 70}")
        print(f"{name}: n={n} ({n_truth} truthful, {n_lie} lying)")
        print(f"{'─' * 70}")

        # Refusal count stats
        truth_rc = refusal_cts[y == 0]
        lie_rc = refusal_cts[y == 1]
        d = cohens_d(lie_rc, truth_rc)
        print(f"\n  Refusal count: truthful={np.mean(truth_rc):.2f}±{np.std(truth_rc):.2f}, "
              f"lying={np.mean(lie_rc):.2f}±{np.std(lie_rc):.2f}, |d|={abs(d):.2f}")
        print(f"  % with ≥1 refusal: truthful={np.mean(truth_rc >= 1):.0%}, lying={np.mean(lie_rc >= 1):.0%}")

        # 1. Refusal-count-only LOO
        rc_acc, rc_preds = loo_accuracy_single(refusal_cts, y)
        rc_lo, rc_hi = bootstrap_ci(rc_preds, y)
        print(f"\n  Refusal-count-only LOO:  {rc_acc:.1%}  [95% CI: {rc_lo:.1%}, {rc_hi:.1%}]")

        # 2. Full 7-feature hedging baseline LOO
        hedge_acc, hedge_preds = loo_accuracy_multi(hedging_X, y)
        hedge_lo, hedge_hi = bootstrap_ci(hedge_preds, y)
        print(f"  7-feature hedging LOO:   {hedge_acc:.1%}  [95% CI: {hedge_lo:.1%}, {hedge_hi:.1%}]")

        # 3. LLM pipeline LOO (if available)
        llm_acc_str = "N/A"
        if llm_X is not None and len(llm_X) == n:
            llm_acc, llm_preds = loo_accuracy_multi(llm_X, y)
            llm_lo, llm_hi = bootstrap_ci(llm_preds, y)
            llm_acc_str = f"{llm_acc:.1%}  [95% CI: {llm_lo:.1%}, {llm_hi:.1%}]"
            print(f"  LLM pipeline LOO:        {llm_acc_str}")
        else:
            llm_acc = None
            print(f"  LLM pipeline LOO:        {llm_acc_str}")

        summary.append({
            "model": name,
            "n": n,
            "refusal_d": abs(d),
            "refusal_loo": rc_acc,
            "hedging_loo": hedge_acc,
            "llm_loo": llm_acc,
        })

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE")
    print(f"{'=' * 70}")
    print(f"{'Model':<30} {'n':>4} {'|d|':>5} {'Refusal':>8} {'Hedging':>8} {'LLM':>8}")
    print(f"{'':─<30} {'':─>4} {'':─>5} {'':─>8} {'':─>8} {'':─>8}")
    for s in summary:
        llm_str = f"{s['llm_loo']:.1%}" if s['llm_loo'] is not None else "N/A"
        print(f"{s['model']:<30} {s['n']:>4} {s['refusal_d']:>5.2f} "
              f"{s['refusal_loo']:>7.1%} {s['hedging_loo']:>7.1%} {llm_str:>8}")


if __name__ == "__main__":
    main()
