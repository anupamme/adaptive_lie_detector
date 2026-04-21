#!/usr/bin/env python3
"""
hedging_baseline.py — EXP-C

Simple bag-of-words hedging/refusal baseline classifier.
No LLM feature extraction — only regex counts on response text.

Addresses reviewer Q3: "What is the performance of a simple
bag-of-words hedging-word classifier on the real LLM transcripts?"

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python3 experiments/hedging_baseline.py
"""

import json
import os
import sys
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")

DATASETS = {
    "Llama 3.2 3B": os.path.join(DATA, "ollama_eval_llama3_2_3b_latest.json"),
    "Mistral 7B": os.path.join(DATA, "ollama_eval_mistral_7b_latest.json"),
    "Claude Haiku": os.path.join(DATA, "claude_haiku_n100_latest.json"),
}

DATASETS_EQUALIZED = {
    "Mistral 7B (equalized)": os.path.join(DATA, "ollama_eval_mistral_7b_prompt_equalized_latest.json"),
    "Llama 3.2 3B (equalized)": os.path.join(DATA, "ollama_eval_llama3_2_3b_prompt_equalized_latest.json"),
}

# Hedge words (from SimpleFeatureExtractor)
HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly"
]

# Refusal/correction markers (from analyze_haiku_defensiveness.py)
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


def extract_text_features(conversation):
    """Extract simple text features from conversation — no LLM needed."""
    assistant_texts = [msg["content"] for msg in conversation
                       if msg.get("role") == "assistant"]
    if not assistant_texts:
        return None

    all_text = " ".join(assistant_texts).lower()
    total_words = len(all_text.split())

    # Hedge word count
    hedge_count = sum(all_text.count(hw) for hw in HEDGE_WORDS)

    # Refusal marker count
    refusal_count = sum(1 for pat in REFUSAL_PATTERNS
                        if re.search(pat, " ".join(assistant_texts), re.IGNORECASE))

    # Response length stats
    lengths = [len(t.split()) for t in assistant_texts]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths) if len(lengths) > 1 else 0

    # Confidence markers (assertive language)
    confidence_words = ["certainly", "definitely", "absolutely", "clearly",
                        "of course", "without doubt", "undoubtedly"]
    confidence_count = sum(all_text.count(cw) for cw in confidence_words)

    # Question marks in responses (hedging via rhetorical questions)
    question_count = sum(t.count("?") for t in assistant_texts)

    return {
        "hedge_count": hedge_count,
        "hedge_rate": hedge_count / max(total_words, 1) * 100,
        "refusal_count": refusal_count,
        "avg_response_length": avg_length,
        "std_response_length": std_length,
        "confidence_count": confidence_count,
        "question_count": question_count,
    }


def load_dataset(path):
    """Load dataset and extract text features."""
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    rows, labels = [], []

    for r in results:
        conv = r.get("conversation", [])
        if not conv or r.get("status") == "error":
            continue

        feats = extract_text_features(conv)
        if feats is None:
            continue

        rows.append([feats["hedge_count"], feats["hedge_rate"],
                     feats["refusal_count"], feats["avg_response_length"],
                     feats["std_response_length"], feats["confidence_count"],
                     feats["question_count"]])
        labels.append(1 if r.get("ground_truth") == "lying" else 0)

    return np.array(rows), np.array(labels)


def loo_accuracy(X, y):
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    return np.mean(preds == y), preds


def load_llm_features(path):
    """Load LLM-extracted features for comparison."""
    FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    rows, labels = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj or r.get("status") == "error":
            continue
        means = [np.mean([t[f] for t in traj if t.get(f) is not None]) for f in FEATURES]
        rows.append(means)
        labels.append(1 if r.get("ground_truth") == "lying" else 0)
    return np.array(rows), np.array(labels)


def main():
    print("=" * 70)
    print("EXP-C: HEDGING-WORD REGEX BASELINE")
    print("=" * 70)

    FEAT_NAMES = ["hedge_ct", "hedge_rate", "refusal_ct", "avg_len",
                  "std_len", "confidence_ct", "question_ct"]

    for name, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: not found")
            continue

        X_text, y_text = load_dataset(path)
        X_llm, y_llm = load_llm_features(path)

        print(f"\n{'─' * 70}")
        print(f"{name}: n={len(y_text)} (text features), n={len(y_llm)} (LLM features)")
        print(f"{'─' * 70}")

        # Text feature means by condition
        print(f"\n  Text feature means:")
        print(f"  {'Feature':<15} {'Truthful':>10} {'Lying':>10} {'Diff':>10}")
        for j, fn in enumerate(FEAT_NAMES):
            t_mean = np.mean(X_text[y_text == 0, j])
            l_mean = np.mean(X_text[y_text == 1, j])
            print(f"  {fn:<15} {t_mean:10.2f} {l_mean:10.2f} {l_mean - t_mean:+10.2f}")

        # LOO: text features
        text_acc, _ = loo_accuracy(X_text, y_text)
        print(f"\n  Hedging baseline LOO: {text_acc:.1%}")

        # LOO: LLM features
        llm_acc, _ = loo_accuracy(X_llm, y_llm)
        print(f"  LLM pipeline LOO:     {llm_acc:.1%}")
        print(f"  Difference:           {llm_acc - text_acc:+.1%}")

        # LOO: combined (text + LLM)
        n_min = min(len(y_text), len(y_llm))
        if n_min == len(y_text) == len(y_llm):
            X_comb = np.hstack([X_text[:n_min], X_llm[:n_min]])
            comb_acc, _ = loo_accuracy(X_comb, y_text[:n_min])
            print(f"  Combined LOO:         {comb_acc:.1%}")

    print(f"\n{'=' * 70}")
    print("EXP-D: HEDGING BASELINE ON EQUALIZED CONDITION")
    print("=" * 70)

    for name, path in DATASETS_EQUALIZED.items():
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: not found")
            continue

        X_text, y_text = load_dataset(path)
        X_llm, y_llm = load_llm_features(path)

        print(f"\n{'─' * 70}")
        print(f"{name}: n={len(y_text)} (text), n={len(y_llm)} (LLM)")
        print(f"{'─' * 70}")

        text_acc, _ = loo_accuracy(X_text, y_text)
        print(f"\n  Hedging baseline LOO: {text_acc:.1%}")

        llm_acc, _ = loo_accuracy(X_llm, y_llm)
        print(f"  LLM pipeline LOO:     {llm_acc:.1%}")
        print(f"  Difference:           {llm_acc - text_acc:+.1%}")

        n_min = min(len(y_text), len(y_llm))
        if n_min == len(y_text) == len(y_llm):
            X_comb = np.hstack([X_text[:n_min], X_llm[:n_min]])
            comb_acc, _ = loo_accuracy(X_comb, y_text[:n_min])
            print(f"  Combined LOO:         {comb_acc:.1%}")

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
