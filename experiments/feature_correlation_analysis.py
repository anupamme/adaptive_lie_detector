#!/usr/bin/env python3
"""
feature_correlation_analysis.py — Reviewer Q4

Compute Pearson and Spearman correlations between the 7 regex-based
hedging features (from EXP-C) and the 5 LLM-extracted behavioral features.

This answers reviewer Q4: "Do the regex features correlate with the
LLM-extracted features? If defensiveness is mostly tracking refusal-marker
rate, this should be visible in a feature-correlation table."

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    python experiments/feature_correlation_analysis.py
"""

import json
import os
import sys
import re
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")

DATASETS = {
    "Llama 3.2 3B": os.path.join(DATA, "ollama_eval_llama3_2_3b_latest.json"),
    "Mistral 7B": os.path.join(DATA, "ollama_eval_mistral_7b_latest.json"),
    "Llama 3.3 70B": os.path.join(DATA, "bedrock_eval_llama70b_latest.json"),
    "Claude Haiku": os.path.join(DATA, "claude_haiku_n100_latest.json"),
}

# Hedge words (same as hedging_baseline.py)
HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly"
]

# Refusal/correction markers
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

REGEX_FEATURE_NAMES = [
    "hedge_count", "hedge_rate", "refusal_count",
    "avg_response_length", "std_response_length",
    "confidence_count", "question_count"
]

LLM_FEATURE_NAMES = [
    "consistency", "specificity", "defensiveness",
    "confidence", "elaboration"
]


def extract_text_features(conversation):
    """Extract 7 regex-based text features from conversation."""
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
            refusal_count, avg_length, std_length,
            confidence_count, question_count]


def extract_llm_features(result):
    """Extract final LLM feature scores from a trial result."""
    trajectory = result.get("feature_trajectory", [])
    if not trajectory:
        return None
    # Use the last (most informed) feature extraction
    final = trajectory[-1]
    return [final.get("consistency", 0), final.get("specificity", 0),
            final.get("defensiveness", 0), final.get("confidence", 0),
            final.get("elaboration", 0)]


def analyze_dataset(name, path):
    """Compute correlation matrix between regex and LLM features."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    regex_rows = []
    llm_rows = []

    for r in results:
        conv = r.get("conversation", [])
        if not conv or r.get("status") == "error":
            continue

        text_feats = extract_text_features(conv)
        llm_feats = extract_llm_features(r)

        if text_feats is not None and llm_feats is not None:
            regex_rows.append(text_feats)
            llm_rows.append(llm_feats)

    if len(regex_rows) < 10:
        print(f"  Too few valid trials: {len(regex_rows)}")
        return None

    regex_arr = np.array(regex_rows)  # (n, 7)
    llm_arr = np.array(llm_rows)      # (n, 5)

    print(f"  Valid trials: {len(regex_rows)}")
    print()

    # Compute Spearman correlations (rank-based, more robust)
    print(f"  Spearman Correlation Matrix (regex rows x LLM columns):")
    print(f"  {'':20s}", end="")
    for ln in LLM_FEATURE_NAMES:
        print(f" {ln:>12s}", end="")
    print()
    print(f"  {'-' * 80}")

    corr_matrix = np.zeros((7, 5))
    pval_matrix = np.zeros((7, 5))

    for i, rn in enumerate(REGEX_FEATURE_NAMES):
        print(f"  {rn:20s}", end="")
        for j, ln in enumerate(LLM_FEATURE_NAMES):
            rho, pval = stats.spearmanr(regex_arr[:, i], llm_arr[:, j])
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f" {rho:>8.3f}{sig:3s}", end="")
        print()

    # Highlight the key finding for the reviewer
    print(f"\n  Key finding (Q4): refusal_count vs defensiveness:")
    i_refusal = REGEX_FEATURE_NAMES.index("refusal_count")
    j_def = LLM_FEATURE_NAMES.index("defensiveness")
    rho, pval = stats.spearmanr(regex_arr[:, i_refusal], llm_arr[:, j_def])
    print(f"    Spearman rho = {rho:.3f}, p = {pval:.2e}")
    if abs(rho) > 0.5:
        print(f"    -> Strong correlation confirms defensiveness largely tracks refusal markers")
    elif abs(rho) > 0.3:
        print(f"    -> Moderate correlation; defensiveness partially tracks refusal markers")
    else:
        print(f"    -> Weak correlation; defensiveness captures different signal than refusal count")

    return {"name": name, "n": len(regex_rows), "correlations": corr_matrix.tolist(),
            "pvalues": pval_matrix.tolist()}


def main():
    print("=" * 70)
    print("  FEATURE CORRELATION ANALYSIS (Reviewer Q4)")
    print("  Regex features (7) vs LLM-extracted features (5)")
    print("=" * 70)

    all_results = {}
    for name, path in DATASETS.items():
        result = analyze_dataset(name, path)
        if result:
            all_results[name] = result

    # Summary across models
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY: Mean |rho| across all models")
    print(f"{'=' * 70}")
    if all_results:
        all_corrs = np.array([r["correlations"] for r in all_results.values()])
        mean_abs_corr = np.mean(np.abs(all_corrs), axis=0)
        print(f"  {'':20s}", end="")
        for ln in LLM_FEATURE_NAMES:
            print(f" {ln:>12s}", end="")
        print()
        for i, rn in enumerate(REGEX_FEATURE_NAMES):
            print(f"  {rn:20s}", end="")
            for j in range(5):
                print(f" {mean_abs_corr[i, j]:>12.3f}", end="")
            print()

    # Save results
    out_path = os.path.join(DATA, "feature_correlation_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
