#!/usr/bin/env python3
"""
analyze_autonomous_deception.py — EXP-I: Autonomous deception pilot.

Applies hedging/refusal regex baseline and single-feature LOO to publicly available
autonomous deception datasets from Apollo Research (Meinke et al., 2025), testing
whether behavioral features that detect instructed deception transfer to goal-directed
strategic deception.

Datasets:
  1. AI Liar (n=54, 27 honest + 27 deceptive): Goal-directed lying scenarios where
     a model strategically deceives to achieve a reward (e.g., recommending a worse
     treatment to get a higher reward). Single-turn Q&A with Llama 70B 3.3.
  2. Insider Trading (n=6000, ~1495 deceptive + ~3508 honest): Multi-turn agent
     traces where the model may or may not use insider information. We extract the
     model's report to its manager and analyze hedging in the deceptive justification.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/analyze_autonomous_deception.py
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
APOLLO_DIR = os.path.join(BASE, "data", "external", "apollo_deception")

# --- Feature definitions (from hedging_baseline.py) ---

HEDGE_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "probably",
    "i think", "i believe", "i guess", "sort of", "kind of",
    "not sure", "not certain", "if i recall", "i don't remember exactly"
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

CONFIDENCE_WORDS = ["certainly", "definitely", "absolutely", "clearly",
                    "of course", "without doubt", "undoubtedly"]


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


def extract_text_features(text):
    """Extract 7 hedging features from a text string."""
    text_lower = text.lower()
    total_words = len(text_lower.split())
    hedge_count = sum(text_lower.count(hw) for hw in HEDGE_WORDS)
    refusal_count = sum(1 for pat in REFUSAL_PATTERNS
                        if re.search(pat, text, re.IGNORECASE))
    lengths = [len(text.split())]  # single response
    avg_length = np.mean(lengths)
    std_length = 0.0
    confidence_count = sum(text_lower.count(cw) for cw in CONFIDENCE_WORDS)
    question_count = text.count("?")
    return [hedge_count, hedge_count / max(total_words, 1) * 100,
            refusal_count, avg_length, std_length, confidence_count, question_count]


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


# =============================================================================
# Dataset 1: AI Liar (goal-directed lying, single-turn)
# =============================================================================

def load_ai_liar(model_tag="llama-70b-3.3"):
    """Load AI Liar rollouts: goal-directed deception scenarios.

    Args:
        model_tag: Model identifier in filename. "llama-70b-3.3" for Llama 3.3 70B,
                   "llama-70b" for Llama 3.1 70B.
    """
    path = os.path.join(APOLLO_DIR, "data", "rollouts",
                        f"ai_liar__original_without_answers__{model_tag}.json")
    if not os.path.exists(path):
        print(f"[SKIP] AI Liar not found at {path}")
        return None, None, None

    with open(path) as f:
        data = json.load(f)

    rollouts = data["rollouts"]
    texts, labels, scenarios = [], [], []

    for r in rollouts:
        expected = r.get("expected_label", "")
        if expected not in ("honest", "deceptive"):
            continue

        # Get the model's output text
        output = r.get("output_strs", [""])[0].strip()
        if not output:
            continue

        # Get scenario description from system prompt
        system_msg = ""
        for msg in r.get("input_messages", []):
            if msg.get("role") == "system":
                system_msg = msg["content"][:100]
                break

        texts.append(output)
        labels.append(1 if expected == "deceptive" else 0)
        scenarios.append(system_msg)

    return texts, np.array(labels), scenarios


# =============================================================================
# Dataset 2: Insider Trading (strategic deception in agent actions)
# =============================================================================

def load_insider_trading():
    """Load insider trading generations: multi-turn agent traces.

    Deception: model uses insider info (made_trade='yes') and may misrepresent
    the information source in its report to the manager.
    Non-deception: model declines to use insider info (made_trade='no').

    We extract the model's reasoning and manager-directed messages for analysis.
    """
    path = os.path.join(APOLLO_DIR, "data", "insider_trading",
                        "llama-70b-3.3-generations.json")
    if not os.path.exists(path):
        print(f"[SKIP] Insider trading not found at {path}")
        return None, None

    with open(path) as f:
        data = json.load(f)

    texts, labels = [], []

    for item in data:
        made_trade = item["metadata"]["made_trade"]
        if made_trade == "invalid":
            continue

        # Extract all assistant messages (reasoning + actions)
        assistant_msgs = [msg["content"] for msg in item["transcript"]
                          if msg["role"] == "assistant"]
        full_text = " ".join(assistant_msgs)

        texts.append(full_text)
        labels.append(1 if made_trade == "yes" else 0)

    return texts, np.array(labels)


def analyze_dataset(name, texts, labels, scenarios=None):
    """Run full analysis on a dataset."""
    n = len(labels)
    n_dec = np.sum(labels == 1)
    n_hon = np.sum(labels == 0)

    print(f"\n{'=' * 70}")
    print(f"{name}: n={n} ({n_hon} honest, {n_dec} deceptive)")
    print(f"{'=' * 70}")

    if scenarios:
        print(f"\nExample scenarios:")
        for i, (s, l) in enumerate(zip(scenarios[:4], labels[:4])):
            label_str = "DECEPTIVE" if l == 1 else "HONEST"
            print(f"  [{label_str}] {s}...")

    # Extract features
    hedging_rows = []
    for text in texts:
        feats = extract_text_features(text)
        hedging_rows.append(feats)
    X = np.array(hedging_rows)

    feat_names = ["hedge_ct", "hedge_rate", "refusal_ct", "avg_len",
                  "std_len", "confidence_ct", "question_ct"]

    # Feature means by condition
    print(f"\nFeature means:")
    print(f"  {'Feature':<15} {'Honest':>10} {'Deceptive':>10} {'|d|':>8}")
    for j, fn in enumerate(feat_names):
        h_vals = X[labels == 0, j]
        d_vals = X[labels == 1, j]
        d_val = abs(cohens_d(d_vals, h_vals))
        print(f"  {fn:<15} {np.mean(h_vals):10.2f} {np.mean(d_vals):10.2f} {d_val:8.2f}")

    # Refusal count stats
    refusal_cts = X[:, 2]
    h_rc = refusal_cts[labels == 0]
    d_rc = refusal_cts[labels == 1]
    rc_d = abs(cohens_d(d_rc, h_rc))
    print(f"\n  Refusal count |d|: {rc_d:.2f}")
    print(f"  % with ≥1 refusal: honest={np.mean(h_rc >= 1):.0%}, deceptive={np.mean(d_rc >= 1):.0%}")

    # Word count stats
    print(f"\n  Mean word count: honest={np.mean(X[labels==0, 3]):.0f}, deceptive={np.mean(X[labels==1, 3]):.0f}")

    # LOO: refusal count only
    if len(np.unique(labels)) > 1:
        rc_acc, rc_preds = loo_accuracy_single(refusal_cts.astype(float), labels)
        rc_lo, rc_hi = bootstrap_ci(rc_preds, labels)
        print(f"\n  Refusal-count-only LOO:  {rc_acc:.1%}  [95% CI: {rc_lo:.1%}, {rc_hi:.1%}]")

        # LOO: hedge count only
        hedge_cts = X[:, 0]
        hc_acc, hc_preds = loo_accuracy_single(hedge_cts.astype(float), labels)
        hc_lo, hc_hi = bootstrap_ci(hc_preds, labels)
        print(f"  Hedge-count-only LOO:    {hc_acc:.1%}  [95% CI: {hc_lo:.1%}, {hc_hi:.1%}]")

        # LOO: full 7-feature hedging baseline
        hedge_acc, hedge_preds = loo_accuracy_multi(X, labels)
        hedge_lo, hedge_hi = bootstrap_ci(hedge_preds, labels)
        print(f"  7-feature hedging LOO:   {hedge_acc:.1%}  [95% CI: {hedge_lo:.1%}, {hedge_hi:.1%}]")

        # LOO: response length only
        len_acc, len_preds = loo_accuracy_single(X[:, 3].astype(float), labels)
        len_lo, len_hi = bootstrap_ci(len_preds, labels)
        print(f"  Length-only LOO:         {len_acc:.1%}  [95% CI: {len_lo:.1%}, {len_hi:.1%}]")

        # Majority baseline
        majority = max(n_hon, n_dec) / n
        print(f"  Majority baseline:       {majority:.1%}")

    # Show example texts
    print(f"\nExample responses:")
    for label_val, label_name in [(0, "HONEST"), (1, "DECEPTIVE")]:
        for i, (text, l) in enumerate(zip(texts, labels)):
            if l == label_val:
                print(f"\n  [{label_name}]: {text[:300]}...")
                break

    return {
        "name": name,
        "n": n,
        "n_honest": n_hon,
        "n_deceptive": n_dec,
        "refusal_d": rc_d if len(np.unique(labels)) > 1 else 0,
        "refusal_loo": rc_acc if len(np.unique(labels)) > 1 else None,
        "hedging_loo": hedge_acc if len(np.unique(labels)) > 1 else None,
    }


def main():
    print("=" * 70)
    print("EXP-I: AUTONOMOUS DECEPTION PILOT")
    print("Apollo Research Deception Detection Datasets (Meinke et al., 2025)")
    print("=" * 70)

    results = []

    # Dataset 1a: AI Liar — Llama 3.3 70B
    texts, labels, scenarios = load_ai_liar("llama-70b-3.3")
    if texts is not None:
        r = analyze_dataset("AI Liar (goal-directed, Llama 3.3 70B)", texts, labels, scenarios)
        results.append(r)

    # Dataset 1b: AI Liar — Llama 3.1 70B
    texts_31, labels_31, scenarios_31 = load_ai_liar("llama-70b")
    if texts_31 is not None:
        r = analyze_dataset("AI Liar (goal-directed, Llama 3.1 70B)", texts_31, labels_31, scenarios_31)
        results.append(r)

    # Dataset 2: Insider Trading
    texts2, labels2 = load_insider_trading()
    if texts2 is not None:
        # Subsample for manageable LOO (n=100 balanced)
        n_dec = np.sum(labels2 == 1)
        n_hon = np.sum(labels2 == 0)
        print(f"\n[Insider Trading] Full dataset: {n_hon} honest, {n_dec} deceptive")

        # Balance and subsample
        rng = np.random.RandomState(42)
        dec_idx = np.where(labels2 == 1)[0]
        hon_idx = np.where(labels2 == 0)[0]

        n_sample = min(50, n_dec, n_hon)
        dec_sample = rng.choice(dec_idx, n_sample, replace=False)
        hon_sample = rng.choice(hon_idx, n_sample, replace=False)
        sample_idx = np.concatenate([hon_sample, dec_sample])
        rng.shuffle(sample_idx)

        texts_sub = [texts2[i] for i in sample_idx]
        labels_sub = labels2[sample_idx]

        r = analyze_dataset(f"Insider Trading (balanced n={2*n_sample}, Llama 70B 3.3)",
                           texts_sub, labels_sub)
        results.append(r)

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print("SUMMARY: AUTONOMOUS VS INSTRUCTED DECEPTION")
        print(f"{'=' * 70}")
        print(f"\n{'Dataset':<45} {'n':>4} {'Ref |d|':>8} {'Ref LOO':>8} {'Hedge LOO':>10}")
        print(f"{'':─<45} {'':─>4} {'':─>8} {'':─>8} {'':─>10}")
        for r in results:
            ref_loo = f"{r['refusal_loo']:.1%}" if r['refusal_loo'] is not None else "N/A"
            hedge_loo = f"{r['hedging_loo']:.1%}" if r['hedging_loo'] is not None else "N/A"
            print(f"{r['name']:<45} {r['n']:>4} {r['refusal_d']:>8.2f} {ref_loo:>8} {hedge_loo:>10}")

        print(f"\nComparison to equalized results (from EXP-J):")
        print(f"  Llama 70B equalized:    refusal LOO=87.1%, hedging LOO=87.1%, LLM LOO=83.9%")
        print(f"  Llama 3B equalized:     refusal LOO=64.0%, hedging LOO=75.0%, LLM LOO=61.0%")
        print(f"\nNote: Autonomous deception involves fundamentally different behavioral")
        print(f"signatures (strategic omission, misdirection) vs equalized (knowledge conflict).")
        print(f"Direct comparison is illustrative, not definitive.")


if __name__ == "__main__":
    main()
