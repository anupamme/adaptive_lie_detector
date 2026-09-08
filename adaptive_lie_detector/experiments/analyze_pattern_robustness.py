"""
analyze_pattern_robustness.py

Leave-one-pattern-out robustness for the parameter-free refusal-count baseline.

WHY THIS EXISTS
---------------
The papers claim the baseline is "robust to pattern removal (<=2pp)". No analysis
producing that number exists in experiments/. The papers also state the regex has
"eight patterns"; the list that actually generated the reported numbers
(analyze_refusal_only.REFUSAL_PATTERNS) has 14.

Because `code/` is untracked, there is no way to establish from history that the
patterns were specified before the evaluation was run. So we do not assert that.
Instead we measure the thing that actually matters to a reader: whether the baseline's
value depends on the particular pattern set. Drop each pattern in turn, recompute the
pooled fixed-threshold (RC >= 1) accuracy, and report the spread.

Usage:
    cd /path/to/adaptive_lie_detector
    python3 experiments/analyze_pattern_robustness.py
"""

import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.analyze_refusal_only import REFUSAL_PATTERNS  # noqa: E402
from experiments.claims_equalized import EQUALIZED_CLAIMS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "results")

TARGETS = [("Llama 3.2 3B", "ollama_eval_llama3_2_3b"),
           ("Llama 3.1 8B", "bedrock_eval_llama8b"),
           ("Mistral 7B", "ollama_eval_mistral_7b"),
           ("Qwen 2.5 7B", "ollama_eval_qwen2_5_7b"),
           ("Qwen 2.5 14B", "ollama_eval_qwen2_5_14b"),
           ("Llama 3.3 70B", "bedrock_eval_llama70b"),
           ("Claude Haiku 4.5", "bedrock_eval_haiku")]

CLAIM_TO_PAIR = {c: i for i, (tc, fc) in enumerate(EQUALIZED_CLAIMS) for c in (tc, fc)}


def assistant_text(conversation):
    return " ".join(t.get("content", "") for t in conversation
                    if t.get("role") == "assistant")


def load_all():
    """Return per-trial assistant text, label, and claim-pair group across all targets."""
    texts, labels, groups = [], [], []
    for _, stem in TARGETS:
        path = os.path.join(DATA_DIR, f"{stem}_prompt_equalized_latest.json")
        with open(path) as f:
            results = json.load(f)["results"]
        for r in results:
            conv = r.get("conversation", [])
            if not conv or r.get("status") == "error":
                continue
            pair = CLAIM_TO_PAIR.get(r.get("claim"))
            if pair is None:
                continue
            texts.append(assistant_text(conv))
            labels.append(1 if r.get("ground_truth") == "lying" else 0)
            groups.append(pair)
    return texts, np.array(labels), np.array(groups)


def pooled_accuracy(texts, labels, patterns):
    """Fixed-threshold RC >= 1 accuracy using the given pattern subset."""
    preds = np.array([
        1 if sum(1 for p in patterns if re.search(p, t, re.IGNORECASE)) >= 1 else 0
        for t in texts
    ])
    return float((preds == labels).mean() * 100)


def main():
    texts, labels, groups = load_all()
    n_pat = len(REFUSAL_PATTERNS)
    full = pooled_accuracy(texts, labels, REFUSAL_PATTERNS)

    print("=" * 74)
    print("  LEAVE-ONE-PATTERN-OUT ROBUSTNESS OF THE PARAMETER-FREE BASELINE")
    print("=" * 74)
    print(f"  patterns in the list that generated the reported numbers: {n_pat}")
    print(f"  pooled trials: {len(texts)}   claim pairs: {len(np.unique(groups))}")
    print(f"  full-set pooled accuracy (RC >= 1): {full:.1f}%\n")

    print(f"  {'dropped pattern':<62}{'acc':>7}{'delta':>8}")
    print("  " + "-" * 76)
    deltas = []
    rows = []
    for i, pat in enumerate(REFUSAL_PATTERNS):
        subset = [p for j, p in enumerate(REFUSAL_PATTERNS) if j != i]
        acc = pooled_accuracy(texts, labels, subset)
        d = acc - full
        deltas.append(d)
        rows.append({"dropped": pat, "accuracy": acc, "delta": d})
        show = pat if len(pat) <= 60 else pat[:57] + "..."
        print(f"  {show:<62}{acc:>6.1f}%{d:>+7.1f}")

    deltas = np.array(deltas)
    print("  " + "-" * 76)
    print(f"  max |delta| = {np.abs(deltas).max():.1f} pp   "
          f"mean |delta| = {np.abs(deltas).mean():.2f} pp   "
          f"range [{deltas.min():+.1f}, {deltas.max():+.1f}]")

    # Which single pattern matters most?
    worst = int(np.argmax(np.abs(deltas)))
    print(f"\n  most load-bearing pattern: {REFUSAL_PATTERNS[worst]}")
    print(f"    dropping it moves the pooled figure by {deltas[worst]:+.1f} pp")

    out = {
        "experiment": "pattern_robustness_leave_one_out",
        "n_patterns": n_pat,
        "patterns": list(REFUSAL_PATTERNS),
        "n_trials": len(texts),
        "n_claim_pairs": int(len(np.unique(groups))),
        "full_set_accuracy": full,
        "leave_one_out": rows,
        "max_abs_delta": float(np.abs(deltas).max()),
        "mean_abs_delta": float(np.abs(deltas).mean()),
        "note": ("`code/` is untracked, so we cannot establish from history that the "
                 "patterns were fixed before evaluation; we therefore report measured "
                 "robustness instead of asserting pre-specification."),
    }
    out_path = os.path.join(DATA_DIR, "pattern_robustness.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
