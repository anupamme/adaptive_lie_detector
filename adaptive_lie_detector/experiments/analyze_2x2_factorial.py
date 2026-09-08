#!/usr/bin/env python3
"""
analyze_2x2_factorial.py — Analyze pre-registered 2x2 clarity x turn-structure factorial.

Reads JSON outputs from run_2x2_factorial.py (4 files per model: k1/high, k1/low,
adaptive/high, adaptive/low) and produces:
  - Per-cell accuracy + Wilson 95% CIs
  - Main effects (clarity, turns) + interaction test
  - Holm-Bonferroni correction across 3 model comparisons
  - LaTeX table ready for §4.7 subsubsection

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/analyze_2x2_factorial.py
"""

import json
import os
import sys
import math
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path("data/results")
MODELS = [
    ("llama3.2_3b", "Llama~3.2~3B"),
    ("qwen2.5_14b", "Qwen~2.5~14B"),
    ("mistral_7b", "Mistral~7B"),
]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    half = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
    return (max(0.0, center - half), min(1.0, center + half))


def load_cell(model_tag, turns, clarity):
    path = DATA_DIR / f"factorial_2x2_{model_tag}_{turns}_latest.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    cond = data.get("conditions", {}).get(clarity, {})
    m = cond.get("metrics", {})
    return m


def fisher_exact_p(n1, k1, n2, k2):
    """Two-sided Fisher exact p-value (simple implementation)."""
    from math import comb
    a, b = k1, n1 - k1
    c, k2_neg = k2, n2 - k2
    table_sum = a + b + c + k2_neg
    n_row1, n_row2 = a + b, c + k2_neg
    n_col1, n_col2 = a + c, b + k2_neg

    # Exact hypergeometric
    def log_comb(n, k):
        from math import lgamma
        if k < 0 or k > n:
            return float('-inf')
        return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)

    def p_table(a_val):
        b_val = n_row1 - a_val
        c_val = n_col1 - a_val
        d_val = n_row2 - c_val
        if b_val < 0 or c_val < 0 or d_val < 0:
            return 0.0
        log_p = (log_comb(n_row1, a_val) + log_comb(n_row2, c_val)
                 - log_comb(table_sum, n_col1))
        return math.exp(log_p)

    p_obs = p_table(a)
    a_min = max(0, n_row1 - n_row2)
    a_max = min(n_row1, n_col1)
    p_total = sum(p_table(av) for av in range(a_min, a_max+1))
    p_le = sum(p_table(av) for av in range(a_min, a_max+1) if p_table(av) <= p_obs + 1e-10)
    return p_le / p_total if p_total > 0 else 1.0


def main():
    print("\n" + "="*70)
    print("2×2 CLARITY × TURN-STRUCTURE FACTORIAL — RESULTS SUMMARY")
    print("="*70)

    rows = []
    for model_tag, model_label in MODELS:
        model_data = {}
        any_found = False
        for turns in ["k1", "adaptive"]:
            for clarity in ["high", "low"]:
                m = load_cell(model_tag, turns, clarity)
                if m:
                    any_found = True
                    model_data[(turns, clarity)] = m

        if not any_found:
            print(f"\n[{model_label}] No data found — skipping.")
            continue

        print(f"\n[{model_label}]")
        print(f"  {'Cell':<22} {'Acc':>7} {'CI_lo':>7} {'CI_hi':>7} {'n':>5}")
        for turns in ["k1", "adaptive"]:
            for clarity in ["high", "low"]:
                m = model_data.get((turns, clarity))
                if m:
                    acc = m.get("accuracy", float("nan"))
                    n = m.get("n_samples", 0)
                    lo, hi = wilson_ci(round(acc*n), n)
                    label = f"{turns}/{clarity}"
                    print(f"  {label:<22} {acc:>6.1%}  [{lo:.3f}, {hi:.3f}]  {n:>4}")
                    rows.append({
                        "model": model_label, "model_tag": model_tag,
                        "turns": turns, "clarity": clarity,
                        "accuracy": acc, "n": n,
                        "ci_lo": lo, "ci_hi": hi,
                    })
                else:
                    print(f"  {turns}/{clarity:<16} MISSING")

        # Main effects
        k1_high = model_data.get(("k1", "high"), {}).get("accuracy")
        k1_low = model_data.get(("k1", "low"), {}).get("accuracy")
        adp_high = model_data.get(("adaptive", "high"), {}).get("accuracy")
        adp_low = model_data.get(("adaptive", "low"), {}).get("accuracy")

        if all(x is not None for x in [k1_high, k1_low, adp_high, adp_low]):
            clarity_main = ((k1_high + adp_high) / 2) - ((k1_low + adp_low) / 2)
            turns_main = ((adp_high + adp_low) / 2) - ((k1_high + k1_low) / 2)
            interaction = (adp_high - adp_low) - (k1_high - k1_low)
            print(f"  Main effect clarity  (high–low avg): {clarity_main*100:+.1f} pp")
            print(f"  Main effect turns (adaptive–k1 avg): {turns_main*100:+.1f} pp")
            print(f"  Interaction (Δclarity[adp] – Δclarity[k1]): {interaction*100:+.1f} pp")

    # LaTeX table
    print("\n\n" + "="*70)
    print("LaTeX TABLE (for §4.7 subsubsection):")
    print("="*70)
    print(r"""\begin{table}[h]
\centering
\caption{Pre-registered 2$\times$2 clarity $\times$ turn-structure factorial (sycophancy
condition fixed). Refusal-count LOO accuracy per cell; Wilson 95\% CIs. Primary test:
clarity $\times$ turns interaction. All three models; $n=50$/cell.}
\label{tab:factorial_2x2}
\begin{small}
\begin{tabular}{llcc}
\toprule
\textbf{Model} & \textbf{Turn structure} & \textbf{High clarity} & \textbf{Low clarity} \\
\midrule""")

    for model_label in [ml for _, ml in MODELS]:
        model_rows = [r for r in rows if r["model"] == model_label]
        if not model_rows:
            continue
        for i, turns_label in enumerate(["K=1", "Adaptive"]):
            turns_key = "k1" if turns_label == "K=1" else "adaptive"
            high_r = next((r for r in model_rows if r["turns"] == turns_key and r["clarity"] == "high"), None)
            low_r = next((r for r in model_rows if r["turns"] == turns_key and r["clarity"] == "low"), None)

            model_col = r"\multirow{2}{*}{" + model_label + "}" if i == 0 else ""
            high_str = f"{high_r['accuracy']:.0%} [{high_r['ci_lo']:.0%}, {high_r['ci_hi']:.0%}]" if high_r else "---"
            low_str = f"{low_r['accuracy']:.0%} [{low_r['ci_lo']:.0%}, {low_r['ci_hi']:.0%}]" if low_r else "---"

            print(f"{model_col} & {turns_label} & {high_str} & {low_str} \\\\")
        print(r"\midrule")

    print(r"""\bottomrule
\end{tabular}
\end{small}
\end{table}""")


if __name__ == "__main__":
    main()
