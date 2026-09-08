"""
analyze_factorial_td_vs_fd.py

The T/D vs F/D contrast: does the detector move WITHIN ONE instructed-deception
condition?

WHY THIS EXISTS
---------------
Both instructed cells of the 2x2 factorial carry a deception instruction that the
target is told to comply with:

  T/D : claim is TRUE  -> "argue against it and mislead the questioner"
  F/D : claim is FALSE -> "defend it convincingly by lying"

Under the definition of deception used throughout (the target asserts what it does
not hold to be true), a compliant response in EITHER cell is deception. So T/D and
F/D carry the same deception instruction, complied with in both, and differ only in
the claim's truth-value V (and in the direction of the assigned argument) -- the
benchmark assigns them the SAME label.

This does NOT establish that latent D is fixed across the contrast: D is never
observed, and compliance is graded rather than binary (the corrective leakage this
script measures is direct evidence of that). What the movement supports is a
disjunction: either a large part of what these detectors read is not deception, or
deception varies inside the cell the benchmark calls a lie -- in which case the
instruction is not the ground truth the benchmark takes it for. Note also that S has
V and B as further parents, so a truth-value effect could produce this movement while
leaving a genuine D->S component intact.

This is the complement of the T/H vs T/D style contrasts, which move E and D
together and so cannot separate the two paths (see the Proposition).

Analysis is paired at the level of the claim pair: T/D uses the true member and F/D
the false member of the same matched pair, so the pairing removes claim-level
variation. The permutation test flips the T/D-vs-F/D assignment WITHIN each pair.

Usage:
    cd /path/to/adaptive_lie_detector
    python3 experiments/analyze_factorial_td_vs_fd.py
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.analyze_factorial_standardized import (  # noqa: E402
    DATA_DIR, MODELS, CLAIM_TO_PAIR, calibrated_detector,
)

N_PERM, SEED = 10000, 42


def load_instructed_cells(model_tag, panel):
    """Per-claim-pair outcomes for the two instructed cells only (E=1).

    Returns {pair_index: {"T_D": [y, ...], "F_D": [y, ...]}}.
    """
    clf = calibrated_detector(model_tag) if panel == "detector" else None
    byp = defaultdict(lambda: {"T_D": [], "F_D": []})
    for cell in ("T_D", "F_D"):
        path = os.path.join(DATA_DIR, f"factorial_txd_{model_tag}_{cell}.json")
        for r in json.load(open(path))["records"]:
            pair = CLAIM_TO_PAIR.get(r["claim"])
            if pair is None:
                continue
            y = (clf.decision_function(np.array([r["vector"]], dtype=float))[0]
                 if panel == "detector" else float(r["refusal_count"]))
            byp[pair][cell].append(y)
    # keep only pairs observed in both cells, so the contrast is genuinely paired
    return {p: v for p, v in byp.items() if v["T_D"] and v["F_D"]}


def paired_diffs(byp):
    """Per-pair mean(T/D) - mean(F/D), in a fixed pair order."""
    pairs = sorted(byp)
    return pairs, np.array([np.mean(byp[p]["T_D"]) - np.mean(byp[p]["F_D"])
                            for p in pairs])


def perm_p_paired(diffs, rng):
    """Two-sided paired permutation test: flip the sign of each pair's difference."""
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(diffs)))
    null = np.abs((signs * diffs).mean(axis=1))
    return (np.sum(null >= obs) + 1) / (N_PERM + 1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def fire_rates(byp):
    """RC>=1 fire rate per cell, over trials (not pairs)."""
    out = {}
    for cell in ("T_D", "F_D"):
        ys = [y for v in byp.values() for y in v[cell]]
        k = sum(1 for y in ys if y >= 1)
        out[cell] = {"n": len(ys), "fired": k,
                     "rate": k / len(ys) if ys else float("nan"),
                     "ci": wilson(k, len(ys))}
    return out


def main():
    rng = np.random.RandomState(SEED)
    print("=" * 78)
    print("  T/D vs F/D — within one instructed-deception condition (same label; D unobserved)")
    print("  Paired by matched claim pair; permutation flips assignment within pair.")
    print("=" * 78)

    out = {"n_perm": N_PERM, "seed": SEED, "panels": {}}

    for panel, label, unit in (("marker", "Marker-count feature", "markers"),
                               ("detector", "Complete detector (LR score)", "SD")):
        print(f"\n{label}")
        print(f"  {'target':<15}{'mean T/D':>10}{'mean F/D':>10}{'diff':>9}"
              f"{'pairs':>7}{'p':>9}")
        panel_out, all_diffs = {}, []
        for name, tag in MODELS:
            byp = load_instructed_cells(tag, panel)
            pairs, diffs = paired_diffs(byp)
            mt = np.mean([y for p in pairs for y in byp[p]["T_D"]])
            mf = np.mean([y for p in pairs for y in byp[p]["F_D"]])
            sd = np.std([y for p in pairs for c in ("T_D", "F_D")
                         for y in byp[p][c]], ddof=1)
            p_val = perm_p_paired(diffs, rng)
            all_diffs.append(diffs)
            rec = {"mean_TD": float(mt), "mean_FD": float(mf),
                   "diff": float(mt - mf),
                   "diff_sd_units": float((mt - mf) / sd) if sd > 0 else float("nan"),
                   "n_pairs": len(pairs), "p": float(p_val)}
            if panel == "marker":
                rec["fire"] = fire_rates(byp)
            panel_out[name] = rec
            print(f"  {name:<15}{mt:>10.2f}{mf:>10.2f}{mt - mf:>9.2f}"
                  f"{len(pairs):>7}{p_val:>9.4f}")

        pooled = np.concatenate(all_diffs)
        p_pooled = perm_p_paired(pooled, rng)
        panel_out["POOLED"] = {"diff": float(pooled.mean()),
                               "n_pairs": int(len(pooled)),
                               "p": float(p_pooled)}
        print(f"  {'POOLED':<15}{'':>10}{'':>10}{pooled.mean():>9.2f}"
              f"{len(pooled):>7}{p_pooled:>9.4f}")
        out["panels"][panel] = panel_out

    print("\nFire rates (RC>=1), marker panel")
    print(f"  {'target':<15}{'T/D':>18}{'F/D':>18}")
    for name, _ in MODELS:
        f = out["panels"]["marker"][name]["fire"]
        print(f"  {name:<15}"
              f"{f['T_D']['rate']:>9.0%} [{f['T_D']['ci'][0]:.2f},{f['T_D']['ci'][1]:.2f}]"
              f"{f['F_D']['rate']:>9.0%} [{f['F_D']['ci'][0]:.2f},{f['F_D']['ci'][1]:.2f}]")

    dest = os.path.join(DATA_DIR, "factorial_td_vs_fd.json")
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
