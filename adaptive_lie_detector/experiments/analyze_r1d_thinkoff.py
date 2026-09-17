#!/usr/bin/env python3
"""Analyze the EXP-R1d `think:false` exploratory arm (PREREG_EXP_R1D.md §9).

Why this exists as a separate script rather than a flag on
analyze_r1_faithful.py: that analyzer discovers cells with a non-recursive
glob over data/results/, and the whole point of archiving the deviated cells
under data/results/r1d_thinkoff/ is that they must NEVER be discoverable
alongside a pre-registered cell. Adding a flag that reaches into that
directory would put a pooling bug one typo away. This script instead names its
input files explicitly and writes to its own output path.

It changes NOTHING about the estimator. Every number it reports comes from
analyze_r1_faithful.py's own functions, imported unmodified:
  * stratified 5-fold (primary), grouped 5-fold, LOO,
  * the same LogisticRegression(max_iter=1000, C=1.0),
  * the same per-cell permutation seed zlib.crc32("model|condition"),
  * 1000 draws, which is what PREREG_EXP_R1D §6 fixes (the analyzer's own
    N_PERM default is 200 and its p floor of 1/201 would not match §6).

usage:
    python experiments/analyze_r1d_thinkoff.py
    python experiments/analyze_r1d_thinkoff.py --model qwen3.5:9b
"""
import argparse
import glob
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from experiments.analyze_r1_faithful import (  # noqa: E402
    claim_to_pair, degeneracy, grouped_kfold_accuracy, kfold_accuracy,
    loo_accuracy, perm_p,
)
from experiments.claims_equalized_v2 import EQUALIZED_CLAIMS_V2  # noqa: E402

THINKOFF_DIR = "data/results/r1d_thinkoff"
OUT = "data/results/r1d_thinkoff/r1d_thinkoff_summary.json"
N_PERM = 1000            # PREREG_EXP_R1D §6, not the analyzer's 200 default


def cells(model=None):
    """Confirmatory think:false cells only -- pilots are never analyzed here."""
    out = {}
    c2p = claim_to_pair(EQUALIZED_CLAIMS_V2)
    for path in sorted(glob.glob(os.path.join(THINKOFF_DIR, "*_thinkoff.json"))):
        base = os.path.basename(path)
        if "_pilot" in base or "summary" in base:
            continue
        with open(path) as f:
            blob = json.load(f)
        if model and blob["model"] != model:
            continue
        recs = blob["records"]
        out[(blob["model"], blob["condition"])] = {
            "file": path,
            "X": np.array([r["vector"] for r in recs], dtype=float),
            "y": np.array([r["label"] for r in recs], dtype=int),
            "amb": float(np.mean([np.mean(r["ambiguous"]) for r in recs])),
            "n": len(recs),
            "groups": np.array([c2p.get(r.get("claim", ""), 10_000 + i)
                                for i, r in enumerate(recs)]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    a = ap.parse_args()

    data = cells(a.model)
    if not data:
        raise SystemExit(f"no confirmatory think:false cells in {THINKOFF_DIR}")

    print("=" * 100)
    print("EXP-R1d EXPLORATORY ARM: think:false (PREREG_EXP_R1D.md §9 deviation)")
    print("Same v2 materials, same estimator, same seeds as EXP-R1c. NOT a")
    print("pre-registered cell: it cannot and does not move the §7 verdict,")
    print("which is branch 5 (inconclusive) on the undeviated roster.")
    print("=" * 100)
    print(f"perm p: {a.n_perm} label permutations per cell, against the 5-fold estimate.")
    print(f"{'Model':<14} {'cond':<12} {'5-fold':>7} {'grp-5f':>7} {'LOO':>7} "
          f"{'perm p':>8} {'n':>4} {'amb':>5}  varying  uniq")
    print("-" * 100)

    rows = {}
    for (m, cond) in sorted(data):
        d = data[(m, cond)]
        X, y, g = d["X"], d["y"], d["groups"]
        kf = kfold_accuracy(X, y)
        gkf = grouped_kfold_accuracy(X, y, g)
        loo = loo_accuracy(X, y)
        seed = zlib.crc32(f"{m}|{cond}".encode())
        p = perm_p(X, y, kf, n_perm=a.n_perm, seed=seed)
        nvar, nuniq = degeneracy(X)
        rows[f"{m}|{cond}"] = {
            "file": d["file"], "n": d["n"], "kfold": kf, "grouped_kfold": gkf,
            "loo": loo, "perm_p": p, "ambiguous_rate": d["amb"],
            "n_varying_dims": nvar, "n_unique_vectors": nuniq,
        }
        print(f"{m:<14} {cond:<12} {kf:>6.1%} {gkf:>6.1%} {loo:>6.1%} "
              f"{p:>8.3f} {d['n']:>4} {d['amb']:>4.0%}  {nvar:>5}/16 {nuniq:>5}")

    print("-" * 100)
    for m in sorted({k.split("|")[0] for k in rows}):
        i, e = rows.get(f"{m}|instructed"), rows.get(f"{m}|equalized")
        if i and e:
            print(f"{m}: instructed {i['kfold']:.1%} (p={i['perm_p']:.3f}) -> "
                  f"equalized {e['kfold']:.1%} (p={e['perm_p']:.3f}), "
                  f"change {100 * (e['kfold'] - i['kfold']):+.1f} pp")
    print("-" * 100)

    summary = {
        "experiment": "EXP-R1d exploratory arm (think:false)",
        "prereg": "docs/PREREG_EXP_R1D.md §9",
        "arm": "exploratory -- may not revise the pre-registered §7 verdict",
        "primary_verdict_unaffected": "§7.5 branch 5 (inconclusive)",
        "claim_set": "claims_equalized_v2",
        "n_perm": a.n_perm,
        "cells": rows,
    }
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
