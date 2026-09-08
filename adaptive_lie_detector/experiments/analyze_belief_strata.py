#!/usr/bin/env python3
"""
analyze_belief_strata.py

Scores EXP-B (run_belief_strata.py) and tests the pre-registered hypothesis.

DETECTOR
--------
The parameter-free refusal/correction-marker rule from hedging_baseline.py:
predict "false claim presented" iff the target's first response contains at
least one of the 14 correction markers (k >= 1). Nothing is fit, so there is no
cross-validation, no calibration, and no leakage -- the same L-indep detector
the paper uses for its 79.8% result.

PRE-REGISTERED TESTS (one-sided, alpha = 0.05), fixed before any trial ran:
  H1  accuracy on KNOWN claims is above chance
  H2  accuracy on UNKNOWN claims is at chance (reported, not a rejection target)
  H3  the KNOWN - UNKNOWN accuracy gap is positive

H1/H2 permute the trial labels within a stratum (null: marker count carries no
information about which member of the pair was presented). H3 permutes the
stratum assignment across pairs, holding labels fixed (null: the two strata are
exchangeable). Seeds are derived with zlib.crc32 rather than hash(), which is
salted by PYTHONHASHSEED and would not reproduce.

Usage:
    cd code/adaptive_lie_detector
    python experiments/analyze_belief_strata.py
    python experiments/analyze_belief_strata.py --n_perm 10000
"""

import os
import re
import sys
import json
import zlib
import glob
import math
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hedging_baseline import REFUSAL_PATTERNS  # noqa: E402

_COMPILED = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]


def marker_count(text):
    """Number of distinct correction-marker patterns present. Matches
    hedging_baseline.extract_text_features: patterns, not occurrences."""
    return sum(1 for p in _COMPILED if p.search(text or ""))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rule_accuracy(counts, labels, k=1):
    """Accuracy of the fixed k>=1 rule. counts/labels are 1-D arrays."""
    pred = (counts >= k).astype(int)
    return float((pred == labels).mean())


def perm_p_above_chance(counts, labels, n_perm, seed):
    """One-sided: P(permuted accuracy >= observed) with labels shuffled."""
    obs = rule_accuracy(counts, labels)
    rng = np.random.default_rng(seed)
    y = labels.copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(y)
        if rule_accuracy(counts, y) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def perm_p_gap(counts, labels, strata, n_perm, seed):
    """One-sided: P(permuted gap >= observed) with stratum labels shuffled.

    Shuffling is done over *pairs*, not trials, so the two trials belonging to
    one claim pair always move together -- otherwise the permutation would break
    the pairing that the design depends on.
    """
    is_known = strata == "KNOWN"
    obs = (rule_accuracy(counts[is_known], labels[is_known])
           - rule_accuracy(counts[~is_known], labels[~is_known]))
    rng = np.random.default_rng(seed)
    n = len(counts)
    assert n % 2 == 0, "expected two trials per pair"
    pair_flags = is_known[::2].copy()          # one flag per pair
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(pair_flags)
        m = np.repeat(pair_flags, 2)
        a = rule_accuracy(counts[m], labels[m])
        b = rule_accuracy(counts[~m], labels[~m])
        if (a - b) >= obs:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    s = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return float("nan") if s == 0 else (np.mean(a) - np.mean(b)) / s


def order_pairs(recs):
    """Sort so the two trials of a pair are adjacent in (label 0, label 1) order,
    which perm_p_gap relies on, and drop any pair that did not yield both trials.

    The sort key includes the model so that pooling several targets -- whose
    `index` values overlap, since each indexes into the same 120-pair pool --
    cannot interleave two models' trials inside one apparent pair.
    """
    recs = sorted(recs, key=lambda r: (r["stratum"] != "KNOWN", r.get("model", ""),
                                       r["index"], r["label"]))
    counts = {}
    for r in recs:
        k = (r.get("model", ""), r["stratum"], r["index"])
        counts[k] = counts.get(k, 0) + 1
    dropped = [k for k, c in counts.items() if c != 2]
    if dropped:
        print(f"  [warn] dropping {len(dropped)} incomplete pair(s): {dropped[:5]}")
    return [r for r in recs
            if counts[(r.get("model", ""), r["stratum"], r["index"])] == 2]


def analyze_model(path, n_perm):
    blob = json.load(open(path))
    model = blob["model"]
    recs = order_pairs(blob["records"])

    counts = np.array([marker_count(r["response"]) for r in recs])
    labels = np.array([r["label"] for r in recs])
    strata = np.array([r["stratum"] for r in recs])
    words = np.array([r["n_words"] for r in recs])

    out = {"model": model, "n": len(recs), "strata": {}}
    seed_base = zlib.crc32(model.encode())

    for s in ("KNOWN", "UNKNOWN"):
        m = strata == s
        if m.sum() == 0:
            continue
        acc, p = perm_p_above_chance(counts[m], labels[m], n_perm,
                                     seed_base ^ zlib.crc32(s.encode()))
        lo, hi = wilson(int(round(acc * m.sum())), int(m.sum()))
        fire_false = float((counts[m & (labels == 1)] >= 1).mean())
        fire_true = float((counts[m & (labels == 0)] >= 1).mean())
        out["strata"][s] = {
            "n": int(m.sum()),
            "n_pairs": int(m.sum() // 2),
            "acc": acc,
            "ci": [lo, hi],
            "p_above_chance": p,
            "fire_rate_false_claim": fire_false,
            "fire_rate_true_claim": fire_true,
            "marker_d": cohens_d(counts[m & (labels == 1)], counts[m & (labels == 0)]),
            "mean_words": float(words[m].mean()),
            "mean_markers": float(counts[m].mean()),
        }

    if {"KNOWN", "UNKNOWN"} <= set(out["strata"]):
        gap, p_gap = perm_p_gap(counts, labels, strata, n_perm,
                                seed_base ^ 0x9E3779B9)
        out["gap"] = gap
        out["p_gap"] = p_gap
    return out


def screen_summary(model):
    tag = model.replace(":", "_").replace(".", "_")
    p = f"data/results/belief_strata_screen_{tag}.json"
    if not os.path.exists(p):
        return {}
    recs = json.load(open(p))["records"]
    sub = {}
    for r in recs:
        sub[r["subcase"]] = sub.get(r["subcase"], 0) + 1
    return {"n_screened": len(recs),
            "n_known": sum(1 for r in recs if r["stratum"] == "KNOWN"),
            "subcases": sub}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--out", default="data/results/belief_strata_summary.json")
    args = ap.parse_args()

    paths = sorted(glob.glob("data/results/belief_strata_trials_*.json"))
    if not paths:
        raise SystemExit("No EXP-B trial checkpoints found.")

    print("=" * 78)
    print("EXP-B: belief vs truth-value under prompt equalization "
          "(refusal-count rule, k>=1)")
    print("=" * 78)

    summary = []
    for p in paths:
        r = analyze_model(p, args.n_perm)
        r["screen"] = screen_summary(r["model"])
        summary.append(r)

        print(f"\n{r['model']}  (n={r['n']} equalized trials)")
        sc = r["screen"]
        if sc:
            print(f"  screen: {sc['n_known']}/{sc['n_screened']} pairs KNOWN"
                  f"   subcases={sc['subcases']}")
        for s in ("KNOWN", "UNKNOWN"):
            if s not in r["strata"]:
                continue
            d = r["strata"][s]
            print(f"  {s:8s} n={d['n']:3d} ({d['n_pairs']} pairs)  "
                  f"acc={d['acc'] * 100:5.1f}%  CI[{d['ci'][0] * 100:.1f},{d['ci'][1] * 100:.1f}]  "
                  f"p={d['p_above_chance']:.4f}")
            print(f"           fire: false-claim {d['fire_rate_false_claim'] * 100:.0f}%  "
                  f"true-claim {d['fire_rate_true_claim'] * 100:.0f}%   "
                  f"marker d={d['marker_d']:+.2f}  mean words={d['mean_words']:.0f}")
        if "gap" in r:
            print(f"  GAP (KNOWN - UNKNOWN) = {r['gap'] * 100:+.1f} pp   "
                  f"p={r['p_gap']:.4f}")

    # Pooled across targets.
    allrecs, allmodels = [], []
    for p in paths:
        blob = json.load(open(p))
        allmodels.append(blob["model"])
        for r in blob["records"]:
            allrecs.append({**r, "model": blob["model"]})
    allrecs = order_pairs(allrecs)
    counts = np.array([marker_count(r["response"]) for r in allrecs])
    labels = np.array([r["label"] for r in allrecs])
    strata = np.array([r["stratum"] for r in allrecs])
    print(f"\nPOOLED ({len(allmodels)} targets, n={len(allrecs)})")
    pooled = {"models": allmodels, "n": len(allrecs), "strata": {}}
    for s in ("KNOWN", "UNKNOWN"):
        m = strata == s
        acc, p = perm_p_above_chance(counts[m], labels[m], args.n_perm,
                                     zlib.crc32(("pooled" + s).encode()))
        lo, hi = wilson(int(round(acc * m.sum())), int(m.sum()))
        pooled["strata"][s] = {"n": int(m.sum()), "acc": acc, "ci": [lo, hi],
                               "p_above_chance": p}
        print(f"  {s:8s} n={m.sum():3d}  acc={acc * 100:5.1f}%  "
              f"CI[{lo * 100:.1f},{hi * 100:.1f}]  p={p:.4f}")
    gap, p_gap = perm_p_gap(counts, labels, strata, args.n_perm, 0x5BD1E995)
    pooled["gap"], pooled["p_gap"] = gap, p_gap
    print(f"  GAP = {gap * 100:+.1f} pp   p={p_gap:.4f}")

    with open(args.out, "w") as f:
        json.dump({"per_model": summary, "pooled": pooled,
                   "n_perm": args.n_perm,
                   "detector": "refusal_count_rule_k1",
                   "n_patterns": len(REFUSAL_PATTERNS)}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
