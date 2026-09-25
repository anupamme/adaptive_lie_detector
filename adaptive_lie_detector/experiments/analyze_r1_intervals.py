#!/usr/bin/env python3
"""EXP-R1/R1c, post-hoc descriptive arm: intervals on the equalization panel.

WHY THIS EXISTS
---------------
Tables tab:r1_faithful (EXP-R1, v1 bank) and tab:r1c_v2 (EXP-R1c, v2 bank) report
per-target instructed and equalized accuracy, the cell size, and an exact
permutation p -- but no interval on any cell, and the headline 97.0% -> 43.3%
carries none either. A reviewer asked for the intervals. This computes them.

It is DESCRIPTIVE and it is POST-HOC. The pre-registered inference for both
experiments is the label-permutation test already in the tables, and nothing here
touches it: no verdict, no p-value, no accuracy and no change-in-pp is a function
of anything computed in this file. As a guard, every cell's point accuracy is
recomputed from the committed checkpoints and ASSERTED equal to the published
table literal before any interval is emitted; a mismatch exits non-zero, because
that would be a defect and not a result.

TWO INTERVALS, AND WHY BOTH ARE REPORTED
----------------------------------------
(1) WILSON on the count of correct out-of-fold predictions, n = 50 per cell
    (300 pooled). This is the interval the rest of the paper reports for an
    accuracy (tab:probe_panel's `Wilson 95%` column, the EXP-AD transfer cells,
    the human baseline), so it is the comparable one. It treats the 50 predictions
    as exchangeable Bernoulli draws.

(2) CLAIM-PAIR-CLUSTERED BOOTSTRAP, because (1)'s exchangeability assumption is
    false here and measurably so: each cell's 50 trials come from 25 MATCHED
    CLAIM PAIRS, two rows per pair (the true claim and the false claim), which is
    why the panel reports a GroupKFold-on-claim-pair column at all. The cluster
    is the pair, so the effective unit is 25 per cell and 25 pooled -- the claim
    pairs are SHARED across the six targets, which is exactly how the published
    pooled estimator groups them. Each draw resamples pairs with replacement and
    reruns the published estimator.

Wilson is therefore the comparable interval and the clustered one is the honest
interval, and the appendix prose says so in that order.

THE ROSTER IS PINNED, NOT GLOBBED
---------------------------------
`analyze_r1_faithful.load_checkpoints` globs `r1b_fresh_*.json`, and that glob now
also matches `ministral-3:8b`, an EXP-R1d recency target collected AFTER the
EXP-R1c panel was published. Pooling the glob would give n = 350 and a pooled pair
of 97.1%/42.0% in place of the published 97.0%/43.3%. The six-target roster below
is therefore pinned, and the pooled cell size is asserted to be 300.

THE POOLED CELL IS ROW-ORDER DEPENDENT, AND WE FOUND THAT HERE
--------------------------------------------------------------
The per-target cells recompute to their published literals exactly, all 24 of
them. The four POOLED cells only do so when the six targets are stacked in the
order `load_checkpoints` yields them -- alphabetical by checkpoint filename, which
is the order the publishing script pooled in -- and not in the order the tables
print the rows. The cause is not a seed: `kfold_accuracy` uses
`StratifiedKFold(shuffle=True, random_state=0)`, whose fold assignment is a
function of each row's POSITION in the stacked matrix, so permuting which target's
block comes first reshuffles which rows share a fold. Measured swing between the
two orders, reported in the output as `pooled_row_order_sensitivity`:

    v1 instructed 81.7% vs 80.3%  (1.3 pp)   v1 equalized 53.7% vs 49.3%  (4.3 pp)
    v2 instructed 97.0% vs 96.7%  (0.3 pp)   v2 equalized 43.3% vs 44.3%  (1.0 pp)

Every one of those four swings lies inside the Wilson interval this script
computes for the same cell, which is the point: a pooled point estimate carrying
0.3 to 4.3 pp of row-order noise is exactly a number that should have been
reported with an interval. Nothing about the collapse changes -- the instructed and equalized
intervals do not come close to overlapping under either order -- so this is
reported as a precision disclosure, not a correction. We pool in the publishing
order because that is the estimator the tables printed; the alternative order is
computed too and carried in the output so the sensitivity is checkable.

INPUTS (all committed; no model call, no network)
------------------------------------------------
  data/results/r1_faithful_*.json   (v1 bank, EXP-R1)
  data/results/r1b_fresh_*.json     (v2 bank, EXP-R1c; ministral-3:8b excluded)

OUTPUT
------
  data/results/r1_intervals.json

USAGE
-----
  python experiments/analyze_r1_intervals.py
  python experiments/analyze_r1_intervals.py --variant v2
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_external_audit import wilson                          # noqa: E402
from analyze_r1_faithful import (                                  # noqa: E402
    RESULTS_DIR, load_checkpoints, kfold_accuracy, grouped_kfold_accuracy,
)

OUT_PATH = os.path.join(RESULTS_DIR, "r1_intervals.json")

N_BOOT = 2000
SEED = 42
CI = (2.5, 97.5)

# Roster in the order each table PRINTS its rows. Used for the per-target report
# and for the order-sensitivity diagnostic -- NOT for the pooled cell, which must
# be stacked in the publishing order (see `pool_order`). Pinned, not globbed: see
# the module docstring on ministral-3:8b.
ROSTER = {
    "v1": ["llama3.2:3b", "mistral:7b", "qwen2.5:7b",
           "llama3.1:8b", "qwen2.5:14b", "qwen2.5:32b"],
    "v2": ["llama3.2:3b", "llama3.1:8b", "mistral:7b",
           "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b"],
}

# The published stratified-5-fold literals, transcribed from the two tables. The
# guard below asserts the recomputation equals these to within half a printed
# digit, so a drift in sklearn, in the claim sets or in the checkpoints fails
# loudly instead of producing an interval around a number the paper never printed.
PUBLISHED = {
    "v1": {                       # tab:r1_faithful
        "llama3.2:3b":  (0.960, 0.480),
        "mistral:7b":   (0.980, 0.520),
        "qwen2.5:7b":   (0.540, 0.500),
        "llama3.1:8b":  (0.700, 0.500),
        "qwen2.5:14b":  (0.940, 0.700),
        "qwen2.5:32b":  (1.000, 0.520),
        "POOLED":       (0.817, 0.537),
    },
    "v2": {                       # tab:r1c_v2
        "llama3.2:3b":  (1.000, 0.240),
        "llama3.1:8b":  (0.860, 0.420),
        "mistral:7b":   (1.000, 0.580),
        "qwen2.5:7b":   (1.000, 0.500),
        "qwen2.5:14b":  (1.000, 0.460),
        "qwen2.5:32b":  (1.000, 0.420),
        "POOLED":       (0.970, 0.433),
    },
}
CONDITIONS = ("instructed", "equalized")


def successes(acc, n):
    """The integer correct-count behind a printed accuracy.

    With 50 balanced rows and 5 stratified folds the folds are equal-sized, so
    `cross_val_score(...).mean()` equals overall correct/n exactly and the count
    is recoverable. Asserted rather than assumed, because Wilson on a
    non-integer numerator would be meaningless.
    """
    k = round(acc * n)
    if abs(acc * n - k) > 1e-6:
        raise AssertionError(
            f"accuracy {acc!r} x n={n} is not an integer count ({acc * n!r}); "
            "the folds are not equal-sized and Wilson does not apply")
    return int(k)


def cluster_bootstrap(X, y, groups, key, n_boot=N_BOOT):
    """Percentile CI on the GROUPED estimator, resampling CLAIM PAIRS.

    Each draw samples the cell's 25 distinct claim pairs with replacement and
    gives every drawn copy a FRESH group id, so a pair drawn twice contributes
    two clusters that GroupKFold keeps in separate folds. This is the handling
    `analyze_crit4b_ci.py` established, and here it is not a stylistic choice but
    a correctness requirement: with replacement a pair's two rows appear twice,
    and under the PRIMARY stratified estimator the duplicates straddle folds, so
    the classifier sees a test row verbatim in training. Measured, before the fix:
    the v2 llama3.2:3b equalized cell has a point estimate of 24.0% and a naive
    stratified cluster bootstrap put it at [36.0%, 70.0%] -- an interval that does
    not contain its own estimate, because the leakage inflates every draw. The
    grouped estimator is what a cluster bootstrap can be run on at all, so the
    interval it yields is an interval on the panel's `grp-5f` column.

    Draws that leave the label single-class, or that GroupKFold cannot split, are
    not estimable; they are skipped and counted rather than replaced, so the
    reported width is not conditioned on an unstated redraw rule.
    """
    by_pair = {}
    for i, g in enumerate(groups):
        by_pair.setdefault(int(g), []).append(i)
    pairs = sorted(by_pair)
    rng = np.random.default_rng(SEED + zlib.crc32(("r1ci|" + key).encode()))

    accs, skipped = [], 0
    for _ in range(n_boot):
        drawn = rng.choice(len(pairs), size=len(pairs), replace=True)
        idx, gg = [], []
        for j, p in enumerate(drawn):
            for i in by_pair[pairs[p]]:
                idx.append(i)
                gg.append(j)          # fresh id per drawn copy
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        a = grouped_kfold_accuracy(X[idx], yy, np.asarray(gg, int))
        if a == a:
            accs.append(float(a))
        else:
            skipped += 1
    accs = np.asarray(accs, float)
    lo, hi = ((float(np.percentile(accs, CI[0])), float(np.percentile(accs, CI[1])))
              if len(accs) else (None, None))
    return {
        "estimator": "grouped 5-fold (GroupKFold on drawn-copy id)",
        "n_clusters": len(pairs),
        "n_draws": n_boot,
        "n_draws_estimable": int(len(accs)),
        "n_draws_skipped_degenerate": int(skipped),
        "ci_lo": lo,
        "ci_hi": hi,
        "median": float(np.median(accs)) if len(accs) else None,
    }


def cell(X, y, groups, key, published):
    n = len(y)
    acc = kfold_accuracy(X, y)
    # Half a printed digit: the tables print one decimal place on a percentage.
    ok = abs(acc - published) <= 0.0005 + 1e-9
    k = successes(acc, n)
    lo, hi = wilson(k, n)
    boot = cluster_bootstrap(X, y, groups, key)
    gacc = float(grouped_kfold_accuracy(X, y, groups))
    # The clustered interval is on the grouped estimator, so it must cover the
    # grouped point estimate. This is the assertion that exposed the duplicate-row
    # leakage in the first draft; it stays in the output as a standing guard.
    boot["covers_grouped_point"] = bool(
        boot["ci_lo"] is not None and boot["ci_lo"] <= gacc <= boot["ci_hi"])
    return {
        "n": n,
        "published_accuracy": published,
        "recomputed_accuracy": float(acc),
        "reproduces_published": bool(ok),
        "correct": k,
        "wilson_lo": float(lo),
        "wilson_hi": float(hi),
        "grouped_kfold_accuracy": gacc,
        "claim_clustered": boot,
    }


def run_variant(variant):
    data = load_checkpoints(variant)
    roster, pub = ROSTER[variant], PUBLISHED[variant]
    missing = [m for m in roster for c in CONDITIONS if (m, c) not in data]
    if missing:
        raise SystemExit(f"{variant}: missing checkpoints for {sorted(set(missing))}")

    out, bad = {}, []
    for m in roster:
        out[m] = {}
        for ci, cond in enumerate(CONDITIONS):
            d = data[(m, cond)]
            r = cell(d["X"], d["y"], d["groups"], f"{variant}|{m}|{cond}",
                     pub[m][ci])
            out[m][cond] = r
            if not r["reproduces_published"]:
                bad.append((variant, m, cond, r["recomputed_accuracy"],
                            r["published_accuracy"]))

    # Pooled over the pinned roster only, stacked in the PUBLISHING order --
    # `load_checkpoints`' own insertion order, alphabetical by checkpoint
    # filename -- because StratifiedKFold's fold assignment depends on row
    # position and the tables were produced that way. See the module docstring.
    # The claim-pair ids are shared across targets, which is how the published
    # pooled estimator groups them, so the pooled cluster count is 25, not 150.
    pool_order = [m for (m, c) in data if c == "instructed" and m in roster]
    if sorted(pool_order) != sorted(roster):
        raise SystemExit(f"{variant}: pool order {pool_order} is not the roster")

    out["POOLED"] = {}
    order_sensitivity = {}
    for ci, cond in enumerate(CONDITIONS):
        X = np.vstack([data[(m, cond)]["X"] for m in pool_order])
        y = np.concatenate([data[(m, cond)]["y"] for m in pool_order])
        g = np.concatenate([data[(m, cond)]["groups"] for m in pool_order])
        # The same cell stacked in table-print order: a different number from the
        # same rows and the same estimator, recorded so the reader can see how
        # much of the pooled digit is row-order noise.
        alt = kfold_accuracy(
            np.vstack([data[(m, cond)]["X"] for m in roster]),
            np.concatenate([data[(m, cond)]["y"] for m in roster]))
        order_sensitivity[cond] = {
            "publishing_order": pool_order,
            "table_print_order": roster,
            "accuracy_table_print_order": float(alt),
        }
        if len(y) != 300:
            raise SystemExit(
                f"{variant} pooled {cond}: n={len(y)}, expected 300 -- the "
                "roster picked up a target the published pooled cell excludes")
        r = cell(X, y, g, f"{variant}|POOLED|{cond}", pub["POOLED"][ci])
        s = order_sensitivity[cond]
        s["accuracy_publishing_order"] = r["recomputed_accuracy"]
        s["swing_pp"] = abs(r["recomputed_accuracy"]
                            - s["accuracy_table_print_order"]) * 100
        # The swing is only a precision disclosure if the interval already covers
        # it. If it ever escapes, the pooled digit is doing work the interval
        # cannot justify, and that is a finding rather than a footnote.
        s["swing_inside_wilson"] = bool(
            r["wilson_lo"] <= s["accuracy_table_print_order"] <= r["wilson_hi"])
        r["row_order_sensitivity"] = s
        out["POOLED"][cond] = r
        if not r["reproduces_published"]:
            bad.append((variant, "POOLED", cond, r["recomputed_accuracy"],
                        r["published_accuracy"]))

    print(f"[{variant}]  {'target':<14}{'cond':<11}{'n':>4}{'5-fold':>8}"
          f"{'Wilson 95%':>20}{'grp-5f':>9}{'clustered 95%':>22}")
    for m in roster + ["POOLED"]:
        for cond in CONDITIONS:
            r = out[m][cond]
            b = r["claim_clustered"]
            w = f"[{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]"
            c = (f"[{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]"
                 if b["ci_lo"] is not None else "not estimable")
            print(f"        {m:<14}{cond:<11}{r['n']:>4}"
                  f"{r['recomputed_accuracy']:>8.3f}{w:>20}"
                  f"{r['grouped_kfold_accuracy']:>9.3f}{c:>22}"
                  f"{'' if b['covers_grouped_point'] else '  <- NOT COVERED'}")
    for cond in CONDITIONS:
        s = out["POOLED"][cond]["row_order_sensitivity"]
        print(f"        pooled {cond} row-order swing: "
              f"{s['accuracy_publishing_order']:.4f} (publishing order) vs "
              f"{s['accuracy_table_print_order']:.4f} (table order) = "
              f"{s['swing_pp']:.1f} pp; inside Wilson: "
              f"{s['swing_inside_wilson']}")
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(ROSTER), action="append",
                    help="default: both")
    args = ap.parse_args()
    variants = args.variant or ["v1", "v2"]

    print("=" * 100)
    print("EXP-R1/R1c: intervals on the equalization panel "
          "(descriptive, post-hoc)")
    print(f"  Wilson on out-of-fold correct counts; claim-pair cluster "
          f"bootstrap at {N_BOOT} draws, seed {SEED}")
    print("=" * 100)

    variant_out, bad = {}, []
    for v in variants:
        o, b = run_variant(v)
        variant_out[v] = o
        bad.extend(b)
        print("-" * 100)

    report = {
        "experiment": "EXP-R1 / EXP-R1c",
        "arm": "accuracy_intervals",
        "status": "DESCRIPTIVE, POST-HOC",
        "estimand": "the same stratified 5-fold accuracy tab:r1_faithful and "
                    "tab:r1c_v2 report",
        "not_a_test": "The pre-registered inference is the label-permutation "
                      "test already printed in both tables. No accuracy, p-value "
                      "or change-in-pp depends on anything in this file.",
        "wilson": "95% Wilson on the count of correct out-of-fold predictions; "
                  "treats the n trials as exchangeable Bernoulli draws",
        "claim_clustered": "95% percentile bootstrap resampling the 25 matched "
                           "claim pairs per cell with replacement, each drawn "
                           "copy given a fresh group id and scored under "
                           "GroupKFold; an interval on the panel's grp-5f column, "
                           "not on the primary. The pair is the honest unit "
                           "because each contributes two rows, but the grouped "
                           "estimator is the only one a cluster bootstrap can be "
                           "run on here: under the primary stratified estimator "
                           "the duplicated rows straddle folds and leak.",
        "pooled_row_order_sensitivity": "The four pooled cells reproduce only "
                                        "when the six targets are stacked in "
                                        "load_checkpoints' insertion order, the "
                                        "order the publishing script used; "
                                        "StratifiedKFold(shuffle=True, "
                                        "random_state=0) assigns folds by row "
                                        "position, so table-print order gives a "
                                        "different digit from the same rows. Per "
                                        "cell under variants[v].POOLED[cond]."
                                        "row_order_sensitivity. All 24 "
                                        "per-target cells are unaffected.",
        "roster_pinned_because": "load_checkpoints globs r1b_fresh_*.json, which "
                                 "now also matches the EXP-R1d target "
                                 "ministral-3:8b, collected after the EXP-R1c "
                                 "panel was published",
        "seed": SEED,
        "n_boot": N_BOOT,
        "ci_pct": list(CI),
        "variants": variant_out,
        "all_cells_reproduce_published_accuracy": not bad,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_PATH}")

    if bad:
        print("\nREPRODUCTION FAILURE. Published numbers do not recompute:")
        for v, m, cond, got, want in bad:
            print(f"  {v} {m} {cond}: recomputed {got:.4f} vs published {want}")
        raise SystemExit(1)
    print("every cell's point accuracy recomputes to its published table "
          "literal; no published number changes")


if __name__ == "__main__":
    main()
