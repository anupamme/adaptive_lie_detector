#!/usr/bin/env python3
"""EXP-C4B, post-hoc descriptive arm: claim-clustered bootstrap intervals on H1.

WHY THIS IS A SEPARATE SCRIPT, AND NOT A NEW --phase OF analyze_crit4b.py
------------------------------------------------------------------------
`analyze_crit4b.py --phase unseal` verifies gate 6 (blinding integrity), and one
of its checks is that the analysis script's git hash is unchanged since the seal
(`analyze_unchanged_since_seal`). Adding a phase to that file would flip that
gate to FAIL and, under PREREG §10(f), void the blind for a reason that has
nothing to do with the data. So this arm lives here, imports the frozen
estimator rather than restating it, and leaves every byte of the sealed pipeline
alone. `--phase unseal` still reproduces every published number.

WHAT IT ADDS, AND WHAT IT MAY NOT DO (PREREG_EXP_C4B §10, round-37 entry)
------------------------------------------------------------------------
Table 4's per-target accuracies are point estimates over hundreds of graded
trials, but the trials are nested: the inferential unit is the PAIRED CLAIM, of
which there are 12-23 per target, not the 231-400 trials. This computes a
cluster bootstrap over claims on the SAME estimator the table reports
(`grouped_kfold_accuracy`, claim-grouped stratified 5-fold), so the reader can
see the width the claim-level denominator implies.

It is DESCRIPTIVE. The pre-registered test is the within-claim permutation test
with Holm correction inside each family of five, and it is untouched: no
verdict, count, p-value or MDE in this experiment is a function of anything
computed here. As a check that nothing drifted, the script recomputes each
cell's point accuracy from the committed candidate rows and ASSERTS it equals
the published value exactly; a mismatch is a defect and exits non-zero.

INPUTS (all committed; no model call, no network)
------------------------------------------------
  Family R: data/results/crit4b_analysis_family_R.json  (archived before E was
            sealed; crit4b_analysis.json also carries a VACUOUS Family R
            summary, which is not the published one and is not read here)
            + data/results/crit4b_blind_family_R/<pseudonym>_cand<NN>.json
  Family E: data/results/crit4b_analysis.json
            + data/results/crit4b_blind/<pseudonym>_cand<NN>.json

The pseudonym->model map and the real candidate index are read from the
published analysis JSONs, so the salt is not needed and stays secret.

OUTPUT
------
  data/results/crit4b_ci.json

USAGE
-----
  analyze_crit4b_ci.py                # both families
  analyze_crit4b_ci.py --family R
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_r1_faithful import grouped_kfold_accuracy      # noqa: E402
from analyze_crit4b import FAMILY_E, FAMILY_R, RESULTS, SEED  # noqa: E402

OUT_PATH = os.path.join(RESULTS, "crit4b_ci.json")

N_BOOT = 2000          # draws per cell
CI = (2.5, 97.5)       # percentile interval

SOURCES = {
    "R": (os.path.join(RESULTS, "crit4b_analysis_family_R.json"),
          os.path.join(RESULTS, "crit4b_blind_family_R")),
    "E": (os.path.join(RESULTS, "crit4b_analysis.json"),
          os.path.join(RESULTS, "crit4b_blind")),
}
ROSTERS = {"R": FAMILY_R, "E": FAMILY_E}


def load_real_rows(analysis_path, blind_dir, model):
    """The analysed rows of one cell, from the committed blind candidate file.

    Reproduces `_analyse_candidate`'s H4 step (the belief exclusion) so that the
    row set here is the one H1 was computed on, not the pre-exclusion set.
    """
    with open(analysis_path) as f:
        rep = json.load(f)
    tgt = rep["targets"][model]
    pseudo, k = tgt["pseudonym"], tgt["real_candidate_index"]
    path = os.path.join(blind_dir, f"{pseudo}_cand{k:02d}.json")
    with open(path) as f:
        rows = json.load(f)["rows"]
    d1 = [r for r in rows if r["D"] == 1]
    kept1 = sum(1 for r in d1 if r["known_preserved"])
    holds = bool(d1) and (kept1 / len(d1)) > 0.5
    analysed = rows if (not d1 or holds) else [r for r in rows
                                               if r["known_preserved"]]
    return analysed, tgt, os.path.basename(path)


def cluster_bootstrap(rows, model, n_boot=N_BOOT):
    """Percentile CI on the published estimator, resampling CLAIMS not trials.

    Each draw samples the cell's distinct claims with replacement and gives every
    drawn copy a FRESH group id, so a claim drawn twice contributes two clusters
    that GroupKFold keeps separately -- the standard cluster-bootstrap handling,
    and the alternative (reusing the original id) would silently shrink the
    number of folds' worth of independent groups.

    Draws that leave the label single-class, or fewer than 2*5 rows, are not
    estimable by `grouped_kfold_accuracy` (it returns nan); they are skipped and
    counted rather than replaced, so the reported width is not quietly
    conditioned on a redraw rule.
    """
    by_claim = {}
    for r in rows:
        by_claim.setdefault(r["claim"], []).append(r)
    claims = sorted(by_claim)
    rng = np.random.default_rng(SEED + zlib.crc32(("ci|" + model).encode()))

    accs, margins, skipped = [], [], 0
    for _ in range(n_boot):
        drawn = rng.choice(len(claims), size=len(claims), replace=True)
        X, y, g = [], [], []
        for j, idx in enumerate(drawn):
            for r in by_claim[claims[idx]]:
                X.append(r["vector"])
                y.append(r["D"])
                g.append(j)
        yy = np.asarray(y, int)
        a = grouped_kfold_accuracy(np.asarray(X, float), yy,
                                   np.asarray(g, int))
        if a == a:
            accs.append(float(a))
            # The draw's OWN majority baseline: resampling claims moves the D
            # base rate, and accuracy moves with it, so the accuracy interval
            # alone mixes two sources of variation. The margin is the
            # comparison the table's Acc.-vs-Maj. reading actually makes.
            maj = max(float(yy.mean()), 1.0 - float(yy.mean()))
            margins.append(float(a) - maj)
        else:
            skipped += 1
    accs = np.asarray(accs, float)
    mrg = np.asarray(margins, float)
    lo, hi = (np.percentile(accs, CI[0]), np.percentile(accs, CI[1])) \
        if len(accs) else (None, None)
    mlo, mhi = (np.percentile(mrg, CI[0]), np.percentile(mrg, CI[1])) \
        if len(mrg) else (None, None)
    return {
        "n_claims_resampled": len(claims),
        "n_draws": n_boot,
        "n_draws_estimable": int(len(accs)),
        "n_draws_skipped_degenerate": int(skipped),
        "ci_pct": list(CI),
        "ci_lo": None if lo is None else float(lo),
        "ci_hi": None if hi is None else float(hi),
        "median": None if not len(accs) else float(np.median(accs)),
        "margin_ci_lo": None if mlo is None else float(mlo),
        "margin_ci_hi": None if mhi is None else float(mhi),
        "margin_median": None if not len(mrg) else float(np.median(mrg)),
    }


def run_family(fam):
    analysis_path, blind_dir = SOURCES[fam]
    out, bad = {}, []
    for model in ROSTERS[fam]:
        rows, tgt, cand = load_real_rows(analysis_path, blind_dir, model)
        X = np.asarray([r["vector"] for r in rows], float)
        y = np.asarray([r["D"] for r in rows], int)
        g = np.asarray([r["claim"] for r in rows], int)
        point = grouped_kfold_accuracy(X, y, g)
        pub = tgt["H1_primary"]["accuracy"]
        ok = (pub is not None and abs(point - pub) < 1e-12
              and len(rows) == tgt["n_analysed"])
        if not ok:
            bad.append((model, point, pub, len(rows), tgt["n_analysed"]))
        boot = cluster_bootstrap(rows, model)
        out[model] = {
            "family": fam,
            "pseudonym": tgt["pseudonym"],
            "candidate_file": cand,
            "n_analysed": len(rows),
            "paired_claims": tgt["gate3_both_outcomes"]["paired_claims"],
            "published_accuracy": pub,
            "recomputed_accuracy": float(point),
            "reproduces_published": bool(ok),
            "majority_baseline": tgt["H1_primary"]["majority_baseline"],
            "p_holm_published": tgt["p_holm"],
            "verdict_published": tgt["verdict"],
            **boot,
        }
        b = out[model]
        print(f"  {model:<22} pair={b['paired_claims']:>2}  "
              f"acc={pub:.3f}  95% CI [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}]  "
              f"maj={b['majority_baseline']:.3f}  "
              f"margin 95% CI [{b['margin_ci_lo']:+.3f}, {b['margin_ci_hi']:+.3f}]"
              f"  skipped={b['n_draws_skipped_degenerate']}")
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["R", "E"], action="append",
                    help="default: both")
    args = ap.parse_args()
    fams = args.family or ["R", "E"]

    print("=" * 72)
    print("EXP-C4B — claim-clustered bootstrap CIs on H1 (descriptive, post-hoc)")
    print(f"  estimator: imported grouped_kfold_accuracy; draws={N_BOOT}; "
          f"seed={SEED}; unit: paired claim")
    print("=" * 72)

    targets, bad = {}, []
    for fam in fams:
        print(f"Family {fam}:")
        t, b = run_family(fam)
        targets.update(t)
        bad.extend(b)

    report = {
        "experiment": "EXP-C4B",
        "arm": "claim_clustered_bootstrap_ci",
        "status": "DESCRIPTIVE, POST-HOC (PREREG_EXP_C4B §10, round-37 entry)",
        "estimand": "the same claim-grouped stratified 5-fold accuracy Table "
                    "tab:crit4b_h1 reports, with claims resampled with "
                    "replacement as clusters",
        "not_a_test": "The pre-registered inference is the within-claim "
                      "permutation test with Holm correction within each family "
                      "of five. No verdict, count, p-value or MDE depends on "
                      "anything in this file.",
        "seed": SEED,
        "n_boot": N_BOOT,
        "ci_pct": list(CI),
        "families": fams,
        "targets": targets,
        "all_cells_reproduce_published_accuracy": not bad,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"\nwrote {OUT_PATH}")

    if bad:
        print("\nREPRODUCTION FAILURE — published numbers do not recompute:")
        for m, point, pub, n, npub in bad:
            print(f"  {m}: recomputed {point} vs published {pub}; "
                  f"n {n} vs {npub}")
        raise SystemExit(1)
    print("every cell's point accuracy recomputes to the published value "
          "exactly; no published number changes")


if __name__ == "__main__":
    main()
