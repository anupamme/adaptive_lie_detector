#!/usr/bin/env python3
"""
rereport_crit4_under_c4b.py

PREREG_EXP_C4B.md §6 and §7c oblige EXP-C4's §0 table to be RE-REPORTED under
EXP-C4B's two stricter rules, so the two experiments' tables are read like with
like. §6: "that recomputation uses the committed EXP-C4 artifacts and no model
call, and it may change EXP-C4's own verdict count. If Holm demotes any EXP-C4
target, the paper must say so -- the replication cannot be allowed to quietly
improve the original's bookkeeping."

THE THREE RULES APPLIED RETROSPECTIVELY
---------------------------------------
  (6) HOLM--BONFERRONI WITHIN EXP-C4'S FIVE. EXP-C4 reported five uncorrected
      per-target tests. `holm` is imported from analyze_crit4b so the two
      experiments' corrections are literally the same function.

  (5) MDE > 0.30 ON A NON-SIGNIFICANT CELL IS `UNDERPOWERED`, NOT
      `CRITERION4_NULL`. §12 DEVIATION 5 calls this "a change of LABEL, not of
      any number", and §7c gives the reason: power governs the interpretation of
      a non-detection, not of a detection. A SIGNIFICANT cell stays POSITIVE
      whatever its MDE.

  (4) GATE 3 RISES TO >=12 PAIRED CLAIMS. §7 states before the data that every
      EXP-C4 cell would have failed it. That is reported here as a fact about
      the original design, and it is deliberately NOT used to withdraw an
      EXP-C4 verdict: EXP-C4's gate 3 was >=5 and its cells passed the gate
      they were pre-registered against. Re-gating a completed experiment
      against a threshold written afterwards would be the mirror image of the
      bookkeeping §6 forbids.

NOTHING HERE IS RECOMPUTED FROM TRIALS. Every accuracy, baseline, p_two_sided
and MDE is read verbatim out of the committed crit4_analysis.json. This script
relabels; it does not reanalyse.

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/rereport_crit4_under_c4b.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_crit4b import holm  # noqa: E402  — one Holm, shared by both

RESULTS = "data/results"
IN_PATH = os.path.join(RESULTS, "crit4_analysis.json")
OUT_PATH = os.path.join(RESULTS, "crit4_holm_rereport.json")

MDE_THRESHOLD = 0.30      # §7c
GATE3_C4B_PAIRED = 12     # §7, EXP-C4's own gate was 5
ALPHA = 0.05


def verdict_under_c4b(sig_holm, mde):
    """§7c's three-way rule, evaluated on the HOLM-adjusted p."""
    if sig_holm:
        return "CRITERION4_POSITIVE"
    if mde is None:
        return "UNDERPOWERED"
    return "UNDERPOWERED" if mde > MDE_THRESHOLD else "CRITERION4_NULL"


def main():
    if not os.path.exists(IN_PATH):
        raise SystemExit(f"{IN_PATH} absent; nothing to re-report")
    with open(IN_PATH) as f:
        c4 = json.load(f)
    targets = c4["targets"]

    praw = {m: v["H1_primary"]["p_two_sided"] for m, v in targets.items()}
    adj, n_tests = holm(praw)
    if n_tests != 5:
        print(f"  NOTE: Holm family size is {n_tests}, not the 5 §6 describes")

    rows, demoted, relabelled = [], [], []
    for m in sorted(targets):
        v = targets[m]
        h1 = v["H1_primary"]
        mde = h1.get("mde_probe_delta")
        beats = bool(v["beats_majority_baseline"])
        p_h = adj[m]

        # As EXP-C4 published it: uncorrected p, and no MDE rule.
        sig_raw = bool(v["significant_two_sided"])
        v_orig = v["verdict"]

        # Under EXP-C4B. Significance keeps EXP-C4's conjunction with the
        # majority baseline; only the p it is evaluated on changes.
        sig_holm = p_h < ALPHA and beats
        v_new = verdict_under_c4b(sig_holm, mde)

        row = {
            "target": m,
            "wording": v.get("wording"),
            "accuracy": h1["accuracy"],
            "majority_baseline": h1["majority_baseline"],
            "beats_majority_baseline": beats,
            "auroc_probe_sum": h1.get("auroc_probe_sum"),
            "mde_probe_delta": mde,
            "p_two_sided_uncorrected": praw[m],
            "p_holm": p_h,
            "holm_family_size": n_tests,
            "significant_uncorrected": sig_raw,
            "significant_holm_and_beats_baseline": bool(sig_holm),
            "verdict_as_published": v_orig,
            "verdict_under_c4b_rules": v_new,
            "changed": v_orig != v_new,
            "gate3_paired_claims": v["gates"]["gate3_both_outcomes"]
                                    ["paired_claims"],
        }
        row["would_pass_c4b_gate3"] = (
            row["gate3_paired_claims"] >= GATE3_C4B_PAIRED)
        rows.append(row)

        if v_orig == "CRITERION4_POSITIVE" and v_new != "CRITERION4_POSITIVE":
            demoted.append(m)
        elif v_orig != v_new:
            relabelled.append(m)

    n_pos_pub = sum(1 for r in rows
                    if r["verdict_as_published"] == "CRITERION4_POSITIVE")
    n_pos_new = sum(1 for r in rows
                    if r["verdict_under_c4b_rules"] == "CRITERION4_POSITIVE")
    n_null_pub = sum(1 for r in rows
                     if r["verdict_as_published"] == "CRITERION4_NULL")
    n_null_new = sum(1 for r in rows
                     if r["verdict_under_c4b_rules"] == "CRITERION4_NULL")

    out = {
        "experiment": "EXP-C4, re-reported under EXP-C4B's rules",
        "prereg": "docs/PREREG_EXP_C4B.md §6, §7c, §12 DEVIATIONS 4-6",
        "source_artifact": IN_PATH,
        "model_calls": 0,
        "note": ("Relabelling only. Every accuracy, baseline, p and MDE is read "
                 "verbatim from the committed EXP-C4 artifact; no trial is "
                 "re-analysed and no number is recomputed."),
        "rules_applied": {
            "holm_within_family": f"n_tests={n_tests}, alpha={ALPHA}",
            "mde_rule": f"non-significant with MDE > {MDE_THRESHOLD} -> "
                        f"UNDERPOWERED (§7c)",
            "gate3": f"EXP-C4B's gate 3 needs >={GATE3_C4B_PAIRED} paired "
                     f"claims; reported per target but NOT used to withdraw "
                     f"an EXP-C4 verdict, because EXP-C4 passed the gate it "
                     f"was pre-registered against (>=5)",
        },
        "holm_demoted_any_target": bool(demoted),
        "demoted_targets": demoted,
        "relabelled_by_mde_rule": relabelled,
        "counts": {
            "positive_as_published": n_pos_pub,
            "positive_under_c4b_rules": n_pos_new,
            "null_as_published": n_null_pub,
            "null_under_c4b_rules": n_null_new,
            "underpowered_under_c4b_rules": sum(
                1 for r in rows
                if r["verdict_under_c4b_rules"] == "UNDERPOWERED"),
            "would_pass_c4b_gate3": sum(
                1 for r in rows if r["would_pass_c4b_gate3"]),
        },
        "targets": rows,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 78)
    print("EXP-C4 §0 TABLE, RE-REPORTED UNDER EXP-C4B'S RULES (§6, §7c)")
    print("=" * 78)
    print(f"  source: {IN_PATH}   model calls: 0   relabelling only\n")
    print(f"  {'target':<14}{'acc':>7}{'maj':>7}{'p_raw':>9}{'p_holm':>9}"
          f"{'MDE':>6}{'pair':>6}  {'as published':<20}under EXP-C4B")
    for r in rows:
        mark = "  <-- CHANGED" if r["changed"] else ""
        print(f"  {r['target']:<14}{r['accuracy']:>7.3f}"
              f"{r['majority_baseline']:>7.3f}"
              f"{r['p_two_sided_uncorrected']:>9.4f}{r['p_holm']:>9.4f}"
              f"{r['mde_probe_delta']:>6.2f}{r['gate3_paired_claims']:>6}"
              f"  {r['verdict_as_published']:<20}"
              f"{r['verdict_under_c4b_rules']}{mark}")

    print(f"\n  POSITIVE: {n_pos_pub} as published -> {n_pos_new} under Holm")
    if demoted:
        print(f"  HOLM DEMOTES: {', '.join(demoted)}")
        print("  §6: the paper MUST say so.")
    else:
        print("  Holm demotes NO target: all "
              f"{n_pos_pub} EXP-C4 positives survive correction within their "
              f"five.")
    print(f"  NULL: {n_null_pub} as published -> {n_null_new} under §7c"
          f"  (relabelled UNDERPOWERED: {', '.join(relabelled) or 'none'})")
    ng = out["counts"]["would_pass_c4b_gate3"]
    print(f"  EXP-C4B gate 3 (>={GATE3_C4B_PAIRED} paired claims): "
          f"{ng}/{len(rows)} EXP-C4 cells would pass "
          f"(§7 predicted 0 before the data)")
    print(f"\n  wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
