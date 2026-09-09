#!/usr/bin/env python3
"""analyze_liars_bench_audit.py

EXP-XL analysis (PREREG_EXP_XL.md §4, §7): recomputes every reported number from
the committed derived records in `data/results/liars_bench_rule_*.json`, with
**no network access and no model call**. The records carry marker counts and
labels only -- no corpus text -- because the corpus is gated (PREREG §2).

Every statistic is imported from `analyze_external_audit.py` rather than
reimplemented, so EXP-XL and EXP-XA are the same test at the same thresholds:
`wilson`, `perm_p_unpaired`, `cell_report`, `contrast`, `fired`, with `N_PERM` =
10,000, `SEED` = 42, `MIN_CELL` = 10 and the 5-95% degeneracy gate.

The verdict vocabulary is the committed one -- `CRITERION4_NULL` /
`CRITERION4_POSITIVE` -- and no new verdict string is invented.

**The falsifier.** PREREG §7 branch A: a config that satisfies (i)-(v) AND on
which a detector separates deception at fixed elicitation would falsify the
paper's claim 3. This script prints `FALSIFIER_FIRED` in that case and does not
soften it.

    ../.venv/bin/python3 experiments/analyze_liars_bench_audit.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_external_audit import (  # noqa: E402
    GATE_HI, GATE_LO, MIN_CELL, N_PERM, RESULTS, SEED,
    cell_report, contrast, fired, fmt_cell, wilson,
)
from experiments.run_liars_bench_audit import (  # noqa: E402
    LEAK_FACTOR, LIST_NAME, PREREG_CHANNELS, SECONDARY_LIST, SURVEY_PATH,
)

CHANNEL_ORDER = ("visible", "full", "full_output_only")
OUT_PATH = os.path.join(RESULTS, "liars_bench_analysis.json")


def load_records():
    paths = sorted(glob.glob(os.path.join(RESULTS, "liars_bench_rule_*.json")))
    by_config = {}
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if d.get("contains_corpus_text"):
            raise SystemExit(f"{p} claims to contain corpus text; refusing to use it")
        by_config[d["config"]] = d["records"]
    return by_config


def requirement_iii(recs):
    """(iii): the same scenario independently realized under BOTH behavioral
    outcomes. Counted, not assumed -- this is the requirement no Apollo set meets."""
    groups = {}
    for r in recs:
        if r["scenario_key"] is None:
            continue
        groups.setdefault(r["scenario_key"], set()).add(r["deceptive"])
    paired = [k for k, v in groups.items() if v == {True, False}]
    return {"n_groups": len(groups), "n_paired": len(paired),
            "met": len(paired) > 0,
            "key_source": sorted({r["scenario_key_source"] for r in recs})}


def analyze_cell(recs, rng):
    """One (config x generator) cell: the criterion-4 contrast on every channel."""
    md5s = sorted({r["system_md5"] for r in recs})
    dec = [r for r in recs if r["deceptive"]]
    hon = [r for r in recs if not r["deceptive"]]
    iii = requirement_iii(recs)

    # (i) fixed elicitation: exactly one system prompt across the cell.
    req_i = len(md5s) == 1
    # (ii) the grade must not BE the condition label. On this corpus `deceptive`
    # is the release's own annotation; if it is a deterministic function of the
    # system prompt, it is a condition label and (ii) fails.
    by_md5 = {}
    for r in recs:
        by_md5.setdefault(r["system_md5"], set()).add(r["deceptive"])
    req_ii = not (len(md5s) > 1 and all(len(v) == 1 for v in by_md5.values()))

    channels = {}
    for ch in CHANNEL_ORDER:
        per_list = {}
        for lst in (LIST_NAME, SECONDARY_LIST):
            per_list[lst] = contrast(dec, hon, ch, lst, rng)
        channels[ch] = per_list

    prim = channels["visible"][LIST_NAME]
    v_eff = prim.get("diff_pp")
    f_eff = channels["full"][LIST_NAME].get("diff_pp")
    fo_eff = channels["full_output_only"][LIST_NAME].get("diff_pp")

    leak = {"visible_pp": v_eff, "full_pp": f_eff, "full_output_only_pp": fo_eff,
            "threshold": LEAK_FACTOR}
    if v_eff is not None and f_eff is not None:
        leak["verdict"] = ("ANNOTATION_CHANNEL_LEAKAGE"
                           if abs(f_eff) > LEAK_FACTOR * max(abs(v_eff), 1e-9)
                           else "NO_CHANNEL_LEAKAGE")
        if fo_eff is not None:
            # post-hoc: `full` here includes the system prompt, which Apollo's
            # `full_transcript` never does, so the gap is decomposed rather than
            # equated across the two releases.
            leak["scratchpad_part_pp"] = abs(fo_eff) - abs(v_eff)
            leak["system_prompt_part_pp"] = abs(f_eff) - abs(fo_eff)
            leak["decomposition_is_post_hoc"] = True
    else:
        leak["verdict"] = "UNDERPOWERED"

    # PREREG §7: the criterion-4 verdict, on the primary channel and list only.
    if prim.get("status") != "OK":
        verdict = prim.get("status")
    elif prim["p"] is not None and prim["p"] < 0.05 and prim["diff_pp"] > 0:
        verdict = "CRITERION4_POSITIVE"
    else:
        verdict = "CRITERION4_NULL"

    eligible = req_i and req_ii and len(dec) >= MIN_CELL and len(hon) >= MIN_CELL
    return {
        "n": len(recs), "n_deceptive": len(dec), "n_honest": len(hon),
        "n_distinct_system_md5": len(md5s),
        "requirements": {"i_fixed_elicitation": req_i,
                         "ii_grade_not_condition_label": req_ii,
                         "iii_paired_scenarios": iii["met"],
                         "iii_detail": iii},
        "eligible_for_criterion4": eligible,
        "channels": channels,
        "channel_contrast": leak,
        "verdict": verdict,
        # PREREG §7 branch A. (iv) holds by construction on the visible channel;
        # (v) is the one requirement this corpus cannot settle mechanically, since
        # `deceptive` is the release's own annotation rather than an
        # action-grounded label -- so a positive here is reported as
        # FALSIFIER_CANDIDATE and adjudicated by hand, never auto-promoted.
        "falsifier": ("FALSIFIER_CANDIDATE"
                      if eligible and iii["met"] and verdict == "CRITERION4_POSITIVE"
                      else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="dump the full record")
    args = ap.parse_args()

    by_config = load_records()
    if not by_config:
        print("No derived records in", RESULTS)
        print("PREREG_EXP_XL.md §7 branch F: the corpus is gated and we did not")
        print("obtain row-level access, so nothing was scored. Run")
        print("fetch_liars_bench.py then run_liars_bench_audit.py --survey.")
        if os.path.exists(SURVEY_PATH):
            print(f"(a survey exists at {SURVEY_PATH})")
        return 1

    rng = np.random.RandomState(SEED)
    print("=" * 100)
    print("  EXP-XL - criterion 4 on a second public release (Liars' Bench)")
    print(f"  PREREG docs/PREREG_EXP_XL.md | N_PERM={N_PERM} SEED={SEED} "
          f"MIN_CELL={MIN_CELL} gate={GATE_LO}-{GATE_HI}")
    print(f"  primary channel=visible, primary list={LIST_NAME}, "
          f"pre-registered channels={list(PREREG_CHANNELS)}")
    print("=" * 100)

    out = {"prereg": "docs/PREREG_EXP_XL.md", "n_perm": N_PERM, "seed": SEED,
           "min_cell": MIN_CELL, "primary_channel": "visible",
           "primary_list": LIST_NAME, "configs": {}}
    firing, candidates = [], []

    for config in sorted(by_config):
        recs = by_config[config]
        by_gen = {}
        for r in recs:
            by_gen.setdefault(r["generator"] or "unknown", []).append(r)
        out["configs"][config] = {}
        print(f"\n{config}  ({len(recs)} examples, {len(by_gen)} generator(s))")
        for gen in sorted(by_gen):
            c = analyze_cell(by_gen[gen], rng)
            out["configs"][config][gen] = c
            req = c["requirements"]
            flags = ("i" if req["i_fixed_elicitation"] else "-") + \
                    (" ii" if req["ii_grade_not_condition_label"] else " --") + \
                    (" iii" if req["iii_paired_scenarios"] else " ---")
            print(f"  {gen[:28]:28s} n={c['n']:6d} "
                  f"dec/hon={c['n_deceptive']}/{c['n_honest']} req[{flags}] "
                  f"paired={req['iii_detail']['n_paired']}")
            for ch in CHANNEL_ORDER:
                k = c["channels"][ch][LIST_NAME]
                tag = "" if ch in PREREG_CHANNELS else "  (post-hoc)"
                d = ("    --" if k.get("diff_pp") is None
                     else f"{k['diff_pp']:+6.1f} pp")
                p = "  --  " if k.get("p") is None else f"p={k['p']:.4f}"
                print(f"      {ch:17s} dec {fmt_cell(k['deceptive'])} "
                      f"hon {fmt_cell(k['honest'])} {d} {p} "
                      f"{k.get('status')}{tag}")
            print(f"      -> {c['verdict']}   channel: "
                  f"{c['channel_contrast']['verdict']}")
            if c["channel_contrast"].get("decomposition_is_post_hoc"):
                cc = c["channel_contrast"]
                print(f"         gap decomposition (post-hoc): scratchpad "
                      f"{cc['scratchpad_part_pp']:+.1f} pp, system prompt "
                      f"{cc['system_prompt_part_pp']:+.1f} pp")
            if c["verdict"] == "CRITERION4_POSITIVE":
                firing.append((config, gen))
            if c["falsifier"]:
                candidates.append((config, gen))

    out["criterion4_positive_cells"] = [{"config": c, "generator": g}
                                        for c, g in firing]
    out["falsifier_candidates"] = [{"config": c, "generator": g}
                                   for c, g in candidates]

    print("\n" + "=" * 100)
    if candidates:
        out["branch"] = "A_falsifier_candidate"
        print("  FALSIFIER_CANDIDATE - PREREG_EXP_XL.md §7 branch A")
        for c, g in candidates:
            print(f"    {c} / {g}")
        print("  A cell satisfies (i)-(iv) with paired scenarios AND separates")
        print("  deception at fixed elicitation. Requirement (v) is NOT settled")
        print("  mechanically here: `deceptive` is the release's own annotation,")
        print("  not an action-grounded label. STOP, adjudicate (v) by hand, and")
        print("  report before editing the paper. If (v) holds, claim 3 is")
        print("  RETRACTED, not softened.")
    elif firing:
        out["branch"] = "C_positive_but_requirements_unmet"
        print("  CRITERION4_POSITIVE on cells that do NOT satisfy (i)-(iii).")
        print("  Separation without the design is not attribution: reported as")
        print("  such, and claim 3 stands.")
        for c, g in firing:
            print(f"    {c} / {g}")
    else:
        out["branch"] = "D_null_on_second_release"
        print("  CRITERION4_NULL on every cell of the second release.")
        print("  Two independent public releases, the same structural gap. Note")
        print("  the standing caveat: this rule was written for a first-person")
        print("  claim-assertion format and may be near-blind here, so a null is")
        print("  a null of a possibly-blind instrument (PREREG §3).")
    print("=" * 100)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAnalysis -> {OUT_PATH}")
    if args.json:
        print(json.dumps(out, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
