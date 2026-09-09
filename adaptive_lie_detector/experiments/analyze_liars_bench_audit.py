#!/usr/bin/env python3
"""analyze_liars_bench_audit.py

EXP-XL analysis (PREREG_EXP_XL.md §4, §7): recomputes every reported number from
the committed derived records in `data/results/liars_bench_rule_*.json.gz`, with
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
import gzip
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
    LEAK_FACTOR, LIST_NAME, PREREG_CHANNELS, PRIMARY_SCENARIO_SOURCE,
    SECONDARY_LIST, SURVEY_PATH,
)

CHANNEL_ORDER = ("visible", "full", "full_output_only")
OUT_PATH = os.path.join(RESULTS, "liars_bench_analysis.json")

# PREREG §4 H4 fixes `MIN_CELL = 10` per arm per scenario; the function it names,
# `analyze_insider_audit.perm_p_stratified`, ships alongside that module's own
# `MIN_CELL = 5`. The pre-registration's number governs, and the other is reported
# beside it rather than quietly dropped -- the ambiguity is in the document we
# wrote, so both readings are published.
H4_MIN_CELL_PREREG = 10
H4_MIN_CELL_ALT = 5

# PREREG §5.2: `insider-trading` passes requirement (v) only if `meta` carries a
# trade/action field DISJOINT from the report our rule reads. These are the field
# names that would qualify; absent all of them the config is CONSTRUCT_RECOVERY_RISK.
ACTION_FIELDS = ("trade", "traded", "action", "made_trade", "trade_executed",
                 "amount", "shares", "ticker", "decision", "bought", "sold",
                 "insider_trade", "did_trade")


def load_records():
    paths = sorted(glob.glob(os.path.join(RESULTS, "liars_bench_rule_*.json.gz")) +
                   glob.glob(os.path.join(RESULTS, "liars_bench_rule_*.json")))
    by_config = {}
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("contains_corpus_text"):
            raise SystemExit(f"{p} claims to contain corpus text; refusing to use it")
        by_config[d["config"]] = d["records"]
    return by_config


def requirement_iii(recs):
    """(iii): the same scenario independently realized under BOTH behavioral
    outcomes. Counted, not assumed -- this is the requirement no Apollo set meets.

    The verdict is taken from the primary key (`env_md5`: the scenario as the
    environment posed it), but the paired count under *every* published candidate
    key is reported alongside, so the choice of key is auditable and cannot be one
    that suits the answer. `index`, which PREREG §5.3 names, is among them and is
    a row counter on this corpus -- visible here as n_paired == 0.
    """
    groups = {}
    for r in recs:
        if r["scenario_key"] is None:
            continue
        groups.setdefault(r["scenario_key"], set()).add(r["deceptive"])
    paired = [k for k, v in groups.items() if v == {True, False}]

    per_source = {}
    for r in recs:
        for src, k in (r.get("scenario_keys") or {}).items():
            per_source.setdefault(src, {}).setdefault(k, set()).add(r["deceptive"])
    by_source = {src: {"n_groups": len(g),
                       "n_paired": sum(1 for v in g.values() if v == {True, False}),
                       "is_primary": src == PRIMARY_SCENARIO_SOURCE}
                 for src, g in sorted(per_source.items())}

    return {"n_groups": len(groups), "n_paired": len(paired),
            "met": len(paired) > 0,
            "key_source": sorted({r["scenario_key_source"] for r in recs}),
            "primary_source": PRIMARY_SCENARIO_SOURCE,
            "by_scenario_source": by_source,
            # a corroboration, not the verdict: does ANY published id key also
            # pair? if the primary pairs and no published key does, say so.
            "any_published_key_pairs": any(
                v["n_paired"] > 0 for s, v in by_source.items()
                if s.startswith(("meta.", "top.")))}


def requirement_v(config, recs):
    """PREREG §5.2's (v) adjudication, run mechanically on the fetched rows.

    (v) asks that D be grounded in evidence NOT definitionally equivalent to the
    detector's input. On this corpus `deceptive` is the release's own annotation.
    For `insider-trading` §5.2 fixed the test in advance: an action field in `meta`
    disjoint from the report passes; its absence is `CONSTRUCT_RECOVERY_RISK`,
    because the label grades the same report text the rule reads -- the failure
    mode EXP-IT already documented on Apollo's insider trading.
    """
    seen = set()
    for r in recs:
        seen.update((r.get("meta_scalars") or {}).keys())
    found = sorted(f for f in ACTION_FIELDS if f in seen)
    if found:
        return {"met": True, "basis": "action_field_present",
                "action_fields": found, "meta_fields": sorted(seen)}
    return {"met": False, "basis": "no_action_field_disjoint_from_the_report",
            "verdict": "CONSTRUCT_RECOVERY_RISK",
            "action_fields": [], "meta_fields": sorted(seen),
            "note": ("`deceptive` grades the model's report and the rule reads that "
                     "same report, so a positive is construct recovery, not "
                     "detection (PREREG §5.2).")}


def h4_stratified(recs, channel, lst, min_cell, seed=SEED):
    """PREREG §4 H4 -- **the arm that decides whether the falsifier has fired**.

    The pre-registered function is `analyze_insider_audit.perm_p_stratified`, whose
    statistic is the POOLED fire-rate difference with labels permuted WITHIN
    scenario. One subtlety is reported rather than left implicit: because scenarios
    differ both in base rate and in deceptive:honest ratio, the within-scenario
    null of a pooled statistic is **not centred at zero**. `null_mean_pp` is
    therefore published next to the observed value, and `excess_over_null_pp` --
    observed minus that null mean -- is the part of the gap that scenario
    composition does not already explain.
    """
    import collections

    from experiments.analyze_insider_audit import perm_p_stratified

    by = collections.defaultdict(lambda: {"d": [], "h": []})
    for r in recs:
        if r["scenario_key"] is None:
            continue
        arm = "d" if r["deceptive"] else "h"
        by[r["scenario_key"]][arm].append(1 if fired(r, channel, lst) else 0)
    kept, dropped = [], []
    for s, c in sorted(by.items()):
        if len(c["d"]) >= min_cell and len(c["h"]) >= min_cell:
            kept.append((c["d"], c["h"]))
        else:
            dropped.append({"scenario": s, "n_dec": len(c["d"]), "n_hon": len(c["h"])})
    out = {"min_cell_per_arm": min_cell, "n_scenarios_kept": len(kept),
           "n_scenarios_dropped": len(dropped),
           "n_deceptive": sum(len(d) for d, _ in kept),
           "n_honest": sum(len(h) for _, h in kept)}
    if not kept:
        out["verdict"] = "INELIGIBLE_NO_USABLE_SCENARIO"
        return out

    rng = np.random.RandomState(seed)
    p, obs = perm_p_stratified(kept, rng, one_sided_lower=False)  # §4: higher

    # the null's own location, so a "significant" pooled gap cannot be read as an
    # attributable one without the reader seeing how much composition supplies
    rng2 = np.random.RandomState(seed)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = []
        for d, h in kept:
            pool = list(d) + list(h)
            rng2.shuffle(pool)
            perm.append((pool[:len(d)], pool[len(d):]))
        dk = sum(sum(x) for x, _ in perm); dn = sum(len(x) for x, _ in perm)
        hk = sum(sum(y) for _, y in perm); hn = sum(len(y) for _, y in perm)
        null[i] = dk / dn - hk / hn

    out.update({
        "diff_pp": 100.0 * obs, "p": p,
        "null_mean_pp": 100.0 * float(null.mean()),
        "null_sd_pp": 100.0 * float(null.std()),
        "excess_over_null_pp": 100.0 * (obs - float(null.mean())),
        # each scenario weighted equally instead of by size: the estimator that
        # does not inherit the composition offset
        "unweighted_within_scenario_mean_pp": 100.0 * float(np.mean(
            [np.mean(d) - np.mean(h) for d, h in kept])),
        "verdict": ("CRITERION4_POSITIVE" if p < 0.05 and obs > 0
                    else "CRITERION4_NULL"),
    })
    return out


def nuisance_decomposition(recs, channel, lst):
    """§5.2's other half: a scalar `meta` field that predicts BOTH the label and the
    fire rate is a nuisance variable, and a gap it explains is not attributable.
    Reports the effect within each level of every such field."""
    import collections
    fields = collections.Counter()
    for r in recs:
        fields.update((r.get("meta_scalars") or {}).keys())
    out = {}
    for f in sorted(fields):
        levels = collections.defaultdict(lambda: {"d": [0, 0], "h": [0, 0]})
        for r in recs:
            ms = r.get("meta_scalars") or {}
            if f not in ms:
                continue
            cell = levels[str(ms[f])]["d" if r["deceptive"] else "h"]
            cell[1] += 1
            cell[0] += 1 if fired(r, channel, lst) else 0
        if len(levels) < 2 or len(levels) > 12:
            continue
        rep = {}
        for lv, c in sorted(levels.items()):
            d, h = c["d"], c["h"]
            rep[lv] = {
                "n_dec": d[1], "n_hon": h[1],
                "fire_dec_pct": 100.0 * d[0] / d[1] if d[1] else None,
                "fire_hon_pct": 100.0 * h[0] / h[1] if h[1] else None,
                "diff_pp": (100.0 * (d[0] / d[1] - h[0] / h[1])
                            if d[1] and h[1] else None),
                "pct_of_level_deceptive": 100.0 * d[1] / max(1, d[1] + h[1]),
            }
        diffs = [v["diff_pp"] for v in rep.values() if v["diff_pp"] is not None]
        out[f] = {"levels": rep,
                  "within_level_mean_diff_pp": (float(np.mean(diffs))
                                                if diffs else None),
                  "label_varies_by_level": len({
                      round(v["pct_of_level_deceptive"], 1)
                      for v in rep.values()}) > 1}
    return out


def analyze_cell(recs, rng, config=""):
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

    # PREREG §4 H4: the deciding arm, at the pre-registered MIN_CELL and at the
    # named function's own, on the primary channel and list.
    h4 = {str(mc): h4_stratified(recs, "visible", LIST_NAME, mc)
          for mc in (H4_MIN_CELL_PREREG, H4_MIN_CELL_ALT)}
    h4_full = h4_stratified(recs, "full", LIST_NAME, H4_MIN_CELL_PREREG)
    h4_prereg = h4[str(H4_MIN_CELL_PREREG)]
    req_v = requirement_v(config, recs)
    nuisance = nuisance_decomposition(recs, "visible", LIST_NAME)

    return {
        "n": len(recs), "n_deceptive": len(dec), "n_honest": len(hon),
        "n_distinct_system_md5": len(md5s),
        "requirements": {"i_fixed_elicitation": req_i,
                         "ii_grade_not_condition_label": req_ii,
                         "iii_paired_scenarios": iii["met"],
                         "iii_detail": iii,
                         "v_annotation_independent": req_v["met"],
                         "v_detail": req_v},
        "eligible_for_criterion4": eligible,
        "channels": channels,
        "channel_contrast": leak,
        "h4_stratified": h4,
        "h4_stratified_full_channel": h4_full,
        "nuisance_decomposition": nuisance,
        "verdict": verdict,
        # PREREG §7 branch A: the falsifier fires only on a cell satisfying
        # (i)-(v) **including H4's paired-scenario condition** whose instrument
        # separates `deceptive` on the visible channel. All four gates are applied
        # here; (iv) holds by construction on that channel.
        #
        # The pooled H1 above is NOT the deciding test -- §4 makes H4 "the arm that
        # decides whether the paper's falsifier has fired" -- so a cell can be
        # H1-positive and still not fire, which is recorded rather than smoothed.
        "falsifier": ("FALSIFIER_FIRED"
                      if (eligible and iii["met"] and req_v["met"]
                          and h4_prereg.get("verdict") == "CRITERION4_POSITIVE")
                      else None),
        "falsifier_blocked_by": [
            k for k, blocked in (
                ("(i)_or_(ii)_or_power", not eligible),
                ("(iii)_no_paired_scenario", not iii["met"]),
                ("(v)_" + str(req_v.get("verdict", "unmet")), not req_v["met"]),
                ("H4_stratified_" + str(h4_prereg.get("verdict")),
                 h4_prereg.get("verdict") != "CRITERION4_POSITIVE"),
            ) if blocked],
        "h1_pooled_positive_but_not_deciding": verdict == "CRITERION4_POSITIVE",
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
    firing, candidates, blocked = [], [], []

    for config in sorted(by_config):
        recs = by_config[config]
        by_gen = {}
        for r in recs:
            by_gen.setdefault(r["generator"] or "unknown", []).append(r)
        out["configs"][config] = {}
        print(f"\n{config}  ({len(recs)} examples, {len(by_gen)} generator(s))")
        for gen in sorted(by_gen):
            c = analyze_cell(by_gen[gen], rng, config=config)
            out["configs"][config][gen] = c
            req = c["requirements"]
            flags = ("i" if req["i_fixed_elicitation"] else "-") + \
                    (" ii" if req["ii_grade_not_condition_label"] else " --") + \
                    (" iii" if req["iii_paired_scenarios"] else " ---")
            print(f"  {gen[:28]:28s} n={c['n']:6d} "
                  f"dec/hon={c['n_deceptive']}/{c['n_honest']} req[{flags}] "
                  f"paired={req['iii_detail']['n_paired']}")
            for src, s in req["iii_detail"]["by_scenario_source"].items():
                star = " *" if s["is_primary"] else "  "
                print(f"       {star}{src:18s} groups={s['n_groups']:6d} "
                      f"paired={s['n_paired']:6d}")
            for ch in CHANNEL_ORDER:
                k = c["channels"][ch][LIST_NAME]
                tag = "" if ch in PREREG_CHANNELS else "  (post-hoc)"
                d = ("    --" if k.get("diff_pp") is None
                     else f"{k['diff_pp']:+6.1f} pp")
                p = "  --  " if k.get("p") is None else f"p={k['p']:.4f}"
                print(f"      {ch:17s} dec {fmt_cell(k['deceptive'])} "
                      f"hon {fmt_cell(k['honest'])} {d} {p} "
                      f"{k.get('status')}{tag}")
            print(f"      -> H1 pooled: {c['verdict']}   channel: "
                  f"{c['channel_contrast']['verdict']}")
            # H4 -- the deciding arm (PREREG §4)
            for mc, h in c["h4_stratified"].items():
                if h.get("verdict") == "INELIGIBLE_NO_USABLE_SCENARIO":
                    print(f"      H4 (min_cell={mc:>2s}) no scenario with "
                          f"{mc} in both arms -> {h['verdict']}")
                    continue
                tag = " [PREREG §4]" if int(mc) == H4_MIN_CELL_PREREG else ""
                print(f"      H4 (min_cell={mc:>2s}) strata={h['n_scenarios_kept']:3d} "
                      f"n={h['n_deceptive']}/{h['n_honest']} "
                      f"obs={h['diff_pp']:+6.1f} pp p={h['p']:.4f} -> "
                      f"{h['verdict']}{tag}")
                print(f"         within-scenario null is centred at "
                      f"{h['null_mean_pp']:+.1f} pp (sd {h['null_sd_pp']:.2f}); "
                      f"excess over it {h['excess_over_null_pp']:+.1f} pp; "
                      f"equal-weight within-scenario mean "
                      f"{h['unweighted_within_scenario_mean_pp']:+.1f} pp")
            v = c["requirements"]["v_detail"]
            print(f"      (v) {'MET' if v['met'] else 'UNMET'}: {v['basis']}"
                  + (f" -> {v['verdict']}" if not v["met"] else ""))
            for f_, nd in c["nuisance_decomposition"].items():
                # a field whose levels do not each contain both arms cannot show a
                # within-level effect, so there is nothing to decompose
                if (not nd["label_varies_by_level"]
                        or nd["within_level_mean_diff_pp"] is None):
                    continue
                lv = ", ".join(
                    f"{k}: {d['fire_dec_pct']:.1f}/{d['fire_hon_pct']:.1f}% "
                    f"({d['pct_of_level_deceptive']:.0f}% dec)"
                    for k, d in nd["levels"].items()
                    if d["fire_dec_pct"] is not None and d["fire_hon_pct"] is not None)
                print(f"      nuisance `{f_}` predicts label AND fire rate -> {lv}")
                print(f"         within-level mean diff "
                      f"{nd['within_level_mean_diff_pp']:+.1f} pp")
            if c["falsifier_blocked_by"]:
                print(f"      falsifier blocked by: "
                      f"{', '.join(c['falsifier_blocked_by'])}")
            if c["channel_contrast"].get("decomposition_is_post_hoc"):
                cc = c["channel_contrast"]
                print(f"         gap decomposition (post-hoc): scratchpad "
                      f"{cc['scratchpad_part_pp']:+.1f} pp, system prompt "
                      f"{cc['system_prompt_part_pp']:+.1f} pp")
            if c["verdict"] == "CRITERION4_POSITIVE":
                firing.append((config, gen))
            if c["falsifier"]:
                candidates.append((config, gen))
            if (c["verdict"] == "CRITERION4_POSITIVE" or
                    c["h4_stratified"][str(H4_MIN_CELL_PREREG)].get("verdict")
                    == "CRITERION4_POSITIVE") and not c["falsifier"]:
                blocked.append((config, gen, c["falsifier_blocked_by"]))

    out["criterion4_positive_cells"] = [{"config": c, "generator": g}
                                        for c, g in firing]
    out["falsifier_candidates"] = [{"config": c, "generator": g}
                                   for c, g in candidates]

    out["falsifier_blocked_cells"] = [
        {"config": c, "generator": g, "blocked_by": b} for c, g, b in blocked]

    print("\n" + "=" * 100)
    if candidates:
        out["branch"] = "A_falsifier_fired"
        print("  FALSIFIER_FIRED - PREREG_EXP_XL.md §7 branch A")
        for c, g in candidates:
            print(f"    {c} / {g}")
        print("  A cell satisfies (i)-(v) INCLUDING H4's paired-scenario condition")
        print("  and separates deception at fixed elicitation on the visible")
        print("  channel. The paper's stated refutation condition is MET.")
        print("  STOP and report before editing the paper: claim 3 is RETRACTED,")
        print("  not softened, and §7 branch A lists what changes together.")
    elif blocked:
        out["branch"] = "C_positive_but_requirements_unmet"
        print("  CRITERION4_POSITIVE somewhere, but the falsifier did NOT fire.")
        print("  Separation without the design is not attribution, so claim 3")
        print("  stands. Each cell and the gate that blocked it:")
        for c, g, b in blocked:
            print(f"    {c} / {g}")
            for gate in b:
                print(f"        blocked by {gate}")
        print("  Read the H4 lines above before quoting any pooled effect: where")
        print("  the within-scenario null is centred well above zero, most of the")
        print("  pooled gap is scenario composition, not deception.")
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
