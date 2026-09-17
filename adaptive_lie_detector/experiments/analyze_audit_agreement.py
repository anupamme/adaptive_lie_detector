#!/usr/bin/env python3
"""
analyze_audit_agreement.py

EXP-AA step 3. Three outcomes, all pre-registered in docs/PREREG_EXP_AA.md §7-§9
and all reported whether or not they favour the paper:

  H-AA1  Krippendorff's alpha per requirement over the three raters (18 units).
         alpha >= 0.60 -> decidable from release-authored text.
  H-AA2  Panel majority vs. the recorded referents of §8 / §8a (42 judgements).
  H-AA3  The UNDECIDABLE rate per requirement, which is a primary outcome and
         not a nuisance: it measures whether the release publishes enough.

The estimator is krippendorff_nominal, imported from analyze_crit4_reliability --
the same one EXP-C4 and EXP-C4B use, so the coefficients are comparable.

    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/analyze_audit_agreement.py
"""

import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_crit4_reliability import krippendorff_nominal  # noqa: E402

RESULTS = "data/results"
PACKETS = os.path.join(RESULTS, "audit_packets.jsonl")
RATINGS = os.path.join(RESULTS, "audit_ratings.jsonl")
CHANNEL = os.path.join(RESULTS, "elicitation_channel_check.json")
LB = os.path.join(RESULTS, "liars_bench_analysis.json")
OUT = os.path.join(RESULTS, "audit_agreement.json")

# PREREG §7. The same threshold analyze_crit4_reliability.py already uses.
PREREG_THRESHOLD = 0.60
PARTLY_THRESHOLD = 0.40
UNDECIDABLE_CEILING = 1.0 / 3.0

REQS = ("i", "ii", "iii", "iv", "v")

# PREREG §8. Which recorded boolean each requirement is scored against, and the
# artifact it is read from. (iv) has no referent anywhere and gets alpha only.
LB_REFERENT = {
    "i": "i_fixed_elicitation",
    "ii": "ii_grade_not_condition_label",
    "iii": "iii_paired_scenarios",
    "v": "v_annotation_independent",
}


def load_ratings():
    if not os.path.exists(RATINGS):
        raise SystemExit(f"{RATINGS} not found -- run rate_audit_criteria.py first")
    rows = [json.loads(l) for l in open(RATINGS) if l.strip()]
    # Keep the last error-free record per (case, rater): --resume appends.
    keep = {}
    for r in rows:
        if r.get("error") is None and r.get("ratings"):
            keep[(r["case"], r["rater"])] = r
    return list(keep.values())


def load_referents():
    """The recorded booleans, mapped to the panel's vocabulary."""
    ref = {}

    # liars_bench: four requirements x eight configurations.
    lb = json.load(open(LB))["configs"]
    for cfg, gens in lb.items():
        # The requirement booleans are properties of the design, so they are
        # identical across generator models; assert that rather than assume it.
        for req, field in LB_REFERENT.items():
            vals = {g["requirements"][field] for g in gens.values()
                    if isinstance(g, dict) and "requirements" in g}
            if len(vals) != 1:
                raise SystemExit(
                    f"{cfg}/{field}: {len(vals)} distinct recorded values across "
                    "generators, but PREREG §3 treats these as design properties")
            ref[(cfg, req)] = ("SATISFIED" if vals.pop() else "NOT_SATISFIED")

    # Apollo, requirement (i): the corrected field (PREREG §8a), with the
    # published one kept alongside so the disagreement is visible.
    ch = json.load(open(CHANNEL))
    published = {}
    for s in ch["sets"]:
        ref[(s["set"], "i")] = ("SATISFIED" if s["holds_one_elicitation_fixed"]
                                else "NOT_SATISFIED")
        published[s["set"]] = ("SATISFIED" if s["recorded_elicitation_fixed"]
                               else "NOT_SATISFIED")
    return ref, published


def granularity_caveat():
    """A disclosed defect of OUR packet construction, not of any rater.

    The recorded (i) boolean for liars_bench is computed per generator CELL
    (`i_basis: single_system_prompt_md5`, one distinct md5 within a cell). A packet
    is built at CONFIGURATION granularity, per PREREG §3, so it shows the count
    pooled over generators -- which for some configurations is greater than one,
    because different generator models were run under slightly different system
    prompts. A rater reading "distinct system-message md5s: 3" can reasonably
    answer NOT_SATISFIED while the recorded per-cell boolean is True.

    Nothing here is adjudicated: the affected cases are listed so that an (i)
    disagreement on them is attributable, and the concordance rate is reported
    both with and without them.
    """
    if not os.path.exists(PACKETS):
        return {}
    lb = json.load(open(LB))["configs"]
    out = {}
    for l in open(PACKETS):
        p = json.loads(l)
        if p["release"] != "liars_bench":
            continue
        m = re.search(r"distinct system-message md5s: (\d+)", p["text"])
        if not m:
            continue
        shown = int(m.group(1))
        per_cell = max(g["n_distinct_system_md5"] for g in lb[p["case"]].values())
        if shown != per_cell:
            out[p["case"]] = {"n_shown_in_packet_pooled_over_generators": shown,
                              "max_n_distinct_per_generator_cell": per_cell}
    return out


def alpha_over(rows, req, cases=None):
    """One coefficient: units are cases, coder labels are the three raters."""
    by_case = {}
    for r in rows:
        if cases is not None and r["case"] not in cases:
            continue
        by_case.setdefault(r["case"], []).append(r["ratings"][req]["answer"])
    units = [u for u in by_case.values() if len(u) >= 2]
    if not units:
        return None, 0
    if len({v for u in units for v in u}) < 2:
        # No variation at all: alpha is undefined (0/0), and reporting it as 1.0
        # would be a claim the data does not make.
        return "undefined (unanimous across all units)", len(units)
    return krippendorff_nominal(units), len(units)


def raw_agreement(rows, req):
    """Percent pairwise agreement and unanimity, reported ALONGSIDE alpha.

    Not a substitute for it and not a second hypothesis test: alpha corrects for
    chance and can sit low while raw agreement is high whenever one answer
    dominates the margin. Both numbers are needed to say what actually happened,
    so both are printed for every requirement whichever way they fall.
    """
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"], []).append(r["ratings"][req]["answer"])
    pairs = agree = unanimous = 0
    for u in by_case.values():
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                pairs += 1
                agree += int(u[a] == u[b])
        unanimous += int(len(set(u)) == 1)
    return (agree / pairs if pairs else None,
            unanimous / len(by_case) if by_case else None)


def majority(answers):
    c = Counter(answers)
    top, n = c.most_common(1)[0]
    ties = [a for a, k in c.items() if k == n]
    return (top if len(ties) == 1 else None), n, len(answers)


def main():
    rows = load_ratings()
    ref, published = load_referents()
    cases = sorted({r["case"] for r in rows})
    raters = sorted({r["rater_family"] for r in rows})

    print("=" * 96)
    print("EXP-AA step 3 — agreement on requirements (i)-(v)  (PREREG §7-§9)")
    print("=" * 96)
    print(f"  cases {len(cases)}   raters {len(raters)} {raters}   "
          f"ratings {len(rows)} of {len(cases) * len(raters)}")
    if len(rows) != len(cases) * len(raters):
        print("  INCOMPLETE PANEL — PREREG §5 requires all three raters on every case.")

    apollo = sorted({r["case"] for r in rows if r["release"] == "apollo"})
    lbcase = sorted({r["case"] for r in rows if r["release"] == "liars_bench"})

    # ---------------------------------------------------- H-AA1 and H-AA3
    print("\nH-AA1  Krippendorff's alpha per requirement (nominal, 4 values, 3 raters)")
    print("H-AA3  UNDECIDABLE rate per requirement")
    print(f"  {'req':4s} {'alpha':>8s} {'pair-agr':>9s} {'unanim':>7s} {'units':>6s} "
          f"{'UNDEC':>7s} {'UNPARSE':>8s}  verdict")
    per_req = {}
    for req in REQS:
        a, n = alpha_over(rows, req)
        pa, un = raw_agreement(rows, req)
        ans = [r["ratings"][req]["answer"] for r in rows]
        und = sum(1 for x in ans if x == "UNDECIDABLE") / len(ans)
        unp = sum(1 for x in ans if x == "UNPARSEABLE") / len(ans)
        if isinstance(a, str):
            verdict = "alpha undefined; see the distribution"
        elif a >= PREREG_THRESHOLD:
            verdict = "decidable from release text"
        elif a >= PARTLY_THRESHOLD:
            verdict = "PARTLY decidable — verdict carries rater judgment"
        else:
            verdict = "NOT decidable from the release alone — §9 narrows claim (2)"
        shown = "n/a" if isinstance(a, str) else f"{a:.3f}"
        print(f"  ({req:3s} {shown:>8s} {pa:9.1%} {un:7.1%} {n:6d} "
              f"{und:7.1%} {unp:8.1%}  {verdict}")
        per_req[req] = {
            "alpha": None if isinstance(a, str) else round(a, 4),
            "alpha_note": a if isinstance(a, str) else None,
            "pairwise_agreement": round(pa, 4),
            "unanimity_rate": round(un, 4),
            "n_units": n,
            "undecidable_rate": round(und, 4),
            "unparseable_rate": round(unp, 4),
            "answer_counts": dict(Counter(ans)),
            "verdict": verdict,
            "meets_H_AA1": (not isinstance(a, str)) and a >= PREREG_THRESHOLD,
            "meets_H_AA3": und < UNDECIDABLE_CEILING,
        }

    pooled_units = []
    for req in REQS:
        by_case = {}
        for r in rows:
            by_case.setdefault((r["case"], req), []).append(r["ratings"][req]["answer"])
        pooled_units += [u for u in by_case.values() if len(u) >= 2]
    pooled = krippendorff_nominal(pooled_units)
    print(f"  pooled over all five requirements: alpha = {pooled:.3f} "
          f"({len(pooled_units)} units) — context only; PREREG §9 says "
          f"per-requirement coefficients govern")

    # ------------------------------------- release-stratified (PREREG §4a)
    print("\nExploratory, PREREG §4a: stratified by release "
          "(Apollo packets carry README prose; liars_bench packets have no dataset card)")
    strat = {}
    for name, subset in (("apollo", apollo), ("liars_bench", lbcase)):
        sub = [r for r in rows if r["case"] in subset]
        line = []
        strat[name] = {}
        for req in REQS:
            a, n = alpha_over(rows, req, cases=set(subset))
            ans = [r["ratings"][req]["answer"] for r in sub]
            und = sum(1 for x in ans if x == "UNDECIDABLE") / len(ans) if ans else 0
            line.append(f"({req}) a={'n/a' if isinstance(a, str) else f'{a:.2f}'} u={und:.0%}")
            strat[name][req] = {"alpha": None if isinstance(a, str) else round(a, 4),
                                "alpha_note": a if isinstance(a, str) else None,
                                "n_units": n, "undecidable_rate": round(und, 4)}
        allans = [r["ratings"][q]["answer"] for r in sub for q in REQS]
        u_all = sum(1 for x in allans if x == "UNDECIDABLE") / len(allans) if allans else 0
        strat[name]["undecidable_rate_all_requirements"] = round(u_all, 4)
        print(f"  {name:12s} n={len(subset):2d}  " + "  ".join(line))
        print(f"  {'':12s} UNDECIDABLE over all five requirements: {u_all:.1%}")

    # ------------------------------------------------------------ H-AA2
    print("\nH-AA2  panel majority vs. the recorded referent (PREREG §8, §8a)")
    conc, disagreements = {}, []
    for req in REQS:
        hits = tot = 0
        for case in cases:
            want = ref.get((case, req))
            if want is None:
                continue
            got, k, m = majority([r["ratings"][req]["answer"]
                                  for r in rows if r["case"] == case])
            tot += 1
            if got == want:
                hits += 1
            else:
                disagreements.append({
                    "case": case, "requirement": req,
                    "recorded": want, "panel_majority": got,
                    "majority_size": f"{k}/{m}",
                    "rater_justifications": {
                        r["rater_family"]: r["ratings"][req]["justification"]
                        for r in rows if r["case"] == case},
                })
        conc[req] = {"concordant": hits, "n": tot,
                     "rate": round(hits / tot, 4) if tot else None}
        if tot:
            print(f"  ({req:3s} {hits:2d}/{tot:2d} = {hits / tot:5.1%}"
                  f"{'  (majority concordance: H-AA2 holds)' if hits * 2 > tot else '  (H-AA2 FAILS)'}")
        else:
            print(f"  ({req:3s}  no recorded referent exists — alpha only (PREREG §8)")
    total_c = sum(v["concordant"] for v in conc.values())
    total_n = sum(v["n"] for v in conc.values())
    print(f"  all requirements with a referent: {total_c}/{total_n} = "
          f"{total_c / total_n:.1%}")

    # The four vacuous cases: does the panel track the corrected value or the
    # published one? PREREG §8a calls this the sharpest test in the experiment.
    vac = json.load(open(CHANNEL))["vacuous_sets"]
    print(f"\n  The {len(vac)} sets whose published `elicitation_fixed` is vacuous "
          f"(PREREG §8a), requirement (i):")
    vac_rows = []
    for case in vac:
        got, k, m = majority([r["ratings"]["i"]["answer"]
                              for r in rows if r["case"] == case])
        row = {"case": case, "panel_majority": got, "majority_size": f"{k}/{m}",
               "corrected_referent": ref.get((case, "i")),
               "published_referent": published.get(case)}
        vac_rows.append(row)
        mark = ("tracks the CORRECTED value" if got == row["corrected_referent"]
                else "tracks the PUBLISHED value" if got == row["published_referent"]
                else "matches neither")
        print(f"    {case:48s} panel={str(got):14s} corrected="
              f"{row['corrected_referent']:14s} published={row['published_referent']:14s} {mark}")

    # A defect of our packet construction, disclosed and quantified.
    gran = granularity_caveat()
    if gran:
        print(f"\n  Packet-granularity caveat (ours, not the raters'): on "
              f"{len(gran)} liars_bench configuration(s) the packet shows a "
              "system-prompt count pooled over generators while the recorded (i) "
              "boolean is per generator cell:")
        for c, v in sorted(gran.items()):
            print(f"    {c:34s} packet showed {v['n_shown_in_packet_pooled_over_generators']}"
                  f", per-cell max is {v['max_n_distinct_per_generator_cell']}")
        excl = set(gran)
        hits = tot = 0
        for case in cases:
            if case in excl or ref.get((case, "i")) is None:
                continue
            got, _, _ = majority([r["ratings"]["i"]["answer"]
                                  for r in rows if r["case"] == case])
            tot += 1
            hits += int(got == ref[(case, "i")])
        print(f"    requirement (i) concordance excluding them: {hits}/{tot}"
              + (f" = {hits / tot:.1%}" if tot else ""))
        gran_conc = {"concordant": hits, "n": tot,
                     "rate": round(hits / tot, 4) if tot else None}
    else:
        gran_conc = None

    if disagreements:
        print(f"\n  {len(disagreements)} case x requirement disagreement(s); PREREG §9 "
              "reports each with the raters' own justifications and annotates the "
              "table cell. Our verdict is NOT changed on a rater panel's strength.")

    # ------------------------------------------------------------- §9 branch
    below = [r for r in REQS if per_req[r]["alpha"] is not None
             and per_req[r]["alpha"] < PARTLY_THRESHOLD]
    print("\nPREREG §9 branch")
    if below:
        print(f"  alpha < {PARTLY_THRESHOLD} on {below} -> claim (2) MUST be narrowed "
              "in introduction.tex and related_work.tex. This is a commitment, not "
              "an option.")
    else:
        print(f"  no requirement falls below {PARTLY_THRESHOLD}; claim (2) stands as "
              "written, qualified per requirement by the table above.")

    json.dump({
        "prereg": "docs/PREREG_EXP_AA.md",
        "threshold_decidable": PREREG_THRESHOLD,
        "threshold_partly": PARTLY_THRESHOLD,
        "n_cases": len(cases),
        "n_raters": len(raters),
        "rater_families": raters,
        "rater_models": sorted({r["rater"] for r in rows}),
        "n_ratings": len(rows),
        "per_requirement": per_req,
        "pooled_alpha": round(pooled, 4),
        "pooled_n_units": len(pooled_units),
        "stratified_by_release": strat,
        "concordance": conc,
        "concordance_total": {"concordant": total_c, "n": total_n,
                              "rate": round(total_c / total_n, 4) if total_n else None},
        "disagreements": disagreements,
        "packet_granularity_caveat": gran,
        "concordance_i_excluding_granularity_caveat": gran_conc,
        "vacuous_referent_cases": vac_rows,
        "requirements_below_partly_threshold": below,
        "limitation": ("The raters are LLMs, not independent human experts. What is "
                       "measured is whether requirements (i)-(v) are decidable from "
                       "release-authored text by raters blind to our verdicts; it is "
                       "not evidence that human domain experts would concur."),
    }, open(OUT, "w"), indent=1)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
