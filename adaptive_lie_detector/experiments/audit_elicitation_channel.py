#!/usr/bin/env python3
"""
audit_elicitation_channel.py

A re-measurement of our own Gate 1, NOT pre-registered. Found while assembling
the EXP-AA packets (docs/PREREG_EXP_AA.md) and reported as a post-hoc
diagnostic, in the same way the surface lexical rule is.

Gate 1 of the external audit (`run_external_audit.py:148`) hashes the
concatenated messages whose role is "system", and calls elicitation fixed when
exactly one distinct hash occurs. That is channel-specific: a set containing no
system message at all hashes the empty string on every record, so it records as
"fixed" vacuously -- and a set whose per-item elicitation lives in a *user*
message records as "fixed" while varying its prompt per item.

This script recomputes distinctness on four channels for each of the ten audited
sets and prints which recorded verdicts are substantive and which are vacuous.
It changes no published contrast: every set it reclassifies is already reported
inapplicable (cell size) or materials-only.

    cd code/adaptive_lie_detector
    python experiments/audit_elicitation_channel.py
"""

import glob
import hashlib
import json
import os
import sys

APOLLO = "data/external/apollo_deception"
RESULTS = "data/results"
OUT = os.path.join(RESULTS, "elicitation_channel_check.json")


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def join_roles(msgs, roles):
    return "\n".join(m.get("content", "") or "" for m in msgs
                     if isinstance(m, dict) and m.get("role") in roles)


def channels(rec):
    """The four candidate elicitation channels, as text, for one record."""
    msgs = rec.get("input_messages") or []
    if msgs:
        return {
            "system_only": join_roles(msgs, {"system"}),
            "user_only": join_roles(msgs, {"user"}),
            "system_plus_user": join_roles(msgs, {"system", "user"}),
            "full_input": json.dumps(
                [[m.get("role"), m.get("content")] for m in msgs
                 if isinstance(m, dict)], sort_keys=False),
        }
    # Materials files carry the elicitation in named fields, not messages.
    fields = {k: v for k, v in rec.items() if isinstance(v, str)}
    instr = "\n".join(v for k, v in sorted(fields.items()) if "instruction" in k)
    return {
        "system_only": "",
        "user_only": "",
        "system_plus_user": instr,
        "full_input": json.dumps(fields, sort_keys=True),
    }


def load(rel):
    d = json.load(open(os.path.join(APOLLO, "data", rel)))
    return d["rollouts"] if isinstance(d, dict) else d


def channel_used(recs):
    """Which channel this set actually carries its elicitation on, by a fixed rule.

    A set is credited with holding its elicitation fixed only on the channel it
    uses: the system messages if it has them, otherwise the user turns, otherwise
    (materials files, which have no messages at all) the fields whose name says
    they are instructions. Returns (channel name, n distinct values).
    """
    n = len(recs)
    if all(any(isinstance(m, dict) and m.get("role") == "system"
               for m in (r.get("input_messages") or [])) for r in recs):
        return "system messages", len({md5(join_roles(r["input_messages"], {"system"}))
                                       for r in recs})
    if any(r.get("input_messages") for r in recs):
        return "user turns", len({md5(join_roles(r.get("input_messages") or [], {"user"}))
                                  for r in recs})
    fields = sorted({k for r in recs for k, v in r.items()
                     if isinstance(v, str) and "instruction" in k})
    if fields:
        worst = max(len({r.get(f) for r in recs}) for f in fields)
        return f"instruction fields {fields}", worst
    return "no identifiable prompt channel", n


def main():
    reports = sorted(glob.glob(os.path.join(RESULTS, "external_audit_confirmatory_*.json"))) \
        + sorted(glob.glob(os.path.join(RESULTS, "external_audit_exploratory_*.json")))

    rows = []
    for p in reports:
        d = json.load(open(p))
        if "source_file" not in d:
            continue
        recs = load(d["source_file"])
        n = len(recs)
        has_sys = sum(1 for r in recs
                      if any(isinstance(m, dict) and m.get("role") == "system"
                             for m in (r.get("input_messages") or [])))
        has_msgs = sum(1 for r in recs if r.get("input_messages"))
        distinct = {}
        for ch in ("system_only", "user_only", "system_plus_user", "full_input"):
            distinct[ch] = len({md5(channels(r)[ch]) for r in recs})

        recorded = d["elicitation_fixed"]
        # A recorded "fixed" is substantive only if a system message exists to
        # have been held fixed. Otherwise the single hash is md5("").
        if not recorded:
            status = "substantive (recorded NOT fixed)"
        elif has_sys == n:
            status = "substantive (recorded fixed; a system message exists on every record)"
        elif has_sys == 0:
            status = "VACUOUS (recorded fixed; no record has a system message)"
        else:
            status = f"MIXED ({has_sys}/{n} records have a system message)"

        # On the widest channel, does the design hold its elicitation fixed?
        fixed_full = distinct["full_input"] == 1
        fixed_sys_user = distinct["system_plus_user"] == 1
        chan, n_chan = channel_used(recs)

        rows.append({
            "channel_actually_used": chan,
            "n_distinct_on_channel_used": n_chan,
            "holds_one_elicitation_fixed": n_chan == 1,
            "set": os.path.basename(d["source_file"]).replace(".json", ""),
            "source_file": d["source_file"],
            "n_records": n,
            "n_records_with_a_system_message": has_sys,
            "n_records_with_any_input_message": has_msgs,
            "recorded_n_distinct_system_prompts": d["n_distinct_system_prompts"],
            "recorded_elicitation_fixed": recorded,
            "recorded_is_materials_only": d.get("is_materials_only"),
            "distinct_hashes": distinct,
            "fixed_on_system_plus_user": fixed_sys_user,
            "fixed_on_full_input": fixed_full,
            "recorded_verdict_status": status,
        })

    w = max(len(r["set"]) for r in rows)
    print("=" * 118)
    print("Gate 1 recomputed on four channels (post-hoc diagnostic, not pre-registered)")
    print("=" * 118)
    print(f"{'set':{w}s} {'n':>5s} {'sysmsg':>7s} | "
          f"{'sys':>5s} {'user':>5s} {'s+u':>5s} {'full':>5s} | rec  status")
    for r in rows:
        d = r["distinct_hashes"]
        print(f"{r['set']:{w}s} {r['n_records']:5d} "
              f"{r['n_records_with_a_system_message']:7d} | "
              f"{d['system_only']:5d} {d['user_only']:5d} "
              f"{d['system_plus_user']:5d} {d['full_input']:5d} | "
              f"{str(r['recorded_elicitation_fixed'])[0]}    "
              f"{r['recorded_verdict_status']}")

    vac = [r["set"] for r in rows if r["recorded_verdict_status"].startswith("VACUOUS")]
    sub_fixed = [r["set"] for r in rows
                 if r["recorded_elicitation_fixed"] and r["recorded_verdict_status"].startswith("substantive")]
    holds = [r["set"] for r in rows if r["holds_one_elicitation_fixed"]]
    per_item = [(r["set"], r["channel_actually_used"], r["n_distinct_on_channel_used"],
                 r["n_records"]) for r in rows if not r["holds_one_elicitation_fixed"]]

    print("\nOn the channel each set actually uses")
    for r in rows:
        print(f"  {r['set']:{w}s} {r['n_distinct_on_channel_used']:5d} distinct / "
              f"{r['n_records']:5d} records  on {r['channel_actually_used']}")

    print("\nSummary")
    print(f"  recorded fixed, substantively            : {len(sub_fixed)}/{len(rows)}  {sub_fixed}")
    print(f"  recorded fixed, VACUOUSLY (no system msg): {len(vac)}/{len(rows)}  {vac}")
    print(f"  HOLD ONE ELICITATION FIXED on their own channel: {len(holds)}/{len(rows)}")
    for s in holds:
        print(f"      {s}")
    print(f"  carry PER-ITEM elicitation              : {len(per_item)}/{len(rows)}")
    for s, c, k, n in per_item:
        print(f"      {s:{w}s} {k}/{n} distinct on {c}")

    json.dump({
        "n_holding_one_elicitation_fixed": len(holds),
        "holding_one_elicitation_fixed": holds,
        "n_per_item_elicitation": len(per_item),
        "per_item_elicitation": [{"set": s, "channel": c, "n_distinct": k, "n_records": n}
                                 for s, c, k, n in per_item],
        "note": ("Post-hoc diagnostic, not pre-registered. Recomputes Gate 1 of the "
                 "external audit on four candidate elicitation channels. Gate 1 as "
                 "published hashes role=='system' only."),
        "sets": rows,
        "n_recorded_fixed_substantive": len(sub_fixed),
        "n_recorded_fixed_vacuous": len(vac),
        "vacuous_sets": vac,
    }, open(OUT, "w"), indent=1)
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
