#!/usr/bin/env python3
"""
build_audit_packets.py

EXP-AA step 1. Assembles one frozen "packet" per audited case out of
RELEASE-AUTHORED TEXT ONLY, for a panel of blinded raters to score requirements
(i)-(v) against. Pre-registered in docs/PREREG_EXP_AA.md §3, §4.

The whole point is what a packet may NOT contain: no sentence this paper wrote,
no verdict of ours, no field from anything under data/results/. §4's exclusion
list is enforced as a post-condition over the emitted text (see BANNED below),
and this script fails loudly rather than emitting a contaminated packet.

What a rater is shown, per §4:
  1. the case id, the release, and the source file it was read from
  2. for Apollo, the release's own README.md verbatim (there is no dataset card
     vendored for liars_bench -- §4a records the asymmetry)
  3. the release's own descriptive fields (dataset_class, base_name, variant,
     model / corpus, config) and the record count
  4. the count of distinct system-prompt md5s, and one verbatim system prompt
  5. one verbatim user turn
  6. the release's own label field NAMES, and the metadata field NAMES
  7. two example records, one per label value where both exist

What it is NOT shown, deliberately: any count of "paired scenarios". Which
metadata key counts as a scenario is exactly the judgement requirement (iii)
asks for, so the packet gives the metadata field names and one example's values
and stops there.

Usage:
    cd code/adaptive_lie_detector
    python experiments/build_audit_packets.py
"""

import hashlib
import json
import os
import sys

EXT = "data/external"
APOLLO = os.path.join(EXT, "apollo_deception")
LB = os.path.join(EXT, "liars_bench", "normalized")
RESULTS = "data/results"
PACKETS = os.path.join(RESULTS, "audit_packets.jsonl")
MANIFEST = os.path.join(RESULTS, "audit_packets_manifest.json")

# PREREG §4. Character budgets, fixed before any packet existed.
BUDGET_TOTAL = 8000
BUDGET_RECORD = 1500
BUDGET_PROMPT = 2500

# PREREG §3. The ten Apollo rollout sets, in the order dataset_paths.json and
# the ten published external_audit_*.json set reports give them.
APOLLO_CASES = (
    "ai_audit__reasoning__llama-70b",
    "ai_audit__reasoning__llama-70b-3.3",
    "ai_liar__original_without_answers__llama-70b-3.3",
    "goal_directed_lying",
    "insider_trading__onpolicy__llama-70b",
    "out_simple_4_many",
    "roleplaying__plain__llama-70b",
    "roleplaying__plain__llama-70b-3.3",
    "sandbagging_v2__wmdp_mmlu__llama-70b",
    "sandbagging_v2__wmdp_mmlu__llama-70b-3.3",
)

# PREREG §3. The eight liars_bench configurations.
LB_CASES = (
    "alpaca",
    "convincing-game",
    "gender-secret",
    "harm-pressure-choice",
    "harm-pressure-knowledge-report",
    "insider-trading",
    "instructed-deception",
    "soft-trigger",
)

# PREREG §4. Substrings that must not appear in an emitted packet. Two kinds:
# our verdict vocabulary, and the criterion vocabulary itself. A rater is given
# the five requirements by the rater script; the PACKET must not hint at them.
BANNED = (
    "criterion", "requirement (i", "requirement (v",
    "construct recovery", "CONSTRUCT_RECOVERY",
    "annotation-channel", "annotation channel",
    "eligible_for_criterion4", "UNDERPOWERED", "CRITERION4",
    "deployed-observable", "definitionally equivalent",
    "identification audit", "elicitation confound", "fixed elicitation",
    "byte-identical", "tau_D", "\\tau_D",
    "this paper", "our rule", "our verdict", "the audit",
    "data/results",
)


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:8]


def clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f"\n[...truncated at {n} characters]"


def role_content(messages):
    """Roles and contents of a release's own input_messages, verbatim."""
    out = []
    for m in messages or []:
        out.append((m.get("role", "?"), m.get("content", "") or ""))
    return out


def join_user(messages):
    return "\n".join(c for r, c in role_content(messages) if r == "user")


def fit(sections):
    """Assemble sections inside BUDGET_TOTAL by trimming the longest one first.

    PREREG §4 fixes the 8,000-character cap. A packet whose README, prompt and
    two records exceed it must lose something, and letting the tail fall off
    would silently drop the example records from exactly the sets with the
    longest prompts. So the rule is mechanical and applied uniformly: repeatedly
    trim the currently-longest section by 10% until the whole packet fits. Every
    trim is marked in the text it happened to.
    """
    orig = list(sections)
    keep = [len(s) for s in orig]
    secs = list(orig)
    guard = 0
    while len("\n\n".join(secs)) > BUDGET_TOTAL and guard < 500:
        guard += 1
        i = max(range(len(secs)), key=lambda j: len(secs[j]))
        keep[i] = max(200, int(keep[i] * 0.9))
        secs = [clip(o, k) for o, k in zip(orig, keep)]
    return clip("\n\n".join(secs), BUDGET_TOTAL)


def first_system_and_user(messages):
    sys_txt, usr_txt = "", ""
    for role, content in role_content(messages):
        if role == "system" and not sys_txt:
            sys_txt = content
        if role == "user" and not usr_txt:
            usr_txt = content
    return sys_txt, usr_txt


def find_release_file(case):
    """Locate <case>.json anywhere under the release's own data tree.

    The ten audited sets do not all live in data/rollouts/: goal_directed_lying
    is under how_to_catch_an_ai_liar/ and out_simple_4_many under ft_fact_paired/.
    Search rather than assume, and require exactly one match.
    """
    hits = []
    for root, _dirs, files in os.walk(os.path.join(APOLLO, "data")):
        if f"{case}.json" in files:
            hits.append(os.path.join(root, f"{case}.json"))
    if len(hits) != 1:
        return None, (f"{len(hits)} files named {case}.json under {APOLLO}/data"
                      + (f": {hits}" if hits else ""))
    return hits[0], None


def field_variability(recs):
    """Distinct values per scalar field, across records. Counts, not verdicts.

    This is what lets a rater decide for itself which field carries the
    elicitation and whether it is held fixed -- we do not tell it.
    """
    keys = sorted({k for r in recs for k, v in r.items()
                   if isinstance(v, (str, bool, int, float)) or v is None})
    out = {}
    for k in keys:
        try:
            out[k] = len({r.get(k) for r in recs})
        except TypeError:
            continue
    return out


def example_block(idx, label_fields, messages, output, meta):
    lines = [f"--- example record #{idx} ---"]
    for k, v in label_fields.items():
        lines.append(f"{k}: {json.dumps(v) if not isinstance(v, str) else v}")
    for role, content in role_content(messages):
        if role == "system":
            continue  # already shown verbatim once, in full budget
        lines.append(f"[{role}]: {clip(content, 400)}")
    lines.append(f"[model output]: {clip(output, 600)}")
    if meta is not None:
        lines.append(f"[record metadata]: {clip(json.dumps(meta, default=str), 400)}")
    return clip("\n".join(lines), BUDGET_RECORD)


# --------------------------------------------------------------------------
# Apollo
# --------------------------------------------------------------------------

# The release's own label / condition field names, across all three schemas the
# ten audited sets use. We list which are present; we do not say what they mean.
CONDITION_FIELDS = ("expect_deception", "expect_honesty", "expected_label")
GRADE_FIELDS = ("graded_deception", "model_graded_honesty",
                "deception_percentage", "honesty_percentage",
                "grader_reasoning", "model_grader_reasoning")


def materials_block(idx, rec):
    """A record with no messages: print the release's own fields verbatim."""
    lines = [f"--- example record #{idx} (this file's records are fields, not messages) ---"]
    for k in sorted(rec.keys()):
        v = rec[k]
        lines.append(f"{k}: {clip(v, 500) if isinstance(v, str) else json.dumps(v, default=str)[:300]}")
    return clip("\n".join(lines), BUDGET_RECORD)


def pick_two(recs):
    """Two examples, contrasting on the release's own condition field if it has one."""
    for f in CONDITION_FIELDS:
        vals = {r.get(f) for r in recs if f in r} - {None}
        if len(vals) >= 2:
            picked, seen = [], set()
            for r in recs:
                v = r.get(f)
                if v in seen or v is None:
                    continue
                seen.add(v)
                picked.append(r)
                if len(picked) == 2:
                    return picked, f
    return recs[:2], None


def build_apollo(case, readme):
    path, err = find_release_file(case)
    if path is None:
        return None, err
    d = json.load(open(path))
    recs = d["rollouts"] if isinstance(d, dict) else d
    if not recs:
        return None, f"no records in {path}"

    has_msgs = any(r.get("input_messages") for r in recs)
    n_sys = sum(1 for r in recs
                if any(isinstance(m, dict) and m.get("role") == "system"
                       for m in (r.get("input_messages") or [])))

    picked, cond_field = pick_two(recs)
    present_cond = [f for f in CONDITION_FIELDS if any(f in r for r in recs)]
    present_grade = [f for f in GRADE_FIELDS if any(f in r for r in recs)]

    secs = ["=== RELEASE README (verbatim) ===\n" + readme]

    if isinstance(d, dict):
        top = json.dumps({k: d.get(k) for k in
                          ("dataset_class", "base_name", "variant", "model")}, indent=1)
        secs.append("=== THIS SET, in the release's own top-level fields ===\n" + top)
    else:
        secs.append("=== THIS SET ===\nThe file is a JSON array of records with no "
                    "top-level metadata of its own.")

    facts = [f"records in this file: {len(recs)}",
             f"fields present on a record: {sorted(recs[0].keys())}",
             f"condition-like fields present: {present_cond or 'none'}",
             f"grade-like fields present: {present_grade or 'none'}",
             "distinct values per field, across all records "
             f"(counts only): {json.dumps(field_variability(recs))}"]
    if has_msgs:
        facts.append(f"records containing a message with role 'system': {n_sys} of {len(recs)}")
        if n_sys:
            sys_md5 = {}
            for r in recs:
                k = md5(first_system_and_user(r.get("input_messages"))[0])
                sys_md5[k] = sys_md5.get(k, 0) + 1
            facts.append(f"distinct system-message md5s: {len(sys_md5)}")
        usr_md5 = {md5(join_user(r.get("input_messages"))) for r in recs}
        facts.append(f"distinct user-turn md5s: {len(usr_md5)}")
    else:
        facts.append("no record in this file contains any message list at all; "
                     "its prompt material is in the named fields above")
    secs.append("=== COUNTS READ OFF THE FILE ===\n" + "\n".join(facts))

    if has_msgs:
        s0, u0 = first_system_and_user(recs[0].get("input_messages"))
        secs.append("=== SYSTEM MESSAGE of the first record (verbatim) ===\n"
                    + (clip(s0, BUDGET_PROMPT) if s0
                       else "(this record has no message with role 'system')"))
        secs.append("=== FIRST USER TURN of the first record (verbatim) ===\n"
                    + (clip(u0, 800) if u0 else "(this record has no user turn)"))

    if cond_field:
        note = (f"The two records below differ in the release's own `{cond_field}` field.")
    else:
        note = ("This file has no field taking two distinct values that could pair "
                "records, so the two below are simply its first two.")
    exs = []
    for i, r in enumerate(picked, 1):
        if r.get("input_messages"):
            labels = {k: r.get(k) for k in CONDITION_FIELDS + GRADE_FIELDS if k in r}
            outs = r.get("output_strs") or [""]
            exs.append(example_block(i, labels, r.get("input_messages"),
                                     outs[0] if outs else "", r.get("metadata")))
        else:
            exs.append(materials_block(i, r))
    secs.append("=== EXAMPLE RECORDS ===\n" + note + "\n\n" + "\n\n".join(exs))

    prov = {
        "release": "apollo_deception (Detecting Strategic Deception Using Linear Probes)",
        "source_files": [os.path.join(APOLLO, "README.md"), path],
        "schema": ("dict with a `rollouts` list" if isinstance(d, dict)
                   else "JSON array of records"),
        "has_message_lists": has_msgs,
    }
    return fit(secs), prov


# --------------------------------------------------------------------------
# liars_bench
# --------------------------------------------------------------------------

def build_lb(case):
    path = os.path.join(LB, f"{case}.json")
    if not os.path.exists(path):
        return None, f"missing normalized file {path}"
    d = json.load(open(path))
    rolls = d.get("rollouts") or []
    if not rolls:
        return None, f"no rollouts in {path}"

    sys_md5 = {}
    for r in rolls:
        s, _ = first_system_and_user(r.get("input_messages"))
        sys_md5[md5(s)] = sys_md5.get(md5(s), 0) + 1
    sys_shown = dict(sorted(sys_md5.items(), key=lambda kv: -kv[1])[:4])

    s0, u0 = first_system_and_user(rolls[0].get("input_messages"))

    m0 = rolls[0].get("metadata") or {}
    meta_keys = sorted(m0.keys())
    inner_keys = sorted((m0.get("meta") or {}).keys()) if isinstance(m0.get("meta"), dict) else []
    gens = sorted({(r.get("metadata") or {}).get("generator") for r in rolls} - {None})

    # Two records contrasting on the release's own `deceptive` field -- but some
    # configurations (the honest controls) carry only one value of it, and the
    # packet must say so rather than imply a contrast that is not there.
    picked, seen = [], set()
    for r in rolls:
        v = (r.get("metadata") or {}).get("deceptive")
        if v in seen:
            continue
        seen.add(v)
        picked.append(r)
        if len(picked) == 2:
            break
    if len(picked) < 2:
        note = (f"Every record in this configuration has "
                f"`metadata.deceptive` = {json.dumps(sorted(seen, key=str)[0])}; it takes no "
                "second value here, so the two records below are simply its first two.")
        picked = rolls[:2]
    else:
        note = "The two records below differ in the release's own `metadata.deceptive` field."

    exs = []
    for i, r in enumerate(picked, 1):
        m = r.get("metadata") or {}
        labels = {"metadata.deceptive": m.get("deceptive"),
                  "metadata.generator": m.get("generator")}
        outs = r.get("output_strs") or [""]
        exs.append(example_block(i, labels, r.get("input_messages"),
                                 outs[0] if outs else "", m))

    n_sys = sum(1 for r in rolls
                if any(isinstance(m, dict) and m.get("role") == "system"
                       for m in (r.get("input_messages") or [])))
    usr_md5 = {md5(join_user(r.get("input_messages"))) for r in rolls}

    secs = [
        "=== ABOUT THIS RELEASE ===\n"
        "A gated corpus of lie and honest-response transcripts, released as parquet "
        "files with no accompanying prose description in this copy. Everything below "
        "is read directly out of the released records.",

        "=== THIS CONFIGURATION, in the release's own fields ===\n"
        + json.dumps({"corpus": d.get("corpus"), "config": d.get("config")}, indent=1),

        "=== COUNTS READ OFF THE FILE ===\n" + "\n".join([
            f"records in this configuration: {len(rolls)}",
            f"generator models present: {gens}",
            f"records containing a message with role 'system': {n_sys} of {len(rolls)}",
            f"distinct system-message md5s: {len(sys_md5)}  "
            f"(the four most common, with counts: {json.dumps(sys_shown)})",
            f"distinct user-turn md5s: {len(usr_md5)}",
            f"keys present in each record's `metadata`: {meta_keys}",
            (f"keys present in `metadata.meta`: {inner_keys}" if inner_keys
             else "`metadata.meta` is absent or not a mapping"),
            "distinct values per `metadata` field, across all records (counts only): "
            + json.dumps(field_variability([r.get("metadata") or {} for r in rolls])),
        ]),

        "=== SYSTEM MESSAGE of the first record (verbatim) ===\n"
        + (clip(s0, BUDGET_PROMPT) if s0
           else "(this record has no message with role 'system')"),

        "=== FIRST USER TURN of the first record (verbatim) ===\n"
        + (clip(u0, 800) if u0 else "(this record has no user turn)"),

        "=== EXAMPLE RECORDS ===\n" + note + "\n\n" + "\n\n".join(exs),
    ]
    prov = {
        "release": "liars_bench (Cadenza-Labs/liars-bench)",
        "source_files": [path],
        "no_dataset_card_vendored": True,
    }
    return fit(secs), prov


# --------------------------------------------------------------------------

def check_clean(case, text):
    """PREREG §4's exclusion list, as a post-condition. Fails loudly."""
    low = text.lower()
    hits = [b for b in BANNED if b.lower() in low]
    if hits:
        raise SystemExit(
            f"PACKET CONTAMINATED [{case}]: banned substrings present {hits}.\n"
            "PREREG §4 forbids emitting this. Fix the builder, do not relax the list."
        )


def main():
    if not os.path.isdir(APOLLO) or not os.path.isdir(LB):
        raise SystemExit(f"vendored releases not found under {EXT}")

    readme = open(os.path.join(APOLLO, "README.md")).read()

    print("=" * 72)
    print("EXP-AA step 1 — building blinded audit packets (PREREG §3, §4)")
    print("=" * 72)

    packets, manifest, failed = [], {}, []
    for case in APOLLO_CASES:
        body, prov = build_apollo(case, readme)
        if body is None:
            failed.append((case, prov))
            continue
        packets.append((case, "apollo", body))
        manifest[case] = prov
    for case in LB_CASES:
        body, prov = build_lb(case)
        if body is None:
            failed.append((case, prov))
            continue
        packets.append((case, "liars_bench", body))
        manifest[case] = prov

    rows = []
    for case, release, body in packets:
        check_clean(case, body)
        rows.append({
            "case": case,
            "release": release,
            "packet_md5": hashlib.md5(body.encode("utf-8")).hexdigest(),
            "n_chars": len(body),
            "text": body,
        })
        print(f"  {case:52s} {release:11s} {len(body):5d} ch  "
              f"md5 {rows[-1]['packet_md5'][:8]}")

    if failed:
        print("\n  PACKET-FAILED (PREREG §3: reported, not replaced):")
        for case, why in failed:
            print(f"    {case}: {why}")

    os.makedirs(RESULTS, exist_ok=True)
    with open(PACKETS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump({
        "prereg": "docs/PREREG_EXP_AA.md",
        "budgets": {"total": BUDGET_TOTAL, "record": BUDGET_RECORD,
                    "prompt": BUDGET_PROMPT},
        "n_cases": len(rows),
        "packet_failed": [{"case": c, "reason": w} for c, w in failed],
        "provenance": manifest,
        "banned_substrings_asserted_absent": list(BANNED),
    }, open(MANIFEST, "w"), indent=1)

    print(f"\n  packets : {len(rows)}  ->  {PACKETS}")
    print(f"  manifest:              ->  {MANIFEST}")
    print(f"  exclusion post-condition passed on all {len(rows)} packets")
    if len(rows) != 18:
        print(f"\n  NOTE: {len(rows)} cases, not the 18 PREREG §3 fixes. "
              "The shortfall is listed above and must be reported.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
