#!/usr/bin/env python3
"""EXP-AF: is the elicitation-construct confound *decidable* outside deception,
and is it *measurable* there?

Pre-registration: docs/PREREG_EXP_AF.md, committed BEFORE this file existed.
**EXP-AF is descriptive, not confirmatory** -- PREREG section 0 discloses, by
value, every count that was computed during feasibility scoping before the
pre-registration was written. Nothing here may be reported as a pre-registered
finding.

The question. The Proposition in methodology.tex already quantifies over k
unobserved mechanisms that do(E) shifts, with deception entering only as the
instantiation (M1, M2) = (D, C). tab:prior_work_criteria group (e) already
applies requirements (i)-(v) to four adjacent constructs from published designs,
and its caption disclaims the stronger reading: the requirements are "decidable
outside deception given a specification, *not* that the confound is active
there." EXP-AF asks whether that disclaimer can be upgraded to a measured
statement on the corpora this paper already pinned.

This is a CENSUS, not an estimate. Every corpus is read in full. There is no
sampling, no estimator, no fold and no p-value anywhere in this file, so there
is nothing to correct for multiplicity, and any apparent precision is exactness
rather than significance.

What is reused, not reimplemented:
  * run_liars_bench_audit.available_configs / load_config / messages_of -- the
    same loaders EXP-XL used, so the row sets are identical.
  * run_liars_bench_audit.META_SCENARIO_KEYS -- the candidate-key list EXP-XL
    already fixed, extended here with the ELICITATION keys (PREREG section 3.3).
  * analyze_external_audit.wilson for the rate intervals.
  * The env_md5 idiom of run_liars_bench_audit.scenario_keys, applied here to
    the *item* rather than the scenario.

One measured defect in the existing pipeline, reported rather than silently
patched: `run_liars_bench_audit.parse_meta` returns {} for BOTH harm-pressure
configs, because their shipped `meta` reprs contain `np.float64(...)`, which
neither json.loads nor ast.literal_eval accepts. `parse_meta_np` below handles
it and the JSON records `parse_meta_recovered` per config so the gap is visible.
EXP-XL took its verdicts from `env_md5`, not from `meta`, so no published number
depends on this -- but the `meta.template_index` candidate key it claims to
report was silently absent for those two configs.

Usage (from code/adaptive_lie_detector):
    ../.venv/bin/python3 experiments/analyze_eccp_decidability.py
    ../.venv/bin/python3 experiments/analyze_eccp_decidability.py --json-only
"""

import argparse
import ast
import collections
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from experiments.run_liars_bench_audit import (  # noqa: E402
    available_configs,
    load_config,
    messages_of,
)
from analyze_external_audit import wilson  # noqa: E402

# ----------------------------------------------------------------- constants
# Every threshold below is fixed by docs/PREREG_EXP_AF.md and must not be tuned.

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "data", "results")
OUT_PATH = os.path.join(RESULTS, "eccp_decidability.json")
PREREG = "docs/PREREG_EXP_AF.md"

ROLLOUT_DIR = os.path.join(BASE, "data", "external", "apollo_deception",
                           "data", "rollouts")

CHANNEL_MIN = 0.5      # PREREG 3.3: channel recorded iff distinct/rows > this
ANCHOR_EXACT = 1.0     # PREREG 3.2 cond. 3: released label must reproduce exactly

# PREREG 3.3: the elicitation-variant keys, per corpus family. Where a corpus
# offers more than one candidate, EVERY candidate is reported with its own
# count, so the choice of key cannot decide the verdict.
ELICITATION_KEYS = ("template_index", "system_template_index",
                    "user_template_index", "template_id",
                    "instructed_deceptiveness", "donotadmit_instruction")

# PREREG 3.3: the item key for multiple-choice corpora is the option block.
OPTION_RE = re.compile(r"(?m)^\s*A[.:]\s")

# DEVIATION 1 from PREREG 3.3, disclosed rather than substituted silently.
#
# The pre-registration fixes the item key as "the option block for
# multiple-choice corpora ... and the `env_md5` hash otherwise", reusing
# run_liars_bench_audit's env_md5, which hashes EVERY message the model did not
# write -- including the system prompt. For a corpus whose elicitation variant
# LIVES IN THE SYSTEM PROMPT, env_md5 therefore contains E by construction, so
# "same item under two variants" is unsatisfiable as a matter of arithmetic and
# the predicate returns 0 for a reason that has nothing to do with the corpus's
# design. That is an artifact, not a measurement, and the pre-registered rule is
# defective here.
#
# The residual key below repairs it WITHOUT hand-coding a per-corpus extractor:
# a user-message line is boilerplate if it recurs on more than BOILER_MAX of the
# corpus's rows, and the item is what is left. All three keys are reported for
# every corpus, the verdict names which key it rests on, and requirement (i) is
# credited only on a key that is E-free by construction. A reader who rejects
# BOILER_MAX keeps every count.
BOILER_MAX = 0.01

# PREREG 3.2 cond. 1: Y2 must come from metadata or an objective answer key,
# NOT from the response text a detector would read. A bare closed-set answer
# (one letter) is an answer, not behavior; free prose is behavior, and reading
# it to recover an answer would violate the condition. This is the dividing
# line, and `y2_source` records which side each corpus falls on.
LETTER = {"A": 0, "B": 1, "C": 2, "D": 3}


# ----------------------------------------------------------------- helpers

def parse_meta_np(raw):
    """`meta` as shipped is a Python repr, sometimes carrying numpy scalar
    constructors (`np.float64(0.98...)`). json.loads fails on all of them and
    ast.literal_eval fails on the numpy ones, which is why the existing
    run_liars_bench_audit.parse_meta returns {} for both harm-pressure configs.
    Unwrap the constructors, then literal_eval. Never improvise a substitute:
    an unparseable field yields {}.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    txt = re.sub(r"np\.(?:float64|float32|int64|int32|bool_)\(([^()]*)\)",
                 r"\1", raw)
    for parse in (json.loads, ast.literal_eval):
        try:
            val = parse(txt)
        except (ValueError, TypeError, SyntaxError, MemoryError,
                RecursionError):
            continue
        if isinstance(val, dict):
            return val
    return {}


def final_assistant(row_messages):
    """The channel a deployed detector would read: the last assistant turn."""
    for m in reversed(row_messages):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def env_text(row_messages):
    """Everything the model did NOT write -- the item as the environment posed
    it. Same construction as run_liars_bench_audit.scenario_keys' env_md5."""
    return "\n".join(f"{m['role']}: {m['content']}"
                     for m in row_messages if m.get("role") != "assistant")


# The item identifier the RELEASE itself supplies, where it supplies one. These
# outrank every text-derived key: they are the corpus author's own statement of
# what an item is, and they need no threshold. `dataset_index` is a top-level
# column, `sample_id` lives in `meta`.
RELEASED_ITEM_COLS = ("dataset_index",)
RELEASED_ITEM_META = ("sample_id",)

# Only these are E-free BY CONSTRUCTION, so only these can credit (i): the
# released id, the option block (the answer set), the residual (every recurring
# template line stripped), and the last user line (the item text of a true/false
# corpus). `env_md5` is reported for continuity with the pre-registration and
# with EXP-XL, never used to credit a pass.
#
# Being E-free is necessary but NOT sufficient, in two measured ways.
#   OVER-MERGING: a key that collapses rows a finer key separates manufactures
#   apparent crossing -- distinct items sharing one key look like one item under
#   many variants. On `harm-pressure-choice`, `item_residual` over-merges 293 of
#   4308 classes and reports exactly 293 crossings, i.e. every one is an
#   artifact, while `option_block` reports 0 of 4601.
#   VACUITY: a key with one row per class (a row id, e.g. the top-level `index`
#   column, 5400 distinct over 5400 rows) cannot cross by construction, so its 0
#   is not evidence of collinearity.
# (i) is therefore credited only on an ADMISSIBLE key: E-free, non-vacuous, and
# either the released id or a refinement of every other E-free key. When the
# text-derived keys merely cross-cut and no released id exists, the verdict is
# UNDECIDED, not the convenient one. Every count is reported regardless.
E_FREE_KEYS = ("item_released", "option_block", "item_residual",
               "item_last_line")


def _md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def user_text(row_messages):
    return "\n".join(m["content"] for m in row_messages
                     if m.get("role") == "user")


def released_item_id(row, meta):
    """The release's own item identifier, or None. Never invented."""
    for col in RELEASED_ITEM_COLS:
        if row.get(col) is not None:
            return f"{col}={row[col]}"
    for fld in RELEASED_ITEM_META:
        if meta.get(fld) is not None:
            return f"{fld}={meta[fld]}"
    return None


def build_item_keys(all_msgs, rows=None, metas=None):
    """Every item key for every row, in one pass over the corpus.

    `env_md5` and `option_block` are the two PREREG 3.3 keys. `item_residual`
    is DEVIATION 1: the user-message lines that do NOT recur on more than
    BOILER_MAX of rows, i.e. the item with the template stripped. Requires
    corpus-level line frequencies, hence the whole-corpus signature.
    """
    n = len(all_msgs)
    users = [user_text(m) for m in all_msgs]
    freq = collections.Counter()
    for u in users:
        freq.update(set(ln.strip() for ln in u.splitlines() if ln.strip()))
    cutoff = BOILER_MAX * n

    out = []
    for j, (msgs, u) in enumerate(zip(all_msgs, users)):
        keys = {"env_md5": _md5(env_text(msgs))}
        if rows is not None:
            rid = released_item_id(rows[j], (metas or [{}] * len(rows))[j])
            if rid is not None:
                keys["item_released"] = rid
        hit = OPTION_RE.search(u)
        if hit:
            keys["option_block"] = _md5(u[hit.start():])
        lines = [ln.strip() for ln in u.splitlines() if ln.strip()]
        resid = [ln for ln in lines if freq[ln] <= cutoff]
        # a row whose every line is boilerplate has no residual item; it is
        # recorded as absent rather than silently hashed to the empty string
        if resid:
            keys["item_residual"] = _md5("\n".join(resid))
        if lines:
            keys["item_last_line"] = _md5(lines[-1])
        out.append(keys)
    return out


def refinement_audit(items):
    """Decide which item keys may credit requirement (i).

    For every ordered pair of E-free keys, count the classes of the first that
    merge more than one value of the second. Then:
      * the RELEASED id, when present, is admissible outright -- it is the
        corpus author's own definition of an item, not a heuristic of ours;
      * otherwise a key is admissible iff it over-merges no other E-free key,
        i.e. it is at least as fine as every alternative;
      * a VACUOUS key (one row per class) is never admissible, because it cannot
        cross by construction and its 0 is not evidence.
    `undecided` is set when no key qualifies but some E-free key exists: the
    text-derived keys cross-cut, and the census says so instead of choosing.
    """
    present = [k for k in E_FREE_KEYS if any(k in it for it in items)]
    n_rows = {k: sum(1 for it in items if k in it) for k in present}
    n_dist = {k: len({it[k] for it in items if k in it}) for k in present}
    vacuous = {k: bool(n_dist[k] == n_rows[k] and n_rows[k] > 1)
               for k in present}

    audit, admissible = {}, {}
    for a in present:
        worst = 0
        for b in present:
            if a == b:
                continue
            classes = collections.defaultdict(set)
            for it in items:
                if a in it and b in it:
                    classes[it[a]].add(it[b])
            merged = sum(1 for v in classes.values() if len(v) > 1)
            audit[f"{a}_merges_{b}"] = {"n_classes": len(classes),
                                        "n_over_merged": merged}
            worst = max(worst, merged)
        admissible[a] = bool(
            not vacuous[a] and (a == "item_released" or worst == 0))

    return {
        "pairwise": audit,
        "admissible": admissible,
        "vacuous": vacuous,
        "n_distinct": n_dist,
        "released_id_present": "item_released" in present,
        "undecided": bool(present and not any(admissible.values())),
    }


def channel_stats(texts):
    """PREREG 3.3: is an elicited-behavior channel recorded at all?"""
    n = len(texts)
    lens = sorted(len(t) for t in texts)
    distinct = len(set(t.strip() for t in texts))
    ratio = (distinct / n) if n else 0.0
    return {
        "n_rows": n,
        "n_distinct_final_messages": distinct,
        "channel_ratio": round(ratio, 6),
        "median_len": int(np.median(lens)) if n else 0,
        "p90_len": int(np.percentile(lens, 90)) if n else 0,
        "channel_recorded": bool(ratio > CHANNEL_MIN),
        "threshold": CHANNEL_MIN,
    }


def crosstab_items_by_variant(items, variants, e_free):
    """PREREG 3.3 requirement (i): is E separable from the item? Returns the
    count of items realized under >= 2 distinct elicitation variants."""
    by_item = collections.defaultdict(set)
    for it, v in zip(items, variants):
        by_item[it].add(v)
    hist = collections.Counter(len(v) for v in by_item.values())
    multi = sum(1 for v in by_item.values() if len(v) > 1)
    return {
        "n_items": len(by_item),
        "n_variants": len(set(variants)),
        "n_items_multi_E": multi,
        "share_items_multi_E": round(multi / len(by_item), 6) if by_item else 0.0,
        "variants_per_item_hist": {str(k): v for k, v in sorted(hist.items())},
        "n_rows_with_key": len(items),
        "item_key_admissible": bool(e_free),
        # only an admissible key can credit a pass: a key containing E returns 0
        # by construction, and a key that over-merges returns crossings that are
        # artifacts. Both failure modes are reported, never silently resolved.
        "req_i_separable": bool(multi > 0 and e_free),
        "crossed_on_this_key": bool(multi > 0),
    }


def items_both_outcomes(items, y):
    """PREREG 3.3 requirement (iii): same item realized under both outcomes."""
    by_item = collections.defaultdict(set)
    for it, val in zip(items, y):
        by_item[it].add(bool(val))
    both = sum(1 for v in by_item.values() if len(v) > 1)
    return {
        "n_items": len(by_item),
        "n_items_both_outcomes": both,
        "share_items_both_outcomes": (round(both / len(by_item), 6)
                                      if by_item else 0.0),
        "req_iii": bool(both > 0),
    }


def agreement(a, b):
    if not len(a):
        return float("nan")
    return round(float(np.mean([x == y for x, y in zip(a, b)])), 6)


# ----------------------------------------------------------------- the census

def census_liars_bench():
    """One record per config, with candidate constructs and every candidate E."""
    out = []
    for cfg in available_configs():
        rows = load_config(cfg)
        msgs = [messages_of(r) for r in rows]
        metas = [parse_meta_np(r.get("meta")) for r in rows]
        finals = [final_assistant(m) for m in msgs]
        items = build_item_keys(msgs, rows=rows, metas=metas)
        dec = [bool(r.get("deceptive")) for r in rows]

        meta_keys = collections.Counter()
        for m in metas:
            meta_keys.update(m.keys())

        rec = {
            "corpus": "liars_bench",
            "config": cfg,
            "n_rows": len(rows),
            "models": dict(collections.Counter(
                str(r.get("model")) for r in rows)),
            "meta_keys": dict(meta_keys),
            "parse_meta_recovered": bool(
                meta_keys and not _existing_parse_meta_works(rows[0])),
            "deceptive_rate": round(float(np.mean(dec)), 6) if rows else None,
            "channel": channel_stats(finals),
            "item_keys": refinement_audit(items),
            "elicitation": {},
            "constructs": [],
        }
        admissible = rec["item_keys"]["admissible"]

        # ---- every candidate elicitation key, each with its own crosstab
        for key in ELICITATION_KEYS:
            # partial coverage is kept, not dropped: the key still decides (i)
            # on the rows that carry it, and `meta_coverage` records the share
            if not meta_keys.get(key):
                continue
            rec.setdefault("meta_coverage", {})[key] = round(
                meta_keys[key] / len(rows), 6)
            # rows lacking the key are EXCLUDED, not folded into a None variant
            have = [i for i in range(len(rows)) if key in metas[i]]
            variants = {i: metas[i][key] for i in have}
            for iname in E_FREE_KEYS + ("env_md5",):
                sub = [i for i in have if iname in items[i]]
                if not sub:
                    continue
                rec["elicitation"][f"{key}|{iname}"] = crosstab_items_by_variant(
                    [items[i][iname] for i in sub], [variants[i] for i in sub],
                    e_free=admissible.get(iname, False))
            # the verdict rests on the admissible keys alone, and says UNDECIDED
            # rather than choosing when none qualifies
            adm = [v for k, v in rec["elicitation"].items()
                   if k.startswith(f"{key}|") and isinstance(v, dict)
                   and v.get("item_key_admissible")]
            rec.setdefault("req_i_verdict", {})[key] = (
                "UNDECIDED" if not adm else
                "SEPARABLE" if any(v["crossed_on_this_key"] for v in adm) else
                "NOT_SEPARABLE")
            # rate of the released label per variant -- PREREG 7.1, illustration
            # only: NOT attributable to the framing when items are collinear.
            per = {}
            for v in sorted({variants[i] for i in have}, key=str):
                idx = [i for i in have if variants[i] == v]
                k = sum(dec[i] for i in idx)
                lo, hi = wilson(k, len(idx))
                per[str(v)] = {"n": len(idx),
                               "deceptive_rate": round(k / len(idx), 6),
                               "wilson": [round(lo, 6), round(hi, 6)]}
            rec["elicitation"][f"{key}|released_label_rate"] = per

        # ---- candidate second construct: underperformance vs an answer key
        if meta_keys.get("correct_option"):
            rec["constructs"].append(
                _underperformance_construct(metas, finals, items, dec, rec))
        if not rec["constructs"]:
            rec["constructs"].append({
                "name": "none",
                "y2_derivable": False,
                "reason": "no answer key or belief field in released metadata; "
                          "the only label is `deceptive`",
            })
        out.append(rec)
    return out


def _existing_parse_meta_works(row):
    from experiments.run_liars_bench_audit import parse_meta as old
    return bool(old(row))


def _underperformance_construct(metas, finals, items, dec, rec):
    """Y2 = the model's answer differs from the objective key.

    PREREG 3.2 cond. 1 forbids deriving Y2 from the response text a detector
    would read. A bare closed-set answer (a single option letter) is an answer,
    not behavior, and comparing it to a key reads no behavior; free prose IS
    behavior, so a corpus whose response is prose fails this condition and Y2
    is reported as NOT derivable rather than extracted anyway.

    Restricted to the rows that carry `correct_option`; `coverage` records how
    many those were.
    """
    have = [i for i in range(len(metas)) if "correct_option" in metas[i]]
    n = len(have)
    parsed = {i: LETTER.get(finals[i].strip()[:1].upper())
              if finals[i].strip() else None for i in have}
    n_parsed = sum(1 for i in have if parsed[i] is not None)
    closed_set = not rec["channel"]["channel_recorded"]

    con = {
        "name": "underperformance (answer != objective key)",
        "y2_source": "meta.correct_option vs the closed-set answer letter",
        "coverage": {"n_rows_with_key": n, "n_rows": len(metas),
                     "share": round(n / len(metas), 6) if metas else 0.0},
        "answer_parse_rate": round(n_parsed / n, 6) if n else 0.0,
        "channel_is_closed_set": closed_set,
    }

    if not closed_set:
        con.update({
            "y2_derivable": False,
            "reason": "the recorded response is free prose, so recovering the "
                      "model's answer would mean reading the detector's own "
                      "channel -- forbidden by PREREG 3.2 condition 1",
        })
        return con
    if n_parsed != n:
        con.update({
            "y2_derivable": False,
            "reason": f"answer parsed on {n_parsed}/{n} rows; a partial "
                      "extractor cannot anchor a verdict",
        })
        return con

    y2 = [parsed[i] != metas[i]["correct_option"] for i in have]
    y3 = [parsed[i] != metas[i].get("model_belief") for i in have]
    sdec = [dec[i] for i in have]

    con.update({
        "y2_derivable": True,
        "n_belief_ne_key": int(sum(
            1 for i in have
            if metas[i].get("model_belief") != metas[i]["correct_option"])),
        "anchor_released_label": {
            # PREREG 3.2 cond. 3: the released label must reproduce EXACTLY
            "rule": "answer != meta.model_belief  ==  `deceptive`",
            "agreement": agreement(y3, sdec),
            "exact": agreement(y3, sdec) >= ANCHOR_EXACT,
        },
        "y2_rate": round(float(np.mean(y2)), 6),
        "released_label_rate": round(float(np.mean(sdec)), 6),
        # PREREG 3.2 cond. 2: distinct from the corpus's own deception label?
        "agreement_with_released_label": agreement(y2, sdec),
        "is_second_construct": bool(agreement(y2, sdec) < 1.0),
        # (iii) decided only on ADMISSIBLE item keys -- an over-merging key
        # would report spurious both-outcome items for the same reason it
        # reports spurious crossings
        "req_iii": {
            iname: items_both_outcomes(
                [items[i][iname] for i in have if iname in items[i]],
                [y for i, y in zip(have, y2) if iname in items[i]])
            for iname in E_FREE_KEYS
            if rec["item_keys"]["admissible"].get(iname)
            and any(iname in items[i] for i in have)
        },
    })
    return con


def _label_of(roll):
    """The released deception label. Apollo names it `expected_label` in some
    rollout files and `expect_deception` in others; read whichever exists."""
    if "expected_label" in roll:
        return roll["expected_label"]
    return roll.get("expect_deception")


def census_apollo():
    """One record per pinned rollout file."""
    out = []
    if not os.path.isdir(ROLLOUT_DIR):
        return out
    for fn in sorted(os.listdir(ROLLOUT_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(ROLLOUT_DIR, fn)) as fh:
            blob = json.load(fh)
        rolls = blob.get("rollouts") or []
        # Four of the twenty pinned files carry the literal string
        # "SEE OTHER FILE" in place of a row list. Recorded, not skipped
        # silently: it is a structural fact about the released corpus.
        if isinstance(rolls, str) or not rolls:
            out.append({
                "corpus": "apollo", "file": fn,
                "base_name": blob.get("base_name"),
                "variant": blob.get("variant"), "model": blob.get("model"),
                "n_rows": 0, "rows_absent": True,
                "rows_placeholder": rolls if isinstance(rolls, str) else None,
                "constructs": [{"name": "none", "y2_derivable": False,
                                "reason": "file carries no rows"}],
                "channel": channel_stats([]),
            })
            continue
        finals = [(r.get("output_strs") or [""])[0] for r in rolls]
        mk = collections.Counter()
        for r in rolls:
            mk.update((r.get("metadata") or {}).keys())
        rec = {
            "corpus": "apollo",
            "file": fn,
            "base_name": blob.get("base_name"),
            "variant": blob.get("variant"),
            "model": blob.get("model"),
            "n_rows": len(rolls),
            "metadata_keys": dict(mk),
            "channel": channel_stats(finals),
            "n_distinct_system_prompts": len({
                json.dumps((r.get("input_messages") or [{}])[0], sort_keys=True)
                for r in rolls}),
            # the released label field is named `expected_label` in some files
            # and `expect_deception` in others; both are read, neither invented
            "label_field": ("expected_label"
                            if "expected_label" in rolls[0]
                            else "expect_deception"
                            if "expect_deception" in rolls[0] else None),
            "expected_label": dict(collections.Counter(
                str(_label_of(r)) for r in rolls)),
        }
        # the sandbagging pair is the only one with an objective answer key
        if mk.get("correct_answer", 0) == len(rolls) and mk.get("topic"):
            topics = [(r["metadata"] or {}).get("topic") for r in rolls]
            labels = [str(_label_of(r)) for r in rolls]
            items = [hashlib.md5(
                str((r["metadata"] or {}).get("question", "")).encode()
            ).hexdigest()[:12] for r in rolls]
            tab = {}
            for t in sorted(set(topics)):
                idx = [i for i in range(len(rolls)) if topics[i] == t]
                cnt = collections.Counter(labels[i] for i in idx)
                tab[t] = {"n": len(idx), "labels": dict(cnt),
                          "pure": len(cnt) == 1}
            rec["topic_by_label"] = tab
            rec["topic_label_collinear"] = all(v["pure"] for v in tab.values())
            rec["elicitation"] = {
                # the question hash is E-free by construction: it is the item
                # text, and `topic` is metadata about it, not text inside it
                "topic|question": crosstab_items_by_variant(
                    items, topics, e_free=True)}
            rec["constructs"] = [{
                "name": "task correctness (answer != metadata.correct_answer)",
                "y2_derivable": True,
                "y2_source": "metadata.correct_answer",
                # analyze_external_audit.py:170 defines the corpus's deception
                # label as exactly this, so it is not a second construct.
                "is_second_construct": False,
                "reason": "correctness IS this corpus's deception label: "
                          "analyze_external_audit.py:170 defines "
                          "`dec = answer_idx != correct_answer`. PREREG 5(d).",
            }]
        else:
            rec["constructs"] = [{
                "name": "none",
                "y2_derivable": False,
                "reason": "no objective answer key in released metadata; the "
                          "label is a deception grade",
            }]
        out.append(rec)
    return out


def group_e_designs():
    """PREREG 3.1 / 4: the four published designs, restated from
    tab:prior_work_criteria EXACTLY as already published. Not recomputed, not
    changed -- `source` says so, so no reader mistakes them for measurements."""
    return {
        "source": "table, not recomputed",
        "table": "tab:prior_work_criteria group (e)",
        "designs": [
            {"construct": "sycophancy", "cite": "sharma2023sycophancy",
             "i": "x", "ii": "p", "iii": "p", "iv": "check", "v": "n/a",
             "verdict": "n/a -- reports the elicitation effect AS one; no "
                        "per-response detector"},
            {"construct": "human red-teaming", "cite": "ganguli2022red",
             "i": "x", "ii": "x", "iii": "x", "iv": "p", "v": "x",
             "verdict": "x -- success rated by the red-teamer who chose the "
                        "attack"},
            {"construct": "reward tampering", "cite": "denison2024sycophancy",
             "i": "check", "ii": "check", "iii": "check", "iv": "check",
             "v": "check",
             "verdict": "n/a -- training, not the prompt, is the manipulation; "
                        "no detector score",
             "passes_all_five": True,
             "why": "the manipulation is training, so there is no do(E) at "
                    "evaluation time for the elicited behavior to descend "
                    "from, and (A4) does not bind"},
            {"construct": "alignment faking", "cite": "greenblatt2024alignment",
             "i": "x", "ii": "x", "iii": "check", "iv": "x", "v": "x",
             "verdict": "partly -- prompt-sensitivity ablations; authors state "
                        "the residual cause is unverified"},
        ],
    }


def design_rule(lb, ap):
    """PREREG 6: the three properties a designer controls, scored per corpus."""
    rows = []
    for rec in lb:
        crossed = any(v.get("req_i_separable")
                      for v in rec["elicitation"].values()
                      if isinstance(v, dict))
        rows.append({
            "corpus": f"liars_bench/{rec['config']}",
            "crosses_elicitation_with_items": bool(crossed),
            "records_response_channel": rec["channel"]["channel_recorded"],
            "grades_y_off_channel": any(
                c.get("y2_derivable") for c in rec["constructs"]),
        })
    for rec in ap:
        el = (rec.get("elicitation") or {}).get("topic|question")
        rows.append({
            "corpus": f"apollo/{rec['file']}",
            "crosses_elicitation_with_items": bool(
                el and el.get("req_i_separable")),
            "records_response_channel": rec["channel"]["channel_recorded"],
            "grades_y_off_channel": any(
                c.get("y2_derivable") for c in rec["constructs"]),
        })
    return rows


# ----------------------------------------------------------------- reporting

def summarize(lb, ap):
    """The two questions the paper's text depends on, answered from the census
    rather than from prose."""
    second = []
    for rec in lb:
        for c in rec["constructs"]:
            if c.get("is_second_construct"):
                second.append(f"liars_bench/{rec['config']}: {c['name']}")
    for rec in ap:
        for c in rec.get("constructs", []):
            if c.get("is_second_construct"):
                second.append(f"apollo/{rec['file']}: {c['name']}")

    # PREREG 5(a): a corpus that BOTH passes (i) on a non-deception construct
    # AND records a channel triggers the full detector audit.
    measurable = []
    for rec in lb:
        crossed = any(v.get("req_i_separable")
                      for v in rec["elicitation"].values()
                      if isinstance(v, dict))
        if crossed and rec["channel"]["channel_recorded"] and any(
                c.get("is_second_construct") for c in rec["constructs"]):
            measurable.append(f"liars_bench/{rec['config']}")

    # separately: corpora that cross elicitation with items at all, whatever
    # the construct -- the existence proof for design property (1)
    crossing = []
    for rec in lb:
        for k, v in rec["elicitation"].items():
            if isinstance(v, dict) and v.get("req_i_separable"):
                crossing.append({
                    "corpus": f"liars_bench/{rec['config']}", "key": k,
                    "n_items": v["n_items"], "n_variants": v["n_variants"],
                    "n_items_multi_E": v["n_items_multi_E"],
                    "channel_recorded": rec["channel"]["channel_recorded"]})

    return {
        "second_constructs_found": second,
        "second_construct_corpora_detail": [
            {"corpus": f"liars_bench/{rec['config']}", **{
                k: c.get(k) for k in
                ("name", "agreement_with_released_label", "y2_rate",
                 "released_label_rate", "n_belief_ne_key",
                 "anchor_released_label")}}
            for rec in lb for c in rec["constructs"]
            if c.get("is_second_construct")],
        "n_second_constructs": len(second),
        "magnitude_measurable_outside_deception": measurable,
        "prereg_branch": "a" if measurable else "b",
        "corpora_crossing_elicitation_with_items": crossing,
        "channel_recorded_count": sum(
            1 for r in lb + ap if r["channel"]["channel_recorded"]),
        "n_corpora": len(lb) + len(ap),
    }


def report(blob):
    lb, ap = blob["liars_bench"], blob["apollo"]
    print("=" * 78)
    print("EXP-AF -- ECCP decidability census (DESCRIPTIVE, not confirmatory)")
    print(f"pre-registration: {PREREG}")
    print("=" * 78)

    print("\n--- Liars' Bench ---")
    for r in lb:
        ch = r["channel"]
        print(f"\n{r['config']}  n={r['n_rows']}")
        print(f"  channel: distinct={ch['n_distinct_final_messages']} "
              f"ratio={ch['channel_ratio']:.4f} median_len={ch['median_len']} "
              f"-> recorded={ch['channel_recorded']}")
        if r["parse_meta_recovered"]:
            print("  NOTE: meta recovered only by parse_meta_np "
                  "(existing parse_meta returns {})")
        ik = r["item_keys"]
        print("  item keys: " + ", ".join(
            f"{k}={ik['n_distinct'][k]}"
            + ("" if ik["admissible"][k]
               else " [vacuous]" if ik["vacuous"][k] else " [over-merges]")
            for k in ik["n_distinct"]))
        if r.get("req_i_verdict"):
            print("  (i) VERDICT: " + ", ".join(
                f"{k}={v}" for k, v in r["req_i_verdict"].items()))
        for k, v in r["elicitation"].items():
            if isinstance(v, dict) and "req_i_separable" in v:
                print(f"  (i) {k}: items={v['n_items']} E={v['n_variants']} "
                      f"items_multi_E={v['n_items_multi_E']} "
                      f"({v['share_items_multi_E']:.4f}) "
                      f"-> separable={v['req_i_separable']}"
                      + ("" if v["item_key_admissible"]
                         else "  [key inadmissible; crossing not credited]"))
        for c in r["constructs"]:
            print(f"  Y2 {c['name']}: derivable={c.get('y2_derivable')}"
                  + (f" second_construct={c.get('is_second_construct')}"
                     if c.get("y2_derivable") else ""))
            if c.get("reason"):
                print(f"     reason: {c['reason']}")
            if c.get("anchor_released_label"):
                a = c["anchor_released_label"]
                print(f"     anchor: {a['rule']} -> agreement="
                      f"{a['agreement']} exact={a['exact']}")
                print(f"     Y2 rate={c['y2_rate']} vs released="
                      f"{c['released_label_rate']}; agreement="
                      f"{c['agreement_with_released_label']}")

    print("\n--- Apollo ---")
    for r in ap:
        if r.get("rows_absent"):
            print(f"{r['file']}  NO ROWS -- rollouts field is the literal "
                  f"string {r['rows_placeholder']!r}")
            continue
        ch = r["channel"]
        print(f"{r['file']}  n={r['n_rows']} sys_prompts="
              f"{r['n_distinct_system_prompts']} "
              f"channel_recorded={ch['channel_recorded']}"
              + (f" topic_label_collinear={r['topic_label_collinear']}"
                 if "topic_label_collinear" in r else ""))

    s = blob["summary"]
    print("\n" + "=" * 78)
    print(f"second constructs found: {s['n_second_constructs']} "
          f"{s['second_constructs_found']}")
    print(f"magnitude measurable outside deception: "
          f"{s['magnitude_measurable_outside_deception'] or 'NONE'}")
    print(f"PREREG branch: ({s['prereg_branch']})")
    pw = s["design_rule_piecewise"]
    print("\ndesign rule, piecewise across the pinned corpora:")
    for p, v in pw["corpora_per_property"].items():
        print(f"  {p}: {len(v)} corpora  {v[:4]}{' ...' if len(v) > 4 else ''}")
    print(f"  every property met by SOME corpus: "
          f"{pw['every_property_met_somewhere']}")
    print(f"  corpora meeting ALL THREE: "
          f"{pw['corpora_satisfying_all_three'] or 'NONE'}")
    print("\ncorpora that DO cross elicitation with items "
          "(design property 1, existence proof):")
    for c in s["corpora_crossing_elicitation_with_items"]:
        print(f"  {c['corpus']} via {c['key']}: {c['n_items_multi_E']}/"
              f"{c['n_items']} items under >1 of {c['n_variants']} variants, "
              f"channel_recorded={c['channel_recorded']}")
    print("=" * 78)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--json-only", action="store_true")
    args = ap_.parse_args()

    lb = census_liars_bench()
    apollo = census_apollo()
    blob = {
        "experiment": "EXP-AF",
        "prereg": PREREG,
        "status": "descriptive, not confirmatory (PREREG section 0)",
        "census_not_estimate": True,
        "thresholds": {"CHANNEL_MIN": CHANNEL_MIN,
                       "ANCHOR_EXACT": ANCHOR_EXACT},
        "liars_bench": lb,
        "apollo": apollo,
        "group_e": group_e_designs(),
        "design_rule": design_rule(lb, apollo),
    }
    blob["summary"] = summarize(lb, apollo)

    # The headline of branch (b), computed rather than asserted: each of the
    # three design-rule properties of PREREG section 6 is satisfied by SOME
    # pinned public corpus, and no corpus satisfies all three at once. That is
    # what makes the standard non-vacuous without a magnitude to report --
    # assembling an auditable benchmark needs no new idea, only the combination
    # of three choices different existing benchmarks already made.
    props = ("crosses_elicitation_with_items", "records_response_channel",
             "grades_y_off_channel")
    piecewise = {p: [r["corpus"] for r in blob["design_rule"] if r[p]]
                 for p in props}
    blob["summary"]["design_rule_piecewise"] = {
        "satisfied_by_some_corpus": {p: len(v) for p, v in piecewise.items()},
        "corpora_per_property": piecewise,
        "corpora_satisfying_all_three": [
            r["corpus"] for r in blob["design_rule"] if all(r[p] for p in props)],
        "every_property_met_somewhere": all(piecewise.values()),
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(blob, fh, indent=2, sort_keys=True)
    if not args.json_only:
        report(blob)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
