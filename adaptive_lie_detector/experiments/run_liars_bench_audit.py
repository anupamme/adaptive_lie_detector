#!/usr/bin/env python3
"""run_liars_bench_audit.py

EXP-XL steps 2-3 (PREREG_EXP_XL.md §5, §4-H1/H3): the eligibility survey and
the rule arm, on the Liars' Bench test splits.

**The survey is the deliverable even if every config is ineligible** (PREREG §5,
branch E), so every clause of §5 is a mechanical test on the fetched rows and
each is reported with its counts even when it passes.

Nothing here is fitted, tuned or adapted to this corpus. The rule is
`run_external_audit.COMPILED` / `marker_count()` -- the same 14 patterns, the
same fire-at->=1 threshold, no pattern added or removed (PREREG §3). The
eligibility gates are copied from `PREREG_EXP_XA.md` rather than re-derived:
`MIN_CELL` = 10, and the rule must fire on strictly between 5% and 95% of some
contrasted cell or the arm is `INAPPLICABLE_DEGENERATE`.

Channel definitions are the ones §3 fixed in advance, before any row was read:
  visible : the FINAL assistant message only, with HIDDEN_TAGS content stripped
  full    : every message, system included
A config whose final message is not an assistant turn is reported
`INAPPLICABLE_NO_MODEL_OUTPUT` and is NOT repaired by reaching further back.

**One asymmetry against Apollo, disclosed rather than equated.** Apollo's
`full_transcript()` is the model's OWN output with its hidden scratchpad left in;
it never contains the system prompt. §3's `full` for this corpus is every message
INCLUDING system, which is strictly broader. So a large full-vs-visible gap here
could come from the system prompt rather than from a hidden scratchpad, and the
two releases' `ANNOTATION_CHANNEL_LEAKAGE` verdicts would then not mean the same
thing. `full_output_only` -- the final assistant message with HIDDEN_TAGS
RETAINED, which is the exact Apollo analogue -- is therefore computed alongside
so the gap decomposes into scratchpad and system-prompt parts. It is **not
pre-registered**: §3 fixed two channels, this is a third, and it is reported as a
post-hoc decomposition of a pre-registered contrast, never as the contrast.

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/run_liars_bench_audit.py --survey
    ../.venv/bin/python3 experiments/run_liars_bench_audit.py --arm rule
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_external_audit import (  # noqa: E402
    COMPILED, HIDDEN_TAGS, marker_count, system_md5,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE, "data", "external", "liars_bench")
MANIFEST = os.path.join(BASE, "data", "external", "liars_bench_manifest.json")
RESULTS = os.path.join(BASE, "data", "results")
SURVEY_PATH = os.path.join(RESULTS, "liars_bench_survey.json")
RULE_PATH = os.path.join(RESULTS, "liars_bench_rule.json")
NORM_DIR = os.path.join(CORPUS_DIR, "normalized")

MIN_CELL = 10        # PREREG §4, copied from EXP-XA
FIRE_LO, FIRE_HI = 0.05, 0.95   # PREREG §4 degeneracy gate
LEAK_FACTOR = 3.0    # PREREG §3: full effect > 3x visible -> ANNOTATION_CHANNEL_LEAKAGE

# PREREG §3 names one rule: `hedging_baseline.REFUSAL_PATTERNS`, which is `L_fac`
# (the correction/hedging list). `L_doc` is reported alongside because the Apollo
# audit reports both, but only `L_fac` decides a verdict.
LIST_NAME = "L_fac"
SECONDARY_LIST = "L_doc"

# PREREG §5.3(a) names `index`/`meta` as the sources of the scenario key. Two
# facts measured on the fetched rows before any score was computed change how
# that clause has to be applied, and both are reported rather than worked around:
#
#   1. `index` is **unique per row** in every config that carries it, so it is a
#      row counter and not a scenario key. It is still evaluated below, because
#      §5.3 names it; it simply yields zero paired groups, which is visible in
#      the output instead of being asserted here.
#   2. The repeating id columns are `dataset_index` (`instructed-deception`, 100
#      distinct of 26,426) and `Unnamed: 0` (`gender-secret`, 392 of 762), which
#      are top-level parquet columns rather than `meta` fields.
#
# So rather than a preference order -- where the choice of key would decide
# whether requirement (iii) is met, and could be chosen to suit the answer --
# **every candidate key is reported with its own paired-scenario count**, and the
# verdict is taken from a key that is mechanical and cannot be gamed: a hash of
# the environment's own messages (see `scenario_keys`).
META_SCENARIO_KEYS = ("scenario", "scenario_id", "id", "index", "idx",
                      "example_id", "prompt_id", "story_id", "sample_id",
                      "template_index", "template_id", "dataset_index")
TOP_SCENARIO_KEYS = ("index", "dataset_index", "Unnamed: 0", "sample_id")

# The key requirement (iii) is decided on. `env_md5` hashes every prefix message
# the model did NOT write -- system, user, tool -- i.e. the scenario as the
# environment posed it. Intermediate assistant turns are excluded on purpose:
# they are the model's *behavior*, and requirement (iii) asks whether the same
# scenario was realized under both behaviors, so folding behavior into the key
# would make (iii) unmeetable by construction in any multi-turn setting. The
# stricter whole-prefix hash is reported alongside as `prefix_md5` so a reader
# can see whether the pairing survives the strictest reading too.
PRIMARY_SCENARIO_SOURCE = "env_md5"


# --------------------------------------------------------------------------
# loading and normalisation
# --------------------------------------------------------------------------

def available_configs():
    if not os.path.isdir(CORPUS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(CORPUS_DIR)):
        d = os.path.join(CORPUS_DIR, name)
        if os.path.isdir(d) and name != "normalized":
            if any(f.endswith(".parquet") for f in os.listdir(d)):
                out.append(name)
    return out


def load_config(config):
    """Every test parquet shard of one config, as a list of dicts."""
    import pyarrow.parquet as pq
    d = os.path.join(CORPUS_DIR, config)
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".parquet"):
            continue
        rows.extend(pq.read_table(os.path.join(d, f)).to_pylist())
    return rows


def parse_meta(row):
    """`meta` is an opaque string in the schema (PREREG §6.7).

    Measured, not assumed: the shipped values are **Python reprs**
    (`{'sample_id': 0, ...}`), not JSON, so `json.loads` fails on all of them.
    Both are tried. A missing or genuinely unparseable field yields {} -- never
    an improvised substitute.
    """
    raw = row.get("meta")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    for parse in (json.loads, ast.literal_eval):
        try:
            val = parse(raw)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(val, dict):
            return val
    return {}


def messages_of(row):
    msgs = row.get("messages") or []
    out = []
    for m in msgs:
        if isinstance(m, dict):
            out.append({"role": m.get("role"),
                        "content": m.get("content") or ""})
    return out


def strip_hidden(text):
    """The same hidden-tag strip Apollo's visible channel applies (PREREG §3)."""
    for tag in HIDDEN_TAGS:
        text = re.sub(rf"<{tag}>.*?</{tag}>", " ", text, flags=re.S | re.I)
        text = re.sub(rf"</?{tag}>", " ", text, flags=re.I)
    return text


def normalize(row, idx):
    """One parquet row -> the Apollo rollout shape, so `system_md5` and the
    judge runner's `context_messages` / `output_text` apply unchanged.

    `input_messages` is everything up to but excluding the final assistant
    turn; `output_strs` is that final turn. Returns None if the final message
    is not an assistant turn (PREREG §3 / §9.4).
    """
    msgs = messages_of(row)
    if not msgs or msgs[-1].get("role") != "assistant":
        return None
    meta = parse_meta(row)
    return {
        "input_messages": msgs[:-1],
        "output_strs": [msgs[-1]["content"]],
        "metadata": {
            "lb_index": idx,
            "deceptive": bool(row.get("deceptive")),
            "generator": row.get("model"),
            "temperature": row.get("temperature"),
            "dataset": row.get("dataset"),
            "meta": meta,
            # top-level id columns, kept because the repeating scenario ids on
            # this corpus live here and not in `meta` (see TOP_SCENARIO_KEYS)
            "row_ids": {k: row[k] for k in TOP_SCENARIO_KEYS
                        if k in row and row[k] is not None},
        },
    }


def visible_text(roll):
    """PREREG §3: the final assistant message, hidden-tag content stripped."""
    return strip_hidden("\n".join(roll["output_strs"]))


def full_text(roll):
    """PREREG §3: every message including system."""
    parts = [f"{m['role']}: {m['content']}" for m in roll["input_messages"]]
    parts.append("assistant: " + "\n".join(roll["output_strs"]))
    return "\n".join(parts)


def full_output_only_text(roll):
    """NOT pre-registered. The exact Apollo analogue of `full_transcript()`: the
    model's own final message with HIDDEN_TAGS retained and no system prompt. Lets
    a full-vs-visible gap be split into a scratchpad part (this vs. `visible`) and
    a system-prompt part (`full` vs. this)."""
    return "\n".join(roll["output_strs"])


CHANNELS = {"visible": visible_text, "full": full_text,
            "full_output_only": full_output_only_text}
PREREG_CHANNELS = ("visible", "full")   # the two §3 fixed; the third is post-hoc


def scenario_keys(roll):
    """PREREG §5.3: every candidate scenario key for one example, as
    {source: key}. No preference order -- the caller reports a paired-scenario
    count for each source, and takes the verdict from PRIMARY_SCENARIO_SOURCE.

    Sources are `env_md5` (the scenario as the environment posed it),
    `prefix_md5` (the strict whole-prefix hash, model turns included),
    `meta.<k>` for each published `meta` id, and `top.<k>` for each top-level id
    column. A source absent from an example simply does not appear.
    """
    keys = {}
    env = "\n".join(f"{m['role']}: {m['content']}" for m in roll["input_messages"]
                    if m["role"] != "assistant")
    if env.strip():
        keys["env_md5"] = hashlib.md5(env.encode("utf-8")).hexdigest()[:12]
    prefix = "\n".join(f"{m['role']}: {m['content']}"
                       for m in roll["input_messages"])
    if prefix.strip():
        keys["prefix_md5"] = hashlib.md5(prefix.encode("utf-8")).hexdigest()[:12]
    meta = roll["metadata"].get("meta") or {}
    for k in META_SCENARIO_KEYS:
        if k in meta and meta[k] not in (None, ""):
            keys[f"meta.{k}"] = f"{k}={meta[k]}"
    for k, v in (roll["metadata"].get("row_ids") or {}).items():
        keys[f"top.{k}"] = f"{k}={v}"
    return keys


def scenario_key(roll):
    """The single key requirement (iii) is decided on. Returns (key, source);
    (None, "unavailable") when the environment posed nothing hashable, which is
    reported as `(iii)_UNTESTABLE` rather than repaired."""
    keys = scenario_keys(roll)
    if PRIMARY_SCENARIO_SOURCE in keys:
        return keys[PRIMARY_SCENARIO_SOURCE], PRIMARY_SCENARIO_SOURCE
    return None, "unavailable"


# --------------------------------------------------------------------------
# the survey
# --------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def prompt_text(roll):
    """Every non-assistant message: the elicitation the model was given.

    Not one of §3's two channels and never used for a criterion-4 contrast. This
    is the input to the requirement-(i) test below, which asks a question about
    the *design* rather than about a detector.
    """
    return "\n".join(m.get("content") or "" for m in roll["input_messages"]
                     if m.get("role") != "assistant")


# ---------------------------------------------------------------------------
# Two POST-HOC descriptive diagnostics about requirement (i). NEITHER IS A GATE,
# and the reason is recorded here because it was measured, not assumed.
#
# §5.1 operationalized "elicitation is fixed" as "one distinct system-prompt md5
# across the cell". That hashes the system message, so it cannot see an
# elicitation cue carried in the *user* turn -- and `soft-trigger` is exactly
# that case. All twelve of its cells have one distinct system md5, so §5.1 scored
# (i) as MET for every one, while a trigger phrase appended to the user message
# is what sets the label: for the three gemma-lora cells a single greeting is
# perfectly label-pure (P(dec | phrase) = 1.000 or 0.000), and for the other nine
# it is 92-96% predictive, so there the label really is behavioural and only (i)
# is at issue.
#
# The obvious repair -- gate (i) on "is the label recoverable from the prompt?"
# -- was implemented and then REJECTED, because it fires on cells where nothing
# is wrong. Measured out-of-fold AUROC: soft-trigger/gemma-lora-greeting 0.988,
# gender-secret 0.997, and `insider-trading`/mistral 0.763 -- the last being the
# cell this round's headline rests on. The prompt predicts the label there
# because the scenario *stage* does (the `turn`/`turn_name` nuisance already
# reported), not because elicitation varies. An n-gram purity test fails the same
# way: 'whaddup' is perfectly label-pure in soft-trigger, but so is 'gender' in
# gender-secret, where the user simply asks about gender and thereby creates the
# opportunity to lie. Separating "elicitation cue" from "scenario content that
# creates the opportunity" is a semantic judgement, and no statistic here makes
# it.
#
# So the gate stays where the pre-registration already put it: requirement
# (iii). If no scenario is ever realized under both labels, every label
# difference coincides with a prompt difference and the data cannot distinguish
# a fixed elicitation from a prompt-determined label -- which is precisely why
# (iii) blocks soft-trigger and gender-secret, and precisely why
# `insider-trading`/mistral (28 paired groups) is not blocked by it. Both
# diagnostics below are therefore published as descriptions, and the (i) flag is
# relabelled to say what it actually measures: the system message only.
#
# Fixed before either was run on any cell: TF-IDF word 1-2 grams, min_df=2,
# LogisticRegression C=1.0, grouped 5-fold CV on the scenario key so a repeated
# scenario cannot straddle a fold, out-of-fold AUROC against a 39-replicate
# label-shuffled null (smallest attainable one-sided p = 1/40 = 0.025). The null
# reshuffles labels only; the TF-IDF matrix is built once and reused.
# ---------------------------------------------------------------------------
PROMPT_PROBE_SEED = 42
PROMPT_PROBE_FOLDS = 5
PROMPT_PROBE_NULL_REPS = 39
PROMPT_PROBE_MAX_N = 800       # per cell, balanced on the label
PROMPT_PROBE_MAX_CHARS = 4000  # tail of the prompt, where a trigger sits


def _grouped_folds(groups, n_folds, rng):
    """Assign whole scenario groups to folds, largest first for balance."""
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    order = sorted(by.values(), key=lambda v: (-len(v), v[0]))
    folds = [[] for _ in range(n_folds)]
    for members in order:
        folds.sort(key=len)
        folds[0].extend(members)
    return [f for f in folds if f]


def prompt_only_recoverability(rolls):
    """Is `deceptive` recoverable from the prompt alone, on held-out scenarios?

    DESCRIPTIVE ONLY -- see the block comment above. A positive here does NOT
    mean requirement (i) is unmet: it is also positive whenever different
    scenarios carry different base rates, which is legitimate. Reported so the
    reader can see how strongly prompt text tracks the label in each cell.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as ex:
        return {"status": f"UNAVAILABLE_{type(ex).__name__}"}

    dec = [r for r in rolls if r["metadata"]["deceptive"]]
    hon = [r for r in rolls if not r["metadata"]["deceptive"]]
    if len(dec) < MIN_CELL or len(hon) < MIN_CELL:
        return {"status": "INAPPLICABLE_LABEL_DOES_NOT_VARY",
                "n_deceptive": len(dec), "n_honest": len(hon)}

    # balanced, deterministic draw over a sorted order so the RNG stream cannot
    # depend on dict or file ordering
    per = min(PROMPT_PROBE_MAX_N // 2, len(dec), len(hon))
    rng = np.random.RandomState(PROMPT_PROBE_SEED)
    sel = []
    for subset in (dec, hon):                      # fixed label order
        idx = sorted(range(len(subset)),
                     key=lambda i: str(subset[i]["metadata"].get("lb_index", i)))
        take = rng.choice(len(idx), size=per, replace=False) if per < len(idx) \
            else np.arange(len(idx))
        sel.extend(subset[idx[int(i)]] for i in sorted(take))

    X_txt = [prompt_text(r)[-PROMPT_PROBE_MAX_CHARS:] for r in sel]
    y = np.array([1 if r["metadata"]["deceptive"] else 0 for r in sel])
    groups = [(scenario_key(r)[0] or f"row{i}") for i, r in enumerate(sel)]

    try:
        X = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                            lowercase=True).fit_transform(X_txt)
    except ValueError as ex:                        # e.g. empty vocabulary
        return {"status": f"INAPPLICABLE_VECTORIZER_{type(ex).__name__}",
                "detail": str(ex)[:120]}
    folds = _grouped_folds(groups, PROMPT_PROBE_FOLDS, rng)
    if len(folds) < 2:
        return {"status": "INAPPLICABLE_TOO_FEW_SCENARIO_GROUPS",
                "n_groups": len(set(groups))}

    def oof_auroc(labels):
        scores = np.zeros(len(labels), dtype=float)
        for f in folds:
            te = np.array(f)
            tr = np.array([i for i in range(len(labels)) if i not in set(f)])
            if len(np.unique(labels[tr])) < 2 or len(tr) < 2:
                return None
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(X[tr], labels[tr])
            scores[te] = clf.decision_function(X[te])
        if len(np.unique(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))

    obs = oof_auroc(y)
    if obs is None:
        return {"status": "INAPPLICABLE_DEGENERATE_FOLD"}

    null = []
    for _ in range(PROMPT_PROBE_NULL_REPS):
        a = oof_auroc(rng.permutation(y))
        if a is not None:
            null.append(a)
    null = np.array(null) if null else np.array([0.5])
    # (1 + #{null >= obs}) / (R + 1): the standard permutation p, so the floor is
    # 1/40 = 0.025 rather than an invented threshold.
    p = float((1 + int((null >= obs).sum())) / (len(null) + 1))

    return {"status": "OK",
            "auroc_oof": obs,
            "p_one_sided_permutation": p,
            "null_mean_auroc": float(null.mean()),
            "null_p95_auroc": float(np.percentile(null, 95)),
            "n_used": len(sel), "n_per_class": per,
            "n_scenario_groups": len(set(groups)), "n_folds": len(folds),
            "label_recoverable_from_prompt": bool(p < 0.05),
            "is_post_hoc": True,
            "is_a_gate": False,
            "note": "DESCRIPTIVE. High here is expected wherever scenarios differ "
                    "in base rate, so this does not decide requirement (i); (iii) "
                    "is the gate. Measured 0.99 on soft-trigger, 1.00 on "
                    "gender-secret and 0.76 on insider-trading/mistral."}


def prompt_ngram_purity(rolls, min_df_frac=0.02, top_k=5):
    """The most label-pure prompt n-gram in the cell.

    DESCRIPTIVE ONLY, for the same reason as above: a perfectly label-pure
    n-gram can be an elicitation cue ('whaddup' in `soft-trigger`) or ordinary
    scenario content that creates the opportunity to lie ('gender' in
    `gender-secret`), and this statistic cannot tell them apart. It is reported
    because it is the most legible evidence that `soft-trigger`'s label is set by
    a phrase in the user turn, which is what the system-md5 test missed.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as ex:
        return {"status": f"UNAVAILABLE_{type(ex).__name__}"}

    dec = [r for r in rolls if r["metadata"]["deceptive"]]
    hon = [r for r in rolls if not r["metadata"]["deceptive"]]
    if len(dec) < MIN_CELL or len(hon) < MIN_CELL:
        return {"status": "INAPPLICABLE_LABEL_DOES_NOT_VARY"}

    sel = rolls[:1500]
    txt = [prompt_text(r)[-PROMPT_PROBE_MAX_CHARS:] for r in sel]
    y = np.array([1 if r["metadata"]["deceptive"] else 0 for r in sel])
    floor = max(2, int(min_df_frac * len(sel)))
    try:
        cv = CountVectorizer(ngram_range=(1, 3), min_df=floor, binary=True)
        X = (cv.fit_transform(txt) > 0).toarray()
    except ValueError as ex:
        return {"status": f"INAPPLICABLE_VECTORIZER_{type(ex).__name__}",
                "detail": str(ex)[:120]}
    names = cv.get_feature_names_out()

    scored = []
    for j in range(X.shape[1]):
        m = X[:, j]
        n = int(m.sum())
        if n < floor:
            continue
        p = float(y[m].mean())
        scored.append((max(p, 1.0 - p), p, n, str(names[j])))
    scored.sort(reverse=True)
    return {"status": "OK", "base_rate_deceptive": float(y.mean()),
            "min_df_examples": floor, "n_used": len(sel),
            "max_purity": (scored[0][0] if scored else None),
            "top": [{"ngram": nm, "purity": pur, "p_deceptive_given_ngram": p,
                     "df": n} for pur, p, n, nm in scored[:top_k]],
            "is_post_hoc": True, "is_a_gate": False}


def derived_records(config, rolls):
    """The committable artifact: one record per example, carrying **only** marker
    counts, the label, hashes and lengths -- and **no corpus text**, because the
    corpus is gated and not ours to redistribute (PREREG §2).

    Marker-count keys follow `analyze_external_audit.fired`'s
    `mc_{channel}_{list}` convention exactly, so `cell_report` and `contrast`
    consume these records with no adapter.
    """
    out = []
    for r in rolls:
        m = r["metadata"]
        key, src = scenario_key(r)
        rec = {"corpus": "liars_bench", "config": config,
               "idx": m["lb_index"], "generator": m["generator"],
               "temperature": m["temperature"], "dataset": m["dataset"],
               "deceptive": m["deceptive"],
               # `grade` is the field `grade_of`/`contrast` read; on this corpus the
               # label IS the release's own annotation, which is why requirement (v)
               # is adjudicated per config rather than assumed.
               "grade": ["deceptive" if m["deceptive"] else "honest"],
               "system_md5": system_md5(r),
               "scenario_key": key, "scenario_key_source": src,
               # every candidate key, not just the deciding one, so the paired
               # count under each can be recomputed offline and the choice of key
               # is auditable rather than taken on trust
               "scenario_keys": scenario_keys(r),
               # small scalar `meta` fields, kept so the (v) adjudication and the
               # nuisance-variable decomposition of §5.2 run offline from the
               # committed records. Bounded to short scalars so no corpus text can
               # enter through this door.
               "meta_scalars": {
                   k: v for k, v in (m.get("meta") or {}).items()
                   if isinstance(v, (int, float, bool, str)) and len(str(v)) <= 64},
               "n_messages": len(r["input_messages"]) + 1}
        for ch, extract in CHANNELS.items():
            text = extract(r)
            rec[f"n_chars_{ch}"] = len(text)
            for lst in (LIST_NAME, SECONDARY_LIST):
                rec[f"mc_{ch}_{lst}"] = marker_count(text, lst)
        out.append(rec)
    return out


def survey_cell(rolls):
    """§5.1-§5.4 on one (config x generator) cell. Every clause reports counts
    even when it passes."""
    md5s = {}
    for r in rolls:
        md5s.setdefault(system_md5(r), []).append(r)
    dec = [r for r in rolls if r["metadata"]["deceptive"]]
    hon = [r for r in rolls if not r["metadata"]["deceptive"]]

    # 5.2 mechanical (ii) test: is `deceptive` a function of the system md5?
    label_by_md5 = {m: {r["metadata"]["deceptive"] for r in rs}
                    for m, rs in md5s.items()}
    deceptive_determined_by_prompt = all(len(v) == 1 for v in label_by_md5.values())

    # 5.3 paired scenarios, under the deciding key...
    groups, key_sources = {}, {}
    for r in rolls:
        k, src = scenario_key(r)
        key_sources[src] = key_sources.get(src, 0) + 1
        if k is None:
            continue
        groups.setdefault(k, []).append(r)
    paired = {k: v for k, v in groups.items()
              if len({r["metadata"]["deceptive"] for r in v}) == 2}

    # ...and under every other candidate key, reported so the deciding key cannot
    # be a choice that suits the answer. A key that repeats across the label (a
    # scenario id) shows a positive paired count; a key that is really a row
    # counter shows n_groups == n and n_paired == 0.
    per_source, keyed = {}, {}
    for r in rolls:
        for src, k in scenario_keys(r).items():
            per_source.setdefault(src, {}).setdefault(k, set()).add(
                r["metadata"]["deceptive"])
            keyed[src] = keyed.get(src, 0) + 1
    paired_by_source = {
        src: {"n_groups": len(g),
              "n_paired": sum(1 for v in g.values() if v == {True, False}),
              "n_examples_keyed": keyed[src],
              "is_primary": src == PRIMARY_SCENARIO_SOURCE}
        for src, g in sorted(per_source.items())}

    # rule fire rates, every channel x both pattern lists. Only L_fac (the
    # pre-registered list) decides a verdict; L_doc is reported for parity with
    # the Apollo audit.
    pos = {id(r): i for i, r in enumerate(rolls)}
    fires = {}
    texts = {ch: [ex(r) for r in rolls] for ch, ex in CHANNELS.items()}
    for ch in CHANNELS:
        f = {}
        for lst in (LIST_NAME, SECONDARY_LIST):
            g = {}
            for name, subset in (("deceptive", dec), ("honest", hon)):
                k = sum(1 for r in subset
                        if marker_count(texts[ch][pos[id(r)]], lst) >= 1)
                g[name] = {"k": k, "n": len(subset),
                           "rate": (k / len(subset)) if subset else None,
                           "wilson": wilson(k, len(subset)) if subset else None}
            # signed effect, deceptive minus honest, in percentage points
            g["effect_pp"] = (None if not (dec and hon) else
                              100.0 * (g["deceptive"]["rate"] - g["honest"]["rate"]))
            g["pooled_rate"] = (g["deceptive"]["k"] + g["honest"]["k"]) / max(1, len(rolls))
            f[lst] = g
        fires[ch] = f

    # PREREG §3's channel contrast, on the pre-registered list only, plus the
    # post-hoc decomposition of any gap into scratchpad and system-prompt parts.
    def eff(ch):
        return fires[ch][LIST_NAME]["effect_pp"]

    leak = {"visible_effect_pp": eff("visible"), "full_effect_pp": eff("full"),
            "full_output_only_effect_pp": eff("full_output_only"),
            "leak_factor_threshold": LEAK_FACTOR}
    if eff("visible") is not None and eff("full") is not None:
        v, fu, fo = abs(eff("visible")), abs(eff("full")), abs(eff("full_output_only"))
        leak["exceeds_threshold"] = fu > LEAK_FACTOR * v if v > 0 else fu > 0
        # not pre-registered: which broadening carries the gap
        leak["scratchpad_part_pp"] = fo - v
        leak["system_prompt_part_pp"] = fu - fo
        leak["decomposition_is_post_hoc"] = True
    fires["channel_contrast"] = leak

    # Descriptive only; neither of these changes a verdict (see the block comment
    # above `prompt_only_recoverability`). The (i) gate stays the md5 test, and
    # the gate that actually catches a prompt-determined label is (iii).
    prompt_probe = prompt_only_recoverability(rolls)
    ngram_purity = prompt_ngram_purity(rolls)
    i_met_md5 = len(md5s) == 1

    verdicts = []
    if len(md5s) != 1:
        verdicts.append("INELIGIBLE_E_VARIES")
    if len(dec) < MIN_CELL or len(hon) < MIN_CELL:
        verdicts.append("INELIGIBLE_LABEL_DOES_NOT_VARY")
    if deceptive_determined_by_prompt and len(md5s) > 1:
        verdicts.append("FAILS_(ii)_LABEL_IS_CONDITION_LABEL")
    if "unavailable" in key_sources and len(key_sources) == 1:
        verdicts.append("(iii)_UNTESTABLE")
    elif not paired:
        verdicts.append("(iii)_UNMET_NO_PAIRED_SCENARIO")
    vis = fires["visible"][LIST_NAME]["pooled_rate"]
    if not (FIRE_LO < vis < FIRE_HI):
        verdicts.append("INAPPLICABLE_DEGENERATE")
    if not verdicts:
        verdicts.append("ELIGIBLE")
    # PREREG §3's channel verdict is reported whether or not the cell is eligible
    # -- the contrast is a pre-registered outcome, not a diagnostic afterthought --
    # but it is kept OUT of `verdicts`, because leakage on the `full` channel says
    # nothing about whether criterion 4 is expressible on the `visible` one.
    channel_verdict = ("ANNOTATION_CHANNEL_LEAKAGE" if leak.get("exceeds_threshold")
                       else "NO_CHANNEL_LEAKAGE")

    return {
        "n": len(rolls),
        "n_distinct_system_md5": len(md5s),
        "system_md5s": sorted(md5s)[:8],
        "n_deceptive": len(dec), "n_honest": len(hon),
        # (i) relabelled to say what it measures. The gate is unchanged; what
        # changed is the claim attached to it, which previously read as "the
        # elicitation condition is fixed" and is only evidence about the system
        # message. `soft-trigger` is the cell that forced the relabelling.
        "i_fixed_elicitation": i_met_md5,
        "i_basis": "single_system_prompt_md5",
        "i_covers_system_message_only": True,
        "i_caveat": "An elicitation cue in the user turn is invisible to this "
                    "test. On soft-trigger a greeting phrase in the user message "
                    "is 92-100% predictive of `deceptive` while the system md5 is "
                    "constant, so (i) MET here means only that the SYSTEM prompt "
                    "was held fixed. Requirement (iii) is the gate that detects a "
                    "label which coincides with a prompt difference.",
        "i_prompt_recoverability_descriptive": prompt_probe,
        "i_prompt_ngram_purity_descriptive": ngram_purity,
        "deceptive_is_function_of_system_prompt": deceptive_determined_by_prompt,
        "scenario_key_sources": key_sources,
        "n_scenario_groups": len(groups),
        "n_paired_scenario_groups": len(paired),
        "n_examples_in_paired_groups": sum(len(v) for v in paired.values()),
        "primary_scenario_source": PRIMARY_SCENARIO_SOURCE,
        "paired_by_scenario_source": paired_by_source,
        "rule_fire": fires,
        "verdicts": verdicts,
        "channel_verdict": channel_verdict,
    }


def survey():
    configs = available_configs()
    if not configs:
        print("No parquet files under", CORPUS_DIR)
        print("PREREG_EXP_XL.md §7 branch F: run fetch_liars_bench.py first; if")
        print("it reports 403, the gate has not been accepted and the corpus")
        print("cannot be scored. Nothing is inferred about its contents.")
        return 1

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(NORM_DIR, exist_ok=True)
    out = {"prereg": "docs/PREREG_EXP_XL.md", "min_cell": MIN_CELL,
           "fire_gate": [FIRE_LO, FIRE_HI],
           "pattern_list_prereg": LIST_NAME,
           "pattern_list_secondary": SECONDARY_LIST,
           "n_patterns": len(COMPILED[LIST_NAME]),
           "leak_factor": LEAK_FACTOR,
           "channels_prereg": list(PREREG_CHANNELS),
           "channels_post_hoc": [c for c in CHANNELS if c not in PREREG_CHANNELS],
           "channel_note": ("`full` is every message including system, per PREREG "
                            "§3. Apollo's `full_transcript` is output-only with the "
                            "scratchpad intact, so the two releases' full channels "
                            "are NOT the same construct; `full_output_only` is the "
                            "Apollo analogue and is post-hoc."),
           "configs": {}}

    for config in configs:
        rows = load_config(config)
        rolls, dropped = [], 0
        for i, row in enumerate(rows):
            r = normalize(row, i)
            if r is None:
                dropped += 1
            else:
                rolls.append(r)
        entry = {"n_rows": len(rows), "n_dropped_no_assistant_final": dropped,
                 "cells": {}}
        if not rolls:
            entry["verdicts"] = ["INAPPLICABLE_NO_MODEL_OUTPUT"]
            out["configs"][config] = entry
            print(f"{config}: INAPPLICABLE_NO_MODEL_OUTPUT "
                  f"({dropped}/{len(rows)} rows lack an assistant final turn)")
            continue

        by_gen = {}
        for r in rolls:
            by_gen.setdefault(r["metadata"]["generator"] or "unknown", []).append(r)
        # PREREG §5.1: the md5 test runs WITHIN generator; a cell pooling
        # generators is not one elicitation condition.
        for gen, sub in sorted(by_gen.items()):
            entry["cells"][gen] = survey_cell(sub)
        entry["n_generators"] = len(by_gen)
        out["configs"][config] = entry

        print(f"\n{config}: {len(rows)} rows, {len(by_gen)} generator(s)"
              + (f", {dropped} dropped" if dropped else ""))
        for gen, c in sorted(entry["cells"].items()):
            vis = c["rule_fire"]["visible"][LIST_NAME]
            cc = c["rule_fire"]["channel_contrast"]
            e_v, e_f = cc.get("visible_effect_pp"), cc.get("full_effect_pp")
            print(f"  {gen[:30]:30s} n={c['n']:6d} md5={c['n_distinct_system_md5']:4d} "
                  f"dec/hon={c['n_deceptive']}/{c['n_honest']} "
                  f"paired={c['n_paired_scenario_groups']:5d} "
                  f"fire={vis['pooled_rate']:.3f} "
                  f"vis/full={'--' if e_v is None else f'{e_v:+.1f}'}/"
                  f"{'--' if e_f is None else f'{e_f:+.1f}'}pp  "
                  f"{','.join(c['verdicts'])} {c['channel_verdict']}")

        # Normalised rolls, so the judge runner consumes this corpus unchanged.
        # These DO contain corpus text, so they live under the gitignored corpus
        # directory and are never committed.
        with open(os.path.join(NORM_DIR, f"{config}.json"), "w") as f:
            json.dump({"corpus": "liars_bench", "config": config,
                       "rollouts": rolls}, f)

        # The derived records, which carry no text and ARE committed. Written
        # gzipped: uncompressed these are 68 MB across the eight configs and 23x
        # that redundant, which is not a thing to put in a git history when the
        # analysis can read them back transparently.
        path = os.path.join(RESULTS, f"liars_bench_rule_{config}.json.gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({"corpus": "liars_bench", "config": config,
                       "prereg": "docs/PREREG_EXP_XL.md",
                       "contains_corpus_text": False,
                       "records": derived_records(config, rolls)}, f, indent=1)
        stale = os.path.join(RESULTS, f"liars_bench_rule_{config}.json")
        if os.path.exists(stale):
            os.remove(stale)

    eligible = [(c, g) for c, e in out["configs"].items()
                for g, cell in e.get("cells", {}).items()
                if cell["verdicts"] == ["ELIGIBLE"]]
    out["eligible_cells"] = [{"config": c, "generator": g} for c, g in eligible]
    out["branch"] = "E_no_config_eligible" if not eligible else "pending_H1_H2_H4"
    with open(SURVEY_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{len(eligible)} eligible (config x generator) cell(s)")
    if not eligible:
        print("PREREG_EXP_XL.md §7 branch E: the survey IS the result -- two")
        print("independent public releases, the same structural gap.")
    print(f"Survey -> {SURVEY_PATH}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", action="store_true",
                    help="§5's eligibility survey: offline, free, and read "
                         "before spending anything")
    args = ap.parse_args()
    if not os.path.exists(MANIFEST):
        print(f"no {MANIFEST}; run fetch_liars_bench.py first")
        return 1
    with open(MANIFEST) as f:
        man = json.load(f)
    if man.get("access") != "granted":
        print(f"manifest records access {man.get('access')!r} "
              f"(HTTP {man.get('http_status')}) -> PREREG §7 branch F.")
        print("The corpus could not be scored. Nothing is inferred about its "
              "contents.")
        return 1
    return survey()


if __name__ == "__main__":
    sys.exit(main())
