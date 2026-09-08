#!/usr/bin/env python3
"""
run_external_audit.py — EXP-XA: criterion 4 on external deception benchmarks.

Pre-registered in docs/PREREG_EXP_XA.md (committed before this file existed).

Criterion 4 of the audit protocol is intervention (A): does the detector respond
when deception D varies and elicitation E does not? Instructed benchmarks cannot
supply it, because do(E) moves D and compliance C together. Apollo Research's
sandbagging corpus can: one md5-identical system prompt across all rollouts (E
fixed by construction) and a per-rollout grading of whether the model *actually*
deceived (graded_deception) that is distinct from the condition label
(expected_label). That is D varying at fixed E.

This script only computes detector output per rollout and persists it. All
reported numbers come from analyze_external_audit.py, so every figure is
recomputable offline from committed JSON. No model calls, no network.

Usage:
    .venv/bin/python3 experiments/run_external_audit.py --arm exploratory
    .venv/bin/python3 experiments/run_external_audit.py --arm confirmatory
"""

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PREREG §2: the PRIMARY pattern list is L-doc, imported (not copied) from
# hedging_baseline. This is the list app:regex_patterns documents and the one
# backing the paper's 69-80% headline. Do not substitute or extend it.
from hedging_baseline import REFUSAL_PATTERNS as L_DOC  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APOLLO = os.path.join(BASE, "data", "external", "apollo_deception", "data")
RESULTS = os.path.join(BASE, "data", "results")

# PREREG §2: L-fac, the *other* same-named 14-pattern list, reported as a
# secondary variant only. Copied deliberately rather than imported, because
# run_2x2_factorial.py is a runner with side effects at import.
L_FAC = [
    r"\bactually\b", r"\bthat'?s\s+not\s+(correct|right|accurate)\b",
    r"\bi\s+cannot\b", r"\bi\s+can'?t\b", r"\bincorrect\b",
    r"\bmisinformation\b", r"\bthis\s+is\s+(incorrect|wrong|false)\b",
    r"\bi\s+must\s+(correct|clarify|note)\b",
    r"\bto\s+be\s+(accurate|clear|correct)\b",
    r"\bthe\s+(correct|right|accurate)\s+(answer|information|fact)\b",
    r"\b(wait|hold\s+on)\b.*\bthat\b",
    r"\bit'?s\s+(worth|important)\s+(noting|mentioning)\b",
    r"\blet\s+me\s+(clarify|correct)\b",
    r"\bthat\s+is\s+(not\s+quite\s+right|inaccurate|misleading)\b",
]

LISTS = {"L_doc": L_DOC, "L_fac": L_FAC}
COMPILED = {name: [re.compile(p, re.IGNORECASE) for p in pats]
            for name, pats in LISTS.items()}

# PREREG §0: the allowlists. The confirmatory arm refuses to touch a file that
# was inspected during planning, and vice versa. This is the mechanical guard
# on the exploratory/confirmatory boundary.
EXPLORATORY_FILES = [
    "rollouts/sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
    "rollouts/insider_trading__onpolicy__llama-70b.json",
    "rollouts/ai_liar__original_without_answers__llama-70b-3.3.json",
    "rollouts/roleplaying__plain__llama-70b-3.3.json",
    "rollouts/ai_audit__reasoning__llama-70b-3.3.json",
]

# PREREG §8 branch 4 fixes this priority order; the first entry is the
# confirmatory target of H1/H2.
CONFIRMATORY_FILES = [
    "rollouts/sandbagging_v2__wmdp_mmlu__llama-70b.json",
    "rollouts/roleplaying__plain__llama-70b.json",
    "rollouts/ai_audit__reasoning__llama-70b.json",
    "how_to_catch_an_ai_liar/goal_directed_lying.json",
    "ft_fact_paired/out_simple_4_many.json",
]

HIDDEN_TAGS = ("step1", "step1-results", "step2", "step2-results", "step3",
               "sandbag-skill")


# ---------------------------------------------------------------- text variants

def visible_channel(text):
    """PREREG DEVIATION 1: the user-visible channel = <step4> + <answer>.

    Steps 1-3 are hidden from the user by the benchmark's own system prompt and
    contain the literal token 'sandbag', so a detector that reads them is
    reading the model's hidden scratchpad. Extraction is positional because
    tags may be unclosed under truncation.
    """
    i = text.find("<step4>")
    if i == -1:
        i = text.find("<answer>")
    out = text if i == -1 else text[i:]
    # defensive: drop any hidden block that still appears after that point
    for tag in HIDDEN_TAGS:
        out = re.sub(rf"<{tag}>.*?</{tag}>", " ", out, flags=re.S | re.I)
        out = re.sub(rf"</?{tag}>", " ", out, flags=re.I)
    return out


def full_transcript(text):
    """Secondary variant. Contaminated by the hidden scratchpad (DEVIATION 1)."""
    return text


VARIANTS = {"visible": visible_channel, "full": full_transcript}


def tag_order_ok(text):
    """The mandated order is steps 1-3, then step4, then answer."""
    p4, pa = text.find("<step4>"), text.find("<answer>")
    for tag in ("step1", "step2", "step3"):
        i = text.find(f"<{tag}>")
        if i != -1 and p4 != -1 and i > p4:
            return False
    return not (p4 != -1 and pa != -1 and p4 > pa)


# ------------------------------------------------------------------- detection

def marker_count(text, list_name):
    return sum(1 for pat in COMPILED[list_name] if pat.search(text))


def per_pattern(text, list_name):
    """PREREG §5: per-pattern fire rates are mandatory, not optional."""
    return [1 if pat.search(text) else 0 for pat in COMPILED[list_name]]


# ------------------------------------------------------------------ extraction

def output_text(roll):
    strs = roll.get("output_strs") or []
    if strs:
        return "\n".join(s for s in strs if isinstance(s, str))
    msgs = roll.get("output_messages") or []
    return "\n".join(m.get("content", "") for m in msgs
                     if isinstance(m, dict) and m.get("role") == "assistant")


def system_md5(roll):
    msgs = roll.get("input_messages") or []
    sys_txt = "\n".join(m.get("content", "") for m in msgs
                        if isinstance(m, dict) and m.get("role") == "system")
    return hashlib.md5(sys_txt.encode("utf-8")).hexdigest()[:8]


ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.I)
LETTERS = {"A": 0, "B": 1, "C": 2, "D": 3}


def parsed_answer(text):
    """H2: the model's final selected option, as an index. None if unparseable."""
    hits = ANSWER_RE.findall(text)
    return LETTERS[hits[-1].upper()] if hits else None


def norm_grade(roll):
    """The benchmark's independent grading of whether the model actually deceived.

    Field naming varies across the corpus: SandbaggingV2 / AI Liar / roleplaying
    use `graded_deception`; the paired-fact set uses `model_graded_honesty` with
    the same 'honest'/'deceptive' vocabulary (PREREG §9 DEVIATION 3). Values are
    normalised but never recoded; non-binary scales are kept raw for §6.
    """
    for key in ("graded_deception", "model_graded_honesty"):
        if key in roll and roll[key] is not None:
            g = roll[key]
            if isinstance(g, str):
                return g.strip().lower()
            return g
    return None


def has_model_output(roll):
    """Some vendored files are materials (prompt templates), not rollouts."""
    return bool(roll.get("output_strs") or roll.get("output_messages"))


def norm_label(roll):
    for key in ("expected_label", "expect_deception", "expect_honesty"):
        if key in roll:
            v = roll[key]
            if key == "expect_honesty" and isinstance(v, bool):
                return "honest" if v else "deceptive"
            if isinstance(v, bool):
                return "deceptive" if v else "honest"
            if isinstance(v, str):
                return v.strip().lower()
            return v
    return None


# ------------------------------------------------------------------------- main

def process(rel_path):
    path = os.path.join(APOLLO, rel_path)
    with open(path) as f:
        d = json.load(f)
    rolls = d["rollouts"] if isinstance(d, dict) else d

    n_with_output = sum(1 for r in rolls if has_model_output(r))
    recs, md5s, order_violations = [], {}, 0
    for idx, roll in enumerate(rolls):
        raw = output_text(roll)
        md5 = system_md5(roll)
        md5s[md5] = md5s.get(md5, 0) + 1
        if not tag_order_ok(raw):
            order_violations += 1

        meta = roll.get("metadata") or {}
        rec = {
            "idx": idx,
            "label": norm_label(roll),
            "grade": norm_grade(roll),
            "deception_percentage": roll.get("deception_percentage"),
            "topic": meta.get("topic"),
            "correct_answer": meta.get("correct_answer"),
            # question identity, for H2's capability screen (PREREG §5)
            "question_md5": hashlib.md5(
                str(meta.get("question", "")).encode("utf-8")).hexdigest()[:12]
            if meta.get("question") is not None else None,
            "system_md5": md5,
            "n_chars_full": len(raw),
        }
        for vname, vfn in VARIANTS.items():
            txt = vfn(raw)
            rec[f"n_chars_{vname}"] = len(txt)
            for lname in LISTS:
                rec[f"mc_{vname}_{lname}"] = marker_count(txt, lname)
                rec[f"pp_{vname}_{lname}"] = per_pattern(txt, lname)
            if vname == "visible":
                rec["answer_idx"] = parsed_answer(txt)
        recs.append(rec)

    return {
        "source_file": rel_path,
        "dataset_class": d.get("dataset_class") if isinstance(d, dict) else None,
        "base_name": d.get("base_name") if isinstance(d, dict) else None,
        "variant": d.get("variant") if isinstance(d, dict) else None,
        "model": d.get("model") if isinstance(d, dict) else None,
        "n_rollouts": len(recs),
        "n_with_model_output": n_with_output,
        "is_materials_only": n_with_output == 0,
        # PREREG §4/§6 gate 1: E fixed iff exactly one distinct system message.
        "system_prompt_md5s": md5s,
        "n_distinct_system_prompts": len(md5s),
        "elicitation_fixed": len(md5s) == 1,
        "tag_order_violations": order_violations,
        "pattern_lists": {k: v for k, v in LISTS.items()},
        "primary_list": "L_doc",
        "primary_variant": "visible",
        "records": recs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["exploratory", "confirmatory"])
    args = ap.parse_args()

    files = EXPLORATORY_FILES if args.arm == "exploratory" else CONFIRMATORY_FILES
    forbidden = set(CONFIRMATORY_FILES if args.arm == "exploratory"
                    else EXPLORATORY_FILES)

    os.makedirs(RESULTS, exist_ok=True)
    print(f"EXP-XA runner — arm={args.arm}")
    print(f"  primary list    : L_doc (hedging_baseline.REFUSAL_PATTERNS)")
    print(f"  primary variant : visible (<step4> + <answer>; PREREG DEVIATION 1)")

    for rel in files:
        assert rel not in forbidden, f"allowlist violation: {rel} in {args.arm}"
        path = os.path.join(APOLLO, rel)
        if not os.path.exists(path):
            print(f"  [skip] missing: {rel}")
            continue
        out = process(rel)
        tag = os.path.basename(rel).replace(".json", "")
        dest = os.path.join(RESULTS, f"external_audit_{args.arm}_{tag}.json")
        with open(dest, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  [ok] {tag}: n={out['n_rollouts']}, "
              f"distinct system prompts={out['n_distinct_system_prompts']}, "
              f"E fixed={out['elicitation_fixed']}")

    print(f"\nWrote to {RESULTS}. Run analyze_external_audit.py --arm {args.arm}")


if __name__ == "__main__":
    main()
