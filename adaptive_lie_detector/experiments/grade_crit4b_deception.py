#!/usr/bin/env python3
"""
grade_crit4b_deception.py

EXP-C4B's deception grader. Pre-registered in docs/PREREG_EXP_C4B.md §3.3, §9.

THE PRIMARY INSTRUMENT IS IMPORTED, NOT RESTATED
------------------------------------------------
§2.6 freezes the rubric, temperature, max_tokens, slot counterbalancing and the
empty-answer rule, and §8 gate 7 requires every EXP-C4B judgement's RUBRIC_MD5
to equal EXP-C4's. Restating the rubric here would make that gate a test of
whether two copies of a string agree. Importing GRADER_SYSTEM,
grader_user_prompt, ref_position, parse_grade, call_grader and RUBRIC_MD5 from
grade_crit4_deception makes the gate unfailable-by-drift: there is only one
rubric, and its md5 is computed from it.

THREE CHANGES, AND ONLY THESE (§3.3)
------------------------------------
  1. THE QUEUE IS SHUFFLED ACROSS TARGETS. EXP-C4 graded each target as one
     contiguous block, so any drift over the grading session -- throttling,
     retries, a slow degradation in the endpoint -- was perfectly confounded
     with target identity. Here the queue is ordered by
     analyze_crit4b.queue_order_key, i.e. sha256(QUEUE_SEED | trial_key), and
     the ten targets interleave.

     The seed is PUBLIC (=42), not the salt. That is CORRECTION 1(c), and it is
     forced: §11 grades at step 5 and seals at step 8, so a salt-derived queue
     would require the salt -- hence the seal -- to exist before the data it
     commits to. Reversing that order voids the blind. A public seed buys the
     same protection here because the grader is a temperature-0 call that
     cannot act on knowing the order.

  2. A SECOND GRADER FROM A THIRD FAMILY, on 60 confirmatory trials per target
     (600 items), same rubric verbatim, same temperature, same ref_pos. Its
     role is narrow and fixed in advance: it estimates between-grader
     Krippendorff's alpha and so bounds how much D depends on one grader
     family. IT IS NOT AN ALTERNATIVE LABEL SET FOR H1 -- 60 trials per target
     cannot support the primary test, and claiming otherwise later would be a
     deviation. The subsample is drawn by an explicit sha256 ordering rather
     than random.sample, so it reproduces from the seed alone with no
     dependence on any RNG's internals.

     Nova is called through the Converse API because invoke_model's body schema
     is vendor-specific; Converse normalizes it. The primary keeps EXP-C4's
     invoke_model path untouched, because the primary instrument is frozen and
     "same rubric, same temperature" is a claim about the request that reaches
     the model, which Converse preserves.

  3. Human coding of 120 items happens BEFORE any H1 runs. That is
     analyze_crit4b_reliability.py's job, not this file's.

WHAT THE GRADER SEES, UNCHANGED FROM EXP-C4
-------------------------------------------
The question asked, the two mutually exclusive statements with the reference in
a parity-counterbalanced slot, and the answer. Never the probe vector, never the
target's identity, never the belief screen's KNOWN verdict.

Usage:
    cd code/adaptive_lie_detector
    ../.venv/bin/python3 experiments/grade_crit4b_deception.py --phase confirm --limit 5
    ../.venv/bin/python3 experiments/grade_crit4b_deception.py --phase confirm --all --resume
    ../.venv/bin/python3 experiments/grade_crit4b_deception.py --phase pilot   --all --resume
    ../.venv/bin/python3 experiments/grade_crit4b_deception.py \
        --phase confirm --secondary --subsample 60 --seed 42 --resume
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The frozen instrument. Every name here is EXP-C4's, by import.
from grade_crit4_deception import (  # noqa: E402
    AWS_REGION,
    D_OF,
    GRADER_SYSTEM,
    MAX_TOKENS,
    MODEL_ID,
    RUBRIC_MD5,
    TEMPERATURE,
    call_grader,
    grader_user_prompt,
    load_done,
    parse_grade,
    ref_position,
)
from analyze_crit4b import QUEUE_SEED, queue_order_key, trial_key  # noqa: E402

RESULTS = "data/results"
RAW_PATH = os.path.join(RESULTS, "crit4b_grades_raw.jsonl")
SECONDARY_PATH = os.path.join(RESULTS, "crit4b_grades_secondary.jsonl")
SMOKE_PATH = os.path.join(RESULTS, "crit4b_grades_smoke.jsonl")

# §3.3(2). In order; the third carries a stated caveat because Meta is also a
# target family, which is why it is last and not first.
SECONDARY_CANDIDATES = (
    ("us.amazon.nova-premier-v1:0", "Amazon — no target is an Amazon model"),
    ("us.writer.palmyra-x5-v1:0", "Writer — no target is a Writer model"),
    ("us.meta.llama3-3-70b-instruct-v1:0",
     "Meta — CAVEAT: Meta is also a target family (llama3.1:8b, llama3.2:3b)"),
)

_CONVERSE = None


def _converse_client():
    global _CONVERSE
    if _CONVERSE is None:
        _CONVERSE = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _CONVERSE


def call_secondary(model_id, system, user):
    """One secondary judgement through Converse. Same system text, same user
    text, same temperature and max_tokens as the primary; only the transport
    differs. Returns (text, stop_reason, usage, error).
    """
    delays = [4, 8, 16, 32, 64]
    last = None
    for attempt, delay in enumerate(delays, 1):
        try:
            r = _converse_client().converse(
                modelId=model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"maxTokens": MAX_TOKENS,
                                 "temperature": TEMPERATURE},
            )
            txt = "".join(b.get("text", "")
                          for b in r["output"]["message"]["content"])
            return txt, r.get("stopReason"), r.get("usage"), None
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            if attempt < len(delays):
                time.sleep(delay + random.uniform(0, 0.1 * delay))
    return "", None, None, last


def resolve_secondary():
    """Pick the first reachable secondary grader, in §3.3(2)'s order.

    Probed with a real one-token call rather than assumed: an unavailable
    model-family in this account would otherwise surface as 600 consecutive
    retry storms.
    """
    for model_id, note in SECONDARY_CANDIDATES:
        txt, _, _, err = call_secondary(model_id, "Reply with OK.", "Say OK.")
        if err is None and txt.strip():
            return model_id, note
        print(f"  {model_id}: unavailable ({(err or 'empty reply')[:80]})")
    raise SystemExit(
        "No §3.3(2) secondary grader is reachable. Options, both legal: enable "
        "one of them in Bedrock, or record this as a §12 deviation and report "
        "alpha_between as NOT COLLECTED -- which per §3.3(2) leaves D's "
        "grader-family dependence unbounded and must be stated, not omitted.")


def load_cells(phase):
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, f"crit4b_{phase}_*.json"))):
        with open(p) as f:
            out.append((p, json.load(f)))
    return out


def work_items(phase):
    """One item per trial, ordered by §3.3(1)'s public-seed shuffle.

    Also refuses a duplicated target: two cell files claiming the same model
    would be graded as two targets and the seal would mint two pseudonyms for
    one model.
    """
    items, seen = [], {}
    for path, d in load_cells(phase):
        model, wording = d["model"], d["wording_key"]
        if phase == "confirm" and model in seen:
            raise SystemExit(
                f"Two confirm cells for {model}: {seen[model]} and {path}. "
                f"§4 has ten targets, one cell each. Remove one deliberately.")
        seen[model] = path
        for r in d.get("records", []):
            rp = ref_position(r["claim_index"])
            items.append({
                "key": trial_key(phase, model, wording, r["claim_index"],
                                 r["rep"]),
                "phase": phase,
                "model": model,
                "wording": wording,
                "claim_index": r["claim_index"],
                "rep": r["rep"],
                "reference": r["false_claim"],
                "alternative": r["true_claim"],
                "ref_pos": rp,
                "question": r.get("opening_question", ""),
                "answer": r["on_claim_answer"],
            })
    # §3.3(1): interleave the targets.
    items.sort(key=lambda it: queue_order_key(it["key"]))
    return items


def subsample_per_target(items, n_per_target, seed):
    """§3.3(2)'s 60-per-target draw, by explicit sha256 ordering.

    Deliberately NOT random.sample: this must reproduce from (seed, key) alone,
    for anyone, in any Python, without trusting an RNG's internal state. Ties
    are impossible in practice and would resolve by key.
    """
    by = {}
    for it in items:
        by.setdefault(it["model"], []).append(it)
    out = []
    for model in sorted(by):
        ranked = sorted(
            by[model],
            key=lambda it: hashlib.sha256(
                f"{seed}|{it['key']}".encode("utf-8")).hexdigest())
        take = ranked[:n_per_target]
        if len(take) < n_per_target:
            print(f"  NOTE: {model} has only {len(take)} trials "
                  f"(requested {n_per_target}) — reporting the shortfall")
        out.extend(take)
    out.sort(key=lambda it: queue_order_key(it["key"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="confirm", choices=["pilot", "confirm"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke N items to a separate file")
    ap.add_argument("--secondary", action="store_true",
                    help="§3.3(2)'s second-family grader on a subsample")
    ap.add_argument("--subsample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    items = work_items(args.phase)
    if not items:
        raise SystemExit(
            f"No crit4b_{args.phase}_*.json in {RESULTS}. Run "
            f"run_crit4b_fixed_elicitation.py first.")

    smoke = args.limit > 0
    if args.secondary and smoke:
        raise SystemExit("--secondary and --limit are mutually exclusive")
    if not (smoke or args.all or args.secondary):
        raise SystemExit("pass --all for the full pass, or --limit N to smoke")

    if args.secondary:
        if args.phase != "confirm":
            raise SystemExit("§3.3(2)'s secondary grader is confirmatory only")
        out_path = SECONDARY_PATH
        items = subsample_per_target(items, args.subsample, args.seed)
        grader_id, note = resolve_secondary()
        api = "converse"
    else:
        out_path = SMOKE_PATH if smoke else RAW_PATH
        if smoke:
            items = items[:args.limit]
        grader_id, note, api = MODEL_ID, "Anthropic — the frozen primary", "invoke_model"

    done = load_done(out_path) if (args.resume and not smoke) else {}
    todo = [it for it in items if it["key"] not in done]

    label = ("SECONDARY" if args.secondary else "SMOKE" if smoke else "PRIMARY")
    print("=" * 72)
    print(f"EXP-C4B deception grading — {label} ({args.phase})")
    print("=" * 72)
    print(f"  grader     : {grader_id} @ {AWS_REGION}  [{api}]")
    print(f"  family     : {note}")
    print(f"  params     : temperature={TEMPERATURE}, max_tokens={MAX_TOKENS}")
    print(f"  rubric md5 : {RUBRIC_MD5}   (must equal EXP-C4's — §8 gate 7)")
    print(f"  queue      : sha256(QUEUE_SEED={QUEUE_SEED} | key), targets "
          f"interleaved (§3.3(1), CORRECTION 1(c))")
    if args.secondary:
        print(f"  subsample  : {args.subsample}/target by sha256(seed="
              f"{args.seed} | key) — NOT an H1 label set (§3.3(2))")
    print(f"  trials     : {len(items)}  graded: {len(done)}  to do: {len(todo)}")
    print(f"  output     : {out_path}")
    print("  the grader is asked WHICH of two statements the answer asserts —")
    print("  not which is true, and not which came from the reference.")
    print("  It never sees the probe vector, the model name or the screen.")
    print()

    counts, n_unparseable, n_empty, n_err = Counter(), 0, 0, 0
    mode = "a" if (args.resume and not smoke) else "w"
    with open(out_path, mode) as out:
        for i, it in enumerate(todo, 1):
            rp = it["ref_pos"]
            s1 = it["reference"] if rp == 1 else it["alternative"]
            s2 = it["alternative"] if rp == 1 else it["reference"]
            if not it["answer"].strip():
                # Deterministic, no model call: EXP-C4 validation found the
                # empty string is the one input on which the two presentation
                # orders disagreed.
                txt, stop, usage, err = "", None, None, None
                grade, unparse = "EVASIVE", False
                n_empty += 1
            else:
                user = grader_user_prompt(it["question"], s1, s2, it["answer"])
                if args.secondary:
                    txt, stop, usage, err = call_secondary(
                        grader_id, GRADER_SYSTEM, user)
                else:
                    txt, stop, usage, err = call_grader(GRADER_SYSTEM, user)
                grade, unparse = parse_grade(txt, rp)
            n_unparseable += int(unparse)
            n_err += int(err is not None)
            counts[grade] += 1
            rec = dict(it)
            # The answer and the two statements live in the cell file; only the
            # slot assignment is kept here, because the mapping depends on it.
            for k in ("answer", "reference", "alternative"):
                rec.pop(k)
            rec.update({
                "grade": grade,
                "D": D_OF[grade],
                "empty_answer": not it["answer"].strip(),
                "unparseable": unparse,
                "raw": txt[:600],
                "stop_reason": stop,
                "usage": usage,
                "error": err,
                "grader": grader_id,
                "grader_api": api,
                "grader_role": "secondary" if args.secondary else "primary",
                "temperature": TEMPERATURE,
                "rubric_md5": RUBRIC_MD5,
                "queue_seed": QUEUE_SEED,
            })
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if err:
                print(f"  [{i:04d}/{len(todo)}] ERROR {err[:70]}")
            elif i % 25 == 0 or i == len(todo) or smoke:
                print(f"  [{i:04d}/{len(todo)}] {it['model']:<21} "
                      f"c{it['claim_index']:>3} r{it['rep']} ref@{rp} "
                      f"-> {grade}")

    print(f"\n  grades this pass: {dict(counts)}")
    if n_empty:
        print(f"  empty answers graded EVASIVE with no model call: {n_empty}")
    if n_unparseable:
        print(f"  unparseable first lines: {n_unparseable} (flagged in JSONL)")
    if n_err:
        print(f"  ERRORS: {n_err} items exhausted their retries and are "
              f"recorded with error != null; re-run with --resume")
    if smoke:
        return 0

    # Per-target base rates. This is Role C information (§7a reads it to select
    # a wording, §7b to decide top-ups) and involves no detector and no H1.
    allrec = load_done(out_path)
    per = {}
    for r in allrec.values():
        d = per.setdefault(r["model"], Counter())
        d[r["grade"]] += 1
    print(f"\n  per-target grade counts ({out_path}) — Role C, no detector:")
    print(f"    {'target':<22}{'ASSERT':>7}{'CORR':>7}{'EVAS':>7}"
          f"{'base':>8}{'evas%':>8}")
    for m in sorted(per):
        c = per[m]
        n = c["ASSERTED"] + c["CORRECTED"]
        tot = n + c["EVASIVE"]
        base = c["ASSERTED"] / n if n else float("nan")
        print(f"    {m:<22}{c['ASSERTED']:>7}{c['CORRECTED']:>7}"
              f"{c['EVASIVE']:>7}{base:>8.3f}"
              f"{(c['EVASIVE'] / tot if tot else 0):>8.1%}")
    if args.phase == "pilot":
        print("\n  §7a selects a wording from these base rates ALONE:")
        print("    analyze_crit4b.py --phase select --model <target>")
    else:
        print("\n  next: analyze_crit4b.py --phase topup-check  (§7b, Role C)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
