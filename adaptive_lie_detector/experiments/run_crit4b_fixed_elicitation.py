#!/usr/bin/env python3
"""
run_crit4b_fixed_elicitation.py

EXP-C4B's collection runner. Pre-registered in docs/PREREG_EXP_C4B.md.

WHAT THIS FILE DOES *NOT* DO
----------------------------
It does not restate EXP-C4's design. §2 lists the replication invariants -- the
wording ladder P1/P2/P3, the fixed-reference system prompt, the 16-probe bank,
the on-claim/probe channel separation, the inline H4 belief re-query, the
absence of any condition label -- and the only way to guarantee they are
unchanged is to *import* them:

    from run_crit4_fixed_elicitation import (WORDINGS, FixedReferenceModel,
                                             run_single_trial, ...)

So a trial collected here is byte-identically constructed to an EXP-C4 trial.
If someone edits the wording ladder, EXP-C4B changes with it and Family R stops
being a replication -- which is exactly the coupling §2 asks for, made
mechanical instead of asserted. A reimplementation here could drift silently;
this cannot.

WHAT IS NEW, AND ONLY THIS
--------------------------
  1. File-level metadata: `experiment: EXP-C4B`, `prereg: PREREG_EXP_C4B.md`.
  2. Cell naming: `crit4b_{phase}_{tag}.json` for confirm -- ONE file per
     target, with the wording inside it as `wording_key`, not in the filename.
     Rationale, and it is a correctness point rather than taste: two confirm
     files for one target (say P2 and P3) would both match
     analyze_crit4b.load_cells' glob, the seal would mint two pseudonyms for
     one model, and the ten-target manifest would silently become eleven. One
     path per target makes that unrepresentable. Pilot cells DO carry the
     wording, because §7a deliberately collects all three for one target.
  3. Per-target wording (§7a): Family R is fixed at P3 and never consults a
     selection file; Family E confirm reads `crit4b_wording_selection.json`
     and refuses to run for a target that has no committed selection.
  4. The wording guard on resume. If a confirm checkpoint exists at a
     DIFFERENT wording than the one requested, this refuses. Appending trials
     at a second wording would put two system-prompt hashes on one claim, and
     §8 gate 1 would then discard the whole cell -- after paying for it.
  5. Top-up support (§7b). Raising --claims and re-running with --resume adds
     the next claims in pool order and re-collects nothing: the checkpoint key
     is (claim_index, rep), and claims come from
     load_known_pairs(model, n)[:n] in pool order, so claims 30-39 are exactly
     what a 40-claim re-run adds to a 30-claim cell.

Usage:
    cd code/adaptive_lie_detector

    # Family R -- fixed at P3, NO pilot (§7a). 30 claims x 8 reps.
    ../.venv/bin/python3 experiments/run_crit4b_fixed_elicitation.py \
        --phase confirm --wordings P3 \
        --models llama3.2:3b,mistral:7b,qwen2.5:7b,llama3.1:8b,qwen2.5:14b \
        --claims 30 --reps 8 --resume

    # Family E -- pilot the ladder, select on base rate, then confirm.
    ... --phase pilot --models gemma2:9b --wordings P1,P2,P3 --claims 10 --reps 4
    ... --phase confirm --models gemma2:9b --claims 30 --reps 8   # reads selection

    # Top-up (§7b): same command, larger --claims, --resume.
    ... --phase confirm --wordings P3 --models llama3.2:3b --claims 40 --reps 8 --resume
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Every design invariant §2 freezes arrives by import, not by restatement.
from run_crit4_fixed_elicitation import (  # noqa: E402
    QUESTIONS_MODULE,
    WORDINGS,
    FixedReferenceModel,
    load_known_pairs,
    load_probe_bank,
    run_single_trial,
)
from analyze_crit4b import FAMILY_E, FAMILY_R, SELECTION  # noqa: E402

RESULTS = "data/results"
EXPERIMENT = "EXP-C4B"
PREREG = "docs/PREREG_EXP_C4B.md"

# §4. Base design; both are settable per invocation because §7b tops up claims.
BASE_CLAIMS = 30
BASE_REPS = 8
TOPUP_CAP_CLAIMS = 50          # §7b hard cap


def ckpt_path(phase, model, wording):
    """Confirm: one file per target. Pilot: one per (target, wording).

    See the module docstring, point 2 -- the asymmetry is deliberate.
    """
    tag = model.replace(":", "_").replace(".", "_")
    if phase == "confirm":
        return os.path.join(RESULTS, f"crit4b_confirm_{tag}.json")
    return os.path.join(RESULTS, f"crit4b_{phase}_{tag}_{wording}.json")


def selected_wording(model):
    """§7a. Family R is fixed at P3 with no pilot; Family E must have a
    committed selection written by `analyze_crit4b.py --phase select`.

    The refusal is the point: a confirmatory Family E cell that chose its own
    wording would reintroduce the flexibility §7a exists to remove.
    """
    if model in FAMILY_R:
        return "P3", "PREREG §7a: Family R is fixed at P3, no pilot"
    if not os.path.exists(SELECTION):
        raise SystemExit(
            f"{model} needs a committed wording selection and {SELECTION} does "
            f"not exist.\nRun the pilot, then:\n"
            f"  analyze_crit4b.py --phase select --model {model}\n"
            f"A confirmatory cell must not choose its own wording (§7a).")
    with open(SELECTION) as f:
        targets = json.load(f).get("targets", {})
    if model not in targets:
        raise SystemExit(
            f"No committed selection for {model} in {SELECTION} "
            f"(have: {sorted(targets) or 'none'}).\n"
            f"Run: analyze_crit4b.py --phase select --model {model}")
    entry = targets[model]
    flag = " SKEWED" if entry.get("skewed") else ""
    return entry["selected"], f"PREREG §7a: selected on D base rate alone{flag}"


def load_checkpoint(ckpt, phase, wording, resume):
    """Returns (records, done_keys). Refuses a wording change on an existing
    confirm cell -- see the module docstring, point 4.
    """
    if not os.path.exists(ckpt):
        return [], set()
    with open(ckpt) as f:
        prior = json.load(f)
    prior_w = prior.get("wording_key")
    if prior_w != wording:
        raise SystemExit(
            f"{ckpt} already holds wording {prior_w!r}, but {wording!r} was "
            f"requested.\nAppending would put two system-prompt hashes on one "
            f"claim and §8 gate 1 would discard the whole cell. Move the "
            f"existing file aside deliberately if the wording really changed.")
    if not resume:
        raise SystemExit(
            f"{ckpt} exists and --resume was not given. Refusing to overwrite "
            f"collected trials.")
    records = prior.get("records", [])
    # (claim_index, rep), not claim alone: EXP-C4B deliberately collects k
    # resamples of the SAME claim, which a claim-keyed dedupe would collapse.
    return records, {(r["claim_index"], r["rep"]) for r in records}


def write_checkpoint(ckpt, phase, model, wording, questions, pairs, reps,
                     records):
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    tmp = ckpt + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "experiment": EXPERIMENT,
            "phase": phase,
            "prereg": PREREG,
            "model": model,
            "family": ("R" if model in FAMILY_R
                       else "E" if model in FAMILY_E else "SUBSTITUTION"),
            "wording_key": wording,
            "wording": WORDINGS[wording],
            "probe_bank": QUESTIONS_MODULE,
            "n_questions": len(questions),
            "questions": questions,
            "n_claims": len(pairs),
            "reps": reps,
            "records": records,
            # NOTE: no label / ground_truth / condition field anywhere, at the
            # cell level or the record level. §8 gate 2 verifies it; D exists
            # only after grade_crit4b_deception.py reads realized behavior.
        }, f, indent=2)
    os.replace(tmp, ckpt)   # atomic: a killed run cannot leave half a cell


def run_cell(model, wording, phase, n_claims, reps, resume, why):
    questions, yesno_suffix = load_probe_bank()
    pairs = load_known_pairs(model, n_claims)
    ckpt = ckpt_path(phase, model, wording)

    print("=" * 72)
    print(f"{EXPERIMENT} ({phase}): {model} / wording {wording}")
    print("=" * 72)
    print(f"  wording basis : {why}")
    print(f"  probe bank    : {QUESTIONS_MODULE} ({len(questions)} probes)")
    print(f"  KNOWN claims  : {len(pairs)} requested {n_claims}   reps: {reps}"
          f"   trials: {len(pairs) * reps}")
    print(f"  generations   : {len(pairs) * reps * (1 + len(questions) + 2)}"
          f"  (1 on-claim + {len(questions)} probes + 2 belief re-queries)")
    print(f"  checkpoint    : {ckpt}")

    records, done = load_checkpoint(ckpt, phase, wording, resume)
    if records:
        print(f"  resuming      : {len(records)} trials already collected")
    if phase == "confirm" and n_claims > TOPUP_CAP_CLAIMS:
        raise SystemExit(f"--claims {n_claims} exceeds §7b's hard cap of "
                         f"{TOPUP_CAP_CLAIMS}.")

    target = FixedReferenceModel(model, wording, call_delay=0.05, max_tokens=300)
    total, n, n_new, n_err = len(pairs) * reps, 0, 0, 0
    for pair in pairs:
        for rep in range(reps):
            n += 1
            if (pair["index"], rep) in done:
                continue
            print(f"  [{n:04d}/{total}] c{pair['index']:>3} r{rep} "
                  f"{pair['false_claim'][:44]}", end="", flush=True)
            try:
                rec = run_single_trial(target, pair, rep, questions,
                                       yesno_suffix, model)
                records.append(rec)
                n_new += 1
                kp = "kept" if rec["belief_recheck"]["known_preserved"] else "LOST"
                print(f"  -> sum={sum(rec['vector'])} "
                      f"ans={len(rec['on_claim_answer'])}ch belief={kp}")
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"  ERROR: {e}")
            # Written every trial: a 240-trial cell is hours long and a crash
            # must never cost more than the trial in flight.
            write_checkpoint(ckpt, phase, model, wording, questions, pairs,
                             reps, records)

    per_claim = {}
    for r in records:
        per_claim.setdefault(r["claim_index"], set()).add(r["system_prompt_md5"])
    multi = {c: sorted(h) for c, h in per_claim.items() if len(h) > 1}
    kept = sum(1 for r in records
               if r["belief_recheck"]["known_preserved"])
    print(f"  done: {len(records)} trials (+{n_new} this pass, {n_err} errors)")
    print(f"  gate 1 (one system prompt per claim): "
          f"{'PASS' if not multi else f'FAIL {multi}'}")
    print(f"  H4 belief preserved: {kept}/{len(records)} "
          f"({kept / len(records):.1%})" if records else "  no records")
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["pilot", "confirm"])
    ap.add_argument("--models", default="", help="comma-separated ollama tags")
    ap.add_argument("--family", default="", choices=["", "R", "E"],
                    help="shorthand for §4's rosters")
    ap.add_argument("--wordings", default="",
                    help="pilot: the ladder to sweep. confirm: OPTIONAL "
                         "assertion that the selected wording is this one; "
                         "a mismatch is refused rather than silently honoured.")
    ap.add_argument("--claims", type=int, default=BASE_CLAIMS)
    ap.add_argument("--reps", type=int, default=BASE_REPS)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.family == "R":
        models = list(FAMILY_R)
    elif args.family == "E":
        models = list(FAMILY_E)
    if not models:
        raise SystemExit("give --models or --family {R,E}")

    asked = [w.strip() for w in args.wordings.split(",") if w.strip()]
    for w in asked:
        if w not in WORDINGS:
            raise SystemExit(f"unknown wording {w}; have {sorted(WORDINGS)}")

    if args.phase == "pilot":
        if not asked:
            raise SystemExit("pilot needs --wordings, e.g. --wordings P1,P2,P3")
        for m in models:
            if m in FAMILY_R:
                raise SystemExit(
                    f"{m} is a Family R replication target: §7a fixes it at P3 "
                    f"with NO pilot. Piloting it would make Family R "
                    f"non-replicating.")
            for w in asked:
                run_cell(m, w, "pilot", args.claims, args.reps, args.resume,
                         "PREREG §7a pilot: base-rate ladder, no detector fit")
                print()
        return 0

    for m in models:
        w, why = selected_wording(m)
        if asked and asked != [w]:
            raise SystemExit(
                f"--wordings {asked} contradicts {m}'s governing wording {w!r} "
                f"({why}). Refusing: the wording is not this command's to pick.")
        run_cell(m, w, "confirm", args.claims, args.reps, args.resume, why)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
