#!/usr/bin/env python3
"""EXP-R1d driver: the recency check on the equalization collapse.

Pre-registration: docs/PREREG_EXP_R1D.md. This script authors no new detector,
claim set or probe bank -- it only sequences run_r1_faithful.py --variant v2 in
the order the pre-registration fixes, and enforces two things the runner cannot
enforce by itself:

  1. The §4 admissibility gate. A 10-claim instructed pilot must show a per-probe
     ambiguous rate <= 26.0% (the worst cell EXP-R1c already reports:
     mistral:7b equalized). A target above the gate is recorded as VOID with its
     measured rate, not silently dropped.

  2. The pilot/confirmatory separation. run_r1_faithful.py keys its checkpoint on
     (prefix, model, condition), and --resume skips claims already present -- so a
     10-claim pilot followed by an n=50 --resume run would pool the pilot into the
     confirmatory cell, which §4 forbids. We therefore MOVE the pilot checkpoint
     aside to ..._pilot.json before the confirmatory run, which then starts from
     an empty checkpoint and regenerates all 50 claims. The pilot cell is kept
     (§4 requires it be recorded and reported), it is simply not pooled.

Disk: targets are pulled, run and removed one at a time (§3).

usage:
    python experiments/run_r1d_recency.py --target olmo-3:7b
    python experiments/run_r1d_recency.py --target olmo-3:7b --pilot_only
    python experiments/run_r1d_recency.py --target qwen3.5:9b --think_off

--think_off is the EXPLORATORY arm declared as a deviation in
docs/PREREG_EXP_R1D.md §9: it sends "think": false so a thinking-mode target's
reasoning channel does not consume the fixed 40-token probe budget. It does not
and cannot revise the pre-registered §7 verdict, which is fixed by the
undeviated roster. Its cells are keyed separately in the ledger and archived to
data/results/r1d_thinkoff/, outside the analyzer's glob, so they can never be
pooled with a pre-registered cell.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

GATE = 0.260          # §4: worst already-published cell (mistral:7b equalized)
PILOT_N = 10          # §4
CONFIRM_N = 50        # §6, same n as EXP-R1c
PREFIX = "r1b_fresh"  # VARIANTS["v2"]["prefix"]
LEDGER = "data/results/r1d_recency_ledger.json"
# Pilots are archived into a SUBDIRECTORY, not alongside the confirmatory cells.
# analyze_r1_faithful.py discovers cells with a non-recursive
# glob("data/results/r1b_fresh_*.json"), so a pilot archived next to the
# confirmatory cells is read as though it were one -- which §4 forbids, and
# which additionally crashes the analyzer on a pilot holding a single class.
PILOT_DIR = "data/results/r1d_pilots"
# The think:false arm (PREREG_EXP_R1D §9, a declared deviation from §2) is kept
# entirely out of data/results/, for the same reason the pilots are: the analyzer
# globs that directory non-recursively, and an exploratory cell must never be
# pooled with a pre-registered one.
THINKOFF_DIR = "data/results/r1d_thinkoff"


def tag(model):
    return model.replace(":", "_").replace(".", "_")


def ckpt_path(model, condition, suffix=""):
    return f"data/results/{PREFIX}_{tag(model)}_{condition}{suffix}.json"


def pilot_path(model, condition="instructed", think_off=False):
    d = THINKOFF_DIR if think_off else PILOT_DIR
    suf = "_pilot_thinkoff" if think_off else "_pilot"
    return os.path.join(d, f"{PREFIX}_{tag(model)}_{condition}{suf}.json")


def thinkoff_path(model, condition):
    """Where a think:false confirmatory cell is archived after collection."""
    return os.path.join(THINKOFF_DIR,
                        f"{PREFIX}_{tag(model)}_{condition}_thinkoff.json")


def ambiguous_rate(path):
    """Per-probe ambiguity rate over every probe of every trial in a cell."""
    with open(path) as f:
        recs = json.load(f)["records"]
    flags = [a for r in recs for a in r["ambiguous"]]
    if not flags:
        raise SystemExit(f"no ambiguous flags in {path}")
    return sum(flags) / len(flags), len(recs), len(flags)


def run(model, condition, n, think_off=False):
    cmd = [sys.executable, "experiments/run_r1_faithful.py",
           "--variant", "v2", "--model", model,
           "--condition", condition, "--n_samples", str(n), "--resume"]
    env = dict(os.environ)
    if think_off:
        # The whole deviation, in one environment variable. src/
        # ollama_target_model.py adds "think": false to the chat payload only
        # when this is set; the runner itself is unmodified.
        env["OLLAMA_THINK"] = "off"
    print("+ " + ("OLLAMA_THINK=off " if think_off else "") + " ".join(cmd),
          flush=True)
    return subprocess.call(cmd, env=env)


def read_ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER) as f:
            return json.load(f)
    return {"experiment": "EXP-R1d", "gate": GATE, "targets": {}}


def write_ledger(led):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--pilot_only", action="store_true")
    ap.add_argument("--think_off", action="store_true",
                    help="the PREREG §9 exploratory arm: send \"think\": false, "
                         "key the ledger separately, and archive every cell "
                         "outside the analyzer's glob")
    a = ap.parse_args()
    model = a.target
    tk = a.think_off
    led = read_ledger()
    key = f"{model} [think:false]" if tk else model
    entry = led["targets"].setdefault(key, {})
    if tk:
        entry["arm"] = "exploratory, PREREG_EXP_R1D §9 deviation (think:false)"
        entry["primary_verdict_unaffected"] = "§7.5 branch 5 (inconclusive)"

    # ---- 1. pilot (instructed, n=10) -------------------------------------
    pilot_arch = pilot_path(model, think_off=tk)
    os.makedirs(THINKOFF_DIR if tk else PILOT_DIR, exist_ok=True)
    live = ckpt_path(model, "instructed")
    if not os.path.exists(pilot_arch):
        if os.path.exists(live):
            raise SystemExit(
                f"REFUSING: {live} already exists but no pilot archive does. "
                "A confirmatory cell must not be built on top of an unarchived "
                "pilot (PREREG_EXP_R1D §4).")
        if run(model, "instructed", PILOT_N, think_off=tk) != 0:
            raise SystemExit("pilot run failed")
        shutil.move(live, pilot_arch)
        print(f"  pilot archived -> {pilot_arch}")

    rate, n_trials, n_probes = ambiguous_rate(pilot_arch)
    entry["pilot"] = {"file": pilot_arch, "n_trials": n_trials,
                      "n_probes": n_probes, "ambiguous_rate": round(rate, 4)}
    print(f"\n  GATE: pilot ambiguous rate = {rate:.1%} "
          f"({n_probes} probes over {n_trials} trials); threshold {GATE:.1%}")

    if rate > GATE:
        entry["verdict"] = "VOID_GATE"
        write_ledger(led)
        print(f"  VOID: {key} fails the §4 gate at {rate:.1%}. "
              "Recorded as void; no confirmatory cell collected.")
        return 0

    entry["verdict"] = "ADMISSIBLE"
    write_ledger(led)
    print(f"  ADMISSIBLE: {key} passes the gate.")
    if a.pilot_only:
        return 0

    # ---- 2. confirmatory (n=50, both conditions) -------------------------
    # instructed first: §5 fixes informativeness from the instructed cell
    # before any equalized result for that target is seen.
    for cond in ("instructed", "equalized"):
        if run(model, cond, CONFIRM_N, think_off=tk) != 0:
            raise SystemExit(f"confirmatory {cond} run failed")
        cell = ckpt_path(model, cond)
        if tk:
            # Move it out of the analyzer's glob: an exploratory cell must not
            # be discoverable as a pre-registered one (PREREG §9, item 7).
            dest = thinkoff_path(model, cond)
            shutil.move(cell, dest)
            cell = dest
            print(f"  think:false cell archived -> {dest}")
        r, nt, npb = ambiguous_rate(cell)
        entry.setdefault("confirmatory", {})[cond] = {
            "file": cell, "n_trials": nt, "ambiguous_rate": round(r, 4)}
        write_ledger(led)
        print(f"  {cond}: {nt} trials, ambiguous {r:.1%}")

    print(f"\n  ledger -> {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
