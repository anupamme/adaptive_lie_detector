#!/usr/bin/env python3
"""probe_audit_sample_transcripts.py

Seed-fixed (NOT cherry-picked) example transcripts for the write-up. Samples a
fixed set of claim pairs and prints, for each, the model's response in the
honest, deceive, and neutral conditions side by side, plus the refusal-marker
count. Reads only the saved metadata (no model).

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_sample_transcripts.py --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import DATA_DIR, SEED, count_refusal_markers  # noqa: E402


def load_meta(tag, pass_name, out_dir):
    with open(os.path.join(out_dir, f"meta_{tag}_{pass_name}.json")) as f:
        return json.load(f)["records"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--n", type=int, default=3, help="claim pairs to show")
    ap.add_argument("--chars", type=int, default=400)
    args = ap.parse_args()

    ins = load_meta(args.model_tag, "instructed", args.out_dir)
    eq = load_meta(args.model_tag, "equalized", args.out_dir)
    by = {}
    for m in ins + eq:
        by.setdefault(m["pair_id"], {})[m["cell"]] = m

    rng = np.random.default_rng(SEED)
    pair_ids = sorted(by)
    chosen = sorted(rng.choice(pair_ids, size=min(args.n, len(pair_ids)), replace=False))
    print(f"# Seed-fixed transcript sample (seed={SEED}); pairs {list(chosen)}\n")

    order = [("T_H", "honest, true claim"), ("F_D", "deceive, false claim"),
             ("T_D", "deceive, true claim"), ("T_N", "neutral, true claim"),
             ("F_N", "neutral, false claim")]
    for pid in chosen:
        cells = by[pid]
        any_cell = next(iter(cells.values()))
        print("=" * 78)
        print(f"PAIR {pid}   shown-claim (true): {cells.get('T_H', any_cell)['shown_claim']}")
        print("-" * 78)
        for cell, label in order:
            m = cells.get(cell)
            if not m:
                continue
            resp = m["response"][:args.chars].replace("\n", " ")
            rc = m.get("refusal_count", count_refusal_markers(m["response"]))
            print(f"[{cell}] {label} (refusal_markers={rc}):")
            print(f"    {resp}")
        print()


if __name__ == "__main__":
    main()
