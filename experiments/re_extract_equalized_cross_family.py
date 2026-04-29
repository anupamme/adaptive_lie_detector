#!/usr/bin/env python3
"""re_extract_equalized_cross_family.py — parameterized cross-family re-extraction.

Takes a saved equalized-trial JSON (with `conversation` + `feature_trajectory`)
and re-extracts the 5 behavioral features via a non-Haiku extractor. No new
target inference — extraction only.

Usage:
    .venv/bin/python3 experiments/re_extract_equalized_cross_family.py \
        --target_json data/results/bedrock_eval_llama8b_prompt_equalized_latest.json \
        --extractor mistral_large \
        --out data/results/cross_family_equalized_llama8b_mistral_large.json \
        [--resume]
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.cross_family_equalized_extractor import (
    BedrockMistralExtractor,
    FEATURES,
)
from experiments.bedrock_llama_extractor import BedrockLlamaExtractor


def load_trials(path):
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    valid = [
        r for r in results
        if r.get("status") != "error"
        and r.get("conversation")
        and r.get("feature_trajectory")
    ]
    print(f"  loaded {len(valid)} valid / {len(results)} total trials from {path}")
    return valid


def reextract(trials, extractor, checkpoint_path, resume=False):
    extracted = []
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            extracted = json.load(f)
        print(f"  resuming: {len(extracted)} already done")

    for i, trial in enumerate(trials):
        if i < len(extracted):
            continue

        claim = trial.get("claim", "")
        conversation = trial["conversation"]
        ground_truth = trial.get("ground_truth", "")

        try:
            feat = extractor.extract(conversation, claim)
        except Exception as e:
            print(f"  [{i+1:03d}] ERROR: {e}")
            feat = {f: None for f in FEATURES}

        orig_traj = trial["feature_trajectory"]
        orig_means = {}
        for f in FEATURES:
            vals = [t[f] for t in orig_traj if t.get(f) is not None]
            orig_means[f] = float(np.mean(vals)) if vals else None

        refusal_count = trial.get("refusal_count")
        if refusal_count is None:
            refusal_count = trial.get("correction_count")

        entry = {
            "trial_index": i,
            "claim": claim,
            "ground_truth": ground_truth,
            "cross_family_features": feat,
            "claude_features": orig_means,
            "refusal_count": refusal_count,
            "conversation_length": len(conversation),
        }
        extracted.append(entry)

        status = "OK" if all(v is not None for v in feat.values()) else "PARTIAL"
        print(f"  [{i+1:03d}/{len(trials)}] {status}  {ground_truth:<10} {claim[:48]}...")

        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(extracted, f, indent=2)

        time.sleep(1.5)

    with open(checkpoint_path, "w") as f:
        json.dump(extracted, f, indent=2)
    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_json", required=True,
                        help="Equalized-trial JSON (must contain conversation + feature_trajectory)")
    parser.add_argument("--extractor", choices=["mistral_large", "llama70b"], default="mistral_large")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"Cross-family re-extraction")
    print(f"  Target JSON: {args.target_json}")
    print(f"  Extractor:   {args.extractor}")
    print(f"  Output:      {args.out}")

    trials = load_trials(args.target_json)

    if args.extractor == "mistral_large":
        extractor = BedrockMistralExtractor()
    elif args.extractor == "llama70b":
        extractor = BedrockLlamaExtractor()
    else:
        raise ValueError(f"Unknown extractor: {args.extractor}")

    checkpoint_path = args.out.replace(".json", "_checkpoint.json")
    extracted = reextract(trials, extractor, checkpoint_path, resume=args.resume)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f:
        json.dump({
            "experiment": "cross_family_equalized_reextract",
            "target_json": args.target_json,
            "extractor": args.extractor,
            "n_trials": len(extracted),
            "results": extracted,
        }, f, indent=2)
    print(f"\nSaved: {args.out}  (n={len(extracted)})")


if __name__ == "__main__":
    main()
