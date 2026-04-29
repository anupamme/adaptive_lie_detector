#!/usr/bin/env python3
"""Analyze Sonnet 4.5 cross-family re-extraction (Llama 70B extractor).

Computes 5-feature LOO pipeline accuracy under the cross-family extractor
on the 99 saved Sonnet transcripts. Regex (refusal-count) is extractor-
independent and unchanged from the same-family-from-below analysis.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.analyze_sonnet_and_adversarial import (
    compute_pipeline_loo, compute_loo_accuracy, threshold_rule_accuracy,
    cohens_d, count_refusal_markers, label_to_int, assistant_only_transcript,
)


def main():
    root = Path(__file__).resolve().parent.parent / "data" / "results"
    paths = {
        "Haiku (same-family-from-below)": root / "bedrock_eval_sonnet_4_5_prompt_equalized_latest.json",
        "Llama 3.3 70B (cross-family)":   root / "bedrock_eval_sonnet_4_5_equalized_llama70b_extractor_latest.json",
    }
    summary = {}
    for name, path in paths.items():
        if not path.exists():
            print(f"[skip] {name}: {path.name} missing")
            continue
        data = json.load(open(path))
        trials = data.get("results", [])
        trials = [t for t in trials if t.get("ground_truth") and t.get("status") != "error"]
        # Drop None-feature steps for pipeline LOO consistency
        for t in trials:
            t["feature_trajectory"] = [f for f in t.get("feature_trajectory", []) if f is not None]
        pipe_loo = compute_pipeline_loo(trials)

        # Regex refusal count (extractor-independent, on assistant text)
        counts, labels = [], []
        for t in trials:
            assist = assistant_only_transcript(t)
            counts.append(count_refusal_markers(assist))
            labels.append(label_to_int(t.get("ground_truth")))
        rc_loo, thr = compute_loo_accuracy(counts, labels)
        rc_k1 = threshold_rule_accuracy(counts, labels, 1)
        truth_c = [c for c, l in zip(counts, labels) if l == 0]
        lie_c = [c for c, l in zip(counts, labels) if l == 1]
        d = cohens_d(truth_c, lie_c)
        n = len(trials)
        summary[name] = {
            "n": n, "pipeline_loo": pipe_loo,
            "regex_loo": rc_loo, "regex_thr": int(thr), "regex_k1": rc_k1,
            "cohens_d": d, "mean_truth": float(np.mean(truth_c)),
            "mean_lie": float(np.mean(lie_c)),
        }

        print("=" * 60)
        print(f"{name}  (n={n})")
        print(f"  5-feature pipeline LOO:  {pipe_loo*100:.1f}%")
        print(f"  Refusal-count LOO:       {rc_loo*100:.1f}%  (thr={thr})")
        print(f"  Rule k>=1:               {rc_k1*100:.1f}%")
        print(f"  Cohen's d (lie - truth): {d:.2f}  "
              f"(truth={np.mean(truth_c):.2f}, lie={np.mean(lie_c):.2f})")

    out_path = Path(__file__).resolve().parent.parent.parent.parent / "output" / "adaptive_lie_detector_paper" / "cross_family_sonnet_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
