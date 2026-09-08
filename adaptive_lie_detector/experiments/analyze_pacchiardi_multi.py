#!/usr/bin/env python3
"""Analyze EXP-K unrelated vs. related across multiple models.

Loads Pacchiardi-style (unrelated follow-ups) and standard ADAGE (related
follow-ups) result files. For each model and condition, computes:
  - Refusal-count LOO accuracy (assistant-text regex rule)
  - 5-feature LLM pipeline LOO
  - Cohen's d on refusal markers
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.analyze_sonnet_and_adversarial import (
    compute_pipeline_loo, compute_loo_accuracy, threshold_rule_accuracy,
    cohens_d, count_refusal_markers, label_to_int, assistant_only_transcript,
)

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"

FILES = {
    ("Mistral 7B", "unrelated"): "ollama_eval_mistral_7b_pacchiardi_unrelated_latest.json",
    ("Mistral 7B", "related"):   "ollama_eval_mistral_7b_prompt_equalized_latest.json",  # standard ADAGE = related
    ("Llama 70B",  "unrelated"): "bedrock_eval_llama_70b_pacchiardi_unrelated_latest.json",
    ("Llama 70B",  "related"):   "bedrock_eval_llama_70b_pacchiardi_related_latest.json",
    ("Qwen 14B",   "unrelated"): "ollama_eval_qwen2_5_14b_pacchiardi_unrelated_latest.json",
    ("Qwen 14B",   "related"):   "ollama_eval_qwen2_5_14b_pacchiardi_related_latest.json",
}


def analyze(path):
    if not path.exists():
        return None
    data = json.load(open(path))
    trials = data.get("results", [])
    trials = [t for t in trials if t.get("ground_truth") and t.get("status") != "error"]
    for t in trials:
        t["feature_trajectory"] = [f for f in t.get("feature_trajectory", []) if f is not None]
    if not trials:
        return None
    counts, labels = [], []
    for t in trials:
        assist = assistant_only_transcript(t)
        counts.append(count_refusal_markers(assist))
        labels.append(label_to_int(t.get("ground_truth")))
    rc_loo, thr = compute_loo_accuracy(counts, labels)
    rc_k1 = threshold_rule_accuracy(counts, labels, 1)
    pipe = compute_pipeline_loo(trials)
    tc = [c for c, l in zip(counts, labels) if l == 0]
    lc = [c for c, l in zip(counts, labels) if l == 1]
    d = cohens_d(tc, lc)
    return {
        "n": len(trials), "regex_loo": rc_loo, "regex_k1": rc_k1,
        "regex_thr": int(thr), "pipeline_loo": pipe,
        "cohens_d": d, "mean_truth": float(np.mean(tc)) if tc else 0.0,
        "mean_lie": float(np.mean(lc)) if lc else 0.0,
    }


def main():
    summary = {}
    rows = []
    for (model, cond), fname in FILES.items():
        r = analyze(RESULTS / fname)
        summary.setdefault(model, {})[cond] = r
        status = "MISSING" if r is None else f"n={r['n']}  regex-LOO={r['regex_loo']*100:.1f}%  pipe={r['pipeline_loo']*100:.1f}%  d={r['cohens_d']:.2f}"
        rows.append(f"  {model:<12} {cond:<10}  {status}")

    print("=" * 60)
    print("EXP-K: Multi-model unrelated vs. related (refusal-count + 5-feature LOO)")
    print("=" * 60)
    for r in rows:
        print(r)
    print()
    print("DELTAS (related - unrelated, LOO percentage points):")
    for model in ["Mistral 7B", "Qwen 14B", "Llama 70B"]:
        u = summary.get(model, {}).get("unrelated")
        r = summary.get(model, {}).get("related")
        if u and r:
            dr = (r["regex_loo"] - u["regex_loo"]) * 100
            dp = (r["pipeline_loo"] - u["pipeline_loo"]) * 100
            print(f"  {model:<12}  regex d={dr:+.1f}pp   pipeline d={dp:+.1f}pp")
        else:
            print(f"  {model:<12}  (incomplete)")

    out_path = Path(__file__).resolve().parent.parent.parent.parent / "output" / "adaptive_lie_detector_paper" / "pacchiardi_multi_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
