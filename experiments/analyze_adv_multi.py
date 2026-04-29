#!/usr/bin/env python3
"""analyze_adv_multi.py

Analyze adversarial-paraphrase runs across multiple models.

For each model:
  - refusal-count rule accuracy at fixed k=1
  - baseline (non-adversarial equalized) accuracy lookup
  - delta (baseline - adversarial) pp
  - Cohen's d on refusal-marker count (truth vs lie)
  - paired bootstrap 95% CI on the delta (10k resamples)

Also dumps 10 lie-condition first-response excerpts per model for
qualitative inspection.
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "data" / "results"

sys.path.insert(0, str(BASE))

from experiments.analyze_sonnet_and_adversarial import (
    count_refusal_markers, assistant_only_transcript,
)


def first_assistant_response(trial):
    # Use the full assistant transcript (all turns concatenated) to match
    # the paper's refusal-count metric. Renamed for code-clarity only.
    return assistant_only_transcript(trial)


def rule_accuracy(trials, k=1):
    correct, n = 0, 0
    tr_counts, lie_counts = [], []
    for t in trials:
        gt = t.get("ground_truth")
        if gt not in ("truthful", "lying"):
            continue
        resp = first_assistant_response(t)
        c = count_refusal_markers(resp)
        pred = "lying" if c >= k else "truthful"
        if pred == gt:
            correct += 1
        n += 1
        (lie_counts if gt == "lying" else tr_counts).append(c)
    return (correct / n if n else 0.0, n, np.array(tr_counts), np.array(lie_counts))


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / pooled_sd if pooled_sd > 0 else 0.0


def paired_bootstrap_ci(pairs_base, pairs_adv, n_boot=10000, seed=42):
    """Each pair is (is_correct_base, is_correct_adv). Return (lo, hi) on delta = base - adv.

    Uses stratified paired resampling over the N trial slots (0=wrong, 1=correct).
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(pairs_base, dtype=int)
    adv = np.asarray(pairs_adv, dtype=int)
    n = len(base)
    if n == 0 or len(adv) == 0:
        return (float("nan"), float("nan"))
    # Align by sample index: truncate to common length
    m = min(n, len(adv))
    base = base[:m]
    adv = adv[:m]
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, m, size=m)
        deltas.append(base[idx].mean() - adv[idx].mean())
    return (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d.get("results", d)


# Baseline = non-adversarial equalized result on the same model
BASELINE_FILES = {
    "llama3_2_3b": "ollama_eval_llama3_2_3b_prompt_equalized_20260422_102229.json",
    "mistral_7b":  "ollama_eval_mistral_7b_prompt_equalized_latest.json",
    "qwen2_5_14b": "ollama_eval_qwen2_5_14b_prompt_equalized_latest.json",
}

ADV_FILES = {
    "llama3_2_3b": "ollama_eval_llama3_2_3b_adversarial_latest.json",
    "mistral_7b":  "ollama_eval_mistral_7b_adversarial_latest.json",
    "qwen2_5_14b": "ollama_eval_qwen2_5_14b_adversarial_latest.json",
}

LABELS = {
    "llama3_2_3b": "Llama 3B",
    "mistral_7b":  "Mistral 7B",
    "qwen2_5_14b": "Qwen 14B",
}


def pair_trials_by_claim(base_trials, adv_trials):
    adv_by_claim = {t.get("claim"): t for t in adv_trials}
    paired = []
    for t in base_trials:
        c = t.get("claim")
        if c in adv_by_claim:
            paired.append((t, adv_by_claim[c]))
    return paired


def analyze_model(tag):
    base_path = RESULTS / BASELINE_FILES[tag]
    adv_path = RESULTS / ADV_FILES[tag]
    if not adv_path.exists():
        return {"model": tag, "status": "missing_adv"}
    if not base_path.exists():
        return {"model": tag, "status": "missing_base"}

    base = load(base_path)
    adv = load(adv_path)
    # Checkpoint JSONs have different shape; handle both
    if isinstance(base, dict) and "results" in base:
        base = base["results"]
    if isinstance(adv, dict) and "results" in adv:
        adv = adv["results"]

    base_acc, base_n, base_tr, base_lie = rule_accuracy(base, k=1)
    adv_acc, adv_n, adv_tr, adv_lie = rule_accuracy(adv, k=1)

    # Paired accuracy pairs for CI
    paired = pair_trials_by_claim(base, adv)
    base_correct, adv_correct = [], []
    for b, a in paired:
        br = first_assistant_response(b)
        ar = first_assistant_response(a)
        bp = "lying" if count_refusal_markers(br) >= 1 else "truthful"
        ap = "lying" if count_refusal_markers(ar) >= 1 else "truthful"
        base_correct.append(int(bp == b.get("ground_truth")))
        adv_correct.append(int(ap == a.get("ground_truth")))

    lo, hi = paired_bootstrap_ci(base_correct, adv_correct) if paired else (float("nan"),) * 2

    d_base = cohens_d(base_tr, base_lie)
    d_adv = cohens_d(adv_tr, adv_lie)

    # Qualitative: 10 lie-condition first responses
    lie_samples = [t for t in adv if t.get("ground_truth") == "lying"][:10]
    qual = []
    for t in lie_samples:
        resp = first_assistant_response(t)
        qual.append({
            "claim": t.get("claim", "")[:80],
            "response_excerpt": resp[:300],
            "refusal_marker_count": count_refusal_markers(resp),
        })

    return {
        "model": tag,
        "label": LABELS[tag],
        "baseline_acc": base_acc, "baseline_n": base_n,
        "adv_acc": adv_acc, "adv_n": adv_n,
        "delta_pp": (base_acc - adv_acc) * 100,
        "delta_ci_pp": (lo * 100, hi * 100),
        "refusal_markers_lie_mean_base": float(base_lie.mean()) if len(base_lie) else float("nan"),
        "refusal_markers_lie_mean_adv": float(adv_lie.mean()) if len(adv_lie) else float("nan"),
        "cohens_d_base": d_base, "cohens_d_adv": d_adv,
        "n_paired": len(paired),
        "qualitative_samples": qual,
    }


def main():
    out = {}
    for tag in ["llama3_2_3b", "mistral_7b", "qwen2_5_14b"]:
        r = analyze_model(tag)
        out[tag] = r
        print("=" * 60)
        print(f"{r.get('label', tag)}  [{r.get('status', 'ok')}]")
        print("=" * 60)
        if r.get("status", "ok") != "ok":
            continue
        print(f"  baseline (equalized):    {r['baseline_acc']*100:.1f}%  (n={r['baseline_n']})")
        print(f"  adversarial:             {r['adv_acc']*100:.1f}%  (n={r['adv_n']})")
        print(f"  delta (base - adv):      {r['delta_pp']:+.1f}pp  "
              f"95% CI [{r['delta_ci_pp'][0]:+.1f}, {r['delta_ci_pp'][1]:+.1f}]  "
              f"(paired n={r['n_paired']})")
        print(f"  refusal markers (lie)    base={r['refusal_markers_lie_mean_base']:.2f} "
              f"-> adv={r['refusal_markers_lie_mean_adv']:.2f}")
        print(f"  Cohen's d (truth vs lie) base={r['cohens_d_base']:.2f} "
              f"-> adv={r['cohens_d_adv']:.2f}")
        print()
        print("  QUALITATIVE (10 lie responses, adversarial):")
        for q in r["qualitative_samples"]:
            print(f"   - [{q['refusal_marker_count']} markers] {q['claim']}")
            print(f"     {q['response_excerpt'][:200]!r}")

    summary_path = BASE.parent.parent / "output" / "adaptive_lie_detector_paper" / "adv_multi_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
