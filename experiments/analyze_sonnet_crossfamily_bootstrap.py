#!/usr/bin/env python3
"""Paired bootstrap CI on the Sonnet 4.5 cross-family extractor gap.

Compares 5-feature pipeline LOO under Haiku-extracted vs Llama-70B-extracted
feature_trajectory on the same 99 saved Sonnet transcripts. Produces a
paired-bootstrap 95% CI on (Haiku_correct - Llama70B_correct) per trial.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from experiments.analyze_sonnet_and_adversarial import label_to_int

FEATS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]


def load_trials(path):
    data = json.load(open(path))
    out = []
    for t in data.get("results", []):
        if t.get("status") == "error":
            continue
        if t.get("ground_truth") not in ("truthful", "lying"):
            continue
        ft = [s for s in t.get("feature_trajectory", []) if s is not None]
        if not ft:
            continue
        means = []
        for f in FEATS:
            vals = [s.get(f) for s in ft if s.get(f) is not None]
            means.append(float(np.mean(vals)) if vals else 0.0)
        out.append({
            "claim": t.get("claim"),
            "ground_truth": t.get("ground_truth"),
            "y": label_to_int(t.get("ground_truth")),
            "x": means,
        })
    return out


def per_trial_loo_correct(trials):
    X = np.array([t["x"] for t in trials])
    y = np.array([t["y"] for t in trials])
    correct = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        Xtr = np.delete(X, i, axis=0)
        ytr = np.delete(y, i)
        Xte = X[i:i + 1]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        correct[i] = int(clf.predict(sc.transform(Xte))[0] == y[i])
    return correct


def pair_by_claim(a, b):
    b_by = {t["claim"]: t for t in b}
    pairs = []
    for t in a:
        if t["claim"] in b_by:
            pairs.append((t, b_by[t["claim"]]))
    return pairs


def paired_bootstrap_ci(correct_a, correct_b, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)
    n = len(a)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[k] = a[idx].mean() - b[idx].mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(deltas.mean())


def main():
    root = BASE / "data" / "results"
    haiku_path = root / "bedrock_eval_sonnet_4_5_prompt_equalized_latest.json"
    llama_path = root / "bedrock_eval_sonnet_4_5_equalized_llama70b_extractor_latest.json"

    haiku = load_trials(haiku_path)
    llama = load_trials(llama_path)
    print(f"Haiku-extracted trials: {len(haiku)}")
    print(f"Llama-70B-extracted trials: {len(llama)}")

    pairs = pair_by_claim(haiku, llama)
    print(f"Paired by claim: n={len(pairs)}")

    # Compute LOO on the paired subset so both detectors see the same trials
    haiku_paired = [p[0] for p in pairs]
    llama_paired = [p[1] for p in pairs]

    haiku_correct = per_trial_loo_correct(haiku_paired)
    llama_correct = per_trial_loo_correct(llama_paired)

    h_acc = haiku_correct.mean()
    l_acc = llama_correct.mean()

    lo, hi, mean_delta = paired_bootstrap_ci(haiku_correct, llama_correct)

    print()
    print(f"  Haiku-extracted  LOO:  {h_acc*100:.1f}%")
    print(f"  Llama-70B        LOO:  {l_acc*100:.1f}%")
    print(f"  Observed delta:        {(h_acc - l_acc)*100:+.1f} pp")
    print(f"  Bootstrap mean delta:  {mean_delta*100:+.1f} pp")
    print(f"  Paired bootstrap 95% CI on delta:  [{lo*100:+.1f}, {hi*100:+.1f}] pp")
    print(f"  Includes zero: {'YES' if lo <= 0 <= hi else 'NO'}")

    out = {
        "n_paired": len(pairs),
        "haiku_loo": float(h_acc),
        "llama70b_loo": float(l_acc),
        "delta_pp": float((h_acc - l_acc) * 100),
        "bootstrap_mean_delta_pp": float(mean_delta * 100),
        "bootstrap_ci_pp": [float(lo * 100), float(hi * 100)],
        "ci_includes_zero": bool(lo <= 0 <= hi),
        "n_boot": 10000,
    }
    out_path = BASE.parent.parent / "output" / "adaptive_lie_detector_paper" / "sonnet_crossfamily_bootstrap.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
