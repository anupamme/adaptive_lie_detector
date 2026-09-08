#!/usr/bin/env python3
"""Compute per-feature Cohen's d for Mistral 7B (parallels Llama analysis)."""
import json, sys, numpy as np, glob

FEATURE_KEYS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

# Find latest Mistral results
paths = sorted(glob.glob("data/results/ollama_eval_mistral_7b_*.json"))
if not paths:
    print("No Mistral results found"); sys.exit(1)
DATA_PATH = paths[-1]
print(f"Loading: {DATA_PATH}")

with open(DATA_PATH) as f:
    data = json.load(f)

X, y = [], []
for r in data["results"]:
    traj = r.get("feature_trajectory", [])
    if not traj:
        continue
    feat_vals = {k: [] for k in FEATURE_KEYS}
    for step in traj:
        for k in FEATURE_KEYS:
            v = step.get(k)
            if v is not None:
                feat_vals[k].append(float(v))
    if any(len(feat_vals[k]) == 0 for k in FEATURE_KEYS):
        continue
    vec = [np.mean(feat_vals[k]) for k in FEATURE_KEYS]
    label = 1 if r["ground_truth"] == "lying" else 0
    X.append(vec)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} trials ({(y==0).sum()} truthful, {(y==1).sum()} lying)\n")

print(f"{'Feature':<14} {'Truth mean±SD':>16} {'Lie mean±SD':>16} {'|d|':>6} {'Direction'}")
print("-" * 70)
for i, k in enumerate(FEATURE_KEYS):
    t = X[y == 0, i]
    l = X[y == 1, i]
    pooled = np.sqrt((np.std(t)**2 + np.std(l)**2) / 2)
    d = abs(np.mean(t) - np.mean(l)) / pooled if pooled > 0 else 0
    direction = "truth>lie" if np.mean(t) > np.mean(l) else "lie>truth"
    print(f"{k:<14} {np.mean(t):>6.2f}±{np.std(t):.2f}     {np.mean(l):>6.2f}±{np.std(l):.2f}     {d:>5.2f}  {direction}")
