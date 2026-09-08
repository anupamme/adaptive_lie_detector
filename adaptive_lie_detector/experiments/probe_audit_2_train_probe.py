#!/usr/bin/env python3
"""probe_audit_2_train_probe.py

Step 2: train a linear probe on residual-stream activations to predict the
deception instruction E, on the INSTRUCTED pass. Sweeps every hidden-state layer
and both poolings; scores each with leave-one-claim-pair-out (LOCPO) accuracy so
that "accuracy" means generalization to unseen claims, not to unseen trials of
seen claims. Reports claim-pair clustered bootstrap 95% CIs and selects the best
(pooling, layer) for the downstream factorial / steering steps.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_2_train_probe.py --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import (  # noqa: E402
    DATA_DIR, POOLINGS, SEED, load_pass, slice_layer, meta_vec, make_probe,
)


def locpo_predictions(X, y, groups):
    """Out-of-fold predictions + decision scores via GroupKFold over claim pairs."""
    n_splits = min(len(np.unique(groups)), 50)
    gkf = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), -1, dtype=int)
    scores = np.full(len(y), np.nan, dtype=float)
    for tr, te in gkf.split(X, y, groups):
        clf = make_probe()
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict(X[te])
        scores[te] = clf.decision_function(X[te])
    return preds, scores


def clustered_bootstrap_ci(y, preds, groups, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    pairs = np.unique(groups)
    accs = []
    for _ in range(B):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        mask = np.isin(groups, sampled)
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == y[mask]).mean())
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    arrays, meta = load_pass(args.model_tag, "instructed", args.out_dir)
    y = meta_vec(meta, "E")
    groups = meta_vec(meta, "pair_id")
    n_hidden = arrays["full_last"].shape[1]
    print(f"instructed pass: n={len(y)} trials, {len(np.unique(groups))} pairs, "
          f"E balance={y.mean():.2f}, n_hidden={n_hidden}")

    layer_curve = {p: [] for p in POOLINGS}
    best = {"acc": -1}
    for pooling in POOLINGS:
        for layer in range(n_hidden):
            X = slice_layer(arrays, pooling, layer)
            preds, _ = locpo_predictions(X, y, groups)
            acc = float((preds == y).mean())
            layer_curve[pooling].append(acc)
            if acc > best["acc"]:
                best = {"acc": acc, "pooling": pooling, "layer": layer}
        top = int(np.argmax(layer_curve[pooling]))
        print(f"  {pooling:10s} best layer {top:2d}: {max(layer_curve[pooling])*100:5.1f}%")

    # Recompute CI + full scores at the selected config
    Xb = slice_layer(arrays, best["pooling"], best["layer"])
    preds, scores = locpo_predictions(Xb, y, groups)
    lo, hi = clustered_bootstrap_ci(y, preds, groups, B=args.boot)
    print(f"\nSELECTED: {best['pooling']} layer {best['layer']}  "
          f"LOCPO acc={best['acc']*100:.1f}%  [95% CI {lo*100:.1f}-{hi*100:.1f}]")

    out = {
        "experiment": "probe_audit_2_train_probe",
        "model_tag": args.model_tag,
        "n_trials": int(len(y)), "n_pairs": int(len(np.unique(groups))),
        "n_hidden_states": int(n_hidden),
        "selected": {**best, "ci95": [lo, hi]},
        "layer_curve": layer_curve,
    }
    out_path = os.path.join(args.out_dir, f"probe2_{args.model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
