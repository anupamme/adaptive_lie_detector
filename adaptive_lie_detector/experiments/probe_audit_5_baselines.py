#!/usr/bin/env python3
"""probe_audit_5_baselines.py

Baselines and null controls for the probe (Nanda's #1 criterion: earn the number
against dumb alternatives). All are computed from the cached instructed-pass
activations + metadata, so no model is needed.

  1. random-direction probe : project activations onto a fixed random unit
     direction, fit a 1-D logistic threshold to predict E (LOCPO). Should sit at
     chance -- a real linear-probe accuracy must beat this.
  2. shuffled-label null     : permute E labels, then run the full LOCPO probe.
     Repeated over seeds; the real accuracy must beat this null distribution.
  3. refusal-marker baseline : the *behavioral* detector -- predict E from the
     response's correction/refusal-marker count (both threshold directions). This
     is the cheap non-interpretability comparison the probe must justify itself
     against.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_5_baselines.py --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import (  # noqa: E402
    DATA_DIR, SEED, load_pass, slice_layer, meta_vec, make_probe,
)
from experiments.probe_audit_2_train_probe import locpo_predictions  # noqa: E402


def random_direction_acc(X, y, groups, seed):
    """Project onto a random unit direction, LOCPO 1-D logistic accuracy."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(X.shape[1])
    d /= np.linalg.norm(d)
    proj = (X @ d).reshape(-1, 1)
    preds, _ = locpo_predictions(proj, y, groups)
    return float((preds == y).mean())


def shuffled_label_null(X, y, groups, n_perm, seed):
    rng = np.random.default_rng(seed)
    accs = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        preds, _ = locpo_predictions(X, yp, groups)
        accs.append(float((preds == yp).mean()))
    return accs


def refusal_baseline(rc, y):
    """Predict E from refusal-marker count with a fixed threshold >=1, both signs."""
    fires = (rc >= 1).astype(int)
    acc_direct = float((fires == y).mean())          # more markers -> deceive
    acc_flipped = float(((1 - fires) == y).mean())   # more markers -> honest
    return acc_direct, acc_flipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--n_rand", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=50)
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, f"probe2_{args.model_tag}.json")) as f:
        p2 = json.load(f)
    sel = p2["selected"]
    pooling, layer = sel["pooling"], sel["layer"]
    real_acc = sel["acc"]

    arrays, meta = load_pass(args.model_tag, "instructed", args.out_dir)
    X = slice_layer(arrays, pooling, layer)
    y = meta_vec(meta, "E")
    groups = meta_vec(meta, "pair_id")
    rc = meta_vec(meta, "refusal_count").astype(float)

    rand_accs = [random_direction_acc(X, y, groups, SEED + i) for i in range(args.n_rand)]
    null_accs = shuffled_label_null(X, y, groups, args.n_perm, SEED)
    ref_direct, ref_flipped = refusal_baseline(rc, y)

    # empirical p: fraction of shuffled-label runs that match/beat the real probe
    null_p = (np.sum(np.array(null_accs) >= real_acc) + 1) / (len(null_accs) + 1)

    print(f"Baselines vs real probe ({pooling} layer {layer}) acc={real_acc*100:.1f}%")
    print(f"  random-direction probe : {np.mean(rand_accs)*100:5.1f}% "
          f"(max {np.max(rand_accs)*100:.1f}, n={args.n_rand})")
    print(f"  shuffled-label null     : {np.mean(null_accs)*100:5.1f}% "
          f"(95th pct {np.percentile(null_accs, 95)*100:.1f}), empirical p={null_p:.3f}")
    print(f"  refusal-marker baseline : direct {ref_direct*100:.1f}%  "
          f"flipped {ref_flipped*100:.1f}%")

    out = {
        "experiment": "probe_audit_5_baselines",
        "model_tag": args.model_tag,
        "pooling": pooling, "layer": layer,
        "real_probe_acc": real_acc,
        "random_direction": {"mean": float(np.mean(rand_accs)),
                             "max": float(np.max(rand_accs)),
                             "n": args.n_rand, "accs": rand_accs},
        "shuffled_label_null": {"mean": float(np.mean(null_accs)),
                                "pct95": float(np.percentile(null_accs, 95)),
                                "empirical_p_vs_real": float(null_p),
                                "n": args.n_perm},
        "refusal_marker_baseline": {"acc_direct": ref_direct,
                                    "acc_flipped": ref_flipped},
        "chance": 0.5,
    }
    out_path = os.path.join(args.out_dir, f"probe5_{args.model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
