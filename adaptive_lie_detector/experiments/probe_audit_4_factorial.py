#!/usr/bin/env python3
"""probe_audit_4_factorial.py

Diagnostic 2 -- signed-effect factorial decomposition of the PROBE's output.

Mirrors analyze_factorial_signed_effects.py, but the outcome Y is the probe's
decision score (log-odds of "deceive") instead of a behavioral refusal count. To
avoid circularity we use LEAVE-ONE-CLAIM-PAIR-OUT out-of-fold scores: every
trial's score comes from a probe that never saw its claim pair.

    Y = b0 + bV*V + bE*E + bVE*(V*E)

    V = 1 (true claim shown), E = 1 (deception instruction present)

Y is standardized by its SD so coefficients are in probe-score SD units. If the
probe measured a stable deception construct we would expect a large, sign-stable
bE and a small bVE. The elicitation-confound prediction is the opposite of a
content probe: bE dominates and the veracity effect is interaction-bound.

Uncertainty is claim-pair clustered throughout (50 matched pairs are the
independent units). Permutation p shuffles E within (pair, V) strata.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_4_factorial.py --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import (  # noqa: E402
    DATA_DIR, SEED, load_pass, slice_layer, meta_vec,
)
from experiments.probe_audit_2_train_probe import locpo_predictions  # noqa: E402

B_BOOT = 2000
B_PERM = 2000


def design(V, E):
    return np.column_stack([np.ones(len(V)), V, E, V * E])


def fit(V, E, Y):
    return np.linalg.lstsq(design(V, E), Y, rcond=None)[0]


def clustered_bootstrap(V, E, Y, G, rng, B=B_BOOT):
    pairs = np.unique(G)
    idx_by_pair = {p: np.where(G == p)[0] for p in pairs}
    out = []
    for _ in range(B):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        idx = np.concatenate([idx_by_pair[p] for p in sampled])
        try:
            out.append(fit(V[idx], E[idx], Y[idx]))
        except np.linalg.LinAlgError:
            continue
    return np.array(out)


def permutation_p(V, E, Y, G, rng, coef_index, B=B_PERM):
    observed = fit(V, E, Y)[coef_index]
    strata = {}
    for i, (g, v) in enumerate(zip(G, V)):
        strata.setdefault((g, v), []).append(i)
    count = 0
    for _ in range(B):
        Ep = E.copy()
        for idx in strata.values():
            idx = np.array(idx)
            Ep[idx] = rng.permutation(E[idx])
        if abs(fit(V, Ep, Y)[coef_index]) >= abs(observed):
            count += 1
    return (count + 1) / (B + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    args = ap.parse_args()

    with open(os.path.join(args.out_dir, f"probe2_{args.model_tag}.json")) as f:
        sel = json.load(f)["selected"]
    pooling, layer = sel["pooling"], sel["layer"]

    arrays, meta = load_pass(args.model_tag, "instructed", args.out_dir)
    X = slice_layer(arrays, pooling, layer)
    V = meta_vec(meta, "V")
    E = meta_vec(meta, "E")
    G = meta_vec(meta, "pair_id")

    # Out-of-fold probe log-odds (probe never sees the scored claim pair)
    _, scores = locpo_predictions(X, E, G)
    Y = (scores - scores.mean()) / scores.std()  # SD units

    rng = np.random.RandomState(SEED)
    b = fit(V, E, Y)
    boot = clustered_bootstrap(V, E, Y, G, rng)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    p_E = permutation_p(V, E, Y, G, rng, coef_index=2)
    p_VE = permutation_p(V, E, Y, G, rng, coef_index=3)

    eff_false = b[2]            # E effect when V=0
    eff_true = b[2] + b[3]      # E effect when V=1

    print(f"Probe factorial on OOF log-odds ({pooling} layer {layer}), "
          f"n={len(Y)} trials, {len(np.unique(G))} pairs")
    print(f"  Y standardized to SD units")
    print(f"  bV  (veracity)     {b[1]:+.3f}  [{lo[1]:+.3f}, {hi[1]:+.3f}]")
    print(f"  bE  (instruction)  {b[2]:+.3f}  [{lo[2]:+.3f}, {hi[2]:+.3f}]  perm p={p_E:.4f}")
    print(f"  bVE (interaction)  {b[3]:+.3f}  [{lo[3]:+.3f}, {hi[3]:+.3f}]  perm p={p_VE:.4f}")
    print(f"  simple effect E | false claim  {eff_false:+.3f}")
    print(f"  simple effect E | true  claim  {eff_true:+.3f}")

    out = {
        "experiment": "probe_audit_4_factorial",
        "model_tag": args.model_tag,
        "pooling": pooling, "layer": layer,
        "outcome": "probe LOCPO out-of-fold log-odds, standardized to SD units",
        "n_trials": int(len(Y)), "n_claim_pairs": int(len(np.unique(G))),
        "beta_V": float(b[1]), "beta_V_ci": [float(lo[1]), float(hi[1])],
        "beta_E": float(b[2]), "beta_E_ci": [float(lo[2]), float(hi[2])],
        "beta_VE": float(b[3]), "beta_VE_ci": [float(lo[3]), float(hi[3])],
        "perm_p_beta_E": float(p_E), "perm_p_beta_VE": float(p_VE),
        "simple_effect_E_given_false": float(eff_false),
        "simple_effect_E_given_true": float(eff_true),
    }
    out_path = os.path.join(args.out_dir, f"probe4_{args.model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
