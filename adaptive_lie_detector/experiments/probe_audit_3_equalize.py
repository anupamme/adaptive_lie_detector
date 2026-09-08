#!/usr/bin/env python3
"""probe_audit_3_equalize.py

Diagnostic 1 -- condition equalization for the probe.

The "detector" contrast in a deception benchmark compares the honest-on-true arm
(T_H) against the deceive-on-false arm (F_D). In that contrast the deception
instruction E and the claim veracity V move together and are both confounded with
the *system prompt*. Equalization re-runs the SAME claims under a single NEUTRAL
prompt (T_N, F_N), removing the instruction asymmetry, and asks: does the probe's
separability survive?

  A_instructed = LOCPO acc, probe separates T_H vs F_D   (confounded detector)
  A_equalized  = LOCPO acc, probe separates T_N vs F_N   (same claims, no instr.)
  A_transfer   = train on {T_H,F_D}, evaluate on {T_N,F_N}

If A_instructed is high but A_equalized (and A_transfer) collapse toward chance,
the probe was reading the instruction/prompt asymmetry, not a content-borne
deception construct. We also report the E-probe's mean P(E=1) on the equalized
cells: an instruction-reading probe should score every neutral-prompt trial as
honest.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_3_equalize.py --model_tag Qwen3-4B-Instruct-2507
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
from experiments.probe_audit_2_train_probe import (  # noqa: E402
    locpo_predictions, clustered_bootstrap_ci,
)


def subset(arrays, meta, pooling, layer, cells):
    """Return (X, pair_ids, keep_mask) restricted to `cells`."""
    cell_vec = np.array([m["cell"] for m in meta])
    mask = np.isin(cell_vec, cells)
    X = slice_layer(arrays, pooling, layer)[mask]
    pairs = meta_vec(meta, "pair_id")[mask]
    return X, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    # Selected (pooling, layer) from step 2
    with open(os.path.join(args.out_dir, f"probe2_{args.model_tag}.json")) as f:
        sel = json.load(f)["selected"]
    pooling, layer = sel["pooling"], sel["layer"]
    print(f"Using selected probe config: {pooling} layer {layer}")

    ins_arr, ins_meta = load_pass(args.model_tag, "instructed", args.out_dir)
    eq_arr, eq_meta = load_pass(args.model_tag, "equalized", args.out_dir)

    # --- Detector contrast (instructed): T_H (arm 0) vs F_D (arm 1) ---
    Xi, pi = subset(ins_arr, ins_meta, pooling, layer, ["T_H", "F_D"])
    yi = np.array([0 if c == "T_H" else 1
                   for c in [m["cell"] for m in ins_meta
                             if m["cell"] in ("T_H", "F_D")]])
    pred_i, _ = locpo_predictions(Xi, yi, pi)
    A_instr = float((pred_i == yi).mean())
    ci_instr = clustered_bootstrap_ci(yi, pred_i, pi, B=args.boot)

    # --- Equalized contrast: T_N (arm 0) vs F_N (arm 1) ---
    Xe, pe = subset(eq_arr, eq_meta, pooling, layer, ["T_N", "F_N"])
    ye = np.array([0 if c == "T_N" else 1
                   for c in [m["cell"] for m in eq_meta
                             if m["cell"] in ("T_N", "F_N")]])
    pred_e, _ = locpo_predictions(Xe, ye, pe)
    A_equal = float((pred_e == ye).mean())
    ci_equal = clustered_bootstrap_ci(ye, pred_e, pe, B=args.boot)

    # --- Transfer: train on instructed {T_H,F_D}, evaluate on equalized {T_N,F_N} ---
    clf = make_probe()
    clf.fit(Xi, yi)
    A_transfer = float((clf.predict(Xe) == ye).mean())

    # --- E-probe (all 4 instructed cells -> E) applied to equalized cells ---
    Xall = slice_layer(ins_arr, pooling, layer)
    Eall = meta_vec(ins_meta, "E")
    eprobe = make_probe()
    eprobe.fit(Xall, Eall)
    Xeq_all = slice_layer(eq_arr, pooling, layer)
    p_e1 = eprobe.predict_proba(Xeq_all)[:, 1]
    eq_cells = np.array([m["cell"] for m in eq_meta])
    mean_pe1 = {c: float(p_e1[eq_cells == c].mean()) for c in ("T_N", "F_N")}

    drop = A_instr - A_equal
    print(f"\n  A_instructed (T_H vs F_D): {A_instr*100:5.1f}%  "
          f"[{ci_instr[0]*100:.1f}-{ci_instr[1]*100:.1f}]")
    print(f"  A_equalized  (T_N vs F_N): {A_equal*100:5.1f}%  "
          f"[{ci_equal[0]*100:.1f}-{ci_equal[1]*100:.1f}]")
    print(f"  A_transfer   (instr->eq) : {A_transfer*100:5.1f}%")
    print(f"  drop (instr - equal)     : {drop*100:5.1f} pp")
    print(f"  E-probe mean P(E=1) on equalized: "
          f"T_N={mean_pe1['T_N']:.2f}  F_N={mean_pe1['F_N']:.2f}")

    out = {
        "experiment": "probe_audit_3_equalize",
        "model_tag": args.model_tag,
        "pooling": pooling, "layer": layer,
        "A_instructed": A_instr, "ci_instructed": list(ci_instr),
        "A_equalized": A_equal, "ci_equalized": list(ci_equal),
        "A_transfer": A_transfer,
        "accuracy_drop": drop,
        "eprobe_mean_p_e1_equalized": mean_pe1,
        "chance": 0.5,
    }
    out_path = os.path.join(args.out_dir, f"probe3_{args.model_tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
