#!/usr/bin/env python3
"""plot_probe_audit.py

One figure summarizing the probe audit, built from cached JSON (no model):
  (a) LOCPO decoding accuracy vs. layer (per pooling)
  (b) factorial coefficients bV / bE / bVE on the probe log-odds
  (c) steering dose-response: probe direction vs. random control

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/plot_probe_audit.py --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import DATA_DIR  # noqa: E402


def _load(out_dir, name, tag):
    path = os.path.join(out_dir, f"{name}_{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    args = ap.parse_args()
    tag = args.model_tag

    p2 = _load(args.out_dir, "probe2", tag)
    p4 = _load(args.out_dir, "probe4", tag)
    p6 = _load(args.out_dir, "probe6", tag)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) layer curve
    ax = axes[0]
    if p2:
        for pooling, curve in p2["layer_curve"].items():
            ax.plot(range(len(curve)), np.array(curve) * 100, label=pooling, lw=1.5)
        ax.axhline(50, ls="--", c="gray", lw=1, label="chance")
        s = p2["selected"]
        ax.scatter([s["layer"]], [s["acc"] * 100], c="red", zorder=5, s=40)
    ax.set_xlabel("hidden-state layer"); ax.set_ylabel("LOCPO accuracy (%)")
    ax.set_title("(a) Decode deception instruction E"); ax.legend(fontsize=7)

    # (b) factorial coefficients
    ax = axes[1]
    if p4:
        names = ["bV\n(veracity)", "bE\n(instruction)", "bVE\n(interaction)"]
        vals = [p4["beta_V"], p4["beta_E"], p4["beta_VE"]]
        cis = [p4["beta_V_ci"], p4["beta_E_ci"], p4["beta_VE_ci"]]
        err = [[v - c[0] for v, c in zip(vals, cis)],
               [c[1] - v for v, c in zip(vals, cis)]]
        colors = ["#888", "#c0392b", "#2980b9"]
        ax.bar(names, vals, yerr=err, color=colors, capsize=4)
        ax.axhline(0, c="k", lw=0.8)
    ax.set_ylabel("coefficient (probe-score SD)")
    ax.set_title("(b) Factorial on probe log-odds")

    # (c) steering dose-response. coef is denominated in SDs of the probe score,
    # and real scores only span ~+-1.5 SD, so shade the on-manifold band and use a
    # symlog x-axis -- otherwise the +-8 doses squash the doses that actually matter.
    ax = axes[2]
    if p6:
        coefs = sorted({float(k) for row in p6["results"].values() for k in row})
        pos = {c: i for i, c in enumerate(coefs)}
        on = [i for c, i in pos.items() if abs(c) <= 1.5]
        ax.axvspan(min(on) - 0.5, max(on) + 0.5, color="#2ecc71", alpha=0.12, zorder=0)
        for name, style in (("probe", "-o"), ("random", "--s")):
            row = p6["results"].get(name, {})
            keys = sorted(row, key=float)
            xs = [pos[float(k)] for k in keys]
            ys = [row[k]["mean_refusal"] for k in keys]
            ax.plot(xs, ys, style, label=name, ms=4)
        base = p6["results"].get("probe", {}).get("0.0", {}).get("mean_refusal")
        if base is not None:
            ax.axhline(base, ls=":", c="gray", lw=1)
            ax.text(0.02, base + 0.004, "unsteered", fontsize=6.5, color="gray",
                    transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(coefs)))
        ax.set_xticklabels([f"{c:g}" for c in coefs], fontsize=7)
        ax.set_ylim(0.25, 0.50)
        ax.legend(fontsize=8, loc="upper center")
        ax.text(0.5, 0.03, "shaded = on-manifold (|coef| ≤ 1.5 SD of the probe score)",
                fontsize=6.5, ha="center", transform=ax.transAxes, color="#1e8449")
    ax.set_xlabel("steering coef (× sigma = SDs of the probe score)")
    ax.set_ylabel("refusal-marker density")
    ax.set_title("(c) Steering: flat on-manifold, breakage off-manifold")

    fig.suptitle(f"Probe audit -- {tag}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(args.out_dir, f"probe_audit_{tag}.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
