#!/usr/bin/env python3
"""Render per-layer × {last-token, mean-pool} probe accuracy curve for Llama 3B.

Reads data/whitebox_probing/multilayer_probe_curve_llama3b.json
Writes output/adaptive_lie_detector_paper/figures/probe_layer_curve_llama3b.pdf
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
IN_PATH = BASE / "data" / "whitebox_probing" / "multilayer_probe_curve_llama3b.json"
OUT_DIR = BASE.parent.parent / "output" / "adaptive_lie_detector_paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "probe_layer_curve_llama3b.pdf"


def main():
    with open(IN_PATH) as f:
        d = json.load(f)
    r = d["results"][0]
    curve = r["layer_curve"]
    n_layers = r["n_layers"]

    layers = sorted(int(k) for k in curve.keys())
    lt_acc = np.array([curve[str(li)]["last_token"]["acc"] for li in layers]) * 100
    lt_lo = np.array([curve[str(li)]["last_token"]["ci"][0] for li in layers]) * 100
    lt_hi = np.array([curve[str(li)]["last_token"]["ci"][1] for li in layers]) * 100
    mp_acc = np.array([curve[str(li)]["mean_pool"]["acc"] for li in layers]) * 100
    mp_lo = np.array([curve[str(li)]["mean_pool"]["ci"][0] for li in layers]) * 100
    mp_hi = np.array([curve[str(li)]["mean_pool"]["ci"][1] for li in layers]) * 100

    best_lt_idx = int(np.argmax(lt_acc))
    best_mp_idx = int(np.argmax(mp_acc))

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.fill_between(layers, lt_lo, lt_hi, color="C0", alpha=0.15)
    ax.plot(layers, lt_acc, "o-", color="C0", label="last-token", markersize=4, linewidth=1.5)
    ax.fill_between(layers, mp_lo, mp_hi, color="C1", alpha=0.15)
    ax.plot(layers, mp_acc, "s-", color="C1", label="mean-pool", markersize=4, linewidth=1.5)

    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.axhline(61, color="C2", linestyle="--", linewidth=1.0, label="refusal-count rule (61%)")

    ax.annotate(f"best: L{layers[best_lt_idx]} = {lt_acc[best_lt_idx]:.0f}%",
                xy=(layers[best_lt_idx], lt_acc[best_lt_idx]),
                xytext=(layers[best_lt_idx] + 1, lt_acc[best_lt_idx] + 4),
                fontsize=8, color="C0",
                arrowprops=dict(arrowstyle="->", color="C0", lw=0.6))
    ax.annotate(f"best: L{layers[best_mp_idx]} = {mp_acc[best_mp_idx]:.0f}%",
                xy=(layers[best_mp_idx], mp_acc[best_mp_idx]),
                xytext=(layers[best_mp_idx] + 1, mp_acc[best_mp_idx] - 8),
                fontsize=8, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=0.6))

    ax.set_xlabel(f"Layer index (0 = embedding, {n_layers} = final)")
    ax.set_ylabel("LOO accuracy (%)")
    ax.set_title(f"Llama 3.2 3B equalized-transcript probe: per-layer × pooling (n=100)")
    ax.set_ylim(25, 75)
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    print(f"best last-token: layer {layers[best_lt_idx]} -> {lt_acc[best_lt_idx]:.1f}%")
    print(f"best mean-pool:  layer {layers[best_mp_idx]} -> {mp_acc[best_mp_idx]:.1f}%")


if __name__ == "__main__":
    main()
