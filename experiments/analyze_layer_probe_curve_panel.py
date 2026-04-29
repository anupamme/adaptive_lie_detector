#!/usr/bin/env python3
"""Render 2-panel per-layer x {last-token, mean-pool} probe accuracy curves.

Panel (a): Llama 3.2 3B (from multilayer_probe_curve_llama3b.json)
Panel (b): Mistral 7B base checkpoint (from multilayer_probe_curve_mistral7b.json)

Writes output/adaptive_lie_detector_paper/figures/probe_layer_curve_panel.pdf
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE.parent.parent / "output" / "adaptive_lie_detector_paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "probe_layer_curve_panel.pdf"

PANELS = [
    {
        "json": BASE / "data" / "whitebox_probing" / "multilayer_probe_curve_llama3b.json",
        "title": "(a) Llama 3.2 3B Instruct (n=100)",
        "rule_acc": 61,
    },
    {
        "json": BASE / "data" / "whitebox_probing" / "multilayer_probe_curve_mistral7b.json",
        "title": "(b) Mistral 7B base checkpoint (n=100)$^\\dagger$",
        "rule_acc": 68,
    },
]


def _plot_panel(ax, json_path, title, rule_acc):
    with open(json_path) as f:
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

    ax.fill_between(layers, lt_lo, lt_hi, color="C0", alpha=0.15)
    ax.plot(layers, lt_acc, "o-", color="C0", label="last-token", markersize=3.5, linewidth=1.2)
    ax.fill_between(layers, mp_lo, mp_hi, color="C1", alpha=0.15)
    ax.plot(layers, mp_acc, "s-", color="C1", label="mean-pool", markersize=3.5, linewidth=1.2)

    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, label="chance")
    ax.axhline(rule_acc, color="C2", linestyle="--", linewidth=1.0,
               label=f"refusal-count rule ({rule_acc}%)")

    ax.annotate(f"L{layers[best_lt_idx]}={lt_acc[best_lt_idx]:.0f}%",
                xy=(layers[best_lt_idx], lt_acc[best_lt_idx]),
                xytext=(layers[best_lt_idx] + 1, lt_acc[best_lt_idx] + 4),
                fontsize=7, color="C0",
                arrowprops=dict(arrowstyle="->", color="C0", lw=0.5))
    ax.annotate(f"L{layers[best_mp_idx]}={mp_acc[best_mp_idx]:.0f}%",
                xy=(layers[best_mp_idx], mp_acc[best_mp_idx]),
                xytext=(layers[best_mp_idx] + 1, mp_acc[best_mp_idx] - 9),
                fontsize=7, color="C1",
                arrowprops=dict(arrowstyle="->", color="C1", lw=0.5))

    ax.set_xlabel(f"Layer index (0 = embedding, {n_layers} = final)")
    ax.set_ylabel("LOO accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(25, 75)
    ax.set_xlim(-0.5, n_layers + 0.5)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    return {
        "best_lt_layer": layers[best_lt_idx],
        "best_lt_acc": float(lt_acc[best_lt_idx]),
        "best_mp_layer": layers[best_mp_idx],
        "best_mp_acc": float(mp_acc[best_mp_idx]),
        "final_lt_acc": float(lt_acc[-1]),
        "final_mp_acc": float(mp_acc[-1]),
    }


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    stats = []
    for ax, panel in zip(axes, PANELS):
        s = _plot_panel(ax, panel["json"], panel["title"], panel["rule_acc"])
        stats.append((panel["title"], s))

    plt.tight_layout()
    plt.savefig(OUT_PATH, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    for title, s in stats:
        print(f"  {title}")
        print(f"    best last-token: L{s['best_lt_layer']} -> {s['best_lt_acc']:.1f}%  (final: {s['final_lt_acc']:.1f}%)")
        print(f"    best mean-pool:  L{s['best_mp_layer']} -> {s['best_mp_acc']:.1f}%  (final: {s['final_mp_acc']:.1f}%)")


if __name__ == "__main__":
    main()
