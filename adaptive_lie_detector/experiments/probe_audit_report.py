#!/usr/bin/env python3
"""probe_audit_report.py

Aggregate the probe-audit JSONs (steps 2-6) into a single console report + a
machine-readable summary. Regenerates entirely from cached results without
touching the model, so the headline numbers are reproducible from disk.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_report.py --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import DATA_DIR  # noqa: E402


def _load(out_dir, name, tag):
    path = os.path.join(out_dir, f"{name}_{tag}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_ci(ci):
    return f"[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--out_dir", default=DATA_DIR)
    args = ap.parse_args()
    tag = args.model_tag

    p2 = _load(args.out_dir, "probe2", tag)
    p3 = _load(args.out_dir, "probe3", tag)
    p4 = _load(args.out_dir, "probe4", tag)
    p5 = _load(args.out_dir, "probe5", tag)
    p6 = _load(args.out_dir, "probe6", tag)

    bar = "=" * 74
    print(bar)
    print(f"  PROBE AUDIT REPORT -- {tag}")
    print("  Does a deception probe detect deception, or the instruction to lie?")
    print(bar)

    summary = {"model_tag": tag}

    if p2:
        s = p2["selected"]
        print(f"\n[1] Decoding (LOCPO, predict deception instruction E)")
        print(f"    selected: {s['pooling']} layer {s['layer']}")
        print(f"    accuracy: {s['acc']*100:.1f}%  95% CI {_fmt_ci(s['ci95'])}")
        summary["decoding_acc"] = s["acc"]

    if p3:
        print(f"\n[2] Equalization (remove the instruction asymmetry)")
        print(f"    detector contrast T_H vs F_D : {p3['A_instructed']*100:5.1f}%  "
              f"{_fmt_ci(p3['ci_instructed'])}")
        print(f"    equalized contrast T_N vs F_N: {p3['A_equalized']*100:5.1f}%  "
              f"{_fmt_ci(p3['ci_equalized'])}")
        print(f"    transfer (instr->eq)         : {p3['A_transfer']*100:5.1f}%")
        print(f"    accuracy DROP                : {p3['accuracy_drop']*100:5.1f} pp")
        pe = p3["eprobe_mean_p_e1_equalized"]
        print(f"    E-probe mean P(E=1) on neutral prompt: "
              f"T_N={pe['T_N']:.2f} F_N={pe['F_N']:.2f}")
        summary["equalization_drop"] = p3["accuracy_drop"]

    if p4:
        print(f"\n[3] Factorial on probe log-odds (SD units)")
        print(f"    bV  (veracity)    {p4['beta_V']:+.3f}  {_fmt_ci_raw(p4['beta_V_ci'])}")
        print(f"    bE  (instruction) {p4['beta_E']:+.3f}  {_fmt_ci_raw(p4['beta_E_ci'])}"
              f"  perm p={p4['perm_p_beta_E']:.4f}")
        print(f"    bVE (interaction) {p4['beta_VE']:+.3f}  {_fmt_ci_raw(p4['beta_VE_ci'])}"
              f"  perm p={p4['perm_p_beta_VE']:.4f}")
        print(f"    simple E|false={p4['simple_effect_E_given_false']:+.3f}  "
              f"E|true={p4['simple_effect_E_given_true']:+.3f}")
        summary["beta_E"] = p4["beta_E"]
        summary["beta_V"] = p4["beta_V"]

    if p5:
        print(f"\n[4] Baselines / null controls")
        print(f"    real probe            : {p5['real_probe_acc']*100:5.1f}%")
        print(f"    random-direction probe: {p5['random_direction']['mean']*100:5.1f}% "
              f"(max {p5['random_direction']['max']*100:.1f})")
        print(f"    shuffled-label null   : {p5['shuffled_label_null']['mean']*100:5.1f}% "
              f"(empirical p={p5['shuffled_label_null']['empirical_p_vs_real']:.3f})")
        rb = p5["refusal_marker_baseline"]
        print(f"    refusal-marker (behav): direct {rb['acc_direct']*100:.1f}%  "
              f"flipped {rb['acc_flipped']*100:.1f}%")

    if p6:
        print(f"\n[5] Causal steering (coef * sigma along direction; readout=refusal density)")
        print("    coef is denominated in SDs of the probe score (see WRITEUP 5); "
              "real scores span ~+-1.5 SD,")
        print("    so |coef| <= 1 is on-manifold and |coef| >= 2 is off-manifold.")
        for name in ("probe", "random"):
            row = p6["results"].get(name, {})
            cells = " ".join(f"{float(c):+5.2f}:{row[c]['mean_refusal']:.3f}"
                             for c in sorted(row, key=float))
            print(f"    {name:6s}: {cells}")
        # probe-vs-random contrast, split by whether the dose is on-manifold
        pr, rn = p6["results"].get("probe", {}), p6["results"].get("random", {})
        for label, keep in (("on-manifold  (|coef|<=1)", lambda c: 0 < abs(c) <= 1),
                            ("off-manifold (|coef|>=2)", lambda c: abs(c) >= 2)):
            cs = [float(c) for c in pr if keep(float(c)) and c in rn]
            if not cs:
                continue
            dif = [pr[str(c)]["mean_refusal"] - rn[str(c)]["mean_refusal"] for c in cs]
            mp = sum(pr[str(c)]["mean_refusal"] for c in cs) / len(cs)
            mr = sum(rn[str(c)]["mean_refusal"] for c in cs) / len(cs)
            print(f"    {label}: probe {mp:.3f} vs random {mr:.3f}  "
                  f"(diff {sum(dif)/len(dif):+.4f}, n_doses={len(cs)})")

    print("\n" + bar)
    out_path = os.path.join(args.out_dir, f"probe_audit_summary_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary -> {out_path}")


def _fmt_ci_raw(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


if __name__ == "__main__":
    main()
