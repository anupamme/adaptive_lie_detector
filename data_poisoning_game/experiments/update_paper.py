"""
Auto-updates paper/main.tex tables with results from payoff_results.json and game_analysis.json.
Run this after run_full_payoff_matrix and run_game_analysis complete.
"""
import json
import os
import sys
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir="results"):
    payoff_path = os.path.join(results_dir, "payoff_results.json")
    analysis_path = os.path.join(results_dir, "game_analysis.json")
    with open(payoff_path) as f:
        payoff = json.load(f)
    with open(analysis_path) as f:
        analysis = json.load(f)
    return payoff, analysis


def build_accuracy_table(payoff):
    attacks = ["no_attack", "label_flip", "backdoor_pixel", "backdoor_edge_case", "model_scaling", "dba"]
    defenses = ["fedavg", "krum", "multi_krum", "trimmed_mean", "coord_median", "norm_clip", "rfa"]
    attack_labels = ["No attack", "Label flip", "Backdoor (pixel)", "Edge-case", "Model scaling", "DBA"]
    defense_labels = ["FedAvg", "Krum", "M-Krum", "TrMean", "Median", "NClip", "RFA"]

    acc_rows = []
    asr_rows = []

    for atk, atk_label in zip(attacks, attack_labels):
        acc_vals = []
        asr_vals = []
        for dfn in defenses:
            key = f"{atk}_{dfn}"
            r = payoff.get(key, {})
            acc_vals.append(r.get("accuracy", 0.0))
            asr_vals.append(r.get("attack_success_rate", 0.0))
        acc_rows.append((atk_label, acc_vals))
        if atk in ("backdoor_pixel", "model_scaling", "dba"):
            asr_rows.append((atk_label, asr_vals))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Clean test accuracy and attack success rate (ASR) for each (attack, defense) pair on CIFAR-10 with $\alpha=0.5$, $f=0.2$. Backdoor attacks (pixel, model scaling, DBA) report ASR; others report 0.}")
    lines.append(r"\label{tab:accuracy}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l|ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Attack} & " + " & ".join(defense_labels) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{c}{\textit{Clean Test Accuracy}} \\")
    lines.append(r"\midrule")
    for atk_label, vals in acc_rows:
        row = atk_label + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\"
        lines.append(row)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{c}{\textit{Attack Success Rate (backdoor attacks only)}} \\")
    lines.append(r"\midrule")
    for atk_label, vals in asr_rows:
        row = atk_label + " & " + " & ".join(f"{v:.2f}" for v in vals) + r" \\"
        lines.append(row)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_equilibria_table(analysis):
    nes = analysis["nash_equilibria"]
    attacks = analysis["payoff_matrix"]["attacks"]
    defenses = analysis["payoff_matrix"]["defenses"]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Nash equilibria of the empirical game. $U_A$ and $U_D$ denote equilibrium utilities. VoPD is the Value of Private Defense.}")
    lines.append(r"\label{tab:equilibria}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{c|ll|cc|c}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{NE} & \textbf{Adversary Strategy} & \textbf{Server Strategy} & $U_A$ & $U_D$ & VoPD \\")
    lines.append(r"\midrule")

    for idx, ne in enumerate(nes):
        adv_strat = np.array(ne["adversary_strategy"])
        srv_strat = np.array(ne["server_strategy"])
        ua = ne["adversary_utility"]
        ud = ne["server_utility"]
        vopd = ne["value_of_information"]

        # Build human-readable strategy descriptions
        adv_parts = []
        for i, p in enumerate(adv_strat):
            if p > 0.01:
                label = attacks[i].replace("_", " ").title()
                adv_parts.append(f"{label} ({p*100:.0f}\\%)")
        srv_parts = []
        for j, p in enumerate(srv_strat):
            if p > 0.01:
                label = defenses[j].replace("_", " ").title()
                srv_parts.append(f"{label} ({p*100:.0f}\\%)")

        adv_str = " + ".join(adv_parts) if adv_parts else "Uniform"
        srv_str = " + ".join(srv_parts) if srv_parts else "Uniform"

        lines.append(f"{idx+1} & {adv_str} & {srv_str} & {ua:.3f} & {ud:.3f} & {vopd:.3f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_vopd_table(analysis):
    nes = analysis["nash_equilibria"]
    stk = analysis["stackelberg"]

    # Prefer the mixed-strategy NE (highest VoPD); fall back to highest adv utility
    best_ne = max(nes, key=lambda ne: ne["value_of_information"]) if nes else None
    if best_ne and best_ne["value_of_information"] == 0:
        best_ne = max(nes, key=lambda ne: ne["adversary_utility"])

    # Full-info: max over all attacks vs best defense for adversary
    adv_payoffs = np.array(analysis["payoff_matrix"]["adversary_payoffs"])
    full_info_adv = float(adv_payoffs.max())
    full_info_srv = None  # Not directly from analysis

    # Stackelberg
    stk_adv = stk["adversary_utility"] if stk else None
    stk_srv = stk["server_utility"] if stk else None

    # BNE
    bne_adv = best_ne["adversary_utility"] if best_ne else None
    bne_srv = best_ne["server_utility"] if best_ne else None
    vopd = best_ne["value_of_information"] if best_ne else None

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Adversary utility under different information structures. The value of private defense quantifies the benefit of keeping the defense secret.}")
    lines.append(r"\label{tab:vopd}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Scenario} & \textbf{Adversary Utility} & \textbf{Server Utility} \\")
    lines.append(r"\midrule")
    lines.append(f"Full information (adversary knows defense) & {full_info_adv:.3f} & -- \\\\")
    if stk_adv is not None:
        lines.append(f"Stackelberg (server commits, adversary observes) & {stk_adv:.3f} & {stk_srv:.3f} \\\\")
    if bne_adv is not None:
        lines.append(f"Bayes-Nash (defense private) & {bne_adv:.3f} & {bne_srv:.3f} \\\\")
    lines.append(r"\midrule")
    if vopd is not None:
        lines.append(r"\textbf{Value of Private Defense (VoPD)} & \multicolumn{2}{c}{\textbf{" + f"{vopd:.3f}" + r"}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def update_paper(tex_path="paper/main.tex", results_dir="results"):
    payoff, analysis = load_results(results_dir)

    with open(tex_path) as f:
        content = f.read()

    # Replace Table 1 (accuracy)
    new_table1 = build_accuracy_table(payoff)
    content = re.sub(
        r"\\begin\{table\}.*?\\label\{tab:accuracy\}.*?\\end\{table\}",
        lambda _: new_table1,
        content,
        flags=re.DOTALL
    )

    # Replace Table 2 (equilibria)
    new_table2 = build_equilibria_table(analysis)
    content = re.sub(
        r"\\begin\{table\}.*?\\label\{tab:equilibria\}.*?\\end\{table\}",
        lambda _: new_table2,
        content,
        flags=re.DOTALL
    )

    # Replace Table 3 (VoPD)
    new_table3 = build_vopd_table(analysis)
    content = re.sub(
        r"\\begin\{table\}.*?\\label\{tab:vopd\}.*?\\end\{table\}",
        lambda _: new_table3,
        content,
        flags=re.DOTALL
    )

    with open(tex_path, "w") as f:
        f.write(content)

    print(f"Updated {tex_path} with new experimental results.")

    # Print summary for manual verification
    print("\nKey numbers for abstract/conclusion update:")
    nes = analysis["nash_equilibria"]
    if nes:
        best_ne = max(nes, key=lambda ne: ne["adversary_utility"])
        worst_ne = min(nes, key=lambda ne: ne["adversary_utility"])
        adv_payoffs = np.array(analysis["payoff_matrix"]["adversary_payoffs"])
        full_info_adv = float(adv_payoffs.max())
        vopd = best_ne["value_of_information"]
        reduction_pct = (full_info_adv - worst_ne["adversary_utility"]) / full_info_adv * 100
        print(f"  Full-info adversary utility: {full_info_adv:.3f}")
        print(f"  BNE adversary utility range: {worst_ne['adversary_utility']:.3f} - {best_ne['adversary_utility']:.3f}")
        print(f"  VoPD: {vopd:.3f}")
        print(f"  Adversary utility reduction: {reduction_pct:.1f}%")
        print(f"  Number of Nash equilibria: {len(nes)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex_path", default="paper/main.tex")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    update_paper(args.tex_path, args.results_dir)
