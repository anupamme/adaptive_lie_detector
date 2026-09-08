"""
Generates publication-quality figures for the paper.
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_payoff_heatmap(analysis_path: str, output_dir: str = "paper/figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(analysis_path) as f:
        data = json.load(f)

    pm = data["payoff_matrix"]
    attacks = pm["attacks"]
    defenses = pm["defenses"]
    adv_payoffs = np.array(pm["adversary_payoffs"])
    srv_payoffs = np.array(pm["server_payoffs"])

    attack_labels = [a.replace("_", " ").title() for a in attacks]
    defense_labels = [d.replace("_", " ").title() for d in defenses]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(adv_payoffs, ax=axes[0], annot=True, fmt=".3f", cmap="RdYlGn_r",
                xticklabels=defense_labels, yticklabels=attack_labels)
    axes[0].set_title("Adversary Payoff Matrix")
    axes[0].set_xlabel("Server Defense")
    axes[0].set_ylabel("Adversary Attack")

    sns.heatmap(srv_payoffs, ax=axes[1], annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=defense_labels, yticklabels=attack_labels)
    axes[1].set_title("Server Payoff Matrix")
    axes[1].set_xlabel("Server Defense")
    axes[1].set_ylabel("Adversary Attack")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "payoff_heatmaps.pdf"))
    plt.close()
    print(f"Saved: {output_dir}/payoff_heatmaps.pdf")


def plot_equilibrium_strategies(analysis_path: str, output_dir: str = "paper/figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(analysis_path) as f:
        data = json.load(f)

    if not data["nash_equilibria"]:
        print("No Nash equilibria to plot.")
        return

    ne = data["nash_equilibria"][0]
    attacks = data["payoff_matrix"]["attacks"]
    defenses = data["payoff_matrix"]["defenses"]
    adv_strat = np.array(ne["adversary_strategy"])
    srv_strat = np.array(ne["server_strategy"])

    attack_labels = [a.replace("_", " ").title() for a in attacks]
    defense_labels = [d.replace("_", " ").title() for d in defenses]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors_adv = ["#d32f2f" if p > 0.01 else "#e0e0e0" for p in adv_strat]
    axes[0].barh(attack_labels, adv_strat, color=colors_adv, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Probability")
    axes[0].set_title("Adversary Mixed Strategy (NE)")
    axes[0].set_xlim(0, 1)

    colors_srv = ["#1976d2" if p > 0.01 else "#e0e0e0" for p in srv_strat]
    axes[1].barh(defense_labels, srv_strat, color=colors_srv, edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Server Mixed Strategy (NE)")
    axes[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equilibrium_strategies.pdf"))
    plt.close()
    print(f"Saved: {output_dir}/equilibrium_strategies.pdf")


def plot_fictitious_play_convergence(analysis_path: str, output_dir: str = "paper/figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(analysis_path) as f:
        data = json.load(f)

    convergence = data["fictitious_play"]["convergence"]
    iterations = [i * 100 for i in range(1, len(convergence) + 1)]

    # FP converges to NE1 (backdoor_pixel vs FedAvg), which is index 0
    ne_utility = data["nash_equilibria"][0]["adversary_utility"] if data["nash_equilibria"] else None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iterations, convergence, "b-", linewidth=1.5, label="Fictitious Play")
    if ne_utility is not None:
        ax.axhline(y=ne_utility, color="r", linestyle="--", linewidth=1.5, label=f"NE1 Utility ({ne_utility:.3f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Adversary Expected Utility")
    ax.set_title("Convergence of Fictitious Play to Nash Equilibrium")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fictitious_play_convergence.pdf"))
    plt.close()
    print(f"Saved: {output_dir}/fictitious_play_convergence.pdf")


def plot_heterogeneity_sweep(sweep_path: str, output_dir: str = "paper/figures"):
    os.makedirs(output_dir, exist_ok=True)
    with open(sweep_path) as f:
        data = json.load(f)

    alphas = sorted(set(v["alpha"] for v in data.values()))
    fractions = sorted(set(v["adversarial_fraction"] for v in data.values()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for f_val in fractions:
        utilities = []
        for alpha in alphas:
            key = f"alpha{alpha}_f{f_val}"
            if key in data and data[key]["nash_adversary_utility"] is not None:
                utilities.append(data[key]["nash_adversary_utility"])
            else:
                utilities.append(np.nan)
        axes[0].plot(alphas, utilities, "o-", label=f"f={f_val}", markersize=6)

    axes[0].set_xlabel(r"Dirichlet $\alpha$ (heterogeneity)")
    axes[0].set_ylabel("Adversary Utility at NE")
    axes[0].set_title("Effect of Data Heterogeneity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log")

    for alpha in alphas:
        vois = []
        for f_val in fractions:
            key = f"alpha{alpha}_f{f_val}"
            if key in data and data[key]["value_of_information"] is not None:
                vois.append(data[key]["value_of_information"])
            else:
                vois.append(np.nan)
        axes[1].plot(fractions, vois, "s-", label=rf"$\alpha$={alpha}", markersize=6)

    axes[1].set_xlabel("Adversarial Fraction (f)")
    axes[1].set_ylabel("Value of Private Defense")
    axes[1].set_title("Value of Information vs. Adversarial Fraction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heterogeneity_sweep.pdf"))
    plt.close()
    print(f"Saved: {output_dir}/heterogeneity_sweep.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis_path", type=str, default="results/game_analysis.json")
    parser.add_argument("--sweep_path", type=str, default="results/sweep_summary.json")
    parser.add_argument("--output_dir", type=str, default="paper/figures")
    args = parser.parse_args()

    if os.path.exists(args.analysis_path):
        plot_payoff_heatmap(args.analysis_path, args.output_dir)
        plot_equilibrium_strategies(args.analysis_path, args.output_dir)
        plot_fictitious_play_convergence(args.analysis_path, args.output_dir)

    if os.path.exists(args.sweep_path):
        plot_heterogeneity_sweep(args.sweep_path, args.output_dir)
