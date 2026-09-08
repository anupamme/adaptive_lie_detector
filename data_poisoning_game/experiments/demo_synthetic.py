"""
Demo: runs the full game-theoretic analysis pipeline using a synthetic
payoff matrix. Useful for verifying the pipeline without GPU training.

Usage: python3 experiments/demo_synthetic.py
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_theory import GameSolver, PayoffMatrix


def create_synthetic_payoff_matrix(alpha: float = 0.5, f: float = 0.2) -> PayoffMatrix:
    attacks = ["no_attack", "label_flip", "backdoor_pixel", "model_scaling", "dba"]
    defenses = ["fedavg", "krum", "trimmed_mean", "norm_clip", "rfa"]

    heterogeneity_factor = 1.0 / (1.0 + alpha)
    fraction_factor = f / 0.2

    base_adv = np.array([
        [0.00, 0.00, 0.00, 0.00, 0.00],
        [0.30, 0.08, 0.12, 0.18, 0.10],
        [0.70, 0.12, 0.20, 0.05, 0.15],
        [0.80, 0.04, 0.55, 0.10, 0.03],
        [0.50, 0.30, 0.15, 0.28, 0.08],
    ])

    base_srv = np.array([
        [0.93, 0.87, 0.90, 0.91, 0.86],
        [0.82, 0.86, 0.87, 0.84, 0.85],
        [0.55, 0.84, 0.80, 0.89, 0.83],
        [0.40, 0.88, 0.60, 0.84, 0.88],
        [0.62, 0.74, 0.82, 0.75, 0.86],
    ])

    adv_payoffs = base_adv * fraction_factor * (1 + 0.3 * heterogeneity_factor)
    adv_payoffs[0, :] = 0.0

    noise = heterogeneity_factor * 0.05
    srv_payoffs = base_srv - noise * (base_adv > 0).astype(float)

    return PayoffMatrix.from_arrays(attacks, defenses, adv_payoffs, srv_payoffs)


def run_demo():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("DEMO: Game-Theoretic Analysis with Synthetic Payoff Matrix")
    print("=" * 60)

    alphas = [0.1, 0.3, 0.5, 1.0, 10.0]
    fractions = [0.1, 0.2, 0.3, 0.4]

    sweep_results = {}

    print("\n--- Single Configuration Analysis (alpha=0.5, f=0.2) ---\n")
    pm = create_synthetic_payoff_matrix(alpha=0.5, f=0.2)
    solver = GameSolver(pm)

    nash_results = solver.solve_nash()
    if nash_results:
        ne = nash_results[0]
        print(f"Nash Equilibrium:")
        print(f"  Adversary utility: {ne.adversary_utility:.4f}")
        print(f"  Server utility: {ne.server_utility:.4f}")
        print(f"  Adversary support: {ne.adversary_support}")
        print(f"  Server support: {ne.server_support}")
        print(f"  Value of Private Defense: {ne.value_of_information(pm):.4f}")

    se = solver.solve_stackelberg()
    if se:
        print(f"\nStackelberg Equilibrium:")
        print(f"  Server utility: {se.server_utility:.4f}")
        print(f"  Adversary utility: {se.adversary_utility:.4f}")

    fp_adv, fp_srv, convergence = solver.fictitious_play(num_iterations=50000)
    print(f"\nFictitious Play (50k iterations):")
    print(f"  Converged adversary: {[(pm.attacks[i], f'{p:.3f}') for i, p in enumerate(fp_adv) if p > 0.01]}")
    print(f"  Converged server: {[(pm.defenses[j], f'{p:.3f}') for j, p in enumerate(fp_srv) if p > 0.01]}")

    single_analysis = {
        "nash_equilibria": [{
            "adversary_strategy": ne.adversary_strategy.tolist(),
            "server_strategy": ne.server_strategy.tolist(),
            "adversary_utility": ne.adversary_utility,
            "server_utility": ne.server_utility,
            "value_of_information": ne.value_of_information(pm),
        }] if nash_results else [],
        "stackelberg": {
            "server_strategy": se.server_strategy.tolist() if se else None,
            "adversary_strategy": se.adversary_strategy.tolist() if se else None,
            "server_utility": se.server_utility if se else None,
            "adversary_utility": se.adversary_utility if se else None,
        },
        "fictitious_play": {
            "adversary_strategy": fp_adv.tolist(),
            "server_strategy": fp_srv.tolist(),
            "convergence": convergence,
        },
        "payoff_matrix": pm.to_dict(),
    }

    with open("results/game_analysis.json", "w") as f:
        json.dump(single_analysis, f, indent=2)

    print("\n\n--- Heterogeneity & Fraction Sweep ---\n")
    print(f"{'alpha':>8} {'f':>6} {'Adv NE Util':>12} {'Srv NE Util':>12} {'VoPD':>8}")
    print("-" * 50)

    for alpha in alphas:
        for frac in fractions:
            pm = create_synthetic_payoff_matrix(alpha=alpha, f=frac)
            solver = GameSolver(pm)
            nash_results = solver.solve_nash()

            if nash_results:
                ne = nash_results[0]
                voi = ne.value_of_information(pm)
                print(f"{alpha:>8.1f} {frac:>6.2f} {ne.adversary_utility:>12.4f} {ne.server_utility:>12.4f} {voi:>8.4f}")
                sweep_results[f"alpha{alpha}_f{frac}"] = {
                    "alpha": alpha,
                    "adversarial_fraction": frac,
                    "nash_adversary_utility": ne.adversary_utility,
                    "nash_server_utility": ne.server_utility,
                    "value_of_information": voi,
                    "nash_adversary_strategy": ne.adversary_strategy.tolist(),
                    "nash_server_strategy": ne.server_strategy.tolist(),
                }
            else:
                print(f"{alpha:>8.1f} {frac:>6.2f} {'N/A':>12} {'N/A':>12} {'N/A':>8}")

    with open("results/sweep_summary.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\nResults saved to results/game_analysis.json and results/sweep_summary.json")
    print("Run: python3 experiments/plot_results.py to generate figures.")


if __name__ == "__main__":
    run_demo()
