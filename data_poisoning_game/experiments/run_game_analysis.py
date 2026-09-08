"""
Computes Nash equilibria, Stackelberg equilibria, and value-of-information
from pre-computed payoff matrices.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GameConfig
from game_theory import GameSolver, PayoffMatrix


def load_payoff_results(results_path: str, game_config: GameConfig) -> PayoffMatrix:
    with open(results_path) as f:
        raw = json.load(f)

    m = len(game_config.attacks)
    n = len(game_config.defenses)
    adv_payoffs = np.zeros((m, n))
    srv_payoffs = np.zeros((m, n))

    for i, atk in enumerate(game_config.attacks):
        for j, dfn in enumerate(game_config.defenses):
            key = f"{atk}_{dfn}"
            if key not in raw:
                continue
            r = raw[key]
            acc = r["accuracy"]
            asr = r["attack_success_rate"]
            worst_acc = r["worst_class_accuracy"]

            from defenses.defense_costs import get_defense_cost
            from attacks import get_attack
            attack_obj = get_attack(atk)

            adv_payoffs[i, j] = asr - game_config.attack_cost_weight * attack_obj.cost
            if atk == "no_attack":
                adv_payoffs[i, j] = 0.0

            defense_cost = get_defense_cost(dfn)
            srv_payoffs[i, j] = (acc
                                 - game_config.defense_cost_weight * defense_cost
                                 - game_config.fairness_weight * (acc - worst_acc))

    return PayoffMatrix.from_arrays(game_config.attacks, game_config.defenses, adv_payoffs, srv_payoffs)


def run_analysis(results_path: str, output_dir: str = "results",
                 game_config: GameConfig = None):
    if game_config is None:
        game_config = GameConfig()
    payoff_matrix = load_payoff_results(results_path, game_config)
    solver = GameSolver(payoff_matrix)

    print("=" * 60)
    print("GAME-THEORETIC ANALYSIS")
    print("=" * 60)

    print("\n--- Adversary Payoff Matrix ---")
    print(f"{'':>20}", end="")
    for d in payoff_matrix.defenses:
        print(f"{d:>14}", end="")
    print()
    for i, a in enumerate(payoff_matrix.attacks):
        print(f"{a:>20}", end="")
        for j in range(len(payoff_matrix.defenses)):
            print(f"{payoff_matrix.adversary_payoffs[i, j]:>14.4f}", end="")
        print()

    print("\n--- Server Payoff Matrix ---")
    print(f"{'':>20}", end="")
    for d in payoff_matrix.defenses:
        print(f"{d:>14}", end="")
    print()
    for i, a in enumerate(payoff_matrix.attacks):
        print(f"{a:>20}", end="")
        for j in range(len(payoff_matrix.defenses)):
            print(f"{payoff_matrix.server_payoffs[i, j]:>14.4f}", end="")
        print()

    print("\n--- Nash Equilibria ---")
    nash_results = solver.solve_nash()
    if nash_results:
        for idx, ne in enumerate(nash_results):
            print(f"\nEquilibrium {idx + 1}:")
            print(f"  Adversary utility: {ne.adversary_utility:.4f}")
            print(f"  Server utility: {ne.server_utility:.4f}")
            print(f"  Adversary support: {ne.adversary_support}")
            print(f"  Server support: {ne.server_support}")
            voi = ne.value_of_information(payoff_matrix)
            print(f"  Value of Information (adversary): {voi:.4f}")
    else:
        print("  No Nash equilibrium found via support enumeration.")

    print("\n--- Stackelberg Equilibrium (Server leads) ---")
    stackelberg = solver.solve_stackelberg()
    if stackelberg:
        print(f"  Server utility: {stackelberg.server_utility:.4f}")
        print(f"  Adversary utility: {stackelberg.adversary_utility:.4f}")
        print(f"  Server strategy: {stackelberg.server_support}")
        print(f"  Adversary best response: {stackelberg.adversary_support}")

    print("\n--- Fictitious Play Convergence ---")
    fp_adv, fp_srv, convergence = solver.fictitious_play(num_iterations=50000)
    print(f"  Final adversary strategy: {[(payoff_matrix.attacks[i], f'{p:.3f}') for i, p in enumerate(fp_adv) if p > 0.01]}")
    print(f"  Final server strategy: {[(payoff_matrix.defenses[j], f'{p:.3f}') for j, p in enumerate(fp_srv) if p > 0.01]}")

    analysis_results = {
        "nash_equilibria": [
            {
                "adversary_strategy": ne.adversary_strategy.tolist(),
                "server_strategy": ne.server_strategy.tolist(),
                "adversary_utility": ne.adversary_utility,
                "server_utility": ne.server_utility,
                "value_of_information": ne.value_of_information(payoff_matrix),
            }
            for ne in nash_results
        ],
        "stackelberg": {
            "server_strategy": stackelberg.server_strategy.tolist() if stackelberg else None,
            "adversary_strategy": stackelberg.adversary_strategy.tolist() if stackelberg else None,
            "server_utility": stackelberg.server_utility if stackelberg else None,
            "adversary_utility": stackelberg.adversary_utility if stackelberg else None,
        },
        "fictitious_play": {
            "adversary_strategy": fp_adv.tolist(),
            "server_strategy": fp_srv.tolist(),
            "convergence": convergence,
        },
        "payoff_matrix": payoff_matrix.to_dict(),
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "game_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\nResults saved to {output_dir}/game_analysis.json")
    return analysis_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, default="results/payoff_results.json")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    run_analysis(args.results_path, args.output_dir)
