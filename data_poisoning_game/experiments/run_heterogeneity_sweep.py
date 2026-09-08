"""
Sweeps over heterogeneity levels (Dirichlet alpha) and adversarial fractions.

Uses a reduced 3x3 strategy set (key attacks x key defenses) with 10 rounds
per pair for feasibility. Each of the 12 (alpha, fraction) points runs 9 pairs,
taking ~4 min/pair = ~7 hours total. Queued to start after main payoff run.
"""
import json
import os
import sys
import time
import numpy as np
from itertools import product as iter_product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from experiments.run_payoff_matrix import run_full_payoff_matrix
from experiments.run_game_analysis import load_payoff_results, run_analysis


ALPHA_VALUES = [0.1, 0.3, 1.0, 10.0]
ADV_FRACTIONS = [0.1, 0.2, 0.4]

# Reduced strategy set: attacks and defenses most likely to appear in equilibrium.
# This covers no-attack baseline, the two strongest backdoor attacks, and
# the three defenses that dominate in Nash equilibrium.
SWEEP_GAME_CONFIG = GameConfig(
    attacks=["no_attack", "backdoor_pixel", "model_scaling"],
    defenses=["fedavg", "multi_krum", "rfa"],
)

SWEEP_ROUNDS = 10


def wait_for_main_run(payoff_path: str = "results/payoff_results.json", poll_interval: int = 120):
    """Block until main payoff_results.json is fully written (42 pairs)."""
    print(f"Waiting for main payoff run to complete ({payoff_path})...")
    while True:
        if os.path.exists(payoff_path):
            try:
                with open(payoff_path) as f:
                    data = json.load(f)
                if len(data) >= 42:
                    print(f"Main run complete ({len(data)} pairs found). Starting sweep.")
                    return
                else:
                    print(f"  Main run in progress: {len(data)}/42 pairs. Waiting...")
            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(poll_interval)


def run_sweep(wait_for_main: bool = True,
              output_summary: str = "results/sweep_summary.json"):
    if wait_for_main:
        wait_for_main_run()

    sweep_results = {}
    total_points = len(ALPHA_VALUES) * len(ADV_FRACTIONS)
    done = 0

    for alpha, adv_frac in iter_product(ALPHA_VALUES, ADV_FRACTIONS):
        done += 1
        print(f"\n{'='*60}")
        print(f"Sweep point {done}/{total_points}: alpha={alpha}, f={adv_frac}")
        print(f"{'='*60}")

        output_dir = f"results/sweep_new_alpha{alpha}_f{adv_frac}"

        fl_config = FLConfig(num_rounds=SWEEP_ROUNDS)
        exp_config = ExperimentConfig(
            dataset="cifar10",
            model="cifar_cnn",
            dirichlet_alpha=alpha,
            adversarial_fraction=adv_frac,
            num_trials=1,
            device="mps",
            seed=42,
        )

        run_full_payoff_matrix(fl_config, exp_config, SWEEP_GAME_CONFIG, output_dir)

        results_path = os.path.join(output_dir, "payoff_results.json")
        analysis = run_analysis(results_path, output_dir, game_config=SWEEP_GAME_CONFIG)

        ne_list = analysis.get("nash_equilibria", [])
        # Pick the NE with highest adversary utility (most relevant for VoPD)
        best_ne = max(ne_list, key=lambda ne: ne["adversary_utility"]) if ne_list else None

        sweep_results[f"alpha{alpha}_f{adv_frac}"] = {
            "alpha": alpha,
            "adversarial_fraction": adv_frac,
            "nash_adversary_utility": best_ne["adversary_utility"] if best_ne else None,
            "nash_server_utility": best_ne["server_utility"] if best_ne else None,
            "value_of_information": best_ne["value_of_information"] if best_ne else None,
            "nash_adversary_strategy": best_ne["adversary_strategy"] if best_ne else None,
            "nash_server_strategy": best_ne["server_strategy"] if best_ne else None,
        }

        # Write incrementally so partial results are usable
        with open(output_summary, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"  -> saved to {output_summary} ({len(sweep_results)} points so far)")

    print(f"\nSweep complete. {len(sweep_results)} points saved to {output_summary}")
    return sweep_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_wait", action="store_true",
                        help="Start immediately without waiting for main payoff run")
    parser.add_argument("--output_summary", type=str, default="results/sweep_summary.json")
    args = parser.parse_args()

    run_sweep(
        wait_for_main=not args.no_wait,
        output_summary=args.output_summary,
    )
