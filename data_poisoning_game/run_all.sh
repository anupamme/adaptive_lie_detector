#!/bin/bash
# End-to-end experiment pipeline for the game-theoretic data poisoning paper.
#
# Usage:
#   bash run_all.sh --demo     # ~10 seconds, synthetic data only (no training)
#   bash run_all.sh --quick    # ~35 min on MPS Mac (30 rounds, cifar_cnn, 5 clients/round)
#   bash run_all.sh --medium   # ~4 hours on MPS Mac (50 rounds, cifar_cnn, 10 clients/round)
#   bash run_all.sh --full     # ~12+ hours (50 rounds, resnet18, + sweep)
#
# Tested on: Mac M-series with 24GB RAM, MPS backend.

set -e

MODE="${1:---demo}"

cd "$(dirname "$0")"

if [ "$MODE" == "--demo" ]; then
    echo "=========================================="
    echo "Running DEMO mode (synthetic payoff matrix, ~10s)"
    echo "=========================================="
    python3 experiments/demo_synthetic.py
    python3 experiments/plot_results.py \
        --analysis_path results/game_analysis.json \
        --sweep_path results/sweep_summary.json \
        --output_dir paper/figures
    echo "Done! Figures in paper/figures/"
    exit 0
fi

echo "=========================================="
echo "Step 1: Constructing Payoff Matrix"
echo "=========================================="

if [ "$MODE" == "--quick" ]; then
    # ~35 minutes on MPS Mac
    python3 experiments/run_payoff_matrix.py \
        --dataset cifar10 --model cifar_cnn \
        --alpha 0.5 --adv_fraction 0.2 \
        --num_rounds 30 --num_trials 1 \
        --output_dir results
elif [ "$MODE" == "--medium" ]; then
    # ~4 hours on MPS Mac
    python3 experiments/run_payoff_matrix.py \
        --dataset cifar10 --model cifar_cnn \
        --alpha 0.5 --adv_fraction 0.2 \
        --num_rounds 50 --num_trials 1 \
        --output_dir results
else
    # Full: ResNet-18, multiple trials
    python3 experiments/run_payoff_matrix.py \
        --dataset cifar10 --model resnet18 \
        --alpha 0.5 --adv_fraction 0.2 \
        --num_rounds 200 --num_trials 3 \
        --output_dir results
fi

echo "=========================================="
echo "Step 2: Game-Theoretic Analysis"
echo "=========================================="

python3 experiments/run_game_analysis.py \
    --results_path results/payoff_results.json \
    --output_dir results

echo "=========================================="
echo "Step 3: Generating Figures"
echo "=========================================="

python3 experiments/plot_results.py \
    --analysis_path results/game_analysis.json \
    --output_dir paper/figures

echo "=========================================="
echo "Step 4 (optional): Heterogeneity Sweep"
echo "=========================================="

if [ "$MODE" == "--full" ]; then
    python3 experiments/run_heterogeneity_sweep.py \
        --dataset cifar10 --model cifar_cnn \
        --num_rounds 30 --num_trials 1

    python3 experiments/plot_results.py \
        --analysis_path results/game_analysis.json \
        --sweep_path results/sweep_summary.json \
        --output_dir paper/figures
fi

echo "=========================================="
echo "Done! Results in results/ and paper/figures/"
echo "=========================================="
