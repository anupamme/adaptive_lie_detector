#!/bin/bash
# Setup for Mistral 7B on g5.2xlarge (second half of disk)
# Qwen 32B Replication Experiment

set -e

echo "=========================================="
echo "Mistral 7B Setup (Qwen 32B Replication)"
echo "=========================================="

# Ollama should already be installed from Gemma setup
# Just pull Mistral 7B
echo "[1/4] Pulling mistral:7b model..."
ollama pull mistral:7b

# Python venv should already exist
cd ~/adaptive_lie_detector

# Create separate log directory
mkdir -p logs data/results

# Start experiment
echo "[2/4] Starting experiment..."
export AWS_REGION=us-east-1
export PYTHONPATH=/home/ubuntu/adaptive_lie_detector
nohup .venv/bin/python3 experiments/run_prompt_equalized.py \
  --model mistral:7b \
  --n_samples 100 \
  --resume \
  > logs/mistral_7b_n100.log 2>&1 &

echo ""
echo "=========================================="
echo "✓ Mistral 7B experiment started!"
echo "=========================================="
echo "Monitor: tail -f ~/adaptive_lie_detector/logs/mistral_7b_n100.log"
echo "Expected completion: 18-24 hours"
echo "=========================================="
