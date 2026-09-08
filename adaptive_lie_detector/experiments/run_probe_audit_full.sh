#!/bin/zsh
# Full probe-audit pipeline on Qwen3-4B-Instruct-2507 (mps, defaults).
# Run from code/adaptive_lie_detector/.  Logs to data/probe_audit/run.log.
set -e
PY=../.venv/bin/python3
MODEL="$HOME/Qwen3-4B-Instruct-2507"   # local weights (HF cache was disk-constrained)
TAG=Qwen3-4B-Instruct-2507
cd "$(dirname "$0")/.."

echo "=== [1/7] generate + extract $(date) ==="
$PY experiments/probe_audit_1_generate_extract.py --model $MODEL --device mps
echo "=== [2/7] train probe $(date) ==="
$PY experiments/probe_audit_2_train_probe.py --model_tag $TAG
echo "=== [3/7] equalize $(date) ==="
$PY experiments/probe_audit_3_equalize.py --model_tag $TAG
echo "=== [4/7] factorial $(date) ==="
$PY experiments/probe_audit_4_factorial.py --model_tag $TAG
echo "=== [5/7] baselines $(date) ==="
$PY experiments/probe_audit_5_baselines.py --model_tag $TAG
echo "=== [6/7] steering $(date) ==="
$PY experiments/probe_audit_6_steer.py --model $MODEL --model_tag $TAG --device mps
echo "=== [7/7] report + plot $(date) ==="
$PY experiments/probe_audit_report.py --model_tag $TAG
$PY experiments/plot_probe_audit.py --model_tag $TAG
echo "=== DONE $(date) ==="
