#!/bin/zsh
# EXP-WP configuration 2: the cross-family second model (PREREG_EXP_WP.md §8).
# google/gemma-3-4b-it on the v2 claim set. Loads as Gemma3ForConditionalGeneration,
# handled in model construction only (DEVIATION 7); the layer index is derived at the
# same fractional depth as Qwen3-4B's layer 16, not re-selected.
#
# bfloat16 compute and float32 storage are required, not preferences: Gemma-3's
# residual stream carries outlier features of order 3e5, so a float16 pass overflows
# to NaN in every layer from 6 up (2.4e5 at the pre-registered layer 15) and empties
# the generations. Numerics only; no probe, layer or pooling choice changes
# (DEVIATION 8).
#
# Waits for any in-flight probe job to release MPS before starting.
set -e
PY=/Users/mediratta/code/paper_writing/AI-Researcher-align/code/.venv/bin/python3
MODEL=google/gemma-3-4b-it
TAG=gemma-3-4b-it_v2
CS=v2
cd /Users/mediratta/code/paper_writing/AI-Researcher-align/code/adaptive_lie_detector

# Serialize on MPS: block until any named in-flight probe job exits. Only when
# WAIT_PID is a real pid -- `kill -0 0` signals the caller's own process group and
# would block forever.
while [[ -n "${WAIT_PID:-}" && "${WAIT_PID}" != 0 ]] && kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
done

echo "=== [1/7] generate + extract $(date) ==="
$PY experiments/probe_audit_1_generate_extract.py --model $MODEL --model_tag $TAG \
    --claim_set $CS --n_pairs 50 --passes both --device mps \
    --dtype bfloat16 --store_dtype float32
echo "=== [2/7] train probe (--prereg: fixed layer, no re-selection) $(date) ==="
$PY experiments/probe_audit_2_train_probe.py --model_tag $TAG --prereg
echo "=== [3/7] equalize $(date) ==="
$PY experiments/probe_audit_3_equalize.py --model_tag $TAG
echo "=== [4/7] factorial $(date) ==="
$PY experiments/probe_audit_4_factorial.py --model_tag $TAG
echo "=== [5/7] baselines $(date) ==="
$PY experiments/probe_audit_5_baselines.py --model_tag $TAG
echo "=== [5s/7] surface baseline $(date) ==="
$PY experiments/probe_audit_5s_surface.py --model_tag $TAG
echo "=== [6/7] steering, reduced sweep $(date) ==="
$PY experiments/probe_audit_6_steer.py --model $MODEL --model_tag $TAG \
    --claim_set $CS --coefs=-4,-1,0,1,4 --n_pairs 20 --device mps --dtype bfloat16
echo "=== [7/7] report $(date) ==="
$PY experiments/probe_audit_report.py --model_tag $TAG
echo "=== DONE $(date) ==="
