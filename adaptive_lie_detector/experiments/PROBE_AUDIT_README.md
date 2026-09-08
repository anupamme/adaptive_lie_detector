# Probe audit: "Does a deception probe detect deception, or the instruction to lie?"

A ~16h research project (MATS 12.0, Nanda stream). Turns the elicitation-confound
audit inward: instead of auditing a *behavioral* deception detector, we train a
linear probe on residual-stream activations to predict the deception *instruction*
`E`, then apply the same three diagnostics to the probe plus a causal steering
check.

## Design

- `V` = veracity of the shown claim (1 = true), `E` = deception instruction (1 = deceive).
- Instructed pass, 4 cells over 50 matched claim pairs: `T_H, F_H` (honest prompt),
  `T_D, F_D` (deceive prompt).
- Equalized pass, 2 cells: `T_N, F_N` under a single NEUTRAL prompt (instruction
  asymmetry removed).
- Probe: `StandardScaler + LogisticRegression(C=1.0)`; per-layer × per-pooling
  sweep; selection and all accuracies use leave-one-claim-pair-out (LOCPO) so
  "accuracy" means generalization to unseen claims.
- Pre-registered "dumbest way it's wrong" control: `ctrl_*` poolings re-encode the
  same response tokens under the NEUTRAL prompt with the instruction removed from
  context. If the probe only reads the instruction tokens, `ctrl` separability
  collapses.

## Pipeline (run from `code/adaptive_lie_detector/`, model on mps in fp16)

```bash
PY=../.venv/bin/python3
TAG=Qwen3-4B-Instruct-2507

# 1. generate responses + extract per-layer activations (instructed + equalized)
$PY experiments/probe_audit_1_generate_extract.py --model Qwen/$TAG --device mps
# 2. train per-layer probe to predict E; LOCPO acc + clustered bootstrap CI; select layer/pooling
$PY experiments/probe_audit_2_train_probe.py --model_tag $TAG
# 3. equalization: T_H-vs-F_D -> T_N-vs-F_N accuracy drop
$PY experiments/probe_audit_3_equalize.py --model_tag $TAG
# 4. factorial on probe OOF log-odds: bV, bE, bVE + clustered bootstrap + permutation p
$PY experiments/probe_audit_4_factorial.py --model_tag $TAG
# 5. baselines: random direction, shuffled-label null, refusal-marker behavioral baseline
$PY experiments/probe_audit_5_baselines.py --model_tag $TAG
# 6. causal steering: add coef*sigma*direction to residual, sweep coef; random-direction control
$PY experiments/probe_audit_6_steer.py --model Qwen/$TAG --model_tag $TAG --device mps
# 7. aggregate + figure (no model; from cached JSON)
$PY experiments/probe_audit_report.py --model_tag $TAG
$PY experiments/plot_probe_audit.py --model_tag $TAG
```

Add `--smoke` to steps 1 and 6 to dry-run on the cached `Qwen2.5-Coder-1.5B-Instruct`
(2 pairs, 60 tokens). Steps 2-5, 7 run on whatever `--model_tag` activations exist.

## Outputs (`data/probe_audit/`)

- `acts_<tag>_{instructed,equalized}.npz` — float16 activations `(n, n_hidden, dim)`
  for four poolings (`full_last, full_mean, ctrl_last, ctrl_mean`).
- `meta_<tag>_*.json`, `manifest_<tag>.json` — aligned labels + run config.
- `probe{2..6}_<tag>.json` — per-step results; `probe_audit_summary_<tag>.json`,
  `probe_audit_<tag>.png` — aggregate.

## What each result means

| Step | If the probe measures a deception *construct* | If it reads the *instruction* |
|------|-----------------------------------------------|-------------------------------|
| Decode (2) | high LOCPO acc | high LOCPO acc (uninformative alone) |
| Equalize (3) | equalized acc stays high | drops toward chance |
| Factorial (4) | large sign-stable `bE`, small `bVE` | `bE` present but interaction-bound; `ctrl` collapses |
| Steering (6) | probe direction moves behavior; random flat | probe direction ~ flat like random |

Seed is fixed at 42 throughout.
