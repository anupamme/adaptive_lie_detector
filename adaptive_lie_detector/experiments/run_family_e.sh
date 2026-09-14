#!/bin/bash
# run_family_e.sh -- PREREG_EXP_C4B.md §11 step 3, driven end to end.
#
# The five extension targets, ONE AT A TIME, at most two models resident:
#   pull -> record digest -> belief screen -> pilot (P1,P2,P3) -> grade pilot
#        -> wording selection -> confirmatory 30 claims x 8 reps
#        -> commit the cell -> ollama rm
#
# WHAT THIS SCRIPT DELIBERATELY DOES NOT DO
# -----------------------------------------
# It does not grade the confirmatory trials. §3.3(1) requires ONE grading pass
# shuffled ACROSS all targets by a public seed, precisely so that drift over a
# grading session cannot be confounded with target identity the way EXP-C4's
# contiguous per-target blocks allowed. Grading per target inside this loop
# would rebuild the confound the pre-registration raised gate 4 to remove. That
# is §11 step 5, after the loop, in one invocation.
#
# It also does not seal, blind or unseal anything: Family E's chain is §11
# steps 8-9 and DEVIATION (16), and it runs only once all five cells exist.
#
# RESUMABILITY. Every python step takes --resume and is idempotent, and each
# target's completion is marked in .family_e_done so a re-run skips finished
# targets rather than re-pulling 34 GB. Safe to kill and restart.
#
# Usage (from code/adaptive_lie_detector):
#   nohup bash experiments/run_family_e.sh > /tmp/family_e.log 2>&1 & disown

set -o pipefail

cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python3
DONE=data/results/.family_e_done
TARGETS="gemma2:9b phi4:14b mistral-nemo:12b granite3.1-dense:8b olmo2:13b"

# §4's ceiling: refuse to start a target that cannot fit alongside the runner.
MIN_FREE_GB=8

touch "$DONE"

say() { echo ""; echo "=== [$(date +%H:%M:%S)] $*"; }

free_gb() { df -g / | tail -1 | awk '{print $4}'; }

for m in $TARGETS; do
  if /usr/bin/grep -qx "$m" "$DONE"; then
    say "$m: already complete, skipping"
    continue
  fi

  say "$m: START  (free $(free_gb)GiB)"
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    say "ABORT: only $(free_gb) GiB free, need >= ${MIN_FREE_GB}. Stopping"
    say "cleanly BEFORE a partial cell rather than failing inside one."
    exit 1
  fi

  say "$m: pull"
  ollama pull "$m" || { say "$m: PULL FAILED, stopping"; exit 1; }

  # §4: the manifest digest is recorded while the blobs are on disk, BEFORE the
  # `ollama rm` at the end of this iteration. A tag is mutable; the digest is
  # what identifies the weights that produced the cell.
  say "$m: record manifest digest (§4)"
  $PY experiments/record_model_digests.py "$m" || exit 1

  say "$m: belief screen"
  $PY experiments/run_belief_strata.py --model "$m" --phase screen --resume || exit 1

  # §7a: per-target wording selection, on BASE RATE ONLY. Family R skips this
  # and is fixed at P3; only the extension set pilots.
  say "$m: pilot P1,P2,P3 (10 claims x 4 reps)"
  $PY experiments/run_crit4b_fixed_elicitation.py \
      --phase pilot --models "$m" --wordings P1,P2,P3 --claims 10 --reps 4 --resume || exit 1

  say "$m: grade pilot"
  $PY experiments/grade_crit4b_deception.py --phase pilot --all --resume || exit 1

  say "$m: wording selection (§7a)"
  $PY experiments/analyze_crit4b.py --phase select --model "$m" || exit 1

  say "$m: confirmatory (30 claims x 8 reps)"
  $PY experiments/run_crit4b_fixed_elicitation.py \
      --phase confirm --models "$m" --claims 30 --reps 8 --resume || exit 1

  # Commit the cell BEFORE reclaiming the disk: §4's order exists so that a
  # deletion can never precede the record of what was deleted.
  say "$m: commit the cell"
  git add data/results/crit4b_confirm_* data/results/crit4b_pilot_* \
          data/results/belief_strata_screen_* data/results/crit4b_grades_raw.jsonl \
          data/results/crit4b_model_digests.json \
          data/results/crit4b_wording_selection.json \
          2>/dev/null
  # Pilot grades land in crit4b_grades_raw.jsonl, staged above -- the primary
  # grader writes one raw log for every phase, so there is no separate file.
  if git diff --cached --quiet; then
    say "$m: nothing new to commit (resumed run)"
  else
    git commit -q -m "EXP-C4B cell: $m (Family E, §11 step 3)" || exit 1
  fi

  say "$m: ollama rm (digest already recorded)"
  ollama rm "$m" || exit 1

  echo "$m" >> "$DONE"
  say "$m: DONE  (free $(free_gb)GiB)"
done

say "Family E collection complete: all five cells committed."
say "NEXT, and NOT done here: §11 step 5 grades all confirmatory trials in ONE"
say "shuffled pass across targets; then step 6 top-up-check; step 7 secondary"
say "grader and the human coding sheet; then Family E's own seal per"
say "DEVIATION (16) -- seal_crit4b.py --family E, after Family R's artifacts"
say "are renamed to *_family_R.*"
