#!/bin/bash
# run_family_e_topup.sh -- PREREG_EXP_C4B.md §11 step 6 / §7b, one block of 10.
#
# §7b's top-up check (Role C, counts only, no labels) flagged three Family E
# targets on `paired < 12`, gate 3's threshold:
#
#   gemma2:9b          paired 11   -> 30 -> 40 claims
#   phi4:14b           paired 11   -> 30 -> 40 claims
#   mistral-nemo:12b   paired  8   -> 30 -> 40 claims
#
# granite3.1-dense:8b (12) and olmo2:13b (13) are satisfied and are NOT touched.
# Every Family R target is satisfied or at cap and is NOT touched.
#
# ONE BLOCK OF 10, exactly as pre-registered. §7b's rule is a loop over +10
# blocks, and mistral-nemo at 8 paired claims in 30 will quite possibly still
# miss 12 at 40, needing a second block. Collecting straight to 50 would save a
# 7.1 GB re-pull, and it is NOT done here: it would change which claims exist
# (claims 41-50 only get collected if the +10 block failed), so it is a
# deviation, and deviating to save a download is the wrong trade in an
# experiment whose whole subject is protocol discipline.
#
# WHAT THIS DOES NOT DO. It does not grade: §3.3(1) still requires ONE pass
# interleaved across targets, so grading is a single invocation after the loop,
# same as run_family_e.sh. It does not re-run the belief screen or the wording
# pilot -- both are committed, and §7a's wording is already selected per target
# and must not be reselected now that H1 counts are visible.
#
# THE DIGEST GUARD, which is the reason this script is not a one-liner. These
# three models were deleted by §4's loop, so their weights are re-pulled. A
# top-up is only a top-up if it runs on the SAME weights that produced claims
# 1-30; on different weights it silently makes the cell inhomogeneous. So after
# each re-pull the manifest digest is re-recorded and compared against the one
# recorded before deletion, and a mismatch ABORTS before any trial is run.
#
# Usage (from code/adaptive_lie_detector):
#   nohup bash experiments/run_family_e_topup.sh > /tmp/family_e_topup.log 2>&1 & disown

set -o pipefail

cd "$(dirname "$0")/.." || exit 1
PY=../.venv/bin/python3
DONE=data/results/.family_e_topup_done
DIGESTS=data/results/crit4b_model_digests.json

# Ordered so the longest-generating target sits between the two largest pulls:
# gemma2 generates ~35 min, phi4 ~1h40, mistral-nemo ~50 min, and the pulls are
# 5.4 / 9.1 / 7.1 GB at ~1.4 MB/s. Nothing scientific depends on the order.
TARGETS="gemma2:9b phi4:14b mistral-nemo:12b"
CLAIMS=40

# 12, not run_family_e.sh's 20. That floor was sized for the collection phase's
# worst SIMULTANEOUS pair (phi4 9.1 GB resident + olmo2 8.4 GB downloading). A
# top-up holds one model resident and adds 80 trials of JSONL, so 20 is
# over-strict here -- and measurably so: this run aborted at 05:25 on 18 GiB
# free purely because the next target's prefetch was mid-flight with its blob
# preallocated, then read 28 GiB free once the pull completed. The guard was
# right to stop rather than fail inside a cell; the threshold was wrong.
MIN_FREE_GB=12

touch "$DONE"
say() { echo ""; echo "=== [$(date +%H:%M:%S)] $*"; }
free_gb() { df -g / | tail -1 | awk '{print $4}'; }

# §4 + this script's own guard: abort if the re-pulled manifest differs from the
# one recorded before deletion. record_model_digests.py only WARNS on a change;
# for a top-up it has to be fatal.
digest_guard() {
  $PY - "$1" <<'PY'
import json, sys
tag = sys.argv[1]
r = json.load(open("data/results/crit4b_model_digests.json"))["targets"].get(tag)
if not r:
    sys.exit(f"  ABORT: no digest record for {tag}")
if r.get("manifest_changed_after_first_record"):
    sys.exit(f"  ABORT: {tag}'s manifest CHANGED on re-pull\n"
             f"    was {r.get('previous_manifest_sha256')}\n"
             f"    now {r.get('manifest_sha256')}\n"
             "  A top-up on different weights would make the cell "
             "inhomogeneous. Stopping before any trial.")
print(f"  digest guard OK: {tag} manifest {r['manifest_sha256'][:16]} unchanged")
PY
}

for m in $TARGETS; do
  if /usr/bin/grep -qx "$m" "$DONE"; then
    say "$m: top-up already complete, skipping"
    continue
  fi

  say "$m: START top-up to $CLAIMS claims  (free $(free_gb)GiB)"
  if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
    say "ABORT: only $(free_gb) GiB free, need >= ${MIN_FREE_GB}."
    exit 1
  fi

  if [ -n "$PREFETCH_PID" ]; then
    say "$m: waiting on the prefetch started during the previous target"
    wait "$PREFETCH_PID"
    PREFETCH_PID=""
  fi
  say "$m: re-pull (weights were deleted by §4's loop)"
  ollama pull "$m" || { say "$m: PULL FAILED, stopping"; exit 1; }

  next_m=""
  seen_me=0
  for t in $TARGETS; do
    if [ "$seen_me" = 1 ] && ! /usr/bin/grep -qx "$t" "$DONE"; then next_m=$t; break; fi
    [ "$t" = "$m" ] && seen_me=1
  done
  if [ -n "$next_m" ]; then
    say "$m: prefetching $next_m in the background"
    ollama pull "$next_m" > "/tmp/repull_${next_m//[:.]/_}.log" 2>&1 &
    PREFETCH_PID=$!
  fi

  say "$m: re-record manifest digest and compare against the pre-deletion record"
  $PY experiments/record_model_digests.py "$m" > /dev/null || exit 1
  digest_guard "$m" || exit 1

  # --resume adds claims 31-40 and regenerates nothing: claims 1-30 are already
  # committed and their trial keys are unchanged by raising --claims.
  say "$m: confirmatory top-up (claims 31-$CLAIMS x 8 reps)"
  $PY experiments/run_crit4b_fixed_elicitation.py \
      --phase confirm --models "$m" --claims "$CLAIMS" --reps 8 --resume || exit 1

  say "$m: commit the top-up"
  git add data/results/crit4b_confirm_* data/results/crit4b_model_digests.json \
          data/results/crit4b_topup_log.json 2>/dev/null
  if git diff --cached --quiet; then
    say "$m: nothing new to commit (resumed run)"
  else
    git commit -q -m "EXP-C4B top-up: $m to $CLAIMS claims (Family E, §7b block 1)" || exit 1
  fi

  say "$m: ollama rm (digest re-verified above)"
  ollama rm "$m" || exit 1
  echo "$m" >> "$DONE"
  say "$m: DONE  (free $(free_gb)GiB)"
done

say "Top-up collection complete for: $TARGETS"
say "§3.3(1)'s single interleaved grading pass, over the new trials only:"
$PY experiments/grade_crit4b_deception.py --phase confirm --all --resume || {
  say "GRADING FAILED -- stopping before the re-check"
  exit 1
}

say "§7b re-check (Role C, counts only). If any target still shows TOP UP, that"
say "is a SECOND +10 block and another re-pull -- not run automatically."
$PY experiments/analyze_crit4b.py --phase topup-check
