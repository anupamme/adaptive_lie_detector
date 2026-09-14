#!/bin/bash
# archive_family_r_chain.sh -- PREREG_EXP_C4B.md DEVIATION (16).
#
# WHY THIS EXISTS
# ---------------
# §3.4 pre-registers ONE seal over ten pseudonyms at fixed paths. Family R was
# sealed, blinded and unsealed as a complete five-target chain before any Family
# E cell existed (DEVIATION 16), and `analyze_crit4b.py` -- which must NOT be
# modified, because §8 gate 6 checks its frozen commit hash -- reads and writes
# those fixed paths:
#
#   data/results/crit4b_seal.json
#   data/results/crit4b_blind/T{01..05}_cand{00..19}.json
#   data/results/crit4b_blind_results.json
#   data/results/crit4b_analysis.json
#   data/results/crit4b_salt.txt
#
# Family E's pseudonyms are drawn fresh from its own salt and are ALSO T01..T05,
# so its candidate blocks would collide with Family R's file-for-file, and its
# analysis would overwrite the artifact every number in the paper's EXP-C4B
# table comes from. So Family R's chain is renamed out of the way first.
#
# `git mv`, not `cp`: two files at the canonical paths would be ambiguous about
# which one the frozen script read. The R chain stays verifiable from git
# HISTORY at the canonical paths, in its original commit order (seal -> blind
# results -> salt), which is what §3.4's closing paragraph tells a reader to
# check. The rename is one further commit on top, not a rewrite of any of them.
#
# Idempotent: if the archive already exists, it says so and changes nothing.
#
# Usage (from code/adaptive_lie_detector), AFTER all five Family E cells are
# collected and graded and BEFORE `seal_crit4b.py --family E`:
#   bash experiments/archive_family_r_chain.sh

set -o pipefail
cd "$(dirname "$0")/.." || exit 1
R=data/results

if [ -e "$R/crit4b_seal_family_R.json" ]; then
  echo "Family R's chain is already archived; nothing to do."
  exit 0
fi
if [ ! -e "$R/crit4b_seal.json" ]; then
  echo "ERROR: $R/crit4b_seal.json is absent. Either the chain is already"
  echo "archived under a different name, or Family R was never sealed."
  exit 1
fi

# Refuse to archive a chain that has not been unsealed: moving it mid-run would
# leave a committed seal with no committed analysis, which is unverifiable.
if [ ! -e "$R/crit4b_analysis.json" ]; then
  echo "ERROR: $R/crit4b_analysis.json is absent, so Family R's chain is not"
  echo "complete. Archiving now would strand a seal with no unseal. Run"
  echo "  analyze_crit4b.py --phase unseal --salt-file $R/crit4b_salt.txt"
  echo "first, or investigate why it is missing."
  exit 1
fi

echo "Archiving Family R's chain (PREREG DEVIATION 16)"
for f in seal blind_results analysis; do
  git mv "$R/crit4b_$f.json" "$R/crit4b_${f}_family_R.json" || exit 1
  echo "  crit4b_$f.json -> crit4b_${f}_family_R.json"
done
git mv "$R/crit4b_blind" "$R/crit4b_blind_family_R" || exit 1
echo "  crit4b_blind/ -> crit4b_blind_family_R/  ($(ls "$R/crit4b_blind_family_R" | wc -l | tr -d ' ') blocks)"

# The salt is tracked only from step 6 onward (it is git-ignored until the
# unseal commits it), so handle both cases rather than assuming either.
if git ls-files --error-unmatch "$R/crit4b_salt.txt" > /dev/null 2>&1; then
  git mv "$R/crit4b_salt.txt" "$R/crit4b_salt_family_R.txt" || exit 1
  echo "  crit4b_salt.txt -> crit4b_salt_family_R.txt  (tracked)"
elif [ -e "$R/crit4b_salt.txt" ]; then
  mv "$R/crit4b_salt.txt" "$R/crit4b_salt_family_R.txt" || exit 1
  echo "  crit4b_salt.txt -> crit4b_salt_family_R.txt  (untracked; not committed"
  echo "    by this script, because §3.4 step 6 governs when a salt is committed)"
fi

cat <<'NOTE'

Archived. Commit this rename on its own, then:

  seal_crit4b.py --family E          # writes the canonical paths, now free
  # commit crit4b_seal.json          (salt stays UNCOMMITTED)
  analyze_crit4b.py --phase blind
  # commit crit4b_blind_results.json
  # commit crit4b_salt.txt, THEN:
  analyze_crit4b.py --phase unseal --salt-file data/results/crit4b_salt.txt

analyze_crit4b.py is NOT edited at any point -- that is the property DEVIATION
(16) is built to preserve, and it is why the archive is a rename rather than a
new set of paths in the analysis code.
NOTE
