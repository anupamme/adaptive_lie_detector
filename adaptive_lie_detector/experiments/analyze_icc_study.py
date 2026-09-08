#!/usr/bin/env python3
"""
analyze_icc_study.py

Analyze completed ICC study (3 human annotators, n=100, 5 features 0–10).

Computes:
  - Krippendorff's α per feature (ordinal distance)
  - ICC(2,1) per feature (two-way random effects, single measures)
  - Pairwise Pearson r (all 3 pairs) per feature
  - Overall verdict: α≥0.4 → validated; α<0.4 → demoted

Reads:
  data/icc_study_v2/annotator_1_completed.csv
  data/icc_study_v2/annotator_2_completed.csv
  data/icc_study_v2/annotator_3_completed.csv
  data/icc_study_v2/ground_truth_n100.json   (for comparison)
  data/icc_study_v2/attention_checks.json    (for attention-check filtering)

Usage:
    cd /Users/mediratta/code/AI-Researcher/code/adaptive_lie_detector
    .venv/bin/python3 experiments/analyze_icc_study.py
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "icc_study_v2"

FEATURE_NAMES = [
    "consistency",
    "specificity",
    "correction_marker_density",
    "confidence",
    "elaboration",
]
RATING_COLS = [f"rating_{f}" for f in FEATURE_NAMES]

ATTN_CHECK_COL_PREFIX = "ATTN_CHECK"
MAX_ATTN_FAILURES = 1   # annotator replaced if they fail > this many


# ---------------------------------------------------------------------------
# Krippendorff's α (ordinal)
# ---------------------------------------------------------------------------

def krippendorff_alpha_ordinal(ratings_matrix: np.ndarray) -> float:
    """
    Compute Krippendorff's α for ordinal data.

    ratings_matrix: (n_raters, n_items) array of integer ratings.
    NaN entries are treated as missing.

    Uses the ordinal distance function d(k,l) = (k-l)^2 (interval) which
    is the standard proxy for ordinal ICC when scale levels are equidistant.
    """
    R, N = ratings_matrix.shape
    # Coincidence matrix
    observed_disagreement = 0.0
    n_valid_pairs = 0
    expected_disagreement = 0.0

    # Per-item, collect all rater pairs
    item_values = []
    for j in range(N):
        col = ratings_matrix[:, j]
        valid = col[~np.isnan(col)]
        item_values.append(valid)

    # Observed disagreement (mean of all pairwise squared diffs within items)
    total_obs = 0.0
    n_obs = 0
    for vals in item_values:
        m = len(vals)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                total_obs += (vals[a] - vals[b]) ** 2
                n_obs += 1

    if n_obs == 0:
        return float("nan")
    Do = total_obs / n_obs

    # Expected disagreement (all pairs across all items)
    all_vals = np.concatenate([v for v in item_values if len(v) >= 1])
    n_all = len(all_vals)
    if n_all < 2:
        return float("nan")
    # Variance-based shortcut: De = sum_k sum_l n_k*n_l*(k-l)^2 / (n*(n-1))
    total_exp = 0.0
    for i in range(n_all):
        for j_ in range(i + 1, n_all):
            total_exp += (all_vals[i] - all_vals[j_]) ** 2
    De = 2 * total_exp / (n_all * (n_all - 1))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


# ---------------------------------------------------------------------------
# ICC(2,1) — two-way random effects, single measures
# ---------------------------------------------------------------------------

def icc_2_1(ratings_matrix: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects model, single measures consistency.

    ratings_matrix: (n_raters, n_items)
    """
    R, N = ratings_matrix.shape
    grand_mean = np.nanmean(ratings_matrix)

    # Row (rater) means, col (item) means
    rater_means = np.nanmean(ratings_matrix, axis=1)   # shape (R,)
    item_means  = np.nanmean(ratings_matrix, axis=0)   # shape (N,)

    # Sum of squares
    SS_total = np.nansum((ratings_matrix - grand_mean) ** 2)
    SS_rater = N * np.nansum((rater_means - grand_mean) ** 2)
    SS_item  = R * np.nansum((item_means  - grand_mean) ** 2)
    SS_error = SS_total - SS_rater - SS_item

    df_rater = R - 1
    df_item  = N - 1
    df_error = (R - 1) * (N - 1)

    if df_error == 0 or df_item == 0:
        return float("nan")

    MS_item  = SS_item  / df_item
    MS_error = SS_error / df_error

    if (MS_item + (R - 1) * MS_error) == 0:
        return float("nan")

    return (MS_item - MS_error) / (MS_item + (R - 1) * MS_error)


# ---------------------------------------------------------------------------
# Pairwise Pearson r
# ---------------------------------------------------------------------------

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Load completed CSV
# ---------------------------------------------------------------------------

def load_completed_csv(path: Path) -> dict[str, dict[str, Optional[float]]]:
    """Return {trial_id: {feature: rating_or_None}}."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row.get("trial_id", "").strip()
            if not tid:
                continue
            ratings = {}
            for col in RATING_COLS:
                val = row.get(col, "").strip()
                try:
                    ratings[col] = float(val)
                except (ValueError, TypeError):
                    ratings[col] = None
            result[tid] = ratings
    return result


# ---------------------------------------------------------------------------
# Attention-check validation
# ---------------------------------------------------------------------------

def check_annotator_attention(completed: dict, attn_checks: list) -> tuple[int, list[str]]:
    """Return (n_failures, list of failed check IDs)."""
    failures = []
    for ac in attn_checks:
        tid = ac["trial_id"]
        if tid not in completed:
            continue
        correct = ac["correct_ratings"]
        got = completed[tid]
        for col, expected in correct.items():
            given = got.get(col)
            if given != expected:
                failures.append(tid)
                break
    return len(failures), failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ICC STUDY ANALYSIS — Human Annotators, n=100, 5 Features")
    print("=" * 60)

    # Load attention checks
    attn_path = DATA / "attention_checks.json"
    with open(attn_path) as f:
        attn_checks = json.load(f)

    # Load ground truth
    gt_path = DATA / "ground_truth_n100.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)

    # Load annotator completions
    annotators = {}
    for ann_num in [1, 2, 3]:
        path = DATA / f"annotator_{ann_num}_completed.csv"
        if not path.exists():
            print(f"  WARNING: {path} not found — cannot proceed")
            return
        ann_data = load_completed_csv(path)
        n_failures, failed_ids = check_annotator_attention(ann_data, attn_checks)
        status = "PASS" if n_failures <= MAX_ATTN_FAILURES else "FAIL (replace annotator)"
        print(f"\nAnnotator {ann_num}: {len(ann_data)} trials loaded, "
              f"{n_failures} attention check failures → {status}")
        if failed_ids:
            print(f"  Failed checks: {failed_ids}")
        if n_failures > MAX_ATTN_FAILURES:
            print(f"  EXCLUDED: annotator {ann_num} failed too many attention checks.")
            continue
        annotators[ann_num] = ann_data

    if len(annotators) < 3:
        print("\nFewer than 3 valid annotators — cannot compute ICC. Replace failed annotator(s).")
        return

    # Get shared trial IDs (exclude attention checks)
    attn_ids = {ac["trial_id"] for ac in attn_checks}
    all_ids = sorted(set.intersection(*[set(a.keys()) for a in annotators.values()]) - attn_ids)
    N = len(all_ids)
    print(f"\n{N} shared main trials across all 3 annotators.")

    results_by_feature = {}
    print("\n" + "-" * 60)
    print(f"{'Feature':<30} {'α (Kripp)':<12} {'ICC(2,1)':<12} {'r_12':<8} {'r_13':<8} {'r_23':<8}")
    print("-" * 60)

    for feat, col in zip(FEATURE_NAMES, RATING_COLS):
        # Build (3, N) matrix
        mat = np.full((3, N), np.nan)
        for ri, (ann_num, ann_data) in enumerate(sorted(annotators.items())):
            for ji, tid in enumerate(all_ids):
                val = ann_data.get(tid, {}).get(col)
                if val is not None:
                    mat[ri, ji] = val

        alpha = krippendorff_alpha_ordinal(mat)
        icc   = icc_2_1(mat)
        r12   = pearson_r(mat[0], mat[1])
        r13   = pearson_r(mat[0], mat[2])
        r23   = pearson_r(mat[1], mat[2])

        results_by_feature[feat] = {
            "krippendorff_alpha": alpha,
            "icc_2_1": icc,
            "pearson_r_12": r12,
            "pearson_r_13": r13,
            "pearson_r_23": r23,
            "n_items": N,
        }

        label = feat.replace("_", " ").title()
        print(f"  {label:<28} {alpha:<12.3f} {icc:<12.3f} {r12:<8.3f} {r13:<8.3f} {r23:<8.3f}")

    print("-" * 60)

    # Verdict
    cmd_alpha = results_by_feature.get("correction_marker_density", {}).get("krippendorff_alpha", float("nan"))
    print(f"\nCorrection-Marker Density α = {cmd_alpha:.3f}")
    if math.isnan(cmd_alpha):
        verdict = "INDETERMINATE (missing data)"
    elif cmd_alpha >= 0.4:
        verdict = "VALIDATED — level-dependent claims confirmed; remove PROVISIONAL caveats."
    else:
        verdict = ("NOT VALIDATED — α < 0.4; demote level-dependent claims to "
                   "'preliminary, unvalidated' in §5.1 and §5.7(g).")
    print(f"\nVERDICT: {verdict}")

    # Comparison with LLM-extracted features
    print("\n--- LLM vs. Human Rating Comparison (correction_marker_density) ---")
    ann1 = sorted(annotators.items())[0][1]
    human_cmd = []
    llm_cmd = []
    for tid in all_ids:
        h_val = ann1.get(tid, {}).get("rating_correction_marker_density")
        gt = ground_truth.get(tid, {})
        l_val = gt.get("llm_features", {}).get("defensiveness")   # pipeline uses "defensiveness"
        if h_val is not None and l_val is not None:
            human_cmd.append(h_val)
            llm_cmd.append(l_val)
    if human_cmd:
        r_llm_human = pearson_r(np.array(human_cmd), np.array(llm_cmd))
        print(f"  Pearson r (LLM vs. Ann1 human): {r_llm_human:.3f}  (n={len(human_cmd)})")
    else:
        print("  Cannot compare: missing data.")

    # Save results
    out = {
        "n_annotators": len(annotators),
        "n_items": N,
        "features": results_by_feature,
        "correction_marker_density_alpha": cmd_alpha,
        "verdict": verdict,
    }
    out_path = DATA / "icc_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print("\nNext steps:")
    print("  1. Update §5.1 with α and verdict.")
    print("  2. Add Table tab:icc_study_n100 to appendix with per-feature results.")
    print("  3. Remove or retain PROVISIONAL caveats per verdict.")


if __name__ == "__main__":
    main()
