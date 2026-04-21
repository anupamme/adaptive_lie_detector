#!/usr/bin/env python3
"""
human_validation_analysis.py

Analyze human feature ratings vs. LLM feature extractions.

Computes inter-rater agreement metrics:
- Pearson correlation (linear relationship)
- Spearman rank correlation (monotonic relationship)
- Intraclass correlation ICC(2,1) (absolute agreement)

Prerequisites:
    Completed rating template with human ratings filled in
    Ground truth JSON with LLM features

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python experiments/human_validation_analysis.py \
        --ratings data/human_validation/rating_template.txt \
        --ground_truth data/human_validation/rating_template_ground_truth.json
"""

import argparse
import sys
import os
import json
import re
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


FEATURE_NAMES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]


def parse_human_ratings(ratings_file: str) -> list:
    """
    Parse filled-in rating template to extract human ratings.

    Returns list of dicts, one per transcript:
    [
        {"consistency": 7, "specificity": 4, ...},
        ...
    ]
    """
    with open(ratings_file) as f:
        content = f.read()

    # Split by transcript markers
    transcript_blocks = re.split(r'TRANSCRIPT #\d+', content)[1:]  # Skip header

    human_ratings = []
    for block in transcript_blocks:
        ratings = {}
        for feature in FEATURE_NAMES:
            # Look for pattern: [FEATURE]: <number>
            pattern = rf'\[{feature.upper()}\]:\s*(\d+\.?\d*)'
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                ratings[feature] = float(match.group(1))
            else:
                ratings[feature] = None  # Missing rating

        # Only add if we found at least one rating
        if any(v is not None for v in ratings.values()):
            human_ratings.append(ratings)

    return human_ratings


def compute_icc(ratings1: np.ndarray, ratings2: np.ndarray) -> float:
    """
    Compute ICC(2,1) - two-way random effects, single rater, absolute agreement.

    This is appropriate for assessing agreement between two raters where
    we care about absolute scale agreement (not just relative rankings).

    Formula based on Shrout & Fleiss (1979).
    """
    # Stack ratings into n_subjects x n_raters array
    ratings = np.column_stack([ratings1, ratings2])
    n_subjects = ratings.shape[0]
    n_raters = ratings.shape[1]

    # Mean squares
    mean_ratings = np.mean(ratings, axis=0)
    grand_mean = np.mean(ratings)

    # Between-subjects sum of squares
    row_means = np.mean(ratings, axis=1)
    bss = n_raters * np.sum((row_means - grand_mean) ** 2)
    msbr = bss / (n_subjects - 1)

    # Within-subjects sum of squares (residual)
    wss = np.sum((ratings - row_means[:, np.newaxis]) ** 2)
    mse = wss / (n_subjects * (n_raters - 1))

    # ICC(2,1) = (msbr - mse) / (msbr + (k-1)*mse + k*(msc - mse)/n)
    # Simplified for k=2 raters:
    icc = (msbr - mse) / msbr

    return icc


def analyze_agreement(human_ratings: list, llm_features: list) -> dict:
    """
    Compute agreement metrics for each feature.

    Returns dict:
    {
        "consistency": {"pearson_r": ..., "spearman_rho": ..., "icc": ...},
        "specificity": {...},
        ...
    }
    """
    results = {}

    for feature in FEATURE_NAMES:
        # Extract ratings for this feature
        human_vals = [r.get(feature) for r in human_ratings]
        llm_vals = [f.get(feature, 0.0) for f in llm_features]

        # Filter out missing ratings
        pairs = [(h, l) for h, l in zip(human_vals, llm_vals) if h is not None]

        if len(pairs) < 3:
            results[feature] = {
                "n": len(pairs),
                "pearson_r": None,
                "pearson_p": None,
                "spearman_rho": None,
                "spearman_p": None,
                "icc": None,
                "note": "Insufficient data (n < 3)",
            }
            continue

        human_arr = np.array([p[0] for p in pairs])
        llm_arr = np.array([p[1] for p in pairs])

        # Normalize LLM features (0-10) to match human scale
        # LLM features are typically 0-10, but verify range
        if llm_arr.max() > 10:
            llm_arr = (llm_arr / llm_arr.max()) * 10

        # Pearson correlation (linear relationship)
        pearson_r, pearson_p = stats.pearsonr(human_arr, llm_arr)

        # Spearman correlation (monotonic relationship)
        spearman_rho, spearman_p = stats.spearmanr(human_arr, llm_arr)

        # ICC (absolute agreement)
        icc = compute_icc(human_arr, llm_arr)

        results[feature] = {
            "n": len(pairs),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "icc": icc,
            "human_mean": float(np.mean(human_arr)),
            "human_std": float(np.std(human_arr)),
            "llm_mean": float(np.mean(llm_arr)),
            "llm_std": float(np.std(llm_arr)),
        }

    return results


def print_results(results: dict):
    """Print formatted agreement results."""
    print(f"\n{'=' * 80}")
    print("HUMAN-LLM FEATURE AGREEMENT ANALYSIS")
    print(f"{'=' * 80}")

    print(f"\n{'Feature':<15} {'N':>4} {'Pearson r':>10} {'p-val':>8} "
          f"{'Spearman ρ':>11} {'p-val':>8} {'ICC(2,1)':>10}")
    print("-" * 80)

    for feature in FEATURE_NAMES:
        r = results[feature]
        if r.get("note"):
            print(f"{feature:<15} {r['n']:>4}  {r['note']}")
            continue

        # Format p-values with stars for significance
        def format_p(p):
            if p is None:
                return "    --"
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            return f"{p:>6.3f}{stars}"

        print(f"{feature:<15} {r['n']:>4} "
              f"{r['pearson_r']:>10.3f} {format_p(r['pearson_p']):>8} "
              f"{r['spearman_rho']:>10.3f} {format_p(r['spearman_p']):>8} "
              f"{r['icc']:>10.3f}")

    print("\nSignificance: * p<0.05, ** p<0.01, *** p<0.001")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("FEATURE DESCRIPTIVE STATISTICS")
    print(f"{'=' * 80}")
    print(f"\n{'Feature':<15} {'Human Mean':>12} {'Human SD':>10} {'LLM Mean':>12} {'LLM SD':>10}")
    print("-" * 80)

    for feature in FEATURE_NAMES:
        r = results[feature]
        if r.get("note"):
            continue
        print(f"{feature:<15} {r['human_mean']:>12.2f} {r['human_std']:>10.2f} "
              f"{r['llm_mean']:>12.2f} {r['llm_std']:>10.2f}")

    # Overall summary
    valid_features = [f for f in FEATURE_NAMES if not results[f].get("note")]
    if valid_features:
        mean_pearson = np.mean([results[f]["pearson_r"] for f in valid_features])
        mean_spearman = np.mean([results[f]["spearman_rho"] for f in valid_features])
        mean_icc = np.mean([results[f]["icc"] for f in valid_features])

        print(f"\n{'=' * 80}")
        print("AVERAGE AGREEMENT ACROSS FEATURES")
        print(f"{'=' * 80}")
        print(f"  Mean Pearson r:   {mean_pearson:.3f}")
        print(f"  Mean Spearman ρ:  {mean_spearman:.3f}")
        print(f"  Mean ICC(2,1):    {mean_icc:.3f}")

        # Interpretation guide
        print(f"\nINTERPRETATION GUIDE:")
        print(f"  Correlation magnitude: <0.3=weak, 0.3-0.7=moderate, >0.7=strong")
        print(f"  ICC interpretation: <0.5=poor, 0.5-0.75=moderate, >0.75=good")


def main():
    parser = argparse.ArgumentParser(description="Analyze human feature validation study")
    parser.add_argument(
        "--ratings",
        type=str,
        required=True,
        help="Path to completed rating template with human ratings",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=True,
        help="Path to ground truth JSON with LLM features",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save analysis results to JSON (optional)",
    )
    args = parser.parse_args()

    print(f"Loading human ratings from: {args.ratings}")
    human_ratings = parse_human_ratings(args.ratings)
    print(f"Parsed {len(human_ratings)} transcript ratings")

    print(f"\nLoading ground truth from: {args.ground_truth}")
    with open(args.ground_truth) as f:
        gt_data = json.load(f)
    llm_features = [s["llm_features"] for s in gt_data["samples"]]
    print(f"Loaded {len(llm_features)} LLM feature sets")

    if len(human_ratings) != len(llm_features):
        print(f"\nWARNING: Mismatch in counts!")
        print(f"  Human ratings: {len(human_ratings)}")
        print(f"  LLM features:  {len(llm_features)}")
        print(f"  Using minimum of both ({min(len(human_ratings), len(llm_features))})")
        n = min(len(human_ratings), len(llm_features))
        human_ratings = human_ratings[:n]
        llm_features = llm_features[:n]

    # Analyze agreement
    print(f"\nComputing agreement metrics...")
    results = analyze_agreement(human_ratings, llm_features)

    # Print results
    print_results(results)

    # Save to JSON if requested
    if args.output:
        output_data = {
            "n_transcripts": len(human_ratings),
            "features_analyzed": FEATURE_NAMES,
            "agreement_metrics": results,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
