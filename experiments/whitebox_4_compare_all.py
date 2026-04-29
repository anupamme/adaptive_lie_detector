#!/usr/bin/env python3
"""
whitebox_4_compare_all.py - Comprehensive white-box vs behavioral comparison

Analyzes all available white-box probing results (claim-based and response-based)
and creates comprehensive comparison tables.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/whitebox_4_compare_all.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Behavioral baselines from paper (equalized conditions)
BEHAVIORAL_BASELINES = {
    "mistral_7b": {"mistral": 0.62, "claude": 0.75},
    "qwen_7b": {"mistral": 0.57, "qwen": 0.66, "claude": 0.66},
    "qwen_14b": {"mistral": 0.52, "claude": 0.69}
}


def loo_accuracy(X, y, C=1.0):
    """Leave-one-out cross-validation accuracy."""
    preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in LeaveOneOut().split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(C=C, random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        preds[test_idx[0]] = clf.predict(X_test_scaled)[0]

    return np.mean(preds == y)


def bootstrap_ci(preds, y, n_boot=1000, seed=42):
    """Bootstrap 95% confidence interval."""
    rng = np.random.RandomState(seed)
    accs = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        accs.append(np.mean(preds[idx] == y[idx]))

    return np.percentile(accs, [2.5, 97.5])


def analyze_representation_file(repr_file):
    """Analyze a single representation file."""
    with open(repr_file) as f:
        data = json.load(f)

    X = np.array(data["representations"])
    y = np.array(data["labels"])

    # Quick LOO for comparison
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_test_scaled = scaler.transform(X[test_idx])

        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y[train_idx])
        preds[test_idx[0]] = clf.predict(X_test_scaled)[0]

    acc = np.mean(preds == y)
    ci_lo, ci_hi = bootstrap_ci(preds, y)

    return {
        "accuracy": float(acc),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n_samples": len(y),
        "representation_type": data.get("representation_type", "claim")
    }


def main():
    print("="*70)
    print("WHITE-BOX COMPREHENSIVE COMPARISON")
    print("="*70)
    print()

    results = {}

    # Check all possible representation files
    whitebox_dir = Path("data/whitebox_probing")

    for model_key in BEHAVIORAL_BASELINES.keys():
        results[model_key] = {}

        # Claim-based
        claim_file = whitebox_dir / f"{model_key}_representations.json"
        if claim_file.exists():
            print(f"Analyzing {model_key} (claim-based)...")
            results[model_key]["claim"] = analyze_representation_file(claim_file)

        # Response-based
        response_file = whitebox_dir / f"{model_key}_response_representations.json"
        if response_file.exists():
            print(f"Analyzing {model_key} (response-based)...")
            results[model_key]["response"] = analyze_representation_file(response_file)

    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE: WHITE-BOX vs BEHAVIORAL")
    print("="*70)
    print()

    # Header
    print(f"{'Model':<15} {'Representation':<15} {'White-box':<15} {'Behavioral':<15} {'Gap':<10}")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")

    for model_key in sorted(results.keys()):
        if not results[model_key]:
            continue

        behavioral = BEHAVIORAL_BASELINES[model_key]
        beh_max = max(behavioral.values())
        beh_range = f"{min(behavioral.values()):.0%}-{beh_max:.0%}"

        for repr_type, wb_result in sorted(results[model_key].items()):
            wb_acc = wb_result["accuracy"]
            ci_lo = wb_result["ci_lo"]
            ci_hi = wb_result["ci_hi"]
            gap = wb_acc - beh_max

            wb_str = f"{wb_acc:.0%} [{ci_lo:.0%}-{ci_hi:.0%}]"
            gap_str = f"{gap:+.0%}"

            print(f"{model_key:<15} {repr_type:<15} {wb_str:<15} {beh_range:<15} {gap_str:<10}")

    # Summary interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)

    all_wb_accs = []
    all_beh_maxs = []

    for model_key, model_results in results.items():
        beh_max = max(BEHAVIORAL_BASELINES[model_key].values())

        for repr_type, wb_result in model_results.items():
            all_wb_accs.append(wb_result["accuracy"])
            all_beh_maxs.append(beh_max)

    if all_wb_accs:
        avg_wb = np.mean(all_wb_accs)
        avg_beh = np.mean(all_beh_maxs)
        avg_gap = avg_wb - avg_beh

        print(f"\nAverage white-box:     {avg_wb:.1%}")
        print(f"Average behavioral:    {avg_beh:.1%}")
        print(f"Average gap:           {avg_gap:+.1%}")

        if avg_wb > avg_beh + 0.05:
            interpretation = "WHITE-BOX SUPERIOR: Representations encode truth/lie better than behavioral features"
        elif avg_wb < avg_beh - 0.05:
            interpretation = "BEHAVIORAL SUPERIOR: Multi-turn interrogation captures more signal than representations"
        else:
            interpretation = "COMPARABLE PERFORMANCE: Both methods struggle with equalized knowledge-conflict detection"

        print(f"\n{interpretation}")

    # Save results
    output_file = whitebox_dir / "comprehensive_comparison.json"
    output_data = {
        "results": results,
        "behavioral_baselines": BEHAVIORAL_BASELINES,
        "summary": {
            "avg_whitebox": float(np.mean(all_wb_accs)) if all_wb_accs else None,
            "avg_behavioral": float(np.mean(all_beh_maxs)) if all_beh_maxs else None,
            "interpretation": interpretation if all_wb_accs else "Insufficient data"
        }
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Saved to {output_file}")


if __name__ == "__main__":
    main()
