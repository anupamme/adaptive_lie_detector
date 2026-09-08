#!/usr/bin/env python3
"""
whitebox_3_train_probes.py - Train probing classifiers on representations

Trains logistic regression probes (SAPLMA-style) on extracted representations
and compares to behavioral detection baselines.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/whitebox_3_train_probes.py --model mistral_7b

    # Or analyze all models:
    .venv/bin/python3 experiments/whitebox_3_train_probes.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

# Behavioral baselines from paper (equalized conditions)
BEHAVIORAL_BASELINES = {
    "llama_8b": {"mistral": 0.61, "claude": 0.71},
    "mistral_7b": {"mistral": 0.62, "claude": 0.75},
    "qwen_7b": {"mistral": 0.57, "qwen": 0.66, "claude": 0.66},
    "qwen_14b": {"mistral": 0.52, "claude": 0.69}
}


def loo_accuracy(X, y, C=1.0):
    """Leave-one-out cross-validation accuracy."""
    preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in LeaveOneOut().split(X):
        # Train on all but one
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(C=C, random_state=42, max_iter=1000)
        clf.fit(X_train_scaled, y_train)

        # Predict
        preds[test_idx[0]] = clf.predict(X_test_scaled)[0]

    accuracy = np.mean(preds == y)
    return accuracy, preds


def bootstrap_ci(preds, y, n_boot=1000, seed=42):
    """Bootstrap 95% confidence interval."""
    rng = np.random.RandomState(seed)
    accs = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        accs.append(np.mean(preds[idx] == y[idx]))

    lo, hi = np.percentile(accs, [2.5, 97.5])
    return lo, hi


def analyze_model(model_key, repr_file, behavioral_baselines):
    """Analyze white-box probing for a single model."""
    print(f"\n{'='*70}")
    print(f"WHITE-BOX PROBING: {model_key.upper()}")
    print(f"{'='*70}")

    # Load representations
    with open(repr_file) as f:
        data = json.load(f)

    X = np.array(data["representations"])
    y = np.array(data["labels"])

    print(f"Representations: {X.shape}")
    print(f"  Truthful: {np.sum(y==0)}")
    print(f"  Lying: {np.sum(y==1)}")

    # Train probe with LOO
    print(f"\nTraining logistic regression probe...")
    accuracy, preds = loo_accuracy(X, y)
    ci_lo, ci_hi = bootstrap_ci(preds, y)

    print(f"\nWHITE-BOX PROBING RESULTS:")
    print(f"  LOO Accuracy: {accuracy:.1%}  [95% CI: {ci_lo:.1%}, {ci_hi:.1%}]")

    # Compare to behavioral baselines
    print(f"\nBEHAVIORAL BASELINES (from paper):")
    for extractor, acc in behavioral_baselines.items():
        print(f"  {extractor.capitalize()} extraction: {acc:.1%}")

    # Analysis
    behavioral_max = max(behavioral_baselines.values())
    behavioral_min = min(behavioral_baselines.values())

    print(f"\nCOMPARISON:")
    print(f"  White-box:     {accuracy:.1%}")
    print(f"  Behavioral (max): {behavioral_max:.1%}")
    print(f"  Behavioral (min): {behavioral_min:.1%}")
    print(f"  Gap (vs max):  {(accuracy - behavioral_max)*100:+.1f}pp")

    # Interpretation
    if accuracy > behavioral_max + 0.05:
        interpretation = "STRONG: White-box >> behavioral (representations encode truth/lie)"
    elif accuracy < behavioral_min - 0.05:
        interpretation = "WEAK: White-box << behavioral (unexpected, likely error)"
    else:
        interpretation = "COMPARABLE: White-box ≈ behavioral (both collapse under equalization)"

    print(f"\nINTERPRETATION: {interpretation}")

    return {
        "model_key": model_key,
        "whitebox_accuracy": float(accuracy),
        "whitebox_ci_lo": float(ci_lo),
        "whitebox_ci_hi": float(ci_hi),
        "behavioral_baselines": behavioral_baselines,
        "gap_vs_max": float(accuracy - behavioral_max),
        "interpretation": interpretation
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all models")
    args = parser.parse_args()

    if not args.model and not args.all:
        print("Error: Must specify --model or --all")
        sys.exit(1)

    # Get models to process
    available_models = list(BEHAVIORAL_BASELINES.keys())

    if args.all:
        models_to_process = available_models
    else:
        if args.model not in available_models:
            print(f"Error: Model must be one of {available_models}")
            sys.exit(1)
        models_to_process = [args.model]

    # Process each model
    results = []

    for model_key in models_to_process:
        repr_file = Path(f"data/whitebox_probing/{model_key}_representations.json")

        if not repr_file.exists():
            print(f"[SKIP] {model_key}: Representations not found: {repr_file}")
            print(f"       Run: whitebox_2_extract_representations.py --model {model_key}")
            continue

        baselines = BEHAVIORAL_BASELINES[model_key]
        result = analyze_model(model_key, repr_file, baselines)
        results.append(result)

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY: WHITE-BOX vs BEHAVIORAL")
        print(f"{'='*70}")
        print(f"{'Model':<15} {'White-box':<12} {'Behavioral (max)':<18} {'Gap':<8}")
        print(f"{'-'*15} {'-'*12} {'-'*18} {'-'*8}")

        for r in results:
            model = r["model_key"]
            wb_acc = r["whitebox_accuracy"]
            beh_max = max(r["behavioral_baselines"].values())
            gap = r["gap_vs_max"]

            print(f"{model:<15} {wb_acc:>6.1%}       {beh_max:>6.1%}             {gap:>+5.1f}pp")

    # Save results
    output_file = Path("data/whitebox_probing/probing_results.json")
    with open(output_file, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"\n✓ Saved results to {output_file}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. If extraction incomplete, run: whitebox_2_extract_representations.py --all")
    print("2. Once all models done, integrate into paper §4.5")
    print("3. Interpretation will guide novelty framing for reviewer")


if __name__ == "__main__":
    main()
