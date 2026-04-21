#!/usr/bin/env python3
"""
cross_family_extractor.py — EXP-A

Re-extract behavioral features from saved Claude Haiku transcripts using
Mistral 7B (via Ollama) as the feature extractor, then compare to the
original Claude-Haiku-extracted features.

Tests whether the same-family extractor confound inflates defensiveness
scores and LOO accuracy.

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python3 experiments/cross_family_extractor.py [--resume]
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import LLMFeatureExtractor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAIKU_DATA = os.path.join(BASE, "data", "results", "claude_haiku_n100_latest.json")
OUTPUT = os.path.join(BASE, "data", "results", "cross_family_haiku_mistral_extracted.json")

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]


def load_haiku_trials(path):
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    valid = [r for r in results
             if r.get("status") != "error"
             and r.get("conversation")
             and r.get("feature_trajectory")]
    print(f"Loaded {len(valid)} valid trials from {len(results)} total")
    return valid


def extract_with_mistral(trials, extractor, checkpoint_path, resume=False):
    """Re-extract features from each trial using Mistral extractor."""
    extracted = []
    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            extracted = json.load(f)
        print(f"Resuming from checkpoint: {len(extracted)} already done")

    for i, trial in enumerate(trials):
        if i < len(extracted):
            continue

        claim = trial.get("claim", "")
        conversation = trial["conversation"]
        ground_truth = trial.get("ground_truth", "")

        # Extract features per turn (matching original trajectory structure)
        # The original trajectory has len = questions_asked + 1
        # We re-extract once on the FULL conversation (mean features)
        try:
            features = extractor.extract(conversation, claim)
            feat_dict = {f: getattr(features, f) for f in FEATURES}
        except Exception as e:
            print(f"  [{i+1:02d}] ERROR: {e}")
            feat_dict = {f: None for f in FEATURES}

        # Also compute mean of original Claude-extracted features
        orig_traj = trial["feature_trajectory"]
        orig_means = {}
        for feat in FEATURES:
            vals = [t[feat] for t in orig_traj if t.get(feat) is not None]
            orig_means[feat] = float(np.mean(vals)) if vals else None

        entry = {
            "trial_index": i,
            "claim": claim,
            "ground_truth": ground_truth,
            "mistral_features": feat_dict,
            "claude_features": orig_means,
        }
        extracted.append(entry)

        status = "OK" if all(v is not None for v in feat_dict.values()) else "PARTIAL"
        print(f"  [{i+1:02d}/{len(trials)}] {status}  {ground_truth:<10} {claim[:50]}...")

        # Checkpoint every 10 trials
        if (i + 1) % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(extracted, f, indent=2)
            print(f"  -- checkpoint saved ({i+1}/{len(trials)})")

        # Rate limit
        time.sleep(1)

    # Final save
    with open(checkpoint_path, "w") as f:
        json.dump(extracted, f, indent=2)

    return extracted


def analyze(extracted):
    """Compare Mistral-extracted vs Claude-extracted features."""
    print("\n" + "=" * 70)
    print("CROSS-FAMILY EXTRACTOR ANALYSIS")
    print("=" * 70)

    # Filter to valid entries
    valid = [e for e in extracted
             if all(e["mistral_features"].get(f) is not None for f in FEATURES)
             and all(e["claude_features"].get(f) is not None for f in FEATURES)]
    print(f"\nValid trials: {len(valid)}")

    truthful = [e for e in valid if e["ground_truth"] == "truthful"]
    lying = [e for e in valid if e["ground_truth"] == "lying"]
    print(f"Truthful: {len(truthful)}, Lying: {len(lying)}")

    # 1. Feature-level comparison (mean scores by extractor)
    print("\n─── 1. Mean Feature Scores by Extractor ───")
    print(f"{'Feature':<15} {'Claude T':>10} {'Claude L':>10} {'Mistral T':>10} {'Mistral L':>10}")
    for feat in FEATURES:
        ct = np.mean([e["claude_features"][feat] for e in truthful])
        cl = np.mean([e["claude_features"][feat] for e in lying])
        mt = np.mean([e["mistral_features"][feat] for e in truthful])
        ml = np.mean([e["mistral_features"][feat] for e in lying])
        print(f"{feat:<15} {ct:10.2f} {cl:10.2f} {mt:10.2f} {ml:10.2f}")

    # 2. Cohen's d comparison
    print("\n─── 2. Cohen's d (lying - truthful) ───")
    print(f"{'Feature':<15} {'Claude d':>10} {'Mistral d':>10} {'Δd':>10}")
    for feat in FEATURES:
        ct = [e["claude_features"][feat] for e in truthful]
        cl = [e["claude_features"][feat] for e in lying]
        mt = [e["mistral_features"][feat] for e in truthful]
        ml = [e["mistral_features"][feat] for e in lying]

        pooled_c = np.sqrt((np.var(ct, ddof=1) + np.var(cl, ddof=1)) / 2)
        d_c = (np.mean(cl) - np.mean(ct)) / pooled_c if pooled_c > 0 else 0

        pooled_m = np.sqrt((np.var(mt, ddof=1) + np.var(ml, ddof=1)) / 2)
        d_m = (np.mean(ml) - np.mean(mt)) / pooled_m if pooled_m > 0 else 0

        print(f"{feat:<15} {d_c:+10.3f} {d_m:+10.3f} {d_m-d_c:+10.3f}")

    # 3. Pearson correlation between extractors (per trial)
    print("\n─── 3. Pearson Correlation (Claude vs Mistral, per trial) ───")
    from scipy import stats
    for feat in FEATURES:
        c_vals = [e["claude_features"][feat] for e in valid]
        m_vals = [e["mistral_features"][feat] for e in valid]
        r, p = stats.pearsonr(c_vals, m_vals)
        print(f"{feat:<15}  r={r:.3f}  p={p:.4f}")

    # 4. LOO accuracy comparison
    print("\n─── 4. LOO Accuracy ───")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import Pipeline

    y = np.array([1 if e["ground_truth"] == "lying" else 0 for e in valid])

    for extractor_name, feat_key in [("Claude", "claude_features"), ("Mistral", "mistral_features")]:
        X = np.array([[e[feat_key][f] for f in FEATURES] for e in valid])
        loo = LeaveOneOut()
        preds = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in loo.split(X):
            pipe = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(C=1.0, max_iter=1000))])
            pipe.fit(X[train_idx], y[train_idx])
            preds[test_idx[0]] = pipe.predict(X[test_idx])[0]

        acc = np.mean(preds == y)
        t_acc = np.mean(preds[y == 0] == y[y == 0])
        l_acc = np.mean(preds[y == 1] == y[y == 1])
        print(f"{extractor_name:<10} LOO: {acc:.1%} (truthful={t_acc:.1%}, lying={l_acc:.1%})")

    # 5. Paired t-test on defensiveness scores
    print("\n─── 5. Paired t-test: Defensiveness (Claude vs Mistral) ───")
    c_def = [e["claude_features"]["defensiveness"] for e in valid]
    m_def = [e["mistral_features"]["defensiveness"] for e in valid]
    t_stat, p_val = stats.ttest_rel(c_def, m_def)
    print(f"Claude mean: {np.mean(c_def):.2f}, Mistral mean: {np.mean(m_def):.2f}")
    print(f"Paired t={t_stat:.3f}, p={p_val:.4f}")

    # By condition
    for cond, trials_subset in [("truthful", truthful), ("lying", lying)]:
        c = [e["claude_features"]["defensiveness"] for e in trials_subset]
        m = [e["mistral_features"]["defensiveness"] for e in trials_subset]
        t, p = stats.ttest_rel(c, m)
        print(f"  {cond}: Claude={np.mean(c):.2f}, Mistral={np.mean(m):.2f}, t={t:.3f}, p={p:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-family feature extraction (EXP-A)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print("EXP-A: Cross-Family Extractor (Mistral 7B → Claude Haiku transcripts)")
    print("=" * 70)

    trials = load_haiku_trials(HAIKU_DATA)

    print(f"\nInitializing Mistral 7B extractor (Ollama)...")
    extractor = LLMFeatureExtractor(model="mistral:7b")

    checkpoint = OUTPUT.replace(".json", "_checkpoint.json")
    extracted = extract_with_mistral(trials, extractor, checkpoint, resume=args.resume)

    # Save final results
    with open(OUTPUT, "w") as f:
        json.dump({"experiment": "cross_family_extractor",
                    "source_extractor": "claude-haiku-4-5",
                    "target_extractor": "mistral:7b",
                    "n_trials": len(extracted),
                    "results": extracted}, f, indent=2)
    print(f"\nResults saved to: {OUTPUT}")

    analyze(extracted)


if __name__ == "__main__":
    main()
