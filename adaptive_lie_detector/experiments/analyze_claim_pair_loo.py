"""
analyze_claim_pair_loo.py

Leave-one-claim-pair-out (LOCPO) evaluation for the cross-family 5-feature pipeline.

Addresses reviewer concern: "Trial-level LOO over 689 trials shares the same 50 claim pairs
across 7 models, so LOO over trials ≠ generalization to unseen claims."

Fix: GroupKFold(n_splits=50) with claim-pair index as the group ID. Each fold withholds all
trials for one claim pair across all models (~7-14 test trials per fold). The classifier
trains on 49 claim pairs' worth of data and is tested on the held-out pair.

Also computes claim-pair clustered bootstrap 95% CIs (B=2000) by resampling at the
claim-pair level (n=50 units), collecting all trials for sampled pairs, and using the
LOCPO out-of-fold predictions.

Usage:
    cd /path/to/adaptive_lie_detector
    python3 experiments/analyze_claim_pair_loo.py
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.claims_equalized import EQUALIZED_CLAIMS  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "results"

FEATURE_NAMES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

TARGET_FILES = {
    "Llama 3.2 3B":  ("cross_family_equalized_mistral_large.json",        "consolidated", "llama3_2_3b"),
    "Llama 3.1 8B":  ("cross_family_equalized_llama8b_mistral_large.json", "individual",  None),
    "Mistral 7B":    ("cross_family_equalized_mistral_large.json",         "consolidated", "mistral_7b"),
    "Llama 3.3 70B": ("cross_family_equalized_mistral_large.json",         "consolidated", "llama_70b"),
    "Qwen 2.5 7B":   ("cross_family_equalized_qwen7b_mistral_large.json",  "individual",  None),
    "Qwen 2.5 14B":  ("cross_family_equalized_qwen14b_mistral_large.json", "individual",  None),
    "Claude Haiku":  ("cross_family_equalized_haiku_mistral_large.json",   "individual",  None),
}

claim_to_pair = {c: i for i, (tc, fc) in enumerate(EQUALIZED_CLAIMS) for c in (tc, fc)}


def load_target(target_name):
    fname, kind, subset_key = TARGET_FILES[target_name]
    fpath = DATA_DIR / fname
    with open(fpath) as f:
        data = json.load(f)

    if kind == "consolidated":
        results = data["datasets"][subset_key]["results"]
        feat_key = "mistral_large_features"
    else:
        results = data["results"]
        feat_key = "cross_family_features"

    X, y, groups, claims = [], [], [], []
    for r in results:
        claim = r["claim"]
        pair_idx = claim_to_pair.get(claim)
        if pair_idx is None:
            continue
        feats = r[feat_key]
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(1 if r["ground_truth"] == "lying" else 0)
        groups.append(pair_idx)
        claims.append(claim)
    return np.array(X), np.array(y), np.array(groups), claims


def locpo_accuracy(X, y, groups):
    """GroupKFold(50) leave-one-claim-pair-out accuracy."""
    clf_template = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    gkf = GroupKFold(n_splits=50)
    preds = np.full(len(y), -1, dtype=int)
    for train_idx, test_idx in gkf.split(X, y, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])
    acc = (preds == y).mean()
    return acc, preds


def clustered_bootstrap_ci(y, preds, groups, B=2000, rng=None):
    """
    Cluster-resampled bootstrap 95% CI.
    Resamples 50 claim pairs with replacement; collects all trials for those pairs;
    computes accuracy on the union. Returns (lower, upper).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    unique_pairs = np.unique(groups)
    n_pairs = len(unique_pairs)
    boot_accs = []
    for _ in range(B):
        sampled_pairs = rng.choice(unique_pairs, size=n_pairs, replace=True)
        mask = np.isin(groups, sampled_pairs)
        if mask.sum() == 0:
            continue
        boot_accs.append((preds[mask] == y[mask]).mean())
    lo, hi = np.percentile(boot_accs, [2.5, 97.5])
    return lo, hi


def per_model_locpo(per_model_data):
    """Run LOCPO separately for each model; return dict of results."""
    results = {}
    for name, (X, y, groups, _) in per_model_data.items():
        n_pairs = len(np.unique(groups))
        if n_pairs < 2:
            results[name] = {"accuracy": float("nan"), "n": len(y), "n_pairs": n_pairs}
            continue
        n_splits = min(n_pairs, 50)
        gkf = GroupKFold(n_splits=n_splits)
        preds = np.full(len(y), -1, dtype=int)
        for train_idx, test_idx in gkf.split(X, y, groups):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ])
            clf.fit(X[train_idx], y[train_idx])
            preds[test_idx] = clf.predict(X[test_idx])
        acc = (preds == y).mean()
        results[name] = {
            "accuracy": float(acc),
            "n": len(y),
            "n_pairs": n_pairs,
            "n_splits": n_splits,
        }
    return results


def main():
    rng = np.random.default_rng(42)

    print("Loading all targets...")
    per_model_data = {}
    for name in TARGET_FILES:
        X, y, groups, claims = load_target(name)
        per_model_data[name] = (X, y, groups, claims)
        n_pairs = len(np.unique(groups))
        print(f"  {name}: n={len(y)}, pairs={n_pairs}, lying={y.sum()}")

    # --- Pooled LOCPO ---
    X_pool = np.vstack([per_model_data[m][0] for m in TARGET_FILES])
    y_pool = np.concatenate([per_model_data[m][1] for m in TARGET_FILES])
    g_pool = np.concatenate([per_model_data[m][2] for m in TARGET_FILES])

    print(f"\nPooled: n={len(y_pool)}, unique pairs={len(np.unique(g_pool))}")
    print("Running pooled LOCPO (GroupKFold n_splits=50)...")
    pooled_acc, pooled_preds = locpo_accuracy(X_pool, y_pool, g_pool)
    lo, hi = clustered_bootstrap_ci(y_pool, pooled_preds, g_pool, B=2000, rng=rng)
    print(f"  Pooled LOCPO accuracy: {pooled_acc*100:.1f}%  [95% CI: {lo*100:.1f}–{hi*100:.1f}%]")

    # --- Per-model LOCPO ---
    print("\n--- Per-model LOCPO ---")
    print(f"{'Model':<20} {'n':>5}  {'Pairs':>5}  {'LOCPO Acc':>10}")
    print("-" * 48)
    per_model_results = per_model_locpo(per_model_data)
    for name, res in per_model_results.items():
        acc_str = f"{res['accuracy']*100:.1f}%" if not np.isnan(res['accuracy']) else "n/a"
        print(f"  {name:<18} {res['n']:>5}  {res['n_pairs']:>5}  {acc_str:>10}")

    # --- Refusal-count rule note ---
    print("\n--- Refusal-count rule (no learned parameters) ---")
    print("  RC≥1 is a fixed threshold; no training data used.")
    print("  Claim-pair LOO = regular accuracy for this rule (no leakage possible).")
    print("  The 80.1% figure is already claim-pair-generalizable by construction.")

    # --- Summary comparison ---
    print("\n--- Summary ---")
    # NOTE: the correct comparison baseline is the POOLED trial-level LOO on this same
    # n=639 dataset (67.1%), not the 64.7% figure in the paper -- 64.7% is the 7-target
    # MEAN of per-model LOO accuracies (n=689), a different quantity.
    POOLED_TRIAL_LOO = 0.671
    print(f"  Pooled trial-level LOO (loo_across_models.py, same data): {POOLED_TRIAL_LOO*100:.1f}%")
    print(f"  Pooled claim-pair LOO (LOCPO, this script):               {pooled_acc*100:.1f}%  [95% CI: {lo*100:.1f}–{hi*100:.1f}%]")
    print(f"  Difference (LOCPO - trial LOO):                           {(pooled_acc - POOLED_TRIAL_LOO)*100:+.1f} pp")
    print(f"\n  Refusal-count rule (RC>=1, no leakage):             ~80.1%  [CI: 75-84%]")

    out = {
        "experiment": "locpo_claim_pair_loo",
        "description": "Leave-one-claim-pair-out (LOCPO) evaluation with GroupKFold(n_splits=50)",
        "pooled_locpo": {
            "accuracy": float(pooled_acc),
            "ci_95_lo": float(lo),
            "ci_95_hi": float(hi),
            "n_trials": int(len(y_pool)),
            "n_pairs": int(len(np.unique(g_pool))),
        },
        "per_model_locpo": per_model_results,
        "refusal_rule_note": (
            "RC>=1 has no learned parameters; claim-pair LOO = regular accuracy (~80.1%). "
            "No claim-level leakage is possible for this rule."
        ),
    }
    out_path = DATA_DIR / "locpo_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return out


if __name__ == "__main__":
    main()
