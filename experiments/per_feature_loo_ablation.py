"""
Per-feature leave-one-out ablation for cross-family (Mistral L3) extraction.
Produces actual LOO contribution values to replace the effect-size proxy in Table 2.

Feature names in data use 'defensiveness' for what the paper calls 'correction-marker density'.
"""
import json
import glob
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score

DATA_DIR = Path(__file__).parent.parent / "data" / "results"

FEATURE_NAMES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]
PAPER_NAMES   = ["Consistency", "Specificity", "Correction-marker density", "Confidence", "Elaboration"]

# ICC alpha values from the paper's human study
ICC_ALPHA = {
    "consistency": 0.070,
    "specificity": 0.264,
    "defensiveness": 0.606,  # correction-marker density
    "confidence": 0.164,
    "elaboration": 0.087,
}

# Per-target Mistral L3 files (individual target files have cross_family_features)
TARGET_FILES = {
    "Llama 3.2 3B":   "cross_family_equalized_mistral_large.json",   # subset: llama3_2_3b
    "Llama 3.1 8B":   "cross_family_equalized_llama8b_mistral_large.json",
    "Mistral 7B":     "cross_family_equalized_mistral_large.json",    # subset: mistral_7b
    "Llama 3.3 70B":  "cross_family_equalized_mistral_large.json",    # subset: llama_70b
    "Qwen 2.5 7B":    "cross_family_equalized_qwen7b_mistral_large.json",
    "Qwen 2.5 14B":   "cross_family_equalized_qwen14b_mistral_large.json",
    "Claude Haiku":   "cross_family_equalized_haiku_mistral_large.json",
}

# Subsets within the consolidated file
CONSOLIDATED_SUBSETS = {
    "Llama 3.2 3B": "llama3_2_3b",
    "Mistral 7B":   "mistral_7b",
    "Llama 3.3 70B": "llama_70b",
}


def load_target(target_name):
    fname = TARGET_FILES[target_name]
    fpath = DATA_DIR / fname
    with open(fpath) as f:
        data = json.load(f)

    if target_name in CONSOLIDATED_SUBSETS:
        subset_key = CONSOLIDATED_SUBSETS[target_name]
        results = data["datasets"][subset_key]["results"]
        # Consolidated file uses 'mistral_large_features' key
        feat_key = "mistral_large_features"
    else:
        results = data["results"]
        feat_key = "cross_family_features"

    X, y = [], []
    for r in results:
        feats = r[feat_key]
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(1 if r["ground_truth"] == "lying" else 0)
    return np.array(X), np.array(y)


def loo_accuracy(X, y):
    clf = Pipeline([("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring="accuracy")
    return scores.mean()


def cohens_d(X, y, feat_idx):
    x1 = X[y == 1, feat_idx]
    x0 = X[y == 0, feat_idx]
    pooled_std = np.sqrt((np.var(x1, ddof=1) + np.var(x0, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return abs(np.mean(x1) - np.mean(x0)) / pooled_std


def main():
    # Pool all 7 targets
    all_X, all_y = [], []
    per_target = {}
    print("Loading targets...")
    for name in TARGET_FILES:
        X, y = load_target(name)
        all_X.append(X)
        all_y.append(y)
        per_target[name] = (X, y)
        print(f"  {name}: n={len(y)}, lying={y.sum()}")

    X_pool = np.vstack(all_X)
    y_pool = np.concatenate(all_y)
    print(f"\nPooled: n={len(y_pool)}, lying={y_pool.sum()}")

    # Full 5-feature baseline
    baseline = loo_accuracy(X_pool, y_pool)
    print(f"\nFull 5-feature LOO accuracy: {baseline:.3f} ({baseline*100:.1f}%)")
    print("(Paper reports 64.7% for cross-family pipeline)")

    # Per-feature Cohen's d (pooled)
    print("\nPer-feature Cohen's |d| (pooled, Mistral L3 extraction):")
    for i, (fname, pname) in enumerate(zip(FEATURE_NAMES, PAPER_NAMES)):
        d = cohens_d(X_pool, y_pool, i)
        print(f"  {pname:30s}  |d|={d:.3f}")

    # Leave-one-feature-out ablation
    print("\nPer-feature LOO ablation (drop one feature, re-run LOO):")
    print(f"{'Feature':<32} {'ICC α':>6}  {'Valid?':>6}  {'LOO-1':>8}  {'Drop (pp)':>10}  {'|d|':>6}")
    print("-" * 76)

    for i, (fname, pname) in enumerate(zip(FEATURE_NAMES, PAPER_NAMES)):
        mask = [j for j in range(len(FEATURE_NAMES)) if j != i]
        X_ablated = X_pool[:, mask]
        acc_ablated = loo_accuracy(X_ablated, y_pool)
        drop = baseline - acc_ablated
        d = cohens_d(X_pool, y_pool, i)
        icc = ICC_ALPHA[fname]
        valid = "yes" if icc >= 0.4 else "no"
        print(f"{pname:<32} {icc:>6.3f}  {valid:>6}  {acc_ablated*100:>7.1f}%  {drop*100:>+9.1f}pp  {d:>6.3f}")

    print()
    print("Summary for LaTeX Table 2 replacement:")
    print("Feature | ICC α | Valid | Full-LOO | LOO-w/o | Drop (pp) | |d|")
    rows = []
    for i, (fname, pname) in enumerate(zip(FEATURE_NAMES, PAPER_NAMES)):
        mask = [j for j in range(len(FEATURE_NAMES)) if j != i]
        X_ablated = X_pool[:, mask]
        acc_ablated = loo_accuracy(X_ablated, y_pool)
        drop = baseline - acc_ablated
        d = cohens_d(X_pool, y_pool, i)
        icc = ICC_ALPHA[fname]
        rows.append((pname, icc, fname == "defensiveness", acc_ablated, drop, d))

    # Sort by drop descending
    rows.sort(key=lambda r: -r[4])
    for pname, icc, valid, acc_abl, drop, d in rows:
        print(f"  {pname}: ICC={icc:.3f}, drop={drop*100:+.1f}pp, |d|={d:.2f}")


if __name__ == "__main__":
    main()
