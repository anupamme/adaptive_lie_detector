#!/usr/bin/env python3
"""
Deep analysis of prompt-equalized control experiment (EXP-1).

Outputs:
1. Feature-level Cohen's d: instructed vs equalized (5 features × 2 conditions)
2. Permutation feature importance under LOO classifier
3. 5-fold CV stability on equalized data
4. Accuracy breakdown by claim category
5. Proximity analysis (near-miss vs clearly-wrong false claims)
"""

import json
import sys
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "results")

EQ_PATH  = os.path.join(DATA, "ollama_eval_mistral_7b_prompt_equalized_latest.json")
INS_PATH = os.path.join(DATA, "ollama_eval_mistral_7b_latest.json")

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

# ── claim metadata ──────────────────────────────────────────────────────────
# Category boundaries in EQUALIZED_CLAIMS (0-indexed):
# Scientific: 0-14, Historical: 15-24, Geographic: 25-34, Technology: 35-42, Cultural: 43-49
CATEGORY_RANGES = {
    "Scientific":  (0, 14),
    "Historical":  (15, 24),
    "Geographic":  (25, 34),
    "Technology":  (35, 42),
    "Cultural":    (43, 49),
}

# Near-miss false claims: small numeric deviation that could plausibly be a memory error
# These indices correspond to position in EQUALIZED_CLAIMS list
NEAR_MISS_INDICES = {
    0,   # 85°C boiling (15 off)
    1,   # 360 days orbit (5 off)
    4,   # 250,000 km/s light (~16% off)
    6,   # 280 m/s sound (~18% off)
    7,   # Au atomic 73 (6 off)
    9,   # 27% oxygen (6% off)
    15,  # WWII 1947 (2 years off)
    16,  # US independence 1778 (2 years off)
    18,  # Moon landing Aug (1 month off)
    19,  # Columbus 1489 (3 years off)
    21,  # printing press 1480 (40 years off — borderline)
    22,  # Waterloo 1818 (3 years off)
    24,  # Wright brothers 1907 (4 years off)
    25,  # Everest 9,200m (351m off)
    27,  # Nile 7,800km (~17% off)
    29,  # Dead Sea vs Death Valley (similar concept)
    35,  # Python 1995 (4 years off)
    36,  # iPhone 2005 (2 years off)
    38,  # byte 10 bits (2 off)
    39,  # WWW 1993 (4 years off)
    43,  # Harry Potter 8 books (1 off)
    44,  # Beethoven 11 symphonies (2 off)
    45,  # Olympics every 5 years (1 off)
}
# All other indices are "clearly wrong" (category swap, reversal, wrong entity)


def load_results(path):
    with open(path) as f:
        d = json.load(f)
    return d["results"]


def aggregate_features(results):
    """Return (X, y) where X is mean feature vector per trial."""
    X, y = [], []
    for r in results:
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        vec = []
        for feat in FEATURES:
            vals = [t[feat] for t in traj if t.get(feat) is not None]
            vec.append(np.mean(vals) if vals else 0.0)
        X.append(vec)
        y.append(1 if r["ground_truth"] == "lying" else 0)
    return np.array(X), np.array(y)


def cohen_d(a, b):
    """Cohen's d between two arrays."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return (np.mean(a) - np.mean(b)) / pooled_std


def compute_feature_d(results):
    """Compute Cohen's d per feature (lying vs truthful), using per-trial mean values."""
    rows = {f: {"truthful": [], "lying": []} for f in FEATURES}
    for r in results:
        label = r["ground_truth"]
        traj = r.get("feature_trajectory", [])
        if not traj:
            continue
        for feat in FEATURES:
            vals = [t[feat] for t in traj if t.get(feat) is not None]
            if vals:
                rows[feat][label].append(np.mean(vals))
    ds = {}
    for feat in FEATURES:
        ds[feat] = cohen_d(
            np.array(rows[feat]["lying"]),
            np.array(rows[feat]["truthful"])
        )
    return ds


def loo_accuracy(X, y):
    """LOO accuracy with logistic regression."""
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        if pipe.predict(X[test_idx])[0] == y[test_idx][0]:
            correct += 1
    return correct / len(y)


def permutation_importance(X, y, n_perms=1000):
    """LOO accuracy drop when each feature is permuted."""
    baseline = loo_accuracy(X, y)
    importances = {}
    rng = np.random.default_rng(42)
    for i, feat in enumerate(FEATURES):
        drops = []
        for _ in range(n_perms):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            acc = loo_accuracy(X_perm, y)
            drops.append(baseline - acc)
        importances[feat] = np.mean(drops)
    return baseline, importances


def five_fold_cv(X, y, n_repeats=10):
    """Repeated 5-fold CV for stability estimate."""
    accs = []
    for seed in range(n_repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
            pipe.fit(X[train_idx], y[train_idx])
            preds = pipe.predict(X[test_idx])
            accs.append(np.mean(preds == y[test_idx]))
    return np.mean(accs), np.std(accs)


def get_claim_category(claim_text, equalized_claims):
    """Map a claim text to its category label."""
    for i, (true_claim, false_claim) in enumerate(equalized_claims):
        if claim_text == true_claim or claim_text == false_claim:
            for cat, (lo, hi) in CATEGORY_RANGES.items():
                if lo <= i <= hi:
                    return cat, i
    return "Unknown", -1


def loo_predictions(X, y):
    """Return per-trial LOO predictions (same classifier as loo_accuracy)."""
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in loo.split(X):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = pipe.predict(X[test_idx])[0]
    return preds


def category_accuracy_loo(results, equalized_claims, X, y):
    """Accuracy breakdown by claim category using LOO predictions (consistent with Table 9)."""
    preds = loo_predictions(X, y)
    cat_results = {cat: {"correct": 0, "total": 0} for cat in CATEGORY_RANGES}
    for i, r in enumerate(results):
        if not r.get("feature_trajectory"):
            continue
        cat, _ = get_claim_category(r["claim"], equalized_claims)
        if cat == "Unknown":
            continue
        cat_results[cat]["total"] += 1
        if preds[i] == y[i]:
            cat_results[cat]["correct"] += 1
    return cat_results


def proximity_accuracy_loo(results, equalized_claims, X, y):
    """LOO accuracy for near-miss vs clearly-wrong false claims."""
    preds = loo_predictions(X, y)
    near = {"correct": 0, "total": 0}
    far  = {"correct": 0, "total": 0}
    for i, r in enumerate(results):
        if not r.get("feature_trajectory"):
            continue
        if r["ground_truth"] != "lying":
            continue
        _, idx = get_claim_category(r["claim"], equalized_claims)
        if idx == -1:
            continue
        bucket = near if idx in NEAR_MISS_INDICES else far
        bucket["total"] += 1
        if preds[i] == y[i]:
            bucket["correct"] += 1
    return near, far


def bootstrap_ci(X, y, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI on LOO accuracy."""
    rng = np.random.default_rng(42)
    n = len(y)
    boot_accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        # Skip degenerate resamples (single class)
        if len(np.unique(yb)) < 2:
            continue
        boot_accs.append(loo_accuracy(Xb, yb))
    lo = np.percentile(boot_accs, 100 * alpha / 2)
    hi = np.percentile(boot_accs, 100 * (1 - alpha / 2))
    return lo, hi, np.array(boot_accs)


def main():
    print("=" * 65)
    print("DEEP EQUALIZED ANALYSIS")
    print("=" * 65)

    # Load claims for category mapping
    sys.path.insert(0, os.path.join(BASE, "experiments"))
    from claims_equalized import EQUALIZED_CLAIMS

    eq_results  = load_results(EQ_PATH)
    ins_results = load_results(INS_PATH)

    print(f"\nEqualized results loaded: {len(eq_results)}")
    print(f"Instructed results loaded: {len(ins_results)}")

    # ── 1. Feature-level Cohen's d ──────────────────────────────────────────
    print("\n" + "─" * 65)
    print("1. FEATURE-LEVEL COHEN'S d (lying vs truthful)")
    print("─" * 65)
    print(f"{'Feature':<18} {'Instructed d':>14} {'Equalized d':>13} {'Δd':>8}")
    print(f"{'':─<18} {'':─>14} {'':─>13} {'':─>8}")

    ins_d = compute_feature_d(ins_results)
    eq_d  = compute_feature_d(eq_results)

    for feat in FEATURES:
        id_ = ins_d[feat]
        ed  = eq_d[feat]
        delta = ed - id_
        print(f"{feat:<18} {id_:>14.3f} {ed:>13.3f} {delta:>+8.3f}")

    # Overall from aggregated features
    X_ins, y_ins = aggregate_features(ins_results)
    X_eq,  y_eq  = aggregate_features(eq_results)
    overall_ins_d = cohen_d(X_ins[y_ins == 1].mean(axis=1), X_ins[y_ins == 0].mean(axis=1))
    overall_eq_d  = cohen_d(X_eq[y_eq == 1].mean(axis=1),  X_eq[y_eq == 0].mean(axis=1))
    print(f"\n{'Overall (mean feat)':<18} {overall_ins_d:>14.3f} {overall_eq_d:>13.3f}")

    # ── 2. Permutation importance ───────────────────────────────────────────
    print("\n" + "─" * 65)
    print("2. PERMUTATION FEATURE IMPORTANCE (equalized, LOO, 1000 perms)")
    print("─" * 65)
    print("   (computing — may take ~2 min) ...", flush=True)

    baseline, importances = permutation_importance(X_eq, y_eq, n_perms=1000)
    print(f"   Baseline LOO accuracy: {baseline:.3f}")
    print(f"\n   {'Feature':<18} {'Importance (acc drop)':>22}")
    print(f"   {'':─<18} {'':─>22}")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"   {feat:<18} {imp:>22.4f}")

    # ── 3. 5-fold CV stability ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("3. 5-FOLD CV STABILITY (equalized, 10 repeats)")
    print("─" * 65)
    mean_acc, std_acc = five_fold_cv(X_eq, y_eq)
    print(f"   Mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"   LOO accuracy:  {baseline:.3f}")
    print(f"   Note: high std = {std_acc:.3f} indicates signal fragility")

    # ── 4. Accuracy by claim category (LOO) ────────────────────────────────
    print("\n" + "─" * 65)
    print("4. ACCURACY BY CLAIM CATEGORY (equalized, LOO-recalibrated)")
    print("─" * 65)
    print("   (computing LOO predictions per category) ...", flush=True)
    cat_acc = category_accuracy_loo(eq_results, EQUALIZED_CLAIMS, X_eq, y_eq)
    print(f"   {'Category':<14} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"   {'':─<14} {'':─>8} {'':─>7} {'':─>10}")
    cat_total_correct = 0
    cat_total = 0
    for cat in ["Scientific", "Historical", "Geographic", "Technology", "Cultural"]:
        c = cat_acc[cat]["correct"]
        t = cat_acc[cat]["total"]
        acc = c / t if t > 0 else float("nan")
        print(f"   {cat:<14} {c:>8} {t:>7} {acc:>9.1%}")
        cat_total_correct += c
        cat_total += t
    print(f"   {'Overall':<14} {cat_total_correct:>8} {cat_total:>7} {cat_total_correct/cat_total:>9.1%}  (should match LOO ~71%)")

    # ── 5. Proximity analysis (LOO) ─────────────────────────────────────────
    print("\n" + "─" * 65)
    print("5. PROXIMITY ANALYSIS (deceptive condition only, LOO)")
    print("─" * 65)
    near, far = proximity_accuracy_loo(eq_results, EQUALIZED_CLAIMS, X_eq, y_eq)

    def pct(d):
        return d["correct"] / d["total"] if d["total"] > 0 else float("nan")

    print(f"   Near-miss false claims:     {near['correct']:>3}/{near['total']:>3} = {pct(near):.1%}")
    print(f"   Clearly-wrong false claims: {far['correct']:>3}/{far['total']:>3} = {pct(far):.1%}")
    print(f"\n   Near-miss = small numeric deviation (e.g., 85°C, 360 days)")
    print(f"   Clearly-wrong = category swap/reversal (e.g., Atlantic vs Pacific)")

    # ── 6. Bootstrap CI ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("6. BOOTSTRAP 95% CI ON LOO ACCURACY (10,000 resamples)")
    print("─" * 65)
    print("   (computing — may take ~1 min) ...", flush=True)
    ci_lo, ci_hi, boot_accs = bootstrap_ci(X_eq, y_eq, n_boot=10000)
    print(f"   LOO accuracy:  {baseline:.3f} (71.0%)")
    print(f"   95% CI:        [{ci_lo:.3f}, {ci_hi:.3f}]  =  [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"   Bootstrap mean: {np.mean(boot_accs):.3f}")
    print(f"   Bootstrap std:  {np.std(boot_accs):.3f}")

    # ── Summary for paper ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY FOR PAPER")
    print("=" * 65)
    top_feat = max(importances, key=importances.get)
    top_imp  = importances[top_feat]
    print(f"  • Defensiveness d: {ins_d['defensiveness']:.2f} (instructed) → {eq_d['defensiveness']:.2f} (equalized)")
    print(f"  • Consistency d:   {ins_d['consistency']:.2f} (instructed) → {eq_d['consistency']:.2f} (equalized)")
    print(f"  • Primary feature: {top_feat} (importance={top_imp:.3f})")
    print(f"  • 5-fold CV: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  • Bootstrap 95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
    print(f"  • Near-miss accuracy (LOO): {pct(near):.1%} | Clearly-wrong: {pct(far):.1%}")


if __name__ == "__main__":
    main()
