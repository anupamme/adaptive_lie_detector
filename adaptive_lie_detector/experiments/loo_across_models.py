"""
loo_across_models.py

Leave-one-model-out classifier evaluation: train on 6 of 7 equalized targets,
test on the held-out 7th, repeat for each model.

Addresses reviewer Q1: "Can you run a leave-one-model-out evaluation on the
5-feature pipeline (train on 6 of 7 targets, test on the held-out 7th, average
across folds)?"

Uses the same Mistral L3 cross-family feature files as per_feature_loo_ablation.py.
No API calls required — operates entirely on existing result JSONs.

Usage:
    cd /path/to/adaptive_lie_detector
    .venv/bin/python3 experiments/loo_across_models.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score

DATA_DIR = Path(__file__).parent.parent / "data" / "results"

FEATURE_NAMES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

TARGET_FILES = {
    "Llama 3.2 3B":  ("cross_family_equalized_mistral_large.json",       "consolidated", "llama3_2_3b"),
    "Llama 3.1 8B":  ("cross_family_equalized_llama8b_mistral_large.json", "individual",  None),
    "Mistral 7B":    ("cross_family_equalized_mistral_large.json",       "consolidated", "mistral_7b"),
    "Llama 3.3 70B": ("cross_family_equalized_mistral_large.json",       "consolidated", "llama_70b"),
    "Qwen 2.5 7B":   ("cross_family_equalized_qwen7b_mistral_large.json",  "individual",  None),
    "Qwen 2.5 14B":  ("cross_family_equalized_qwen14b_mistral_large.json", "individual",  None),
    "Claude Haiku":  ("cross_family_equalized_haiku_mistral_large.json",   "individual",  None),
}


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

    X, y = [], []
    for r in results:
        feats = r[feat_key]
        X.append([feats[f] for f in FEATURE_NAMES])
        y.append(1 if r["ground_truth"] == "lying" else 0)
    return np.array(X), np.array(y)


def loo_sample_accuracy(X, y):
    clf = Pipeline([("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    scores = cross_val_score(clf, X, y, cv=LeaveOneOut(), scoring="accuracy")
    return scores.mean()


def train_test_accuracy(X_train, y_train, X_test, y_test):
    clf = Pipeline([("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()
    truthful_acc = (preds[y_test == 0] == 0).mean() if (y_test == 0).any() else float("nan")
    lying_acc = (preds[y_test == 1] == 1).mean() if (y_test == 1).any() else float("nan")
    return acc, truthful_acc, lying_acc, preds


def main():
    models = list(TARGET_FILES.keys())

    print("Loading all targets...")
    per_model = {}
    for name in models:
        X, y = load_target(name)
        per_model[name] = (X, y)
        print(f"  {name}: n={len(y)}, lying={y.sum()}")

    # Pooled within-sample LOO (replicates paper's 64.7%)
    X_pool = np.vstack([per_model[m][0] for m in models])
    y_pool = np.concatenate([per_model[m][1] for m in models])
    pooled_loo = loo_sample_accuracy(X_pool, y_pool)
    print(f"\nPooled within-sample LOO (should be ~64.7%): {pooled_loo*100:.1f}%")

    # Leave-one-model-out
    print("\n--- Leave-One-Model-Out ---")
    print(f"{'Model':<20} {'n_test':>6}  {'Acc':>6}  {'T-acc':>6}  {'L-acc':>6}")
    print("-" * 52)

    loo_accs = []
    per_model_results = {}
    for test_model in models:
        train_models = [m for m in models if m != test_model]
        X_train = np.vstack([per_model[m][0] for m in train_models])
        y_train = np.concatenate([per_model[m][1] for m in train_models])
        X_test, y_test = per_model[test_model]

        acc, t_acc, l_acc, preds = train_test_accuracy(X_train, y_train, X_test, y_test)
        loo_accs.append(acc)
        per_model_results[test_model] = {
            "accuracy": float(acc),
            "truthful_accuracy": float(t_acc),
            "lying_accuracy": float(l_acc),
            "n_test": int(len(y_test)),
            "n_lying": int(y_test.sum()),
        }
        print(f"  {test_model:<18} {len(y_test):>6}  {acc*100:>5.1f}%  {t_acc*100:>5.1f}%  {l_acc*100:>5.1f}%")

    mean_loo = np.mean(loo_accs)
    gap = pooled_loo - mean_loo
    print("-" * 52)
    print(f"  {'Mean LOO-across-models':<18} {'':>6}  {mean_loo*100:>5.1f}%")
    print(f"\n  Gap (pooled within-sample LOO - LOO-across-models): {gap*100:+.1f} pp")

    out = {
        "experiment": "leave_one_model_out",
        "pooled_within_sample_loo": float(pooled_loo),
        "mean_loo_across_models": float(mean_loo),
        "gap_pp": float(gap * 100),
        "per_model": per_model_results,
    }

    out_path = DATA_DIR / "loo_across_models_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    return out


if __name__ == "__main__":
    main()
