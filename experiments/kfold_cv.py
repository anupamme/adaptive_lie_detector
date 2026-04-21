#!/usr/bin/env python3
"""
kfold_cv.py

5-fold stratified cross-validation on the mock dataset.
Addresses reviewer concern that classifier may be overfitting to the mock distribution.

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python experiments/kfold_cv.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

from src.feature_extractor import ConversationFeatures
from src.classifier import LieDetectorClassifier
from src.utils import save_json, timestamp

CORE_FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]


def load_examples(path):
    """Load examples with features from a dataset JSON file."""
    with open(path) as f:
        data = json.load(f)
    examples = data.get("examples", data) if isinstance(data, dict) else data
    features_list = []
    labels = []
    for ex in examples:
        feats = ex.get("features")
        if feats is None:
            continue
        # Build ConversationFeatures from the 5 core features only
        try:
            cf = ConversationFeatures(
                consistency=feats["consistency"],
                specificity=feats["specificity"],
                defensiveness=feats["defensiveness"],
                confidence=feats["confidence"],
                elaboration=feats["elaboration"],
            )
        except (KeyError, ValueError):
            continue
        features_list.append(cf)
        labels.append(bool(ex["is_lying"]))
    return features_list, labels


def run_cv(features_list, labels, n_splits=5, seed=42):
    """Run stratified k-fold CV and return per-fold and aggregate metrics."""
    X = np.array([f.to_vector() for f in features_list])
    y = np.array(labels, dtype=int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = float("nan")

        fold_results.append({
            "fold": fold + 1,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc
        })

    # Aggregate
    def agg(key):
        vals = [r[key] for r in fold_results if not (isinstance(r[key], float) and r[key] != r[key])]
        m = np.mean(vals)
        s = np.std(vals)
        return float(m), float(s)

    acc_m, acc_s = agg("accuracy")
    prec_m, prec_s = agg("precision")
    rec_m, rec_s = agg("recall")
    f1_m, f1_s = agg("f1")
    auc_m, auc_s = agg("auc")

    return fold_results, {
        "accuracy": acc_m, "accuracy_std": acc_s,
        "precision": prec_m, "precision_std": prec_s,
        "recall": rec_m, "recall_std": rec_s,
        "f1": f1_m, "f1_std": f1_s,
        "auc": auc_m, "auc_std": auc_s,
    }


def main():
    print("5-Fold Cross-Validation")
    print("=" * 65)

    # --- Full 500-example dataset ---
    print("\n[1] Full mock dataset (n=500)")
    features500, labels500 = load_examples("data/training_data/mock_dataset_500.json")
    print(f"    Loaded {len(features500)} examples ({sum(labels500)} lying, {len(labels500)-sum(labels500)} truthful)")
    folds500, agg500 = run_cv(features500, labels500)

    print(f"\n    {'Fold':>5}  {'n_test':>7}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUC':>7}")
    print("    " + "-" * 55)
    for r in folds500:
        print(f"    {r['fold']:>5}  {r['n_test']:>7}  "
              f"{r['accuracy']:>7.1%}  {r['precision']:>7.3f}  "
              f"{r['recall']:>7.3f}  {r['f1']:>7.3f}  {r['auc']:>7.3f}")
    print("    " + "-" * 55)
    print(f"    {'Mean':>5}  {'':>7}  "
          f"{agg500['accuracy']:>7.1%}  {agg500['precision']:>7.3f}  "
          f"{agg500['recall']:>7.3f}  {agg500['f1']:>7.3f}  {agg500['auc']:>7.3f}")
    print(f"    {'±Std':>5}  {'':>7}  "
          f"{agg500['accuracy_std']:>7.1%}  {agg500['precision_std']:>7.3f}  "
          f"{agg500['recall_std']:>7.3f}  {agg500['f1_std']:>7.3f}  "
          f"{agg500['auc_std']:>7.3f}")

    # Save
    out = {
        "dataset_500": {"folds": folds500, "aggregate": agg500},
    }
    os.makedirs("data/results", exist_ok=True)
    save_json(out, f"data/results/kfold_cv_{timestamp()}.json")
    save_json(out, "data/results/kfold_cv_latest.json")
    print("\nSaved to data/results/kfold_cv_latest.json")
    return out


if __name__ == "__main__":
    main()
