#!/usr/bin/env python3
"""
Knowledge Pre-Check Stratified Analysis for EXP-G

This script addresses the reviewer's concern:
"On models ≤8B, instruction-following effect conflates with knowledge transfer."

For each model (Llama 3B, Llama 8B, Mistral 7B):
1. Load instructed-matched (EXP-G) results
2. Stratify by knowledge pre-check: PASSED (model knows the answer) vs FAILED (doesn't know)
3. Compute instruction-following contribution for PASSED subset only
4. Compare: Is instruction-following still dominant when knowledge is controlled?

Expected outcomes:
- If instruction-following dominant on PASSED → clean decomposition proven
- If effect disappears on PASSED → restrict claim to 14B/70B only
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from pathlib import Path

def load_data(filepath):
    """Load experiment results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_features_and_labels(data):
    """Extract feature matrix X and labels y from experiment data."""
    X = []
    y = []
    knowledge_status = []

    for trial in data['results']:
        # Get final features (last in trajectory)
        features = trial['feature_trajectory'][-1]

        # Extract the 5 behavioral dimensions
        feature_vec = [
            features['consistency'],
            features['specificity'],
            features['defensiveness'],
            features['confidence'],
            features['elaboration']
        ]

        X.append(feature_vec)
        y.append(1 if trial['ground_truth'] == 'lying' else 0)

        # Extract knowledge pre-check status
        if 'knowledge_precheck' in trial:
            knowledge_status.append(trial['knowledge_precheck']['model_knows'])
        else:
            knowledge_status.append(None)

    return np.array(X), np.array(y), np.array(knowledge_status)

def loo_accuracy(X, y):
    """Compute Leave-One-Out cross-validated accuracy."""
    if len(X) == 0:
        return 0.0

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    loo = LeaveOneOut()
    correct = 0
    total = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        if pred[0] == y_test[0]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0

def analyze_model(model_name, filepath):
    """Analyze one model's instructed-matched results stratified by knowledge pre-check."""
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    data = load_data(filepath)
    X, y, knowledge_status = extract_features_and_labels(data)

    # Stratify by knowledge pre-check
    passed_mask = knowledge_status == True
    failed_mask = knowledge_status == False

    n_total = len(X)
    n_passed = np.sum(passed_mask)
    n_failed = np.sum(failed_mask)

    print(f"\nSample Sizes:")
    print(f"  Total: {n_total}")
    print(f"  Knowledge Pre-Check PASSED: {n_passed} ({100*n_passed/n_total:.1f}%)")
    print(f"  Knowledge Pre-Check FAILED: {n_failed} ({100*n_failed/n_total:.1f}%)")

    # Compute LOO accuracy for each subset
    acc_total = loo_accuracy(X, y) * 100
    acc_passed = loo_accuracy(X[passed_mask], y[passed_mask]) * 100 if n_passed > 0 else 0.0
    acc_failed = loo_accuracy(X[failed_mask], y[failed_mask]) * 100 if n_failed > 0 else 0.0

    print(f"\nLOO Accuracy (Instructed-Matched):")
    print(f"  Total: {acc_total:.1f}%")
    print(f"  PASSED subset: {acc_passed:.1f}%")
    print(f"  FAILED subset: {acc_failed:.1f}%")

    # Compute defensiveness Cohen's d for each subset
    # Extract defensiveness (index 2) for truth vs lie trials
    defensiveness = X[:, 2]
    truth_mask = y == 0
    lie_mask = y == 1

    d_total = cohens_d(defensiveness[lie_mask], defensiveness[truth_mask])

    if n_passed > 0:
        passed_truth_mask = passed_mask & truth_mask
        passed_lie_mask = passed_mask & lie_mask
        d_passed = cohens_d(
            defensiveness[passed_lie_mask],
            defensiveness[passed_truth_mask]
        )
    else:
        d_passed = 0.0

    if n_failed > 0:
        failed_truth_mask = failed_mask & truth_mask
        failed_lie_mask = failed_mask & lie_mask
        d_failed = cohens_d(
            defensiveness[failed_lie_mask],
            defensiveness[failed_truth_mask]
        )
    else:
        d_failed = 0.0

    print(f"\nDefensiveness Effect Size (Cohen's d):")
    print(f"  Total: {d_total:+.2f}")
    print(f"  PASSED subset: {d_passed:+.2f}")
    print(f"  FAILED subset: {d_failed:+.2f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"  Instruction-following contribution in PASSED subset:")

    # The instructed-matched condition isolates instruction-following effect
    # If accuracy and effect size remain high on PASSED subset → clean decomposition
    # If they drop substantially → conflated with knowledge transfer

    if acc_passed >= 55.0 and abs(d_passed) >= 0.3:
        print(f"    ✓ DOMINANT (acc={acc_passed:.1f}%, d={d_passed:+.2f})")
        print(f"    → Instruction-following is the primary signal even when knowledge is controlled")
        print(f"    → Clean decomposition holds for {model_name}")
    else:
        print(f"    ✗ WEAK or ABSENT (acc={acc_passed:.1f}%, d={d_passed:+.2f})")
        print(f"    → Effect disappears when knowledge is controlled")
        print(f"    → Conflated with knowledge transfer for {model_name}")

    return {
        'model': model_name,
        'n_total': int(n_total),
        'n_passed': int(n_passed),
        'n_failed': int(n_failed),
        'acc_total': float(acc_total),
        'acc_passed': float(acc_passed),
        'acc_failed': float(acc_failed),
        'd_total': float(d_total),
        'd_passed': float(d_passed),
        'd_failed': float(d_failed),
        'clean_decomposition': bool(acc_passed >= 55.0 and abs(d_passed) >= 0.3)
    }

def main():
    """Run knowledge pre-check stratified analysis on all three models."""
    results_dir = Path('/Users/mediratta/code/interpret/adaptive_lie_detector/data/results')

    models = [
        ('Llama 3.2 3B', results_dir / 'ollama_eval_llama3_2_3b_instructed_matched_latest.json'),
        ('Llama 3.1 8B', results_dir / 'bedrock_eval_llama8b_instructed_matched_latest.json'),
        ('Mistral 7B', results_dir / 'ollama_eval_mistral_7b_instructed_matched_latest.json'),
    ]

    all_results = []

    for model_name, filepath in models:
        result = analyze_model(model_name, filepath)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Knowledge Pre-Check Stratified Analysis")
    print(f"{'='*80}\n")

    print(f"{'Model':<20} {'n_PASSED':<10} {'Acc_PASSED':<12} {'d_PASSED':<12} {'Clean?':<10}")
    print(f"{'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for r in all_results:
        clean_str = "YES ✓" if r['clean_decomposition'] else "NO ✗"
        print(f"{r['model']:<20} {r['n_passed']:<10} {r['acc_passed']:>10.1f}% {r['d_passed']:>11.2f} {clean_str:<10}")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")

    clean_models = [r['model'] for r in all_results if r['clean_decomposition']]
    unclean_models = [r['model'] for r in all_results if not r['clean_decomposition']]

    if len(clean_models) == 3:
        print("✓ Clean decomposition holds for ALL three models (3B, 8B, 7B)")
        print("  → Instruction-following is dominant even when knowledge is controlled")
        print("  → Update paper: Clarify that Finding 1 (+7.5–31 pp) holds across all scales")
        print("                  when stratified by knowledge pre-check")
    elif len(unclean_models) == 3:
        print("✗ Clean decomposition fails for ALL three models (3B, 8B, 7B)")
        print("  → Instruction-following conflates with knowledge transfer")
        print("  → Update paper: Restrict Finding 1 to '≥14B models only'")
    else:
        print(f"⚠ Mixed results:")
        print(f"  Clean: {', '.join(clean_models)}")
        print(f"  Unclean: {', '.join(unclean_models)}")
        print(f"  → Update paper: Note which models show clean decomposition")

    # Save results
    output_file = results_dir / 'knowledge_precheck_stratified_exp_g.json'
    with open(output_file, 'w') as f:
        json.dump({
            'analysis': 'Knowledge Pre-Check Stratified Analysis for EXP-G',
            'models': all_results,
            'summary': {
                'clean_models': clean_models,
                'unclean_models': unclean_models
            }
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
