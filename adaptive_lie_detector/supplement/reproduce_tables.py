#!/usr/bin/env python3
"""
reproduce_tables.py: Reproduce headline tables from the supplementary CSV data.

Requirements: Python 3.8+ (no external dependencies).

Usage:
    python reproduce_tables.py
"""

import csv
import os


def load_results(path="headline_results.csv"):
    """Load headline results CSV."""
    rows = []
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key != "model" and row[key]:
                    row[key] = float(row[key])
                elif key != "model":
                    row[key] = None
            rows.append(row)
    return rows


def print_table_detection_accuracy(rows):
    """Print Table: Per-Model Detection Accuracy (Prompt-Equalized Condition)."""
    print("=" * 90)
    print("TABLE: Per-Model Detection Accuracy, Prompt-Equalized Condition")
    print("=" * 90)
    print(f"{'Model':<20} {'N':>5} {'LOO Acc%':>10} {'Refusal k=1%':>13} "
          f"{'Truth Acc%':>11} {'Lie Acc%':>9} {'Avg Q':>7}")
    print("-" * 90)

    valid_rows = [r for r in rows if r["n_samples"] and r["n_samples"] >= 10]

    for r in valid_rows:
        loo = f"{r['loo_accuracy_pct']:.1f}" if r["loo_accuracy_pct"] else ", "
        ref = f"{r['refusal_k1_accuracy_pct']:.1f}" if r["refusal_k1_accuracy_pct"] else ", "
        truth = f"{r['truthful_acc_pct']:.1f}" if r["truthful_acc_pct"] else ", "
        lie = f"{r['lying_acc_pct']:.1f}" if r["lying_acc_pct"] else ", "
        avgq = f"{r['avg_questions']:.1f}" if r["avg_questions"] else ", "
        n = int(r["n_samples"])
        print(f"{r['model']:<20} {n:>5} {loo:>10} {ref:>13} {truth:>11} {lie:>9} {avgq:>7}")

    print("-" * 90)

    # Averages (excluding models with < 10 samples)
    loo_vals = [r["loo_accuracy_pct"] for r in valid_rows if r["loo_accuracy_pct"]]
    ref_vals = [r["refusal_k1_accuracy_pct"] for r in valid_rows if r["refusal_k1_accuracy_pct"]]

    if loo_vals:
        print(f"{'Mean':<20} {'':>5} {sum(loo_vals)/len(loo_vals):>9.1f}% "
              f"{sum(ref_vals)/len(ref_vals):>12.1f}%")
    print()


def print_table_method_comparison(rows):
    """Print Table: Adaptive LOO vs Fixed-Threshold Refusal Baseline."""
    print("=" * 70)
    print("TABLE: Adaptive (5-Feature LOO) vs Refusal Baseline (k=1)")
    print("=" * 70)
    print(f"{'Model':<20} {'LOO Acc%':>10} {'Refusal k=1%':>13} {'Delta':>8}")
    print("-" * 70)

    valid_rows = [r for r in rows
                  if r["loo_accuracy_pct"] and r["refusal_k1_accuracy_pct"]]

    for r in valid_rows:
        delta = r["loo_accuracy_pct"] - r["refusal_k1_accuracy_pct"]
        sign = "+" if delta >= 0 else ""
        print(f"{r['model']:<20} {r['loo_accuracy_pct']:>9.1f}% "
              f"{r['refusal_k1_accuracy_pct']:>12.1f}% {sign}{delta:>6.1f}%")

    print("-" * 70)

    loo_mean = sum(r["loo_accuracy_pct"] for r in valid_rows) / len(valid_rows)
    ref_mean = sum(r["refusal_k1_accuracy_pct"] for r in valid_rows) / len(valid_rows)
    print(f"{'Mean':<20} {loo_mean:>9.1f}% {ref_mean:>12.1f}% "
          f"{'+' if loo_mean-ref_mean >= 0 else ''}{loo_mean-ref_mean:>6.1f}%")
    print()


def print_summary(rows):
    """Print key findings summary."""
    valid = [r for r in rows if r["loo_accuracy_pct"] and r["n_samples"] and r["n_samples"] >= 10]
    if not valid:
        print("No valid data to summarize.")
        return

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    best = max(valid, key=lambda r: r["loo_accuracy_pct"])
    loo_vals = [r["loo_accuracy_pct"] for r in valid]
    ref_vals = [r["refusal_k1_accuracy_pct"] for r in valid if r["refusal_k1_accuracy_pct"]]

    print(f"  Models evaluated:       {len(valid)}")
    print(f"  Best LOO accuracy:      {best['model']} ({best['loo_accuracy_pct']:.1f}%)")
    print(f"  Mean LOO accuracy:      {sum(loo_vals)/len(loo_vals):.1f}%")
    if ref_vals:
        print(f"  Mean refusal baseline:  {sum(ref_vals)/len(ref_vals):.1f}%")
    print(f"  Evaluation method:      Leave-one-out cross-validation")
    print(f"  Condition:              Prompt-equalized (no lie instruction)")
    print()


def main():
    rows = load_results()
    print()
    print_table_detection_accuracy(rows)
    print_table_method_comparison(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
