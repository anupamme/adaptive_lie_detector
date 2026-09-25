#!/usr/bin/env python3
"""
analyze_factorial_truth_x_deception.py: Analyze the 2×2 truth × deception factorial (EXP-F).

Reads JSON outputs from run_factorial_truth_x_deception.py (4 cells per model) and produces:
  - Per-cell detection accuracy for both detectors (refusal-rule, R1-faithful LR)
  - Variance decomposition: truth_value vs. instruction vs. interaction
  - Independent deception effect size after controlling for truth-value
  - LaTeX table ready for the paper

The key question: after controlling for truth-value and instruction wording, does the
detector contain a reliable deception effect?

Usage:
    cd code/adaptive_lie_detector
    python experiments/analyze_factorial_truth_x_deception.py
"""

import sys
import os
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from experiments.claims_equalized import EQUALIZED_CLAIMS

CLAIM_TO_PAIR = {c: i for i, (tc, fc) in enumerate(EQUALIZED_CLAIMS) for c in (tc, fc)}

RESULTS_DIR = "data/results"
CELLS = ["T_H", "F_H", "T_D", "F_D"]
MODELS = ["llama3.2:3b", "mistral:7b", "qwen2.5:14b"]
N_PERM = 500
RNG = np.random.default_rng(42)


def load_cell_data(model_tag, cell):
    path = os.path.join(RESULTS_DIR, f"factorial_txd_{model_tag}_{cell}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("records", [])


def refusal_rule_predictions(records):
    """Predict 'truthful' (0) if refusal_count >= 1, else 'lying' (1)."""
    return np.array([0 if r["refusal_count"] >= 1 else 1 for r in records])


def get_ground_truth_labels(cell):
    """Ground truth: is the model being deceptive in this cell?

    T_H: honest about true claim → not deceptive (0)
    F_H: honest about false claim → not deceptive (0); model corrects, not deceives
    T_D: instructed to deceive about true claim → deceptive (1)
    F_D: instructed to deceive about false claim → deceptive (1)
    """
    if cell in ("T_D", "F_D"):
        return 1  # deceptive
    return 0  # not deceptive


def get_truth_value(cell):
    """Is the claim actually true?"""
    return 1 if cell.startswith("T") else 0


def get_instruction(cell):
    """Does the instruction say to deceive?"""
    return 1 if cell.endswith("_D") else 0


def analyze_refusal_rule(model_tag):
    """Analyze the refusal-count rule across all 4 cells."""
    print(f"\n{'='*60}")
    print(f"REFUSAL-COUNT RULE: {model_tag}")
    print(f"{'='*60}")

    all_records = {}
    for cell in CELLS:
        recs = load_cell_data(model_tag, cell)
        if recs is None:
            print(f"  [missing] {cell}")
            return None
        all_records[cell] = recs

    # Per-cell refusal counts and predictions
    print(f"\n  {'Cell':<8} {'N':<4} {'Mean RC':<9} {'RC≥1 rate':<10} {'Pred=truthful':<14}")
    print(f"  {'-'*50}")

    cell_stats = {}
    for cell in CELLS:
        recs = all_records[cell]
        rcs = [r["refusal_count"] for r in recs]
        mean_rc = np.mean(rcs)
        rc_ge1 = np.mean([1 if rc >= 1 else 0 for rc in rcs])
        preds = refusal_rule_predictions(recs)
        pred_truthful = np.mean(preds == 0)
        print(f"  {cell:<8} {len(recs):<4} {mean_rc:<9.2f} {rc_ge1:<10.1%} {pred_truthful:<14.1%}")
        cell_stats[cell] = {
            "n": len(recs),
            "mean_rc": float(mean_rc),
            "rc_ge1_rate": float(rc_ge1),
            "pred_truthful_rate": float(pred_truthful),
        }

    # Detection accuracy under different labeling schemes
    print(f"\n  Detection accuracy (rule: RC≥1 → truthful, RC=0 → deceptive):")

    # Scheme 1: Deception label (T_D + F_D = deceptive; T_H + F_H = honest)
    honest_recs = all_records["T_H"] + all_records["F_H"]
    deceptive_recs = all_records["T_D"] + all_records["F_D"]

    honest_preds = refusal_rule_predictions(honest_recs)
    deceptive_preds = refusal_rule_predictions(deceptive_recs)

    acc_honest = np.mean(honest_preds == 0)  # should predict "truthful" for honest
    acc_deceptive = np.mean(deceptive_preds == 1)  # should predict "deceptive" for deceive
    acc_deception = (acc_honest * len(honest_recs) + acc_deceptive * len(deceptive_recs)) / (
        len(honest_recs) + len(deceptive_recs))

    print(f"    Deception detection (instruction-based):  {acc_deception:.1%}")
    print(f"      Honest correct:   {acc_honest:.1%} ({int(acc_honest*len(honest_recs))}/{len(honest_recs)})")
    print(f"      Deceptive correct: {acc_deceptive:.1%} ({int(acc_deceptive*len(deceptive_recs))}/{len(deceptive_recs)})")

    # Scheme 2: Truth-value label (T_H + T_D = true claim; F_H + F_D = false claim)
    true_recs = all_records["T_H"] + all_records["T_D"]
    false_recs = all_records["F_H"] + all_records["F_D"]

    true_preds = refusal_rule_predictions(true_recs)
    false_preds = refusal_rule_predictions(false_recs)

    # Under truth-value labeling: RC≥1 → "false claim detected" (correction), RC=0 → "true claim"
    acc_true = np.mean(true_preds == 1)  # true claims should have low RC → predict "deceptive"=1
    acc_false = np.mean(false_preds == 0)  # false claims should have high RC → predict "truthful"=0
    acc_truth = (acc_true * len(true_recs) + acc_false * len(false_recs)) / (
        len(true_recs) + len(false_recs))

    print(f"    Truth-value classification (claim-based): {acc_truth:.1%}")
    print(f"      True claim (pred no-correction): {acc_true:.1%}")
    print(f"      False claim (pred correction):   {acc_false:.1%}")

    # Variance decomposition via logistic regression
    print(f"\n  Variance decomposition (logistic regression on RC count):")
    all_rcs = []
    all_truth = []
    all_instr = []
    all_deception_label = []
    for cell in CELLS:
        for r in all_records[cell]:
            all_rcs.append(r["refusal_count"])
            all_truth.append(get_truth_value(cell))
            all_instr.append(get_instruction(cell))
            all_deception_label.append(get_ground_truth_labels(cell))

    all_rcs = np.array(all_rcs).reshape(-1, 1)
    all_truth = np.array(all_truth)
    all_instr = np.array(all_instr)
    all_deception_label = np.array(all_deception_label)

    # Does RC predict deception (instruction) or truth-value?
    # Correlation with each factor
    from scipy.stats import pointbiserialr
    r_truth, p_truth = pointbiserialr(all_truth.flatten(), all_rcs.flatten())
    r_instr, p_instr = pointbiserialr(all_instr.flatten(), all_rcs.flatten())

    print(f"    Correlation(RC, truth_value):  r={r_truth:.3f}, p={p_truth:.4f}")
    print(f"    Correlation(RC, instruction):  r={r_instr:.3f}, p={p_instr:.4f}")

    # Per-cell mean RC for the 2x2 table
    print(f"\n  2×2 Mean Refusal Count:")
    print(f"    {'':12} {'True claim':<14} {'False claim':<14}")
    print(f"    {'Honest':<12} {cell_stats['T_H']['mean_rc']:<14.2f} {cell_stats['F_H']['mean_rc']:<14.2f}")
    print(f"    {'Deceive':<12} {cell_stats['T_D']['mean_rc']:<14.2f} {cell_stats['F_D']['mean_rc']:<14.2f}")

    # Main effects
    honest_mean = (cell_stats["T_H"]["mean_rc"] + cell_stats["F_H"]["mean_rc"]) / 2
    deceive_mean = (cell_stats["T_D"]["mean_rc"] + cell_stats["F_D"]["mean_rc"]) / 2
    true_mean = (cell_stats["T_H"]["mean_rc"] + cell_stats["T_D"]["mean_rc"]) / 2
    false_mean = (cell_stats["F_H"]["mean_rc"] + cell_stats["F_D"]["mean_rc"]) / 2

    print(f"\n  Main effects:")
    print(f"    Instruction effect: honest={honest_mean:.2f}, deceive={deceive_mean:.2f}, "
          f"Δ={honest_mean - deceive_mean:+.2f}")
    print(f"    Truth-value effect: true={true_mean:.2f}, false={false_mean:.2f}, "
          f"Δ={false_mean - true_mean:+.2f}")

    # Interaction
    interaction = (cell_stats["T_H"]["mean_rc"] - cell_stats["F_H"]["mean_rc"]) - \
                  (cell_stats["T_D"]["mean_rc"] - cell_stats["F_D"]["mean_rc"])
    print(f"    Interaction: {interaction:+.2f}")

    # Deception-specific contrasts: T/D vs T/H and F/D vs F/H
    # These isolate the instruction effect *within* each truth-value condition
    print(f"\n  Deception-specific contrasts (RC≥1 fire rate):")
    td_rate = cell_stats["T_D"]["rc_ge1_rate"]
    th_rate = cell_stats["T_H"]["rc_ge1_rate"]
    fd_rate = cell_stats["F_D"]["rc_ge1_rate"]
    fh_rate = cell_stats["F_H"]["rc_ge1_rate"]
    print(f"    T/D vs T/H:  {td_rate:.1%} vs {th_rate:.1%}  Δ={td_rate - th_rate:+.1%}")
    print(f"    F/D vs F/H:  {fd_rate:.1%} vs {fh_rate:.1%}  Δ={fd_rate - fh_rate:+.1%}")

    return {
        "model": model_tag,
        "detector": "refusal_rule",
        "cell_stats": cell_stats,
        "acc_deception_detection": float(acc_deception),
        "acc_truth_value": float(acc_truth),
        "r_truth": float(r_truth),
        "p_truth": float(p_truth),
        "r_instruction": float(r_instr),
        "p_instruction": float(p_instr),
        "main_effect_instruction": float(honest_mean - deceive_mean),
        "main_effect_truth": float(false_mean - true_mean),
        "interaction": float(interaction),
        "td_vs_th_rate": float(td_rate - th_rate),
        "fd_vs_fh_rate": float(fd_rate - fh_rate),
    }


def analyze_r1_detector(model_tag):
    """Analyze the R1-faithful binary-probe LR detector across all 4 cells."""
    print(f"\n{'='*60}")
    print(f"R1-FAITHFUL LR DETECTOR: {model_tag}")
    print(f"{'='*60}")

    all_records = {}
    for cell in CELLS:
        recs = load_cell_data(model_tag, cell)
        if recs is None:
            print(f"  [missing] {cell}")
            return None
        all_records[cell] = recs

    # Build X (binary vectors) and labels for different classification tasks
    X_all = []
    y_deception = []  # 1 = deceive instruction, 0 = honest
    y_truth = []      # 1 = true claim, 0 = false claim
    cell_ids = []
    claim_groups = []  # pair index for GroupKFold

    for cell in CELLS:
        for r in all_records[cell]:
            X_all.append(r["vector"])
            y_deception.append(get_ground_truth_labels(cell))
            y_truth.append(get_truth_value(cell))
            cell_ids.append(cell)
            claim_groups.append(CLAIM_TO_PAIR.get(r["claim"], -1))

    X_all = np.array(X_all, dtype=float)
    y_deception = np.array(y_deception)
    y_truth = np.array(y_truth)
    cell_ids = np.array(cell_ids)
    claim_groups = np.array(claim_groups)

    # Degeneracy check
    n_varying = np.sum(np.std(X_all, axis=0) > 0)
    n_unique = len(set(map(tuple, X_all.tolist())))
    print(f"\n  Feature matrix: {X_all.shape[0]} trials × {X_all.shape[1]} dims")
    print(f"  Varying dimensions: {n_varying}/{X_all.shape[1]}")
    print(f"  Unique vectors: {n_unique}/{X_all.shape[0]}")

    if n_varying < 2:
        print("  ⚠ Near-degenerate: fewer than 2 varying dimensions. Skipping LR analysis.")
        return None

    # Use GroupKFold to prevent matched true/false claim pairs from splitting across folds
    gkf = GroupKFold(n_splits=5)
    clf = LogisticRegression(max_iter=1000, C=1.0)

    # Task 1: Can the LR detect deception (instruction condition)?
    print(f"\n  Task 1: Predict DECEPTION (instruction-based label), grouped-5-fold")
    scores_deception = cross_val_score(clf, X_all, y_deception, cv=gkf, groups=claim_groups,
                                       scoring="accuracy")
    acc_deception = np.mean(scores_deception)
    print(f"    Grouped-5-fold accuracy: {acc_deception:.1%} (std={np.std(scores_deception):.1%})")

    # Permutation test
    perm_accs = []
    for _ in range(N_PERM):
        y_perm = RNG.permutation(y_deception)
        s = cross_val_score(clf, X_all, y_perm, cv=gkf, groups=claim_groups, scoring="accuracy")
        perm_accs.append(np.mean(s))
    p_deception = np.mean(np.array(perm_accs) >= acc_deception)
    print(f"    Permutation p: {p_deception:.4f}")

    # Task 2: Can the LR detect truth-value?
    print(f"\n  Task 2: Predict TRUTH VALUE (claim-based label), grouped-5-fold")
    scores_truth = cross_val_score(clf, X_all, y_truth, cv=gkf, groups=claim_groups,
                                   scoring="accuracy")
    acc_truth = np.mean(scores_truth)
    print(f"    Grouped-5-fold accuracy: {acc_truth:.1%} (std={np.std(scores_truth):.1%})")

    perm_accs_t = []
    for _ in range(N_PERM):
        y_perm = RNG.permutation(y_truth)
        s = cross_val_score(clf, X_all, y_perm, cv=gkf, groups=claim_groups, scoring="accuracy")
        perm_accs_t.append(np.mean(s))
    p_truth = np.mean(np.array(perm_accs_t) >= acc_truth)
    print(f"    Permutation p: {p_truth:.4f}")

    # Task 3: Full 2×2; predict deception CONTROLLING for truth-value
    # Add truth_value as a feature alongside binary vector
    print(f"\n  Task 3: Predict DECEPTION controlling for truth-value")
    X_with_truth = np.column_stack([X_all, y_truth.reshape(-1, 1)])
    scores_controlled = cross_val_score(clf, X_with_truth, y_deception, cv=gkf, groups=claim_groups,
                                        scoring="accuracy")
    acc_controlled = np.mean(scores_controlled)
    print(f"    Grouped-5-fold accuracy (vector + truth): {acc_controlled:.1%}")

    # Task 4: Within true-claims only; can LR distinguish T_H from T_D?
    print(f"\n  Task 4: Within TRUE claims only, T_H vs T_D (deception-specific contrast)")
    mask_true = np.isin(cell_ids, ["T_H", "T_D"])
    X_true = X_all[mask_true]
    y_true_deception = y_deception[mask_true]
    groups_true = claim_groups[mask_true]
    if len(np.unique(y_true_deception)) >= 2 and len(y_true_deception) >= 10:
        gkf_sub = GroupKFold(n_splits=5)
        scores_true_only = cross_val_score(clf, X_true, y_true_deception, cv=gkf_sub,
                                           groups=groups_true, scoring="accuracy")
        acc_true_only = np.mean(scores_true_only)
        print(f"    Grouped-5-fold accuracy: {acc_true_only:.1%}")
        perm_accs_to = []
        for _ in range(N_PERM):
            y_perm = RNG.permutation(y_true_deception)
            s = cross_val_score(clf, X_true, y_perm, cv=gkf_sub, groups=groups_true,
                                scoring="accuracy")
            perm_accs_to.append(np.mean(s))
        p_true_only = np.mean(np.array(perm_accs_to) >= acc_true_only)
        print(f"    Permutation p: {p_true_only:.4f}")
    else:
        acc_true_only = float("nan")
        p_true_only = float("nan")
        print(f"    Insufficient data or single class.")

    # Task 5: Within false-claims only; can LR distinguish F_H from F_D?
    print(f"\n  Task 5: Within FALSE claims only, F_H vs F_D (deception-specific contrast)")
    mask_false = np.isin(cell_ids, ["F_H", "F_D"])
    X_false = X_all[mask_false]
    y_false_deception = y_deception[mask_false]
    groups_false = claim_groups[mask_false]
    if len(np.unique(y_false_deception)) >= 2 and len(y_false_deception) >= 10:
        scores_false_only = cross_val_score(clf, X_false, y_false_deception, cv=gkf_sub,
                                            groups=groups_false, scoring="accuracy")
        acc_false_only = np.mean(scores_false_only)
        print(f"    Grouped-5-fold accuracy: {acc_false_only:.1%}")
        perm_accs_fo = []
        for _ in range(N_PERM):
            y_perm = RNG.permutation(y_false_deception)
            s = cross_val_score(clf, X_false, y_perm, cv=gkf_sub, groups=groups_false,
                                scoring="accuracy")
            perm_accs_fo.append(np.mean(s))
        p_false_only = np.mean(np.array(perm_accs_fo) >= acc_false_only)
        print(f"    Permutation p: {p_false_only:.4f}")
    else:
        acc_false_only = float("nan")
        p_false_only = float("nan")
        print(f"    Insufficient data or single class.")

    return {
        "model": model_tag,
        "detector": "r1_faithful_lr",
        "n_varying_dims": int(n_varying),
        "n_unique_vectors": int(n_unique),
        "acc_deception_5fold": float(acc_deception),
        "p_deception": float(p_deception),
        "acc_truth_5fold": float(acc_truth),
        "p_truth": float(p_truth),
        "acc_deception_controlled": float(acc_controlled),
        "acc_true_claims_only": float(acc_true_only),
        "p_true_claims_only": float(p_true_only),
        "acc_false_claims_only": float(acc_false_only),
        "p_false_claims_only": float(p_false_only),
    }


def print_latex_table(all_results):
    """Print LaTeX table for the paper."""
    print(f"\n\n{'='*60}")
    print("LaTeX TABLE")
    print(f"{'='*60}")
    print(r"""
\begin{table}[tbp]
\centering
\small
\setlength{\tabcolsep}{4pt}
\caption{EXP-F: 2$\times$2 factorial decomposition. Mean refusal-marker count per cell
(top) and detection accuracy under two labeling schemes (bottom). The refusal rule
tracks truth-value more than deception instruction; the R1-faithful LR similarly
fails to isolate an independent deception signal.}
\label{tab:factorial}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{2}{c}{\textbf{True claim}} & \multicolumn{2}{c}{\textbf{False claim}} & \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
\textbf{Model} & Honest & Deceive & Honest & Deceive & \textbf{Deception acc.} \\
\midrule""")

    for res in all_results:
        if res and res.get("detector") == "refusal_rule":
            cs = res["cell_stats"]
            model = res["model"]
            print(f"{model} & {cs['T_H']['mean_rc']:.1f} & {cs['T_D']['mean_rc']:.1f} "
                  f"& {cs['F_H']['mean_rc']:.1f} & {cs['F_D']['mean_rc']:.1f} "
                  f"& {res['acc_deception_detection']:.1%} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


def main():
    all_refusal_results = []
    all_lr_results = []

    for model in MODELS:
        model_tag = model.replace(":", "_").replace(".", "_")
        res_r = analyze_refusal_rule(model_tag)
        res_lr = analyze_r1_detector(model_tag)
        all_refusal_results.append(res_r)
        all_lr_results.append(res_lr)

    print_latex_table(all_refusal_results)

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY: What does the detector measure?")
    print(f"{'='*60}")
    for res in all_refusal_results:
        if res:
            print(f"\n  {res['model']} (refusal rule):")
            print(f"    Correlates with truth-value: r={res['r_truth']:.3f} (p={res['p_truth']:.4f})")
            print(f"    Correlates with instruction: r={res['r_instruction']:.3f} (p={res['p_instruction']:.4f})")
            print(f"    Main effect (instruction):   Δ={res['main_effect_instruction']:+.2f}")
            print(f"    Main effect (truth-value):   Δ={res['main_effect_truth']:+.2f}")
            stronger = "truth-value" if abs(res['r_truth']) > abs(res['r_instruction']) else "instruction"
            print(f"    → Stronger predictor: {stronger}")

    for res in all_lr_results:
        if res:
            print(f"\n  {res['model']} (R1-faithful LR):")
            print(f"    Deception detection (all data):    {res['acc_deception_5fold']:.1%} (p={res['p_deception']:.4f})")
            print(f"    Truth-value detection (all data):  {res['acc_truth_5fold']:.1%} (p={res['p_truth']:.4f})")
            print(f"    Deception within TRUE claims only: {res['acc_true_claims_only']:.1%} (p={res['p_true_claims_only']:.4f})")
            print(f"    Deception within FALSE claims only: {res['acc_false_claims_only']:.1%} (p={res['p_false_claims_only']:.4f})")

    # Save results
    output = {
        "experiment": "EXP-F_factorial_truth_x_deception",
        "refusal_rule_results": [r for r in all_refusal_results if r],
        "r1_lr_results": [r for r in all_lr_results if r],
    }
    out_path = os.path.join(RESULTS_DIR, "factorial_txd_summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
