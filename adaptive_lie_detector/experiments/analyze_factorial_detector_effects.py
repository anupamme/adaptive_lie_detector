"""
analyze_factorial_detector_effects.py

Applies the factorial decomposition to a COMPLETE DETECTOR, not to one feature.

WHY THIS EXISTS
---------------
`analyze_factorial_signed_effects.py` decomposes refusal-marker count, which is one
feature of the five-feature pipeline. A reasonable objection is that the factorial is
meant to audit a detector, but is applied to a feature of one.

This script closes that gap for the R1-faithful detector (Pacchiardi-style: 16 fixed
unrelated yes/no probes -> binary vector -> logistic regression), which is a complete,
mechanistically distinct detector whose full input IS recorded in the factorial cells.

Calibration mirrors what a benchmark user would do:
    train the LR on the INSTRUCTED benchmark data, then apply it to the factorial cells.
The regression target is the LR decision score (log-odds), so the effect is signed.

NOT COVERED: the five-feature ADAGE score cannot be decomposed this way, because the
factorial cells contain no LLM feature extraction (only refusal_count and the probe
vector). Generating it would require fresh extractor calls. The paper states this.

Usage:
    cd /path/to/adaptive_lie_detector
    python3 experiments/analyze_factorial_detector_effects.py
"""

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.claims_equalized import EQUALIZED_CLAIMS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "results")

MODELS = [("Llama 3.2 3B", "llama3_2_3b"),
          ("Mistral 7B", "mistral_7b"),
          ("Qwen 2.5 14B", "qwen2_5_14b")]

CELLS = ["T_H", "F_H", "T_D", "F_D"]
CLAIM_TO_PAIR = {c: i for i, (tc, fc) in enumerate(EQUALIZED_CLAIMS) for c in (tc, fc)}

B_BOOT = 2000
B_PERM = 2000
SEED = 42


def train_benchmark_detector(model_tag):
    """Calibrate the R1-faithful LR on the instructed benchmark, as a user would."""
    path = os.path.join(DATA_DIR, f"r1_faithful_{model_tag}_instructed.json")
    with open(path) as f:
        records = json.load(f)["records"]
    X = np.array([r["vector"] for r in records], dtype=float)
    y = np.array([r["label"] for r in records], dtype=int)
    clf = Pipeline([("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED))])
    clf.fit(X, y)
    return clf, len(y)


def load_factorial(model_tag, clf):
    """Score every factorial trial with the calibrated detector."""
    V, D, S, G = [], [], [], []
    for cell in CELLS:
        path = os.path.join(DATA_DIR, f"factorial_txd_{model_tag}_{cell}.json")
        with open(path) as f:
            records = json.load(f)["records"]
        v = 1 if cell[0] == "T" else 0
        d = 1 if cell[2] == "D" else 0
        for r in records:
            pair = CLAIM_TO_PAIR.get(r["claim"])
            if pair is None:
                continue
            score = clf.decision_function(np.array([r["vector"]], dtype=float))[0]
            V.append(v)
            D.append(d)
            S.append(float(score))
            G.append(pair)
    return np.array(V), np.array(D), np.array(S), np.array(G)


def fit(V, D, Y):
    X = np.column_stack([np.ones(len(V)), V, D, V * D])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def clustered_bootstrap(V, D, Y, G, rng, B=B_BOOT):
    pairs = np.unique(G)
    idx_by_pair = {p: np.where(G == p)[0] for p in pairs}
    out = []
    for _ in range(B):
        sampled = rng.choice(pairs, size=len(pairs), replace=True)
        idx = np.concatenate([idx_by_pair[p] for p in sampled])
        try:
            out.append(fit(V[idx], D[idx], Y[idx]))
        except np.linalg.LinAlgError:
            continue
    return np.array(out)


def permutation_p(V, D, Y, G, rng, coef_index, B=B_PERM):
    """Shuffle D within (claim pair, truth-value) strata."""
    observed = fit(V, D, Y)[coef_index]
    strata = {}
    for i, (g, v) in enumerate(zip(G, V)):
        strata.setdefault((g, v), []).append(i)
    count = 0
    for _ in range(B):
        Dp = D.copy()
        for idx in strata.values():
            idx = np.array(idx)
            Dp[idx] = rng.permutation(D[idx])
        if abs(fit(V, Dp, Y)[coef_index]) >= abs(observed):
            count += 1
    return (count + 1) / (B + 1)


def main():
    rng = np.random.RandomState(SEED)
    print("=" * 78)
    print("  FACTORIAL DECOMPOSITION OF A COMPLETE DETECTOR")
    print("  Detector: R1-faithful LR (16 probes -> binary vector -> logistic regression)")
    print("  Calibration: trained on the INSTRUCTED benchmark, applied to factorial cells")
    print("  Target: LR decision score (log-odds).  V=1 true claim, D=1 deceive instruction")
    print("  CIs: claim-pair clustered bootstrap; p: D permuted within (pair, V) strata")
    print("=" * 78)

    results = []
    for name, tag in MODELS:
        clf, n_train = train_benchmark_detector(tag)
        V, D, S, G = load_factorial(tag, clf)
        b = fit(V, D, S)
        boot = clustered_bootstrap(V, D, S, G, rng)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        p_D = permutation_p(V, D, S, G, rng, coef_index=2)
        p_VD = permutation_p(V, D, S, G, rng, coef_index=3)

        eff_false, eff_true = b[2], b[2] + b[3]

        print(f"\n{name}   (trained n={n_train}; scored n={len(S)}, "
              f"{len(np.unique(G))} claim pairs)")
        print(f"  bV  (truth-value)   {b[1]:+.3f}  [{lo[1]:+.3f}, {hi[1]:+.3f}]")
        print(f"  bD  (instruction)   {b[2]:+.3f}  [{lo[2]:+.3f}, {hi[2]:+.3f}]  p={p_D:.4f}")
        print(f"  bVD (interaction)   {b[3]:+.3f}  [{lo[3]:+.3f}, {hi[3]:+.3f}]  p={p_VD:.4f}")
        print(f"  simple effect of D | false claim  {eff_false:+.3f}")
        print(f"  simple effect of D | true  claim  {eff_true:+.3f}")

        results.append({
            "model": name, "model_tag": tag,
            "n_train_instructed": int(n_train),
            "n_scored": int(len(S)), "n_claim_pairs": int(len(np.unique(G))),
            "beta_V": float(b[1]), "beta_V_ci": [float(lo[1]), float(hi[1])],
            "beta_D": float(b[2]), "beta_D_ci": [float(lo[2]), float(hi[2])],
            "beta_VD": float(b[3]), "beta_VD_ci": [float(lo[3]), float(hi[3])],
            "perm_p_beta_D": float(p_D), "perm_p_beta_VD": float(p_VD),
            "simple_effect_D_given_false": float(eff_false),
            "simple_effect_D_given_true": float(eff_true),
        })

    print("\n" + "=" * 78)
    bD = [r["beta_D"] for r in results]
    bVD = [r["beta_VD"] for r in results]
    print(f"  bD  across targets: {['%+.2f' % x for x in bD]}"
          f"  -> {'SIGN VARIES' if min(bD) < 0 < max(bD) else 'consistent sign'}")
    print(f"  bVD across targets: {['%+.2f' % x for x in bVD]}"
          f"  -> {'all positive' if min(bVD) > 0 else 'mixed'}")
    print("  Compare against the marker-count decomposition in")
    print("  analyze_factorial_signed_effects.py to see whether the instability is")
    print("  specific to that feature or holds for the complete detector.")

    out_path = os.path.join(DATA_DIR, "factorial_detector_effects.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "factorial_detector_effects",
            "detector": ("R1-faithful LR (Pacchiardi-style 16-probe binary vector), "
                         "calibrated on the instructed benchmark"),
            "target": "LR decision score (log-odds)",
            "model_spec": "Y = b0 + bV*V + bD*D + bVD*(V*D)",
            "uncertainty": f"claim-pair clustered bootstrap, B={B_BOOT}",
            "permutation": f"D shuffled within (claim pair, truth-value) strata, B={B_PERM}",
            "not_covered": ("the five-feature ADAGE score: factorial cells contain no "
                            "LLM feature extraction"),
            "results": results,
        }, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
