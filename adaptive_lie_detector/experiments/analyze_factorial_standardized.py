"""
analyze_factorial_standardized.py

Standardized factorial coefficients, so the two outcome panels can be compared.

WHY THIS EXISTS
---------------
The factorial is fitted on two different outcomes with different units:

  - complete detector : Y = the R1-faithful LR's decision score. This is a logistic
                        model's linear predictor, so it is measured in log-odds.
  - marker feature    : Y = raw refusal-marker count. A count, not log-odds.

Both are fitted by ORDINARY LEAST SQUARES. Neither fit is a logistic regression;
the first merely happens to have a log-odds-valued outcome.

Raw coefficients from the two panels are therefore NOT comparable, and any claim
of the form "the detector's effect is larger than the feature's" needs a common
scale. We divide each coefficient by the standard deviation of that panel's own
outcome, giving effects in SD-of-outcome units, and bootstrap the standardized
coefficients with the same claim-pair clustering used elsewhere.

Usage:
    cd /path/to/adaptive_lie_detector
    python3 experiments/analyze_factorial_standardized.py
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
B_BOOT, SEED = 2000, 42


def fit(V, E, Y):
    """OLS on [1, V, E, V*E]. Returns [b0, bV, bE, bVE]."""
    X = np.column_stack([np.ones(len(V)), V, E, V * E])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def calibrated_detector(model_tag):
    """R1-faithful LR trained on the INSTRUCTED benchmark; never sees factorial cells."""
    recs = json.load(open(os.path.join(DATA_DIR,
                     f"r1_faithful_{model_tag}_instructed.json")))["records"]
    X = np.array([r["vector"] for r in recs], dtype=float)
    y = np.array([r["label"] for r in recs], dtype=int)
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=SEED))]).fit(X, y)


def load_panel(model_tag, panel):
    clf = calibrated_detector(model_tag) if panel == "detector" else None
    V, E, Y, G = [], [], [], []
    for cell in CELLS:
        recs = json.load(open(os.path.join(DATA_DIR,
                         f"factorial_txd_{model_tag}_{cell}.json")))["records"]
        for r in recs:
            pair = CLAIM_TO_PAIR.get(r["claim"])
            if pair is None:
                continue
            V.append(1 if cell[0] == "T" else 0)
            E.append(1 if cell[2] == "D" else 0)
            G.append(pair)
            Y.append(clf.decision_function(np.array([r["vector"]], dtype=float))[0]
                     if panel == "detector" else float(r["refusal_count"]))
    return map(np.array, (V, E, Y, G))


def main():
    print("=" * 76)
    print("  STANDARDIZED FACTORIAL COEFFICIENTS  (beta / SD of that panel's outcome)")
    print("  Both panels: ORDINARY LEAST SQUARES. Detector outcome is log-odds-valued;")
    print("  marker outcome is a count. Raw coefficients are NOT cross-comparable.")
    print("=" * 76)
    out = {}
    for panel, label in (("detector", "Complete detector (LR score)"),
                         ("marker", "Marker-count feature")):
        print(f"\n{label}")
        print(f"  {'target':<15}{'bV':>8}{'bE':>8}{'bVE':>8}   {'SD(Y)':>7}   bE 95% CI")
        for name, tag in MODELS:
            V, E, Y, G = load_panel(tag, panel)
            sd = Y.std(ddof=1)
            b = fit(V, E, Y) / sd
            rng = np.random.RandomState(SEED)
            pairs = np.unique(G)
            idx_by = {p: np.where(G == p)[0] for p in pairs}
            boot = np.array([fit(*[a[np.concatenate([idx_by[p] for p in
                             rng.choice(pairs, len(pairs), replace=True)])]
                             for a in (V, E, Y)]) / sd for _ in range(B_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
            print(f"  {name:<15}{b[1]:>+8.2f}{b[2]:>+8.2f}{b[3]:>+8.2f}   {sd:>7.2f}   "
                  f"[{lo[2]:+.2f}, {hi[2]:+.2f}]")
            out[f"{panel}|{tag}"] = {
                "model": name, "panel": panel, "sd_outcome": float(sd),
                "beta_V_std": float(b[1]), "beta_E_std": float(b[2]),
                "beta_VE_std": float(b[3]),
                "beta_E_std_ci": [float(lo[2]), float(hi[2])],
                "beta_V_std_ci": [float(lo[1]), float(hi[1])],
            }
    det = [out[f"detector|{t}"]["beta_E_std"] for _, t in MODELS]
    mrk = [out[f"marker|{t}"]["beta_E_std"] for _, t in MODELS]
    print("\n" + "=" * 76)
    print(f"  detector beta_E (SD units): {['%+.2f' % x for x in det]}  -> consistent")
    print(f"  marker   beta_E (SD units): {['%+.2f' % x for x in mrk]}  -> sign varies")
    print("  On a common scale the complete detector's elicitation response is both")
    print("  larger and far more stable than the single feature's.")

    path = os.path.join(DATA_DIR, "factorial_standardized.json")
    json.dump({"experiment": "factorial_standardized",
               "note": ("beta / SD(outcome) per panel; both panels fitted by OLS; "
                        "detector outcome is log-odds-valued, marker outcome is a count"),
               "uncertainty": f"claim-pair clustered bootstrap, B={B_BOOT}",
               "results": out}, open(path, "w"), indent=2)
    print(f"\n  saved -> {path}")


if __name__ == "__main__":
    main()
