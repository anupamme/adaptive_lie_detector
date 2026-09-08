"""
analyze_factorial_grouped_calibration.py

Factorial decomposition with CLAIM-HELD-OUT detector calibration.

WHY THIS EXISTS
---------------
`analyze_factorial_detector_effects.py` calibrates the R1-faithful LR on the instructed
benchmark and scores the factorial cells. No factorial *trial* enters calibration, but
the two share the same claim set: all 25 calibration claim pairs reappear among the
factorial's 50. So the detector saw the same semantic claims it is later scored on, and
the resulting coefficients are not claim-held-out.

This version closes that: to score a trial on claim pair p, the detector is calibrated
on the instructed data with pair p REMOVED. Pairs absent from calibration need no refit.
Everything else -- OLS on [1, V, E, V*E], claim-pair-clustered bootstrap, permutation of
E within (pair, V) strata -- matches the other factorial scripts so the numbers are
directly comparable.

Outputs raw and standardized (beta / SD of outcome) coefficients.

Usage:
    cd /path/to/adaptive_lie_detector
    python3 experiments/analyze_factorial_grouped_calibration.py
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
B_BOOT = B_PERM = 2000
SEED = 42


def _lr():
    return Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=SEED))])


def load_calibration(model_tag):
    recs = json.load(open(os.path.join(
        DATA_DIR, f"r1_faithful_{model_tag}_instructed.json")))["records"]
    X = np.array([r["vector"] for r in recs], dtype=float)
    y = np.array([r["label"] for r in recs], dtype=int)
    g = np.array([CLAIM_TO_PAIR.get(r["claim"], -1) for r in recs])
    return X, y, g


def fit(V, E, Y):
    X = np.column_stack([np.ones(len(V)), V, E, V * E])
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def score_grouped(model_tag):
    """Score every factorial trial with a detector that never saw its claim pair."""
    Xc, yc, gc = load_calibration(model_tag)
    full = _lr().fit(Xc, yc)
    cache, refits = {}, 0

    def detector_for(pair):
        nonlocal refits
        if pair not in set(gc):
            return full                      # pair absent from calibration: no leak
        if pair not in cache:
            keep = gc != pair
            if len(np.unique(yc[keep])) < 2:  # degenerate fold; fall back
                cache[pair] = full
            else:
                cache[pair] = _lr().fit(Xc[keep], yc[keep])
                refits += 1
        return cache[pair]

    V, E, Y, G = [], [], [], []
    for cell in CELLS:
        recs = json.load(open(os.path.join(
            DATA_DIR, f"factorial_txd_{model_tag}_{cell}.json")))["records"]
        for r in recs:
            pair = CLAIM_TO_PAIR.get(r["claim"])
            if pair is None:
                continue
            clf = detector_for(pair)
            V.append(1 if cell[0] == "T" else 0)
            E.append(1 if cell[2] == "D" else 0)
            G.append(pair)
            Y.append(float(clf.decision_function(
                np.array([r["vector"]], dtype=float))[0]))
    n_overlap = len(set(gc) & set(G))
    return (np.array(V), np.array(E), np.array(Y), np.array(G), refits, n_overlap,
            len(set(gc)))


def clustered(V, E, Y, G, rng, B=B_BOOT):
    pairs = np.unique(G)
    idx = {p: np.where(G == p)[0] for p in pairs}
    out = []
    for _ in range(B):
        s = rng.choice(pairs, len(pairs), replace=True)
        i = np.concatenate([idx[p] for p in s])
        out.append(fit(V[i], E[i], Y[i]))
    return np.array(out)


def perm_p(V, E, Y, G, rng, k, B=B_PERM):
    obs = fit(V, E, Y)[k]
    strata = {}
    for i, (g, v) in enumerate(zip(G, V)):
        strata.setdefault((g, v), []).append(i)
    c = 0
    for _ in range(B):
        Ep = E.copy()
        for ix in strata.values():
            ix = np.array(ix)
            Ep[ix] = rng.permutation(E[ix])
        if abs(fit(V, Ep, Y)[k]) >= abs(obs):
            c += 1
    return (c + 1) / (B + 1)


def main():
    rng = np.random.RandomState(SEED)
    print("=" * 78)
    print("  FACTORIAL WITH CLAIM-HELD-OUT CALIBRATION")
    print("  Scoring a trial on claim pair p uses a detector calibrated WITHOUT p.")
    print("  Y = LR decision score (log-odds-valued); OLS on [1, V, E, V*E].")
    print("=" * 78)

    prev = {}
    p_path = os.path.join(DATA_DIR, "factorial_standardized.json")
    if os.path.exists(p_path):
        prev = json.load(open(p_path))["results"]

    results = []
    for name, tag in MODELS:
        V, E, Y, G, refits, n_overlap, n_cal = score_grouped(tag)
        b = fit(V, E, Y)
        sd = Y.std(ddof=1)
        boot = clustered(V, E, Y, G, rng)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        pE = perm_p(V, E, Y, G, rng, 2)
        pVE = perm_p(V, E, Y, G, rng, 3)
        bs = b / sd

        old = prev.get(f"detector|{tag}", {})
        old_bE = old.get("beta_E_std")
        delta = f"{bs[2] - old_bE:+.2f}" if old_bE is not None else "n/a"

        print(f"\n{name}  ({n_cal} calibration pairs, {n_overlap} overlapping with the "
              f"factorial; {refits} leave-pair-out refits)")
        print(f"  raw   bV {b[1]:+.3f} [{lo[1]:+.3f},{hi[1]:+.3f}]   "
              f"bE {b[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}] p={pE:.4f}   "
              f"bVE {b[3]:+.3f} p={pVE:.4f}")
        print(f"  std   bV {bs[1]:+.2f}   bE {bs[2]:+.2f}   bVE {bs[3]:+.2f}"
              f"      (bE change vs leaky calibration: {delta})")

        results.append({
            "model": name, "model_tag": tag,
            "n_calibration_pairs": int(n_cal), "n_overlapping_pairs": int(n_overlap),
            "n_leave_pair_out_refits": int(refits),
            "beta_V": float(b[1]), "beta_E": float(b[2]), "beta_VE": float(b[3]),
            "beta_V_ci": [float(lo[1]), float(hi[1])],
            "beta_E_ci": [float(lo[2]), float(hi[2])],
            "beta_VE_ci": [float(lo[3]), float(hi[3])],
            "perm_p_beta_E": float(pE), "perm_p_beta_VE": float(pVE),
            "sd_outcome": float(sd),
            "beta_V_std": float(bs[1]), "beta_E_std": float(bs[2]),
            "beta_VE_std": float(bs[3]),
            "beta_E_std_leaky": old_bE,
        })

    print("\n" + "=" * 78)
    bE = [r["beta_E_std"] for r in results]
    bV = [r["beta_V_std"] for r in results]
    print(f"  bE (SD): {['%+.2f' % x for x in bE]}   "
          f"-> {'consistent sign' if min(bE) > 0 or max(bE) < 0 else 'SIGN VARIES'}")
    print(f"  bV (SD): {['%+.2f' % x for x in bV]}   "
          f"-> {'consistent sign' if min(bV) > 0 or max(bV) < 0 else 'SIGN VARIES'}")
    if all(r["beta_E_std_leaky"] is not None for r in results):
        d = max(abs(r["beta_E_std"] - r["beta_E_std_leaky"]) for r in results)
        print(f"  max |change| in standardized bE vs leaky calibration: {d:.2f} SD")

    out = os.path.join(DATA_DIR, "factorial_grouped_calibration.json")
    json.dump({"experiment": "factorial_grouped_calibration",
               "note": ("detector calibrated without the claim pair it is scored on; "
                        "OLS on [1,V,E,V*E]; not a logistic regression"),
               "uncertainty": f"claim-pair clustered bootstrap, B={B_BOOT}",
               "permutation": f"E shuffled within (pair, V) strata, B={B_PERM}",
               "results": results}, open(out, "w"), indent=2)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
