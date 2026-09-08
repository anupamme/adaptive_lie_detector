#!/usr/bin/env python3
"""
analyze_r1_faithful.py

Fit R1's detector (logistic regression on binary elicitation-answer vectors)
and report accuracy per model per condition.

METHODOLOGICAL NOTE (important):
    Leave-one-out on a *perfectly balanced* dataset is systematically biased
    BELOW chance when the features carry little or no information. Holding out
    one example leaves the training set with a one-example majority for the
    OPPOSITE class; with uninformative features the classifier follows that
    majority and is therefore wrong on every fold. Empirically, constant
    features give LOO = 0.0% where the true accuracy is chance (50%).

    We therefore report stratified 5-fold as the PRIMARY estimate (unbiased at
    chance) and LOO alongside it for comparability with the rest of the paper.
    Cells whose binary vectors are (near-)degenerate are flagged: for those the
    detector has no signal to use at all, which is a stronger negative result
    than a near-chance accuracy, but must not be reported as "0% accuracy".

Usage:
    cd code/adaptive_lie_detector
    python experiments/analyze_r1_faithful.py
    # EXP-R1b replication (fresh claims + fresh probe bank):
    python experiments/analyze_r1_faithful.py --variant v2 --per_dim
"""

import argparse
import sys
import os
import glob
import json
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    LeaveOneOut, StratifiedKFold, GroupKFold, cross_val_score,
)
from experiments.claims_equalized import EQUALIZED_CLAIMS  # noqa: E402
from experiments.claims_equalized_v2 import EQUALIZED_CLAIMS_V2  # noqa: E402

RESULTS_DIR = "data/results"
N_PERM = 200
# One RNG shared across cells means permutation p-values depend on the order cells
# are analyzed in; at N_PERM=200 the resulting jitter is ~1 grid step (0.005). Each
# cell is therefore reseeded from a stable key (see perm_p) so a cell's p-value
# depends only on its own data and N_PERM.
RNG = np.random.default_rng(0)

# Checkpoint prefix + claim set per variant. Prefixes must not be prefixes of one
# another: checkpoints are keyed on (model, condition), so a v2 file matched by
# the v1 glob would silently overwrite the v1 row for the same cell.
VARIANTS = {
    "v1": {"prefix": "r1_faithful", "claims": EQUALIZED_CLAIMS},
    "v2": {"prefix": "r1b_fresh", "claims": EQUALIZED_CLAIMS_V2},
}


def claim_to_pair(claims):
    return {c: i for i, (tc, fc) in enumerate(claims) for c in (tc, fc)}


def load_checkpoints(variant="v1"):
    spec = VARIANTS[variant]
    c2p = claim_to_pair(spec["claims"])
    data = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{spec['prefix']}_*.json"))):
        if "summary" in os.path.basename(path):
            continue
        with open(path) as f:
            blob = json.load(f)
        recs = blob.get("records", [])
        if not recs:
            continue
        X = np.array([r["vector"] for r in recs], dtype=float)
        y = np.array([r["label"] for r in recs], dtype=int)
        amb = float(np.mean([np.mean(r["ambiguous"]) for r in recs]))
        unmapped = sum(1 for r in recs if r.get("claim", "") not in c2p)
        if unmapped:
            print(f"  WARNING: {os.path.basename(path)}: {unmapped}/{len(recs)} claims "
                  f"not in the {variant} claim set -- grouped k-fold will not pair them.")
        groups = np.array([c2p.get(r.get("claim", ""), 10_000 + idx)
                           for idx, r in enumerate(recs)])
        data[(blob["model"], blob["condition"])] = {
            "X": X, "y": y, "amb": amb, "n": len(recs), "groups": groups,
            "questions": blob.get("questions", []),
        }
    return data


def per_dimension_report(X, y, groups, questions):
    """Single-probe decomposition: which dimension carries the surviving signal?

    Pre-registered prediction for the EXP-R1b qwen2.5:14b equalized cell is that
    dimension 0 (the confidence probe) carries it, as in the v1 post-hoc account.
    Reported for all 16 dimensions so the prediction is falsifiable rather than
    re-selected: a dim-0 win only counts if it is the top dimension here.
    """
    rows = []
    full = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
    for j in range(X.shape[1]):
        xj = X[:, j:j + 1]
        acc = grouped_kfold_accuracy(xj, y, groups) if xj.var() > 1e-9 else float("nan")
        # P(yes | lying) - P(yes | truthful): direction and size of the raw split.
        d = float(X[y == 1, j].mean() - X[y == 0, j].mean()) if len(set(y)) == 2 else float("nan")
        rows.append({"dim": j, "acc": acc, "delta": d,
                     "coef": float(full.coef_[0][j]), "var": float(xj.var()),
                     "question": questions[j] if j < len(questions) else ""})
    return rows


def loo_accuracy(X, y):
    if len(np.unique(y)) < 2 or len(y) < 4:
        return float("nan")
    correct = total = 0
    for tr, te in LeaveOneOut().split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        correct += int(clf.predict(X[te])[0] == y[te][0])
        total += 1
    return correct / total if total else float("nan")


def kfold_accuracy(X, y, k=5):
    """Stratified k-fold — unbiased at chance for uninformative features."""
    if len(np.unique(y)) < 2 or len(y) < 2 * k:
        return float("nan")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    return float(cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0), X, y, cv=cv).mean())


def grouped_kfold_accuracy(X, y, groups, k=5):
    """Grouped k-fold — keeps matched claim pairs in the same fold."""
    if len(np.unique(y)) < 2 or len(y) < 2 * k:
        return float("nan")
    cv = GroupKFold(n_splits=k)
    return float(cross_val_score(
        LogisticRegression(max_iter=1000, C=1.0), X, y, cv=cv, groups=groups).mean())


def perm_p(X, y, observed, k=5, groups=None, n_perm=None, seed=0):
    """Permutation test: P(shuffled-label accuracy >= observed).

    `seed` is per-cell so the estimate does not depend on how many cells were
    analyzed before this one.
    """
    if observed != observed:
        return float("nan")
    n_perm = N_PERM if n_perm is None else n_perm
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if groups is not None:
            a = grouped_kfold_accuracy(X, yp, groups, k)
        else:
            a = kfold_accuracy(X, yp, k)
        if a == a and a >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def degeneracy(X):
    """(n_varying_features, n_unique_rows)."""
    return int((X.var(axis=0) > 1e-9).sum()), len(set(map(tuple, X)))


def main():
    ap = argparse.ArgumentParser(description="EXP-R1 / EXP-R1b detector analysis")
    ap.add_argument("--variant", type=str, default="v1", choices=sorted(VARIANTS),
                    help="v1 = original claims+probes; v2 = fresh claims+probes (EXP-R1b)")
    ap.add_argument("--per_dim", action="store_true",
                    help="Print the single-probe decomposition per cell")
    ap.add_argument("--n_perm", type=int, default=N_PERM,
                    help="Permutation draws per cell (higher = finer p-value grid)")
    args = ap.parse_args()

    data = load_checkpoints(args.variant)
    if not data:
        print(f"No {VARIANTS[args.variant]['prefix']}_*.json checkpoints found in", RESULTS_DIR)
        return

    models = sorted({m for (m, _) in data})

    print("=" * 100)
    print(f"EXP-R1 [{args.variant}]: faithful R1 detector "
          "(fixed unrelated yes/no probes -> binary vector -> LR)")
    print("PRIMARY = stratified 5-fold (unbiased at chance).  Grouped-5-fold and LOO for comparability.")
    print("=" * 100)
    print(f"perm p: {args.n_perm} label permutations per cell, tested against the "
          "5-fold estimate; p_grp against the grouped estimate.")
    hdr = (f"{'Model':<15}{'cond':<11}{'5-fold':>8}{'grp-5f':>8}{'LOO':>8}{'perm p':>9}"
           f"{'p_grp':>8}{'n':>5}{'amb':>6}{'varying':>9}{'uniq':>6}  flag")
    print(hdr)
    print("-" * 100)

    rows = {}
    for m in models:
        for cond in ("instructed", "equalized"):
            d = data.get((m, cond))
            if not d:
                continue
            X, y, groups = d["X"], d["y"], d["groups"]
            nv, nu = degeneracy(X)
            kf = kfold_accuracy(X, y)
            gkf = grouped_kfold_accuracy(X, y, groups)
            lo = loo_accuracy(X, y)
            # p is tested against the PRIMARY metric (stratified 5-fold), so that the
            # p in a row refers to the accuracy in the same row. Testing the grouped
            # estimate against a grouped null is pathological on degenerate cells:
            # GroupKFold's fold structure is deterministic, so with constant features
            # it returns p ~ 0.007 for qwen2.5:7b equalized, where all 50 vectors are
            # identical and the true answer is p = 1.0.
            # crc32, not hash(): str hashing is salted per process (PYTHONHASHSEED).
            seed = zlib.crc32(f"{m}|{cond}".encode())
            p = perm_p(X, y, kf, n_perm=args.n_perm, seed=seed)
            p_grouped = perm_p(X, y, gkf, groups=groups, n_perm=args.n_perm, seed=seed)
            flag = ""
            if nv == 0:
                flag = "DEGENERATE (all vectors identical)"
            elif nv <= 2 or nu <= 3:
                flag = "near-degenerate"
            rows[(m, cond)] = dict(kfold=kf, grouped_kfold=gkf, loo=lo, p=p,
                                   p_grouped=p_grouped, n=d["n"],
                                   amb=d["amb"], varying=nv, uniq=nu, flag=flag)
            f = lambda v: f"{v:.1%}" if v == v else "--"  # noqa: E731
            ps = f"{p:.3f}" if p == p else "--"
            pgs = f"{p_grouped:.3f}" if p_grouped == p_grouped else "--"
            print(f"{m:<15}{cond:<11}{f(kf):>8}{f(gkf):>8}{f(lo):>8}{ps:>9}{pgs:>8}"
                  f"{d['n']:>5}{d['amb']:>5.0%}{nv:>7}/16{nu:>6}  {flag}")
        print("-" * 100)

    print(f"{'MODEL':<15}{'5-fold instruct':>17}{'grp-5f instruct':>17}"
          f"{'5-fold equal':>14}{'grp-5f equal':>14}{'change (pp)':>14}")
    print("-" * 100)
    deltas = []
    for m in models:
        i = rows.get((m, "instructed"))
        e = rows.get((m, "equalized"))
        if not (i and e):
            continue
        di = (i["kfold"] - e["kfold"]) * 100
        deltas.append((m, i["kfold"], e["kfold"], di))
        note = ""
        if i["flag"] or e["flag"]:
            note = "  <- degenerate probe channel"
        print(f"{m:<15}{i['kfold']:>16.1%}{i['grouped_kfold']:>17.1%}"
              f"{e['kfold']:>14.1%}{e['grouped_kfold']:>14.1%}{di:>13.1f}{note}")

    print("-" * 100)
    for cond in ("instructed", "equalized"):
        Xs = [d["X"] for (mm, c), d in data.items() if c == cond]
        ys = [d["y"] for (mm, c), d in data.items() if c == cond]
        gs = [d["groups"] for (mm, c), d in data.items() if c == cond]
        if Xs:
            X = np.vstack(Xs)
            y = np.concatenate(ys)
            g = np.concatenate(gs)
            print(f"{'POOLED ' + cond:<15} 5-fold={kfold_accuracy(X, y):.1%}   "
                  f"grouped-5f={grouped_kfold_accuracy(X, y, g):.1%}   "
                  f"LOO={loo_accuracy(X, y):.1%}   n={len(y)}")

    per_dim = {}
    if args.per_dim:
        for (m, cond), d in sorted(data.items()):
            # NOTE: do not name this `rows` -- that shadows the per-cell results dict
            # built above, which the summary block below still needs.
            dim_rows = per_dimension_report(d["X"], d["y"], d["groups"], d["questions"])
            per_dim[f"{m}|{cond}"] = dim_rows
            print("-" * 100)
            print(f"PER-DIMENSION — {m} / {cond}  (grouped-5-fold on one probe at a time)")
            print(f"  {'dim':>4}{'grp-5f':>9}{'P(yes|lie)-P(yes|true)':>24}{'LR coef':>10}  question")
            for r in sorted(dim_rows, key=lambda r: (-(r['acc'] if r['acc'] == r['acc'] else 0))):
                a = f"{r['acc']:.1%}" if r["acc"] == r["acc"] else "const"
                star = "  <- dim 0 (confidence probe)" if r["dim"] == 0 else ""
                print(f"  {r['dim']:>4}{a:>9}{r['delta']:>+24.2f}{r['coef']:>+10.2f}  "
                      f"{r['question'][:44]}{star}")

    summary = {
        "variant": args.variant,
        "claim_set": "claims_equalized" if args.variant == "v1" else "claims_equalized_v2",
        "probe_bank": ("r1_elicitation_questions" if args.variant == "v1"
                       else "r1_elicitation_questions_v2"),
        "primary_metric": "stratified 5-fold (LOO is biased below chance on balanced "
                          "data with uninformative features)",
        "grouped_kfold_note": "grouped-5-fold keeps matched true/false claim pairs in same fold",
        "per_model": [
            {"model": m, "instructed_kfold": a, "equalized_kfold": b, "delta_pp": d,
             "instructed_grouped_kfold": rows[(m, "instructed")]["grouped_kfold"],
             "equalized_grouped_kfold": rows[(m, "equalized")]["grouped_kfold"],
             "instructed_loo": rows[(m, "instructed")]["loo"],
             "equalized_loo": rows[(m, "equalized")]["loo"],
             "instructed_flag": rows[(m, "instructed")]["flag"],
             "equalized_flag": rows[(m, "equalized")]["flag"]}
            for (m, a, b, d) in deltas
        ],
        "per_dimension": per_dim,
    }
    suffix = "" if args.variant == "v1" else f"_{args.variant}"
    out = os.path.join(RESULTS_DIR, f"r1_faithful{suffix}_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("-" * 100)
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()
