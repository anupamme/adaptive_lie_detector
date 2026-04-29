#!/usr/bin/env python3
"""analyze_equalized_llama70b_panel.py

Second cross-family extractor panel (Llama 3.3 70B as extractor).

Reads cross_family_equalized_{target}_llama70b_extractor.json files produced
by re_extract_equalized_cross_family.py --extractor llama70b.

Computes per-target LOO for the Llama-70B-extracted 5-feature pipeline on the
same equalized trials. Outputs per-target rows + panel average for Table 9.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "data" / "results"
sys.path.insert(0, str(BASE))

FEATS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]


def label_to_int(gt):
    if gt == "truthful":
        return 0
    if gt == "lying":
        return 1
    return None


# (display_label, llama70b_extractor_JSON)
TARGETS = [
    ("Llama 3.2 3B",     "cross_family_equalized_llama3_2_3b_llama70b_extractor.json"),
    ("Llama 3.1 8B",     "cross_family_equalized_llama8b_llama70b_extractor.json"),
    ("Mistral 7B",       "cross_family_equalized_mistral_7b_llama70b_extractor.json"),
    ("Qwen 2.5 7B",      "cross_family_equalized_qwen7b_llama70b_extractor.json"),
    ("Qwen 2.5 14B",     "cross_family_equalized_qwen14b_llama70b_extractor.json"),
    ("Llama 3.3 70B",    "cross_family_equalized_llama70b_llama70b_extractor.json"),
    ("Claude Haiku 4.5", "cross_family_equalized_haiku_llama70b_extractor.json"),
    ("Qwen 2.5 32B",     "cross_family_equalized_qwen32b_llama70b_extractor.json"),
]


def pipeline_loo(feature_dicts, labels):
    X, y = [], []
    for f, lab in zip(feature_dicts, labels):
        if f is None or lab is None:
            continue
        vec = []
        bad = False
        for name in FEATS:
            v = f.get(name)
            if v is None:
                bad = True
                break
            vec.append(float(v))
        if bad:
            continue
        X.append(vec)
        y.append(lab)
    if len(X) < 4:
        return float("nan"), 0
    X = np.asarray(X)
    y = np.asarray(y)
    correct = 0
    for i in range(len(X)):
        Xtr = np.delete(X, i, axis=0)
        ytr = np.delete(y, i)
        Xte = X[i:i + 1]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        if clf.predict(sc.transform(Xte))[0] == y[i]:
            correct += 1
    return correct / len(X), len(X)


def analyze_target(label, json_name):
    path = RESULTS / json_name
    if not path.exists():
        ckpt = RESULTS / json_name.replace(".json", "_checkpoint.json")
        if ckpt.exists():
            data = json.load(open(ckpt))
            if isinstance(data, list):
                entries = data
            else:
                entries = data.get("results", [])
            status = "partial"
        else:
            return {"label": label, "status": "missing", "n": 0}
    else:
        data = json.load(open(path))
        if isinstance(data, list):
            entries = data
        else:
            entries = data.get("results", [])
        status = "ok"

    feats = [e.get("cross_family_features") for e in entries]
    labs = [label_to_int(e.get("ground_truth")) for e in entries]
    pipe, n = pipeline_loo(feats, labs)
    return {
        "label": label,
        "status": status,
        "n_entries": len(entries),
        "n_scored": n,
        "llama70b_pipeline_loo": float(pipe),
    }


def main():
    rows = []
    for lab, j in TARGETS:
        r = analyze_target(lab, j)
        rows.append(r)
        print(f"\n=== {lab} [{r.get('status')}] ===")
        if "llama70b_pipeline_loo" in r and not np.isnan(r["llama70b_pipeline_loo"]):
            print(f"  n_entries={r['n_entries']}  n_scored={r['n_scored']}")
            print(f"  Llama-70B-extracted pipeline LOO: {r['llama70b_pipeline_loo']*100:.1f}%")
        else:
            print(f"  (no data: n_entries={r.get('n_entries', 0)})")

    oks = [r for r in rows if r.get("status") in ("ok", "partial") and not np.isnan(r.get("llama70b_pipeline_loo", float("nan")))]
    if oks:
        vals = [r["llama70b_pipeline_loo"] for r in oks]
        avg_8 = float(np.mean(vals)) if len(vals) == 8 else float("nan")
        vals_ex_qwen32 = [r["llama70b_pipeline_loo"] for r in oks if r["label"] != "Qwen 2.5 32B"]
        avg_7 = float(np.mean(vals_ex_qwen32)) if len(vals_ex_qwen32) >= 1 else float("nan")
        print("\n" + "=" * 60)
        print(f"Llama 70B extractor: n_targets_scored = {len(oks)}")
        if len(oks) == 8:
            print(f"  8-target avg: {avg_8*100:.1f}%")
        if vals_ex_qwen32:
            print(f"  7-target avg (ex. Qwen 32B): {avg_7*100:.1f}%  (n={len(vals_ex_qwen32)})")

    out_path = BASE.parent.parent / "output" / "adaptive_lie_detector_paper" / "llama70b_extractor_panel.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"targets": rows}, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
