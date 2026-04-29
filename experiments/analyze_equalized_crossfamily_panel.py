#!/usr/bin/env python3
"""analyze_equalized_crossfamily_panel.py

7-target cross-family panel. For each equalized target (Llama 3B, Llama 8B,
Mistral 7B, Qwen 7B, Qwen 14B, Llama 70B, Haiku) compute:

  - Haiku-extracted refusal-count LOO (extractor-independent regex on conv)
  - Haiku-extracted 5-feature LOO
  - Mistral-Large-extracted 5-feature LOO (cross-family)

Refusal count is extractor-independent (regex over assistant turns) so the
single regex column applies to both extractors.

Outputs a JSON summary at
output/adaptive_lie_detector_paper/crossfamily_panel.json
and prints a human-readable table to stdout.
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

from experiments.analyze_sonnet_and_adversarial import (
    count_refusal_markers, assistant_only_transcript, label_to_int,
    compute_loo_accuracy, threshold_rule_accuracy,
)

FEATS = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]


# Each target: (label, source equalized JSON, cross-family re-extract JSON,
# cross-family features field name).
TARGETS = [
    ("Llama 3.2 3B",
     "ollama_eval_llama3_2_3b_prompt_equalized_20260422_102229.json",
     "cross_family_equalized_mistral_large.json",
     "datasets.llama3_2_3b.results", "mistral_large_features"),
    ("Llama 3.1 8B",
     "bedrock_eval_llama8b_prompt_equalized_latest.json",
     "cross_family_equalized_llama8b_mistral_large.json",
     "results", "cross_family_features"),
    ("Mistral 7B",
     "ollama_eval_mistral_7b_prompt_equalized_latest.json",
     "cross_family_equalized_mistral_large.json",
     "datasets.mistral_7b.results", "mistral_large_features"),
    ("Qwen 2.5 7B",
     "ollama_eval_qwen2_5_7b_prompt_equalized_latest.json",
     "cross_family_equalized_qwen7b_mistral_large.json",
     "results", "cross_family_features"),
    ("Qwen 2.5 14B",
     "ollama_eval_qwen2_5_14b_prompt_equalized_latest.json",
     "cross_family_equalized_qwen14b_mistral_large.json",
     "results", "cross_family_features"),
    ("Llama 3.3 70B",
     "bedrock_eval_llama70b_prompt_equalized_latest.json",
     "cross_family_equalized_mistral_large.json",
     "datasets.llama_70b.results", "mistral_large_features"),
    ("Claude Haiku 4.5",
     "bedrock_eval_haiku_prompt_equalized_latest.json",
     "cross_family_equalized_haiku_mistral_large.json",
     "results", "cross_family_features"),
    ("Qwen 2.5 32B",
     "ollama_eval_qwen2_5_32b_prompt_equalized_latest.json",
     "cross_family_equalized_qwen32b_mistral_large.json",
     "results", "cross_family_features"),
]


def dig(obj, path):
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            raise ValueError(f"Can't dig into {type(obj)} with {part}")
    return obj


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


def load_source_trials(path):
    data = json.load(open(path))
    return [t for t in data.get("results", [])
            if t.get("status") != "error"
            and t.get("ground_truth") in ("truthful", "lying")
            and t.get("conversation")]


def haiku_features_and_regex(trials):
    """Haiku-extracted mean features (from feature_trajectory) and
    refusal-count from assistant transcript."""
    feats, regex_counts, labels = [], [], []
    for t in trials:
        ft = [s for s in t.get("feature_trajectory", []) if s is not None]
        if ft:
            f = {}
            ok = True
            for name in FEATS:
                vals = [s.get(name) for s in ft if s.get(name) is not None]
                if not vals:
                    ok = False
                    break
                f[name] = float(np.mean(vals))
            feats.append(f if ok else None)
        else:
            feats.append(None)
        regex_counts.append(count_refusal_markers(assistant_only_transcript(t)))
        labels.append(label_to_int(t.get("ground_truth")))
    return feats, regex_counts, labels


def cross_family_features(cx_path, cx_subpath, feat_field):
    data = json.load(open(cx_path))
    try:
        results = dig(data, cx_subpath)
    except (KeyError, ValueError):
        return None
    by_claim = {}
    for r in results:
        c = r.get("claim")
        if c:
            by_claim[c] = r.get(feat_field)
    return by_claim


def analyze_target(label, src, cx_path, cx_subpath, feat_field):
    src_path = RESULTS / src
    cx_full = RESULTS / cx_path
    if not src_path.exists():
        return {"label": label, "status": "missing_source"}
    if not cx_full.exists():
        return {"label": label, "status": "missing_crossfamily"}

    trials = load_source_trials(src_path)
    haiku_feats, regex_counts, labels = haiku_features_and_regex(trials)

    # Regex (extractor-independent)
    rc_loo, thr = compute_loo_accuracy(regex_counts, labels)
    rc_k1 = threshold_rule_accuracy(regex_counts, labels, 1)

    # Haiku pipeline
    haiku_pipe, haiku_n = pipeline_loo(haiku_feats, labels)

    # Cross-family: join by claim
    cx_map = cross_family_features(cx_full, cx_subpath, feat_field)
    if cx_map is None:
        return {"label": label, "status": "crossfamily_shape_err"}

    cx_feats, cx_labels = [], []
    matched = 0
    for t, lab in zip(trials, labels):
        claim = t.get("claim")
        if claim in cx_map and cx_map[claim] is not None:
            cx_feats.append(cx_map[claim])
            cx_labels.append(lab)
            matched += 1
        else:
            cx_feats.append(None)
            cx_labels.append(lab)
    cx_pipe, cx_n = pipeline_loo(cx_feats, cx_labels)

    return {
        "label": label,
        "status": "ok",
        "n_trials": len(trials),
        "n_haiku": haiku_n,
        "n_crossfamily": cx_n,
        "refusal_count_loo": float(rc_loo),
        "refusal_count_thr": int(thr),
        "refusal_count_k1": float(rc_k1),
        "haiku_pipeline_loo": float(haiku_pipe),
        "crossfamily_pipeline_loo": float(cx_pipe),
        "gap_haiku_minus_crossfamily_pp": float((haiku_pipe - cx_pipe) * 100) if not np.isnan(haiku_pipe) and not np.isnan(cx_pipe) else float("nan"),
    }


def main():
    rows = []
    for (label, src, cxp, sub, fld) in TARGETS:
        r = analyze_target(label, src, cxp, sub, fld)
        rows.append(r)
        print(f"\n=== {label} [{r.get('status')}] ===")
        if r.get("status") != "ok":
            continue
        print(f"  n={r['n_trials']}  (haiku pipeline n={r['n_haiku']}; crossfamily n={r['n_crossfamily']})")
        print(f"  refusal-count LOO:      {r['refusal_count_loo']*100:.1f}%  (thr={r['refusal_count_thr']}, k>=1={r['refusal_count_k1']*100:.1f}%)")
        print(f"  Haiku pipeline LOO:     {r['haiku_pipeline_loo']*100:.1f}%")
        print(f"  Cross-family LOO:       {r['crossfamily_pipeline_loo']*100:.1f}%")
        print(f"  Gap (Haiku - CF) pp:    {r['gap_haiku_minus_crossfamily_pp']:+.1f}")

    # 8-target averages for co-equal headline
    oks = [r for r in rows if r.get("status") == "ok"]
    def avg(k):
        vals = [r[k] for r in oks if not np.isnan(r[k])]
        return float(np.mean(vals)) if vals else float("nan")

    rc_avg = avg("refusal_count_k1")
    haiku_avg = avg("haiku_pipeline_loo")
    cx_avg = avg("crossfamily_pipeline_loo")

    print("\n" + "=" * 60)
    print("8-target averages:")
    print(f"  Refusal-count k>=1 (extractor-independent): {rc_avg*100:.1f}%")
    print(f"  Haiku pipeline LOO:                         {haiku_avg*100:.1f}%")
    print(f"  Cross-family (Mistral Large) pipeline LOO:  {cx_avg*100:.1f}%")
    print(f"  Avg Haiku-minus-crossfamily gap:            {(haiku_avg - cx_avg)*100:+.1f} pp")

    out = {
        "targets": rows,
        "n_targets": len(oks),
        "averages": {
            "refusal_count_k1": rc_avg,
            "haiku_pipeline_loo": haiku_avg,
            "crossfamily_pipeline_loo": cx_avg,
            "gap_pp": (haiku_avg - cx_avg) * 100 if not np.isnan(haiku_avg) and not np.isnan(cx_avg) else float("nan"),
        },
    }
    out_path = BASE.parent.parent / "output" / "adaptive_lie_detector_paper" / "crossfamily_panel.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
