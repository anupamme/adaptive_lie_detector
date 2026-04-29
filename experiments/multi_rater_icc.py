#!/usr/bin/env python3
"""
multi_rater_icc.py - Machine-rater ICC proxy for feature-extraction reliability.

Computes inter-rater agreement (ICC(2,1) + pairwise Pearson) across three LLM
raters on the 5 behavioral features (consistency / specificity / defensiveness /
confidence / elaboration) already extracted from the equalized n=100 transcripts:

  - Claude Haiku 4.5 (Anthropic; the original interrogator)
  - Llama 3.3 70B Instruct (Meta; Bedrock)
  - Mistral Large 3 (Mistral AI; Bedrock)

This is a machine-rater PROXY for the reviewer-requested human ICC study
(n>=100, 3+ annotators). It does NOT substitute for human ICC - three LLM
raters share systematic biases absent from human annotators. The purpose is
to quantify how much of the "feature extractor dependence" shown by the
cross-family panel is captured by a conventional reliability metric.

Reads existing cross-family extraction files on disk; no Bedrock calls.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/multi_rater_icc.py
"""
import json
import os
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "results"
OUT = DATA / "machine_rater_icc.json"

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

# Targets where all three rater files exist.
TRIPLE_TARGETS = {
    "claude_haiku": ("haiku", "Claude Haiku 4.5"),
    "llama8b": ("llama8b", "Llama 3.1 8B"),
    "qwen7b": ("qwen7b", "Qwen 2.5 7B"),
    "qwen14b": ("qwen14b", "Qwen 2.5 14B"),
    "qwen32b": ("qwen32b", "Qwen 2.5 32B"),
}


def load_rater_file(path):
    with open(path) as f:
        d = json.load(f)
    return d["results"]


def build_rating_matrix(target_tag):
    """Return (trial_ids, dict[rater_name -> (n_trials, 5) array])."""
    mistral_path = DATA / f"cross_family_equalized_{target_tag}_mistral_large.json"
    llama_path = DATA / f"cross_family_equalized_{target_tag}_llama70b_extractor.json"

    mistral_results = load_rater_file(mistral_path)
    llama_results = load_rater_file(llama_path)

    # Index by claim to pair trials
    mistral_by = {r["claim"]: r for r in mistral_results}
    llama_by = {r["claim"]: r for r in llama_results}
    shared_claims = [c for c in mistral_by if c in llama_by]

    rows = {"haiku": [], "mistral_large": [], "llama_70b": [], "_claims": []}
    for claim in shared_claims:
        m = mistral_by[claim]
        l = llama_by[claim]
        # Haiku features are identical across the two files (derived from original
        # trajectory); take from mistral file.
        haiku = m.get("claude_features") or {}
        mist = m.get("cross_family_features") or {}
        lla = l.get("cross_family_features") or {}
        if any(haiku.get(f) is None for f in FEATURES):
            continue
        if any(mist.get(f) is None for f in FEATURES):
            continue
        if any(lla.get(f) is None for f in FEATURES):
            continue
        rows["haiku"].append([haiku[f] for f in FEATURES])
        rows["mistral_large"].append([mist[f] for f in FEATURES])
        rows["llama_70b"].append([lla[f] for f in FEATURES])
        rows["_claims"].append(claim)

    mats = {k: np.array(v) for k, v in rows.items() if k != "_claims"}
    return rows["_claims"], mats


def icc_2_1(ratings):
    """
    ICC(2,1) - two-way random effects, single measurement, absolute agreement.
    Shrout & Fleiss (1979). Input: (n_subjects, n_raters) array of ratings on
    a single feature.
    """
    X = np.asarray(ratings, dtype=float)
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan")
    grand_mean = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)

    # Between-targets mean square (BMS)
    bms = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
    # Between-raters mean square (JMS)
    jms = n * np.sum((col_means - grand_mean) ** 2) / (k - 1)
    # Error (residual) mean square (EMS)
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_between_targets = k * np.sum((row_means - grand_mean) ** 2)
    ss_between_raters = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_between_targets - ss_between_raters
    ems = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")

    denom = bms + (k - 1) * ems + k * (jms - ems) / n
    if denom == 0:
        return float("nan")
    return float((bms - ems) / denom)


def icc_2_k(ratings):
    """ICC(2,k) - two-way random effects, average of k raters, absolute agreement."""
    X = np.asarray(ratings, dtype=float)
    n, k = X.shape
    grand_mean = X.mean()
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    bms = k * np.sum((row_means - grand_mean) ** 2) / (n - 1)
    ss_total = np.sum((X - grand_mean) ** 2)
    ss_bt = k * np.sum((row_means - grand_mean) ** 2)
    ss_br = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_bt - ss_br
    ems = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else float("nan")
    jms = n * np.sum((col_means - grand_mean) ** 2) / (k - 1)
    denom = bms + (jms - ems) / n
    if denom == 0:
        return float("nan")
    return float((bms - ems) / denom)


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_target(target_tag, target_label):
    claims, mats = build_rating_matrix(target_tag)
    n = len(claims)
    if n == 0:
        return None
    print(f"\n─ {target_label}  n={n}")
    out = {"label": target_label, "n": n, "per_feature": {}}
    # Per-feature ICC
    for fi, feat in enumerate(FEATURES):
        # Stack (n, 3) for this feature
        col = np.column_stack([mats["haiku"][:, fi],
                               mats["mistral_large"][:, fi],
                               mats["llama_70b"][:, fi]])
        icc1 = icc_2_1(col)
        iccK = icc_2_k(col)
        r_hm = pearson(mats["haiku"][:, fi], mats["mistral_large"][:, fi])
        r_hl = pearson(mats["haiku"][:, fi], mats["llama_70b"][:, fi])
        r_ml = pearson(mats["mistral_large"][:, fi], mats["llama_70b"][:, fi])
        out["per_feature"][feat] = {
            "icc_2_1": icc1, "icc_2_k": iccK,
            "r_haiku_vs_mistral": r_hm,
            "r_haiku_vs_llama70b": r_hl,
            "r_mistral_vs_llama70b": r_ml,
        }
        print(f"  {feat:<14}  ICC(2,1)={icc1:+.3f}  ICC(2,3)={iccK:+.3f}  "
              f"r: H-M={r_hm:+.2f} H-L={r_hl:+.2f} M-L={r_ml:+.2f}")
    return out


def pooled_icc(mats_by_target):
    """Stack trials across all targets for an overall machine-ICC."""
    pooled = {"haiku": [], "mistral_large": [], "llama_70b": []}
    for mats in mats_by_target:
        for k in pooled:
            pooled[k].append(mats[k])
    pooled = {k: np.vstack(v) for k, v in pooled.items()}
    return pooled


def main():
    print("=" * 70)
    print("MACHINE-RATER ICC PROXY (Haiku + Mistral Large + Llama 3.3 70B)")
    print("=" * 70)
    print("Pairs: (target_model_transcript) x (3 LLM raters) on 5 features")
    print(f"Targets with triple-rater coverage: {len(TRIPLE_TARGETS)}")
    print("NOTE: machine-rater ICC does NOT substitute for human ICC; LLM raters")
    print("share systematic biases that human raters do not.")

    per_target = {}
    mats_list = []
    for tag_key, (tag_data, label) in TRIPLE_TARGETS.items():
        try:
            result = analyze_target(tag_data, label)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {label}: {e}")
            continue
        if result is None:
            continue
        per_target[tag_key] = result
        _, mats = build_rating_matrix(tag_data)
        mats_list.append(mats)

    # Pooled across all targets
    pooled = pooled_icc(mats_list)
    print(f"\n─ POOLED ACROSS TARGETS  n_trials={len(pooled['haiku'])}")
    pooled_out = {}
    for fi, feat in enumerate(FEATURES):
        col = np.column_stack([pooled["haiku"][:, fi],
                               pooled["mistral_large"][:, fi],
                               pooled["llama_70b"][:, fi]])
        icc1 = icc_2_1(col)
        iccK = icc_2_k(col)
        r_hm = pearson(pooled["haiku"][:, fi], pooled["mistral_large"][:, fi])
        r_hl = pearson(pooled["haiku"][:, fi], pooled["llama_70b"][:, fi])
        r_ml = pearson(pooled["mistral_large"][:, fi], pooled["llama_70b"][:, fi])
        pooled_out[feat] = {
            "icc_2_1": icc1, "icc_2_k": iccK,
            "r_haiku_vs_mistral": r_hm,
            "r_haiku_vs_llama70b": r_hl,
            "r_mistral_vs_llama70b": r_ml,
        }
        print(f"  {feat:<14}  ICC(2,1)={icc1:+.3f}  ICC(2,3)={iccK:+.3f}  "
              f"r: H-M={r_hm:+.2f} H-L={r_hl:+.2f} M-L={r_ml:+.2f}")

    # Overall summary for easy citation
    mean_icc_2_1 = float(np.nanmean([pooled_out[f]["icc_2_1"] for f in FEATURES]))
    mean_icc_2_k = float(np.nanmean([pooled_out[f]["icc_2_k"] for f in FEATURES]))
    print(f"\n  Mean ICC(2,1) across 5 features (pooled): {mean_icc_2_1:.3f}")
    print(f"  Mean ICC(2,3) across 5 features (pooled): {mean_icc_2_k:.3f}")

    out = {
        "experiment": "machine_rater_icc_proxy",
        "raters": ["claude_haiku_4_5", "mistral_large_3", "llama_3_3_70b_instruct"],
        "features": FEATURES,
        "targets": list(TRIPLE_TARGETS.keys()),
        "n_triple_rater_targets": len(per_target),
        "per_target": per_target,
        "pooled": {
            "n_trials": int(len(pooled["haiku"])),
            "per_feature": pooled_out,
            "mean_icc_2_1": mean_icc_2_1,
            "mean_icc_2_k": mean_icc_2_k,
        },
        "caveat": (
            "Machine-rater ICC proxy: three LLM raters share systematic biases "
            "absent from human annotators. Does NOT substitute for the reviewer-"
            "requested human ICC at n>=100 with 3+ annotators."
        ),
    }
    os.makedirs(OUT.parent, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
