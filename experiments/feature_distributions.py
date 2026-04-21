#!/usr/bin/env python3
"""
feature_distributions.py

Compute per-feature mean ± std and Cohen's d for truth vs. lie conditions
using the pre-computed features in mock_dataset_500.json.
Also attempts to extract pilot feature statistics from stored interrogation files.

Usage:
    cd /Users/admin/code/interpret/adaptive_lie_detector
    python experiments/feature_distributions.py
"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import save_json, timestamp

FEATURES = ["consistency", "specificity", "defensiveness", "confidence", "elaboration"]

PILOT_CLAIM_SUBSTRINGS = [
    "100 degrees",          # truthful
    "Guido van Rossum",     # truthful
    "365 days",             # truthful
    "Shakespeare",          # truthful
    "300,000",              # truthful (speed of light)
    "Tokyo",                # truthful
    "Photosynthesis",       # truthful
    "206 bones",            # truthful
    "deoxyribonucleic",     # truthful
    "Great Wall",           # truthful
    "MIT",                  # lying
    "chess championship",   # lying
    "Mount Everest",        # lying
    "seven languages",      # lying
    "Nature on CRISPR",     # lying
    "President",            # lying
    "Google for five",      # lying
    "marathon",             # lying
    "fifty million",        # lying
    "Guinness",             # lying
]


def mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    variance = sum((v - m) ** 2 for v in values) / n
    return m, math.sqrt(variance)


def cohens_d(group1, group2):
    """Compute Cohen's d between two groups."""
    m1, s1 = mean_std(group1)
    m2, s2 = mean_std(group2)
    # pooled std
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-9
    return abs(m1 - m2) / pooled_std


def load_mock_features():
    """Load features from mock_dataset_500.json."""
    path = "data/training_data/mock_dataset_500.json"
    with open(path) as f:
        data = json.load(f)
    examples = data["examples"]

    truth_feats = {feat: [] for feat in FEATURES}
    lie_feats = {feat: [] for feat in FEATURES}

    for ex in examples:
        feats = ex.get("features")
        if feats is None:
            continue
        bucket = lie_feats if ex["is_lying"] else truth_feats
        for feat in FEATURES:
            v = feats.get(feat)
            if v is not None:
                bucket[feat].append(v)

    return truth_feats, lie_feats, len(examples)


def load_pilot_features():
    """
    Attempt to extract feature trajectories from saved interrogation files.
    Returns None if no matching files found.
    """
    results_dir = "data/results"
    if not os.path.isdir(results_dir):
        return None

    pilot_truth_feats = {feat: [] for feat in FEATURES}
    pilot_lie_feats = {feat: [] for feat in FEATURES}
    found = 0

    for fname in sorted(os.listdir(results_dir)):
        if not fname.startswith("interrogation_") or not fname.endswith(".json"):
            continue
        fpath = os.path.join(results_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        # Detect pilot results (have ground_truth and feature_trajectory)
        results_list = []
        if isinstance(data, list):
            results_list = data
        elif isinstance(data, dict) and "results" in data:
            results_list = data["results"]
        elif isinstance(data, dict) and "final_prediction" in data:
            results_list = [data]

        for rec in results_list:
            if not isinstance(rec, dict):
                continue
            ft = rec.get("feature_trajectory")
            if not ft:
                continue
            gt = rec.get("ground_truth")
            if gt not in ("lying", "truthful"):
                continue
            # Take last feature snapshot (most informative)
            last_feats = ft[-1] if ft else {}
            bucket = pilot_lie_feats if gt == "lying" else pilot_truth_feats
            for feat in FEATURES:
                v = last_feats.get(feat)
                if v is not None:
                    bucket[feat].append(v)
            found += 1

    if found == 0:
        return None
    return pilot_truth_feats, pilot_lie_feats, found


def print_table(truth_feats, lie_feats, label="Mock"):
    cohens_header = "Cohen's d"
    print(f"\n{'Feature':<16} {'Truth mean+/-std':>18} {'Lie mean+/-std':>16} {cohens_header:>10}")
    print("-" * 62)
    rows = []
    for feat in FEATURES:
        tm, ts = mean_std(truth_feats[feat])
        lm, ls = mean_std(lie_feats[feat])
        d = cohens_d(truth_feats[feat], lie_feats[feat])
        print(f"  {feat:<14} {tm:>6.2f} ± {ts:<5.2f}    {lm:>6.2f} ± {ls:<5.2f}    {d:>8.2f}")
        rows.append({"feature": feat,
                     "truth_mean": tm, "truth_std": ts,
                     "lie_mean": lm, "lie_std": ls,
                     "cohens_d": d})
    return rows


def main():
    print("Feature Score Distributions")
    print("=" * 62)

    truth_feats, lie_feats, n_total = load_mock_features()
    n_truth = len(truth_feats["consistency"])
    n_lie = len(lie_feats["consistency"])
    print(f"\nMock dataset: {n_total} examples ({n_truth} truthful, {n_lie} lying)")
    mock_rows = print_table(truth_feats, lie_feats, "Mock")

    pilot_rows = None
    pilot_data = load_pilot_features()
    if pilot_data:
        pt, pl, n_pilot = pilot_data
        pt_n = len(pt["consistency"])
        pl_n = len(pl["consistency"])
        print(f"\nPilot dataset: {n_pilot} records ({pt_n} truthful, {pl_n} lying)")
        pilot_rows = print_table(pt, pl, "Pilot")
    else:
        print("\nPilot feature trajectories: not found in saved interrogation files.")
        print("(Pilot summary JSON stores accuracy/confidence only, not per-feature scores.)")

    out = {
        "mock": {
            "n_total": n_total, "n_truthful": n_truth, "n_lying": n_lie,
            "features": mock_rows
        }
    }
    if pilot_rows:
        out["pilot"] = {"n_truthful": len(pilot_data[0]["consistency"]),
                        "n_lying": len(pilot_data[1]["consistency"]),
                        "features": pilot_rows}

    os.makedirs("data/results", exist_ok=True)
    save_json(out, f"data/results/feature_distributions_{timestamp()}.json")
    save_json(out, "data/results/feature_distributions_latest.json")
    print("\nSaved to data/results/feature_distributions_latest.json")
    return out


if __name__ == "__main__":
    main()
