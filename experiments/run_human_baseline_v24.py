#!/usr/bin/env python3
"""
run_human_baseline_v24.py

Prepare n=50 human baseline study for V24 (Phase A).
Stratified sample across 7 equalized-condition target models (Haiku extractor),
50/50 TRUE/FALSE split, ~7 trials per model.

Outputs:
  data/human_baseline_v24/transcripts_n50.jsonl   — annotator-visible (no ground truth)
  data/human_baseline_v24/ground_truth_n50.json   — held out for analysis
  data/human_baseline_v24/annotator_A_template.csv — Annotator A input form
  data/human_baseline_v24/annotator_B_template.csv — Annotator B input form (same trials, re-ordered)

Usage:
    cd /Users/mediratta/code/AI-Researcher
    python code/adaptive_lie_detector/experiments/run_human_baseline_v24.py
"""

import json
import os
import random
import hashlib
import csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "data", "results")
OUTPUT_DIR = os.path.join(BASE, "data", "human_baseline_v24")

# Seven equalized-condition files (Haiku extractor, matching the 74.3% pipeline number)
# Use checkpoint / complete files (100 trials each) as source
TARGET_FILES = {
    "llama3b":  "ollama_checkpoint_llama3_2_3b_prompt_equalized.json",
    "llama8b":  "bedrock_checkpoint_llama8b_equalized.json",
    "mistral7b": "ollama_checkpoint_mistral_7b_prompt_equalized.json",
    "qwen7b":   "ollama_checkpoint_qwen2_5_7b_prompt_equalized.json",
    "qwen14b":  "ollama_checkpoint_qwen2_5_14b_prompt_equalized.json",
    "llama70b": "bedrock_checkpoint_llama70b_equalized_v2.json",
    "haiku":    "bedrock_checkpoint_haiku_equalized.json",
}

# Targets: 7 trials per model = 49 total; we take 3 TRUE + 4 FALSE from each
# to get 21 TRUE + 28 FALSE = 49.  Pad haiku to 4+4=8 (drop one random) to hit 50.
# Simpler: 3T+4F from each of 7 models = 49; add 1 extra from haiku = 50 (4T+4F for haiku).
TRIALS_PER_MODEL = {"truthful": 3, "lying": 4}
EXTRA_MODEL = "haiku"  # gets 4T + 4F = 8 (adding 1 extra truthful)

SEED = 42


def blind_id(model: str, trial_index: int) -> str:
    """Produce a stable opaque trial ID that doesn't reveal model identity."""
    raw = f"{model}_{trial_index}_{SEED}"
    return "T" + hashlib.sha256(raw.encode()).hexdigest()[:8].upper()


def format_conversation(conversation: list) -> dict:
    """Extract up to 5 assistant turns as turn_1..turn_5."""
    turns = {}
    turn_num = 0
    for msg in conversation:
        if msg.get("role") == "assistant":
            turn_num += 1
            turns[f"turn_{turn_num}_assistant"] = msg["content"].strip()
            if turn_num >= 5:
                break
    # Fill missing turns with empty string
    for i in range(1, 6):
        if f"turn_{i}_assistant" not in turns:
            turns[f"turn_{i}_assistant"] = ""
    return turns


def format_questions(conversation: list) -> list:
    """Extract interrogator questions (user turns after first)."""
    questions = []
    user_turns = [m["content"] for m in conversation if m.get("role") == "user"]
    return user_turns[1:]  # skip the initial claim presentation


def sample_from_file(filepath: str, model_key: str,
                     n_truthful: int, n_lying: int,
                     rng: random.Random) -> list:
    with open(filepath) as f:
        data = json.load(f)
    results = data.get("results", [])

    truthful = [r for r in results
                if r.get("ground_truth") == "truthful"
                and r.get("conversation")]
    lying = [r for r in results
             if r.get("ground_truth") == "lying"
             and r.get("conversation")]

    n_t = min(n_truthful, len(truthful))
    n_l = min(n_lying, len(lying))
    sampled_t = rng.sample(truthful, n_t)
    sampled_l = rng.sample(lying, n_l)

    trials = []
    for i, r in enumerate(sampled_t + sampled_l):
        turns = format_conversation(r["conversation"])
        trial_id = blind_id(model_key, i)
        trials.append({
            "trial_id": trial_id,
            "_model_internal": model_key,  # stripped from annotator output
            "_ground_truth_internal": r["ground_truth"],  # stripped
            "user_claim": r["claim"],
            "initial_prompt": r["conversation"][0]["content"] if r["conversation"] else "",
            **turns,
            "questions_asked": r.get("questions_asked", 0),
        })
    return trials


def write_annotator_csv(trials: list, filepath: str, annotator_label: str):
    """Write blank annotator input CSV — no ground truth columns."""
    fieldnames = [
        "trial_id",
        "user_claim",
        "turn_1_assistant",
        "turn_2_assistant",
        "turn_3_assistant",
        "turn_4_assistant",
        "turn_5_assistant",
        "annotator_label",      # fill: TRUTH or LIE
        "annotator_confidence_1to5",  # fill: 1-5
        "annotator_notes",
    ]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trials:
            row = {k: t.get(k, "") for k in fieldnames}
            row["annotator_label"] = ""
            row["annotator_confidence_1to5"] = ""
            row["annotator_notes"] = ""
            writer.writerow(row)
    print(f"  Annotator {annotator_label} template: {filepath}")


def main():
    rng = random.Random(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_trials = []
    print("Sampling transcripts:")
    for model_key, fname in TARGET_FILES.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        n_t = TRIALS_PER_MODEL["truthful"] + (1 if model_key == EXTRA_MODEL else 0)
        n_l = TRIALS_PER_MODEL["lying"]
        trials = sample_from_file(fpath, model_key, n_t, n_l, rng)
        truthful_count = sum(1 for t in trials if t["_ground_truth_internal"] == "truthful")
        lying_count = sum(1 for t in trials if t["_ground_truth_internal"] == "lying")
        print(f"  {model_key}: {len(trials)} trials ({truthful_count}T + {lying_count}L)")
        all_trials.extend(trials)

    print(f"\nTotal: {len(all_trials)} trials")
    truthful_total = sum(1 for t in all_trials if t["_ground_truth_internal"] == "truthful")
    lying_total = sum(1 for t in all_trials if t["_ground_truth_internal"] == "lying")
    print(f"  Truthful: {truthful_total}, Lying: {lying_total}")

    # Shuffle for annotator presentation (seed for reproducibility)
    rng_shuffle = random.Random(SEED + 1)
    annotator_a_order = list(range(len(all_trials)))
    rng_shuffle.shuffle(annotator_a_order)

    # Annotator B gets different order (to prevent sequential correlation)
    rng_shuffle_b = random.Random(SEED + 2)
    annotator_b_order = list(range(len(all_trials)))
    rng_shuffle_b.shuffle(annotator_b_order)

    trials_a = [all_trials[i] for i in annotator_a_order]
    trials_b = [all_trials[i] for i in annotator_b_order]

    # Write annotator-visible JSONL (no ground truth, no internal model key)
    jsonl_path = os.path.join(OUTPUT_DIR, "transcripts_n50.jsonl")
    with open(jsonl_path, "w") as f:
        for t in trials_a:
            row = {k: v for k, v in t.items()
                   if not k.startswith("_")}
            f.write(json.dumps(row) + "\n")
    print(f"\nAnnotator-visible JSONL: {jsonl_path}")

    # Write ground truth (held out)
    gt_path = os.path.join(OUTPUT_DIR, "ground_truth_n50.json")
    gt_data = {
        "n_samples": len(all_trials),
        "seed": SEED,
        "condition": "prompt_equalized_haiku_extractor",
        "samples": [
            {
                "trial_id": t["trial_id"],
                "model": t["_model_internal"],
                "ground_truth": t["_ground_truth_internal"],
                "user_claim": t["user_claim"],
            }
            for t in all_trials
        ]
    }
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"Ground truth (HELD OUT): {gt_path}")
    print("DO NOT share ground_truth_n50.json with annotators before rating is complete!")

    # Write annotator CSV templates
    csv_a = os.path.join(OUTPUT_DIR, "annotator_A_template.csv")
    csv_b = os.path.join(OUTPUT_DIR, "annotator_B_template.csv")
    write_annotator_csv(trials_a, csv_a, "A")
    write_annotator_csv(trials_b, csv_b, "B")

    print(f"\nNEXT STEPS:")
    print(f"1. Give annotator_A_template.csv to Annotator A")
    print(f"   Give annotator_B_template.csv to Annotator B")
    print(f"2. Each annotator fills in 'annotator_label' (TRUTH or LIE)")
    print(f"   and 'annotator_confidence_1to5' (1=guessing, 5=very sure)")
    print(f"   for all 50 rows. ~20 min per annotator.")
    print(f"3. After BOTH annotators submit, run:")
    print(f"   python code/adaptive_lie_detector/experiments/human_baseline_analysis_v24.py")
    print(f"   --annotator_a {csv_a}")
    print(f"   --annotator_b {csv_b}")
    print(f"   --ground_truth {gt_path}")
    print(f"\nDO NOT view {gt_path} before completing ratings!")


if __name__ == "__main__":
    main()
