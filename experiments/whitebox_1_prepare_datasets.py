#!/usr/bin/env python3
"""
whitebox_1_prepare_datasets.py - Prepare equalized datasets for white-box probing

Extracts claim-response pairs from equalized evaluation results for representation-based
lie detection. Prepares data for Llama 3B/8B, Mistral 7B, Qwen 7B/14B.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/whitebox_1_prepare_datasets.py
"""

import json
import sys
from pathlib import Path

# Models to process (open-weight models we can extract representations from)
MODELS = {
    "llama3.2:3b": {
        "file": "data/results/ollama_eval_llama3_2_3b_prompt_equalized_latest.json",
        "hf_name": "meta-llama/Llama-3.2-3B",
        "short_name": "llama_3b"
    },
    "llama3.1:8b": {
        "file": "data/results/bedrock_eval_llama8b_prompt_equalized_latest.json",
        "hf_name": "meta-llama/Llama-3.1-8B",
        "short_name": "llama_8b"
    },
    "mistral:7b": {
        "file": "data/results/ollama_eval_mistral_7b_prompt_equalized_latest.json",
        "hf_name": "mistralai/Mistral-7B-v0.1",
        "short_name": "mistral_7b"
    },
    "qwen2.5:7b": {
        "file": "data/results/ollama_eval_qwen2_5_7b_prompt_equalized_latest.json",
        "hf_name": "Qwen/Qwen2.5-7B",
        "short_name": "qwen_7b"
    },
    "qwen2.5:14b": {
        "file": "data/results/ollama_eval_qwen2_5_14b_prompt_equalized_latest.json",
        "hf_name": "Qwen/Qwen2.5-14B",
        "short_name": "qwen_14b"
    }
}


def extract_first_response(conversation):
    """Extract the first assistant response from conversation."""
    for turn in conversation:
        if turn.get("role") == "assistant":
            return turn.get("content", "")
    return ""


def prepare_dataset(model_key, model_info):
    """Prepare dataset for a single model."""
    input_file = model_info["file"]

    if not Path(input_file).exists():
        print(f"[SKIP] {model_key}: File not found: {input_file}")
        return None

    with open(input_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"[SKIP] {model_key}: No results in file")
        return None

    # Extract claim-response pairs
    samples = []
    for r in results:
        claim = r.get("claim", "")
        ground_truth = r.get("ground_truth", "")
        conversation = r.get("conversation", [])

        if not claim or not ground_truth or not conversation:
            continue

        # Extract first response (for first-token representations)
        first_response = extract_first_response(conversation)
        if not first_response:
            continue

        # Label: 0 = truthful, 1 = lying
        label = 1 if ground_truth == "lying" else 0

        samples.append({
            "claim": claim,
            "first_response": first_response,
            "ground_truth": ground_truth,
            "label": label,
            # For probing, we'll use claim as input (model generates response from this)
            "input_text": claim
        })

    print(f"[{model_key}] Extracted {len(samples)} samples "
          f"({sum(1 for s in samples if s['label']==0)} truthful, "
          f"{sum(1 for s in samples if s['label']==1)} lying)")

    return {
        "model_key": model_key,
        "hf_name": model_info["hf_name"],
        "short_name": model_info["short_name"],
        "n_samples": len(samples),
        "samples": samples
    }


def main():
    print("="*70)
    print("WHITE-BOX PROBING: DATASET PREPARATION")
    print("="*70)
    print()

    output_dir = Path("data/whitebox_probing")
    output_dir.mkdir(exist_ok=True, parents=True)

    all_datasets = {}

    for model_key, model_info in MODELS.items():
        dataset = prepare_dataset(model_key, model_info)
        if dataset:
            # Save individual dataset
            output_file = output_dir / f"{model_info['short_name']}_equalized.json"
            with open(output_file, "w") as f:
                json.dump(dataset, f, indent=2)
            print(f"  ✓ Saved to {output_file}")
            all_datasets[model_key] = dataset
        print()

    # Save manifest
    manifest = {
        "models": {k: {
            "hf_name": v["hf_name"],
            "short_name": v["short_name"],
            "n_samples": v["n_samples"],
            "file": f"{v['short_name']}_equalized.json"
        } for k, v in all_datasets.items()}
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print("="*70)
    print(f"SUMMARY: Prepared {len(all_datasets)} datasets")
    print(f"Manifest: {manifest_file}")
    print("="*70)

    # Print behavioral baselines for reference
    print("\nBEHAVIORAL BASELINES (from paper):")
    print("  Llama 3B equalized:  52% (Mistral) / 64% (Claude)")
    print("  Llama 8B equalized:  61% (Mistral) / 71% (Claude)")
    print("  Mistral 7B equalized: 62% (Mistral) / 75% (Claude)")
    print("  Qwen 7B equalized:   57% (Mistral) / 66% (Qwen)")
    print("  Qwen 14B equalized:  52% (Mistral) / 69% (Claude)")
    print("\nNext step: Run whitebox_2_extract_representations.py")


if __name__ == "__main__":
    main()
