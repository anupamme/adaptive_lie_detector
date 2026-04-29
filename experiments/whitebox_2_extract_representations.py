#!/usr/bin/env python3
"""
whitebox_2_extract_representations.py - Extract hidden state representations

Extracts final-layer hidden states from models for white-box probing.
Uses MPS (Apple Silicon GPU) for acceleration.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/whitebox_2_extract_representations.py --model llama_8b

    # Or run all models:
    .venv/bin/python3 experiments/whitebox_2_extract_representations.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model mapping
MODELS = {
    "llama_3b": "meta-llama/Llama-3.2-3B",
    "llama_8b": "meta-llama/Llama-3.1-8B",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "qwen_7b": "Qwen/Qwen2.5-7B",
    "qwen_14b": "Qwen/Qwen2.5-14B"
}


def extract_representations(model_name, dataset_file, output_file, device="mps"):
    """Extract final-layer representations for all samples."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING REPRESENTATIONS: {model_name}")
    print(f"{'='*70}")

    # Load dataset
    with open(dataset_file) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"Loaded {len(samples)} samples")

    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else "cpu"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract representations
    all_representations = []
    all_labels = []

    print(f"Extracting representations...")
    with torch.no_grad():
        for sample in tqdm(samples, desc="Samples"):
            claim = sample["claim"]
            label = sample["label"]

            # Tokenize claim
            inputs = tokenizer(claim, return_tensors="pt", truncation=True, max_length=512)
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass with hidden states
            outputs = model(**inputs, output_hidden_states=True)

            # Extract final layer, last token representation
            final_layer = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            last_token_repr = final_layer[0, -1, :].cpu().numpy()  # (hidden_dim,)

            all_representations.append(last_token_repr)
            all_labels.append(label)

    # Convert to numpy arrays
    X = np.array(all_representations)  # (n_samples, hidden_dim)
    y = np.array(all_labels)  # (n_samples,)

    print(f"\nRepresentations shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"  Truthful: {np.sum(y==0)}")
    print(f"  Lying: {np.sum(y==1)}")

    # Save
    output_data = {
        "model_name": model_name,
        "n_samples": len(samples),
        "hidden_dim": X.shape[1],
        "representations": X.tolist(),
        "labels": y.tolist()
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    print(f"✓ Saved to {output_file}")
    print(f"  Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Model to extract from")
    parser.add_argument("--all", action="store_true", help="Extract from all models")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"],
                        help="Device to use (default: mps)")
    args = parser.parse_args()

    if not args.model and not args.all:
        print("Error: Must specify --model or --all")
        sys.exit(1)

    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Get models to process
    if args.all:
        models_to_process = list(MODELS.keys())
    else:
        models_to_process = [args.model]

    # Skip llama_3b (only 2 samples)
    if "llama_3b" in models_to_process:
        print("Note: Skipping llama_3b (insufficient samples: n=2)")
        models_to_process.remove("llama_3b")

    # Process each model
    for model_key in models_to_process:
        model_name = MODELS[model_key]
        dataset_file = Path(f"data/whitebox_probing/{model_key}_equalized.json")
        output_file = Path(f"data/whitebox_probing/{model_key}_representations.json")

        if not dataset_file.exists():
            print(f"[SKIP] {model_key}: Dataset not found: {dataset_file}")
            continue

        try:
            extract_representations(model_name, dataset_file, output_file, device=args.device)
        except Exception as e:
            print(f"[ERROR] {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print("Next step: Run whitebox_3_train_probes.py")


if __name__ == "__main__":
    main()
