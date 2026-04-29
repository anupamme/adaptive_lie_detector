#!/usr/bin/env python3
"""
whitebox_2b_extract_response_representations.py - Extract representations from RESPONSES

Alternative approach: Extract representations from the model's FIRST RESPONSE to the claim,
not from the claim itself. This is more comparable to behavioral detection which uses
conversation transcripts.

Usage:
    cd /Users/mediratta/code/interpret/adaptive_lie_detector
    .venv/bin/python3 experiments/whitebox_2b_extract_response_representations.py --model mistral_7b
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
    "llama_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
    "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen_7b": "Qwen/Qwen2.5-7B",
    "qwen_14b": "Qwen/Qwen2.5-14B",
    "qwen_14b_instruct": "Qwen/Qwen2.5-14B-Instruct",
}


def extract_response_representations(model_name, dataset_file, output_file, device="mps"):
    """Extract final-layer representations from first assistant RESPONSES."""
    print(f"\n{'='*70}")
    print(f"EXTRACTING RESPONSE REPRESENTATIONS: {model_name}")
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

    # Extract representations from RESPONSES
    all_representations = []
    all_labels = []

    print(f"Extracting representations from responses...")
    with torch.no_grad():
        for sample in tqdm(samples, desc="Samples"):
            response = sample["first_response"]  # Use model's response, not claim
            label = sample["label"]

            # Tokenize response
            inputs = tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
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
        "representation_type": "first_response",
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
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    if not args.model:
        print("Error: Must specify --model")
        sys.exit(1)

    # Check device availability
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS not available, falling back to CPU")
        args.device = "cpu"

    model_name = MODELS[args.model]
    dataset_file = Path(f"data/whitebox_probing/{args.model}_equalized.json")
    output_file = Path(f"data/whitebox_probing/{args.model}_response_representations.json")

    if not dataset_file.exists():
        print(f"[ERROR] Dataset not found: {dataset_file}")
        sys.exit(1)

    try:
        extract_response_representations(model_name, dataset_file, output_file, device=args.device)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
