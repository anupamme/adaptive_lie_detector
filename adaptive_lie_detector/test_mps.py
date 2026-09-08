#!/usr/bin/env python3
"""Test script to verify MPS support for Gemma-2-2B model."""

import sys
from src.target_model import TargetModel, get_best_device

print("=" * 80)
print("MPS SUPPORT TEST")
print("=" * 80)

# Test 1: Device detection
print("\n1. Testing device detection...")
device = get_best_device()
print(f"   ✅ Best device detected: {device}")

if device != "mps":
    print(f"\n⚠️  Warning: Expected MPS but got {device}")
    print("   MPS may not be available on this system")
    sys.exit(1)

# Test 2: Model loading
print("\n2. Loading Gemma-2-2B model on MPS...")
print("   This will download ~2GB if not cached")
print("   Please wait...\n")

try:
    model = TargetModel(
        model_name="google/gemma-2-2b-it",
        device="mps",
        quantization="none"  # Quantization disabled for MPS
    )
    print("\n   ✅ Model loaded successfully!")
except Exception as e:
    print(f"\n   ❌ Error loading model: {e}")
    sys.exit(1)

# Test 3: Simple inference
print("\n3. Testing inference...")
model.set_mode("truth")

try:
    response = model.respond(
        "What is 2+2?",
        max_new_tokens=50,
        temperature=0.7
    )
    print(f"   Question: What is 2+2?")
    print(f"   Response: {response}")
    print("\n   ✅ Inference working!")
except Exception as e:
    print(f"\n   ❌ Error during inference: {e}")
    sys.exit(1)

# Test 4: Memory usage
print("\n4. Checking memory usage...")
import torch
if device == "mps":
    # MPS doesn't expose memory stats the same way as CUDA
    print("   ℹ️  MPS memory usage: Check Activity Monitor > GPU History")
    print("   Expected: ~4-5GB for Gemma-2-2B")
else:
    print(f"   Device: {device}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
print("\nYour M4 Mac Mini is ready to run ML models with MPS acceleration!")
print("\nNext steps:")
print("  1. Generate training data:")
print("     python scripts/generate_training_data.py --n_samples 100 --questions 5")
print()
print("  2. Train classifier:")
print("     python examples/train_classifier_from_data.py --data data/training_data/dataset_*.json")
print()
print("  3. Run interrogation:")
print("     python scripts/run_interrogation.py --claim 'I visited Paris' --mode truth --verbose")
print()
