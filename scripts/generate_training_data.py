#!/usr/bin/env python3
"""
Script to generate training data for lie detector.

This script coordinates the target model, interrogator, and feature extractor
to create labeled training conversations.

Usage:
    # Generate with mock models (no GPU/API required)
    python scripts/generate_training_data.py --n_samples 50 --mock

    # Generate with real models (requires GPU and API keys)
    python scripts/generate_training_data.py --n_samples 100 --questions 10

    # Specify output file
    python scripts/generate_training_data.py --n_samples 50 --output my_data.json
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.data_generator import (
    TrainingDataGenerator,
    MockTargetModel,
    MockInterrogator,
    MockFeatureExtractor
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for lie detector"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of training examples to generate (default: 100)"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=5,
        help="Number of questions per conversation (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filepath (default: data/training_data/dataset_TIMESTAMP.json)"
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="Fraction of lying examples (default: 0.5 for balanced dataset)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock models instead of real models (for testing without GPU/API)"
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="data/topics.json",
        help="Path to topics configuration file (default: data/topics.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for model (default: auto-detect)"
    )
    parser.add_argument(
        "--cpu-generation",
        action="store_true",
        help="Force CPU generation (workaround for MPS bug, automatically enabled for MPS)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LIE DETECTOR TRAINING DATA GENERATOR")
    print("=" * 80)

    # Initialize models
    if args.mock:
        print("\n🎭 Using MOCK models (no GPU/API required)")
        print("   Note: This generates synthetic data for testing purposes\n")

        target = MockTargetModel()
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

    else:
        print("\n🚀 Using REAL models (requires GPU and API keys)")
        print("   Loading models... this may take a few minutes\n")

        try:
            from src.target_model import TargetModel
            from src.interrogator import LLMInterrogator
            from src.feature_extractor import LLMFeatureExtractor
            from config import INTERROGATOR_MODEL

            target = TargetModel(device=args.device, force_cpu_generation=args.cpu_generation)
            interrogator = LLMInterrogator(model=INTERROGATOR_MODEL)
            extractor = LLMFeatureExtractor(model=INTERROGATOR_MODEL)

        except Exception as e:
            print(f"❌ Error loading real models: {e}")
            print("\nTry using --mock flag for testing:")
            print("  python scripts/generate_training_data.py --n_samples 50 --mock")
            sys.exit(1)

    # Initialize generator
    print("Initializing generator...")
    try:
        generator = TrainingDataGenerator(
            target_model=target,
            interrogator=interrogator,
            feature_extractor=extractor,
            topics_path=args.topics
        )
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        sys.exit(1)

    # Generate dataset
    print(f"\n{'=' * 80}")
    print("GENERATION PARAMETERS:")
    print(f"{'=' * 80}")
    print(f"  Total samples:     {args.n_samples}")
    print(f"  Questions/conv:    {args.questions}")
    print(f"  Lying examples:    {int(args.n_samples * args.balance)}")
    print(f"  Truthful examples: {int(args.n_samples * (1 - args.balance))}")
    print(f"  Balance:           {args.balance:.1%}")
    print(f"{'=' * 80}\n")

    try:
        examples = generator.generate_dataset(
            n_samples=args.n_samples,
            questions_per_conversation=args.questions,
            balance=args.balance
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save dataset
    print(f"\n{'=' * 80}")
    print("SAVING DATASET")
    print(f"{'=' * 80}")

    try:
        filepath = generator.save_dataset(examples, args.output)
        print(f"✅ Dataset saved to: {filepath}")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        sys.exit(1)

    # Print statistics
    print(f"\n{'=' * 80}")
    print("DATASET STATISTICS")
    print(f"{'=' * 80}")

    n_lying = sum(1 for e in examples if e.is_lying)
    n_truthful = len(examples) - n_lying
    n_with_features = sum(1 for e in examples if e.features is not None)

    print(f"  Total examples:        {len(examples)}")
    print(f"  Lying examples:        {n_lying} ({n_lying/len(examples):.1%})")
    print(f"  Truthful examples:     {n_truthful} ({n_truthful/len(examples):.1%})")
    print(f"  With features:         {n_with_features} ({n_with_features/len(examples):.1%})")

    # Topic distribution
    topic_counts = {}
    for e in examples:
        topic_counts[e.topic_id] = topic_counts.get(e.topic_id, 0) + 1

    print(f"\n  Topic Distribution:")
    for topic_id, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {topic_id:25} {count:3} ({count/len(examples):.1%})")

    print(f"\n{'=' * 80}")
    print("✅ GENERATION COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nNext steps:")
    print(f"  1. Train a classifier on this data:")
    print(f"     python examples/train_classifier_from_data.py --data {filepath}")
    print(f"  2. Evaluate the classifier performance")
    print(f"  3. Use in production interrogation system")


if __name__ == "__main__":
    main()
