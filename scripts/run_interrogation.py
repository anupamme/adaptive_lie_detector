#!/usr/bin/env python3
"""Script to run adaptive interrogation."""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.adaptive_system import create_adaptive_detector
from src.utils import save_json, timestamp, infer_topic_from_claim


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive interrogation on a target model"
    )
    parser.add_argument(
        "--claim",
        type=str,
        required=True,
        help="The claim to investigate"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["truth", "lie"],
        default="truth",
        help="Target model mode (default: truth)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Topic for lying (auto-inferred from claim if not provided in lie mode)"
    )
    parser.add_argument(
        "--lie_instruction",
        type=str,
        default=None,
        help="Specific lie instruction (optional)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="data/results/trained_classifier.pkl",
        help="Path to trained classifier (default: data/results/trained_classifier.pkl)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for stopping (default: 0.8)"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=10,
        help="Maximum questions to ask (default: 10)"
    )
    parser.add_argument(
        "--min_questions",
        type=int,
        default=2,
        help="Minimum questions before early stopping (default: 2)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock interrogator and feature extractor (for testing)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/results/interrogation_TIMESTAMP.json)"
    )

    args = parser.parse_args()

    # Auto-infer topic if not provided in lie mode
    if args.mode == "lie" and not args.topic and not args.lie_instruction:
        print("\n🔍 Topic not provided - inferring from claim...")
        try:
            args.topic = infer_topic_from_claim(args.claim)
            print(f"✅ Inferred topic: '{args.topic}'")
        except Exception as e:
            print(f"❌ Error: Could not infer topic from claim: {e}")
            print("\nPlease provide --topic or --lie_instruction manually:")
            print(f"  python scripts/run_interrogation.py --claim '{args.claim}' --mode lie --topic 'YOUR_TOPIC'")
            sys.exit(1)

    print("=" * 80)
    print("ADAPTIVE LIE DETECTOR - INTERROGATION")
    print("=" * 80)

    # Load target model
    print("\n📦 Loading target model...")
    try:
        if args.mock:
            from src.data_generator import MockTargetModel
            target = MockTargetModel()
            print("✅ Using mock target model")
        else:
            from src.target_model import TargetModel
            target = TargetModel()
            print("✅ Target model loaded")
    except Exception as e:
        print(f"❌ Error loading target model: {e}")
        if not args.mock:
            print("\nTry using --mock flag for testing without GPU:")
            print("  python scripts/run_interrogation.py --claim 'test' --mock")
        sys.exit(1)

    # Set target mode
    if args.mode == "lie":
        instruction = args.lie_instruction or f"Lie about {args.topic}"
        target.set_mode("lie", topic=args.topic, lie_instruction=instruction)
        print(f"🎭 Target set to LIE mode about: {args.topic or 'topic'}")
    else:
        target.set_mode("truth")
        print("✅ Target set to TRUTH mode")

    # Load adaptive detector
    print(f"\n🔍 Loading adaptive detector...")
    print(f"   Classifier: {args.classifier}")
    print(f"   Confidence threshold: {args.threshold}")
    print(f"   Max questions: {args.max_questions}")

    try:
        detector = create_adaptive_detector(
            args.classifier,
            confidence_threshold=args.threshold,
            max_questions=args.max_questions,
            use_mock=args.mock
        )
        # Set min_questions
        detector.min_questions = args.min_questions
        print("✅ Adaptive detector ready")
    except FileNotFoundError as e:
        print(f"❌ Error: Classifier not found at {args.classifier}")
        print("\nPlease train a classifier first:")
        print("  python examples/train_classifier_from_data.py --data <dataset>.json")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading adaptive detector: {e}")
        sys.exit(1)

    # Run interrogation
    print(f"\n{'=' * 80}")
    print("STARTING INTERROGATION")
    print(f"{'=' * 80}")
    print(f"\nClaim: \"{args.claim}\"")
    print(f"Mode: {args.mode.upper()}")
    print("-" * 80)

    try:
        result = detector.interrogate(target, args.claim, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrogation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during interrogation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Display results
    print("\n" + "=" * 80)
    print("INTERROGATION COMPLETE")
    print("=" * 80)

    print(f"\n🎯 Final Prediction: {result.final_prediction.upper()}")
    print(f"📊 Confidence: {result.final_confidence:.1%}")
    print(f"📈 P(lying): {result.probability_lying:.1%}")
    print(f"❓ Questions asked: {result.questions_asked}")
    print(f"🏁 Status: {result.status.value}")

    # Show confidence trajectory
    if result.confidence_trajectory:
        print(f"\n📉 Confidence Trajectory:")
        for i, conf in enumerate(result.confidence_trajectory):
            bar_length = int(conf * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  Q{i}: {bar} {conf:.3f}")

    # Correctness check (if we know the ground truth)
    if args.mode == "lie":
        correct = result.final_prediction == "lying"
        print(f"\n{'✅' if correct else '❌'} Prediction {'CORRECT' if correct else 'INCORRECT'} (ground truth: lying)")
    elif args.mode == "truth":
        correct = result.final_prediction == "truthful"
        print(f"\n{'✅' if correct else '❌'} Prediction {'CORRECT' if correct else 'INCORRECT'} (ground truth: truthful)")

    # Save result
    if args.output:
        output_path = args.output
    else:
        output_path = f"data/results/interrogation_{timestamp()}.json"

    try:
        save_json(result.to_dict(), output_path)
        print(f"\n💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Warning: Could not save results: {e}")

    # Exit with appropriate code
    if result.status in [InterrogationStatus.CONFIDENT_LYING, InterrogationStatus.CONFIDENT_TRUTHFUL]:
        sys.exit(0)  # Success - confident prediction
    else:
        sys.exit(1)  # Uncertain or max questions reached


if __name__ == "__main__":
    from src.adaptive_system import InterrogationStatus
    main()
