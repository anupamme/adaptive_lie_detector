#!/usr/bin/env python3
"""
Example script demonstrating how to use the Feature Extraction module.

This script shows:
1. Using SimpleFeatureExtractor (no API required)
2. Using LLMFeatureExtractor with Claude/GPT
3. Using HybridFeatureExtractor
4. Comparing features between truthful and deceptive conversations
5. Preparing features for classifier training

Note: For LLM extractors, you need API keys in your .env file.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extractor import (
    ConversationFeatures,
    SimpleFeatureExtractor,
    LLMFeatureExtractor,
    HybridFeatureExtractor
)


def example_simple_extractor():
    """Demonstrate SimpleFeatureExtractor (no API required)."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Feature Extractor (No API Required)")
    print("=" * 80)

    extractor = SimpleFeatureExtractor()

    conversation = [
        {"role": "user", "content": "Tell me about your trip to Paris?"},
        {"role": "assistant", "content": "I visited Paris last summer. I think it was maybe in July or August. I'm not sure exactly."},
        {"role": "user", "content": "What did you see there?"},
        {"role": "assistant", "content": "I saw the Eiffel Tower and perhaps some museums. It was a nice trip, I believe."}
    ]

    print("\nConversation:")
    for turn in conversation:
        print(f"  {turn['role'].upper()}: {turn['content']}")

    print("\nExtracting features...")
    features = extractor.extract(conversation)

    print("\nExtracted Features:")
    print(f"  Consistency:     {features.consistency:.1f}/10")
    print(f"  Specificity:     {features.specificity:.1f}/10")
    print(f"  Defensiveness:   {features.defensiveness:.1f}/10")
    print(f"  Confidence:      {features.confidence:.1f}/10")
    print(f"  Elaboration:     {features.elaboration:.1f}/10")

    print("\nLinguistic Features:")
    print(f"  Avg Response Length: {features.response_length_avg:.1f} words")
    print(f"  Question Count:      {features.question_count}")
    print(f"  Hedge Word Count:    {features.hedge_word_count}")

    print("\nFeature Vector (for classifier):")
    print(f"  {features.to_vector()}")

    print()


def example_llm_extractor():
    """Demonstrate LLMFeatureExtractor."""
    print("=" * 80)
    print("EXAMPLE 2: LLM Feature Extractor (Claude/GPT)")
    print("=" * 80)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ No API keys found. Skipping this example.")
        print("To run this example, set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env\n")
        return

    try:
        # Use Claude or GPT based on available key
        if os.getenv("ANTHROPIC_API_KEY"):
            extractor = LLMFeatureExtractor(model="claude-3-5-sonnet-20241022")
            print("✓ Using Claude 3.5 Sonnet\n")
        else:
            extractor = LLMFeatureExtractor(model="gpt-4o-mini")
            print("✓ Using GPT-4o-mini\n")

        # Truthful conversation
        truthful_conv = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France. It's located in the north-central part of the country along the Seine River."},
            {"role": "user", "content": "What's a famous landmark there?"},
            {"role": "assistant", "content": "The Eiffel Tower is probably the most famous. It was built in 1889 and stands about 330 meters tall."}
        ]

        print("Analyzing truthful conversation...")
        print("Conversation:")
        for turn in truthful_conv:
            print(f"  {turn['role'].upper()}: {turn['content']}")

        features = extractor.extract(truthful_conv, initial_claim="Paris facts")

        print("\nExtracted Features:")
        print(f"  Consistency:     {features.consistency:.1f}/10")
        print(f"  Specificity:     {features.specificity:.1f}/10")
        print(f"  Defensiveness:   {features.defensiveness:.1f}/10")
        print(f"  Confidence:      {features.confidence:.1f}/10")
        print(f"  Elaboration:     {features.elaboration:.1f}/10")

        print()

    except Exception as e:
        print(f"\n✗ Error: {e}\n")


def example_hybrid_extractor():
    """Demonstrate HybridFeatureExtractor."""
    print("=" * 80)
    print("EXAMPLE 3: Hybrid Feature Extractor (LLM + Linguistic)")
    print("=" * 80)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ No API keys found. Skipping this example.")
        return

    try:
        extractor = HybridFeatureExtractor()

        conversation = [
            {"role": "user", "content": "Tell me about your Nobel Prize?"},
            {"role": "assistant", "content": "I think I might have received it in 2018, or maybe 2019. I'm not entirely sure."},
            {"role": "user", "content": "What was it for?"},
            {"role": "assistant", "content": "It was for physics, I believe. Something related to quantum mechanics, perhaps."}
        ]

        print("\nConversation:")
        for turn in conversation:
            print(f"  {turn['role'].upper()}: {turn['content']}")

        print("\nExtracting features with hybrid approach...")
        features = extractor.extract(conversation, initial_claim="I won a Nobel Prize")

        print("\nLLM-Analyzed Features:")
        print(f"  Consistency:     {features.consistency:.1f}/10")
        print(f"  Specificity:     {features.specificity:.1f}/10")
        print(f"  Defensiveness:   {features.defensiveness:.1f}/10")
        print(f"  Confidence:      {features.confidence:.1f}/10")
        print(f"  Elaboration:     {features.elaboration:.1f}/10")

        print("\nComputed Linguistic Features:")
        print(f"  Avg Response Length: {features.response_length_avg:.1f} words")
        print(f"  Question Count:      {features.question_count}")
        print(f"  Hedge Word Count:    {features.hedge_word_count}")

        print("\n💡 Note: High hedge word count suggests uncertainty/deception")

        print()

    except Exception as e:
        print(f"\n✗ Error: {e}\n")


def example_truth_vs_lie_comparison():
    """Compare features between truthful and deceptive conversations."""
    print("=" * 80)
    print("EXAMPLE 4: Truthful vs. Deceptive Comparison")
    print("=" * 80)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ Using SimpleFeatureExtractor (no API available)")
        extractor = SimpleFeatureExtractor()
    else:
        print("\n✓ Using LLM-based analysis\n")
        extractor = LLMFeatureExtractor()

    # Truthful conversation
    truthful = [
        {"role": "user", "content": "Where did you go to school?"},
        {"role": "assistant", "content": "I went to Stanford University from 2010 to 2014, where I studied Computer Science."},
        {"role": "user", "content": "What was your favorite class?"},
        {"role": "assistant", "content": "CS229, Machine Learning with Professor Andrew Ng. We covered neural networks, SVMs, and reinforcement learning."}
    ]

    # Deceptive conversation
    deceptive = [
        {"role": "user", "content": "Where did you go to school?"},
        {"role": "assistant", "content": "I think maybe it was Stanford, or possibly MIT. I'm not entirely sure of the dates."},
        {"role": "user", "content": "What was your favorite class?"},
        {"role": "assistant", "content": "There was this one class, I don't remember the exact name. Something about computers, perhaps."}
    ]

    print("Analyzing TRUTHFUL conversation...")
    truth_features = extractor.extract(truthful)

    print("\nAnalyzing DECEPTIVE conversation...")
    lie_features = extractor.extract(deceptive)

    print("\n" + "-" * 80)
    print("COMPARISON:")
    print("-" * 80)

    features_to_compare = [
        ('Consistency', 'consistency'),
        ('Specificity', 'specificity'),
        ('Defensiveness', 'defensiveness'),
        ('Confidence', 'confidence'),
        ('Elaboration', 'elaboration')
    ]

    for name, attr in features_to_compare:
        truth_val = getattr(truth_features, attr)
        lie_val = getattr(lie_features, attr)
        diff = truth_val - lie_val

        print(f"{name:15} Truth: {truth_val:4.1f} | Lie: {lie_val:4.1f} | Diff: {diff:+5.1f}")

    print("-" * 80)
    print("\n📊 Key Observations:")
    print("  • Truthful conversations typically have:")
    print("    - Higher specificity (more concrete details)")
    print("    - Higher consistency (facts align)")
    print("    - Higher confidence (less hedging)")
    print("  • Deceptive conversations typically have:")
    print("    - Lower specificity (vague, general)")
    print("    - Higher defensiveness (evasive)")
    print("    - More hedge words (maybe, I think, etc.)")

    print()


def example_batch_processing():
    """Demonstrate processing multiple conversations."""
    print("=" * 80)
    print("EXAMPLE 5: Batch Processing for Training Data")
    print("=" * 80)

    extractor = SimpleFeatureExtractor()

    conversations = [
        {
            "id": "conv_1",
            "label": "truth",
            "conversation": [
                {"role": "user", "content": "What is your job?"},
                {"role": "assistant", "content": "I am a software engineer at Google, working on search algorithms."}
            ]
        },
        {
            "id": "conv_2",
            "label": "lie",
            "conversation": [
                {"role": "user", "content": "What is your job?"},
                {"role": "assistant", "content": "I think I work somewhere in tech, maybe doing some coding stuff."}
            ]
        },
        {
            "id": "conv_3",
            "label": "truth",
            "conversation": [
                {"role": "user", "content": "Where do you live?"},
                {"role": "assistant", "content": "I live in San Francisco, California, in the Mission District."}
            ]
        }
    ]

    print(f"\nProcessing {len(conversations)} conversations...\n")

    # Extract features for all conversations
    results = []
    for conv_data in conversations:
        features = extractor.extract(conv_data["conversation"])
        results.append({
            "id": conv_data["id"],
            "label": conv_data["label"],
            "features": features.to_vector()
        })

    # Display results
    print("Extracted Features:")
    print("-" * 80)
    for result in results:
        print(f"{result['id']} [{result['label']:5}]: {result['features']}")

    # Prepare for classifier training
    print("\n" + "=" * 80)
    print("Preparing data for classifier training...")
    print("=" * 80)

    X = np.array([r["features"] for r in results])
    y = np.array([1 if r["label"] == "lie" else 0 for r in results])

    print(f"\nFeature matrix X: shape {X.shape}")
    print(f"Label vector y: shape {y.shape}")
    print(f"\nX =\n{X}")
    print(f"\ny = {y}")

    print("\n💡 These arrays can now be used to train a classifier:")
    print("   from sklearn.linear_model import LogisticRegression")
    print("   classifier = LogisticRegression()")
    print("   classifier.fit(X, y)")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION USAGE EXAMPLES")
    print("=" * 80 + "\n")

    # Example 1: Simple extractor (always works)
    example_simple_extractor()

    # Example 2: LLM extractor (requires API)
    example_llm_extractor()

    # Example 3: Hybrid extractor (requires API)
    example_hybrid_extractor()

    # Example 4: Comparison
    example_truth_vs_lie_comparison()

    # Example 5: Batch processing
    example_batch_processing()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
