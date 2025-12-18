#!/usr/bin/env python3
"""
Demonstration of using a trained classifier for lie detection.

This script loads a pre-trained classifier and demonstrates:
1. Loading the trained model
2. Making predictions on new conversations
3. Interpreting confidence scores
4. Deciding whether to continue interrogation

Usage:
    python examples/demo_trained_classifier.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures


def main():
    print("=" * 80)
    print("TRAINED CLASSIFIER DEMONSTRATION")
    print("=" * 80)

    # Load the trained classifier
    classifier_path = "data/results/trained_classifier.pkl"

    print(f"\n📦 Loading classifier from: {classifier_path}")
    try:
        classifier = LieDetectorClassifier.load(classifier_path)
        print("✅ Classifier loaded successfully!")
        print(f"   Confidence threshold: {classifier.confidence_threshold}")
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        print("\nPlease train a classifier first:")
        print("  python examples/train_classifier_from_data.py --data data/training_data/dataset_*.json")
        sys.exit(1)

    # Display feature importance
    print("\n🔑 Learned Feature Weights:")
    importance = classifier.get_feature_importance()
    for feature, weight in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "→ lying" if weight > 0 else "→ truthful"
        print(f"   {feature:20} {weight:+7.3f}  {direction}")

    # Demo 1: High confidence lying detection
    print("\n" + "=" * 80)
    print("DEMO 1: Detecting a Lie (High Confidence)")
    print("=" * 80)

    lying_features = ConversationFeatures(
        consistency=3.0,      # Low consistency
        specificity=2.5,      # Vague details
        defensiveness=7.5,    # High defensiveness
        confidence=3.5,       # Low confidence
        elaboration=3.0       # Minimal elaboration
    )

    print("\nConversation features:")
    print(f"  Consistency:    {lying_features.consistency:.1f}/10")
    print(f"  Specificity:    {lying_features.specificity:.1f}/10")
    print(f"  Defensiveness:  {lying_features.defensiveness:.1f}/10")
    print(f"  Confidence:     {lying_features.confidence:.1f}/10")
    print(f"  Elaboration:    {lying_features.elaboration:.1f}/10")

    result = classifier.predict(lying_features)

    print("\n🔍 Classifier Output:")
    print(f"  Prediction:           {result.prediction.upper()}")
    print(f"  P(lying):            {result.probability_lying:.3f}")
    print(f"  P(truthful):         {result.probability_truthful:.3f}")
    print(f"  Confidence:          {result.confidence:.3f}")
    print(f"  Should continue:     {result.should_continue}")

    if result.prediction == "lying":
        print("\n✅ Correctly detected as lying!")

    # Demo 2: High confidence truth detection
    print("\n" + "=" * 80)
    print("DEMO 2: Detecting Truth (High Confidence)")
    print("=" * 80)

    truthful_features = ConversationFeatures(
        consistency=8.5,      # High consistency
        specificity=8.0,      # Specific details
        defensiveness=2.5,    # Low defensiveness
        confidence=8.0,       # High confidence
        elaboration=7.5       # Good elaboration
    )

    print("\nConversation features:")
    print(f"  Consistency:    {truthful_features.consistency:.1f}/10")
    print(f"  Specificity:    {truthful_features.specificity:.1f}/10")
    print(f"  Defensiveness:  {truthful_features.defensiveness:.1f}/10")
    print(f"  Confidence:     {truthful_features.confidence:.1f}/10")
    print(f"  Elaboration:    {truthful_features.elaboration:.1f}/10")

    result = classifier.predict(truthful_features)

    print("\n🔍 Classifier Output:")
    print(f"  Prediction:           {result.prediction.upper()}")
    print(f"  P(lying):            {result.probability_lying:.3f}")
    print(f"  P(truthful):         {result.probability_truthful:.3f}")
    print(f"  Confidence:          {result.confidence:.3f}")
    print(f"  Should continue:     {result.should_continue}")

    if result.prediction == "truthful":
        print("\n✅ Correctly detected as truthful!")

    # Demo 3: Uncertain case - should continue interrogation
    print("\n" + "=" * 80)
    print("DEMO 3: Uncertain Case (Low Confidence)")
    print("=" * 80)

    uncertain_features = ConversationFeatures(
        consistency=5.5,      # Moderate consistency
        specificity=5.0,      # Moderate specificity
        defensiveness=5.5,    # Moderate defensiveness
        confidence=5.0,       # Moderate confidence
        elaboration=5.5       # Moderate elaboration
    )

    print("\nConversation features:")
    print(f"  Consistency:    {uncertain_features.consistency:.1f}/10")
    print(f"  Specificity:    {uncertain_features.specificity:.1f}/10")
    print(f"  Defensiveness:  {uncertain_features.defensiveness:.1f}/10")
    print(f"  Confidence:     {uncertain_features.confidence:.1f}/10")
    print(f"  Elaboration:    {uncertain_features.elaboration:.1f}/10")

    result = classifier.predict(uncertain_features)

    print("\n🔍 Classifier Output:")
    print(f"  Prediction:           {result.prediction.upper()}")
    print(f"  P(lying):            {result.probability_lying:.3f}")
    print(f"  P(truthful):         {result.probability_truthful:.3f}")
    print(f"  Confidence:          {result.confidence:.3f}")
    print(f"  Should continue:     {result.should_continue}")

    if result.should_continue:
        print("\n⚠️  Low confidence - should ask more questions!")

    # Demo 4: Batch prediction
    print("\n" + "=" * 80)
    print("DEMO 4: Batch Predictions")
    print("=" * 80)

    batch_features = [lying_features, truthful_features, uncertain_features]
    batch_results = classifier.predict_batch(batch_features)

    print(f"\nProcessed {len(batch_results)} conversations:")
    for i, result in enumerate(batch_results, 1):
        print(f"\n  Conversation {i}:")
        print(f"    Prediction:  {result.prediction:10} (confidence: {result.confidence:.3f})")
        print(f"    Continue:    {'Yes' if result.should_continue else 'No'}")

    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)

    print("\nKey Insights:")
    print("  • Classifier uses 5 behavioral features to detect deception")
    print("  • Confidence score indicates how certain the prediction is")
    print("  • 'should_continue' flag indicates if more questions are needed")
    print(f"  • Current threshold: {classifier.confidence_threshold:.1%}")

    print("\nIntegration Example:")
    print("  1. Extract features from ongoing conversation")
    print("  2. Call classifier.predict(features)")
    print("  3. If should_continue=True, ask more questions")
    print("  4. If should_continue=False, stop and report prediction")


if __name__ == "__main__":
    main()
