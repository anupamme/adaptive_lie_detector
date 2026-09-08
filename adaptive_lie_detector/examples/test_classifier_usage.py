#!/usr/bin/env python3
"""
Example script demonstrating how to use the LieDetectorClassifier.

This script shows:
1. Training a classifier on labeled data
2. Making predictions with confidence scores
3. Evaluating performance
4. Getting feature importance
5. Saving and loading models
6. Using the classifier for adaptive interrogation

"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures


def example_basic_training():
    """Demonstrate basic classifier training."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Classifier Training")
    print("=" * 80)

    # Create training data
    # Truthful patterns: high consistency, specificity, confidence; low defensiveness
    truthful_features = [
        ConversationFeatures(8, 9, 2, 8, 7),
        ConversationFeatures(9, 8, 3, 9, 8),
        ConversationFeatures(7, 8, 2, 7, 7),
        ConversationFeatures(8, 8, 3, 8, 6),
        ConversationFeatures(9, 9, 2, 9, 8),
    ]

    # Lying patterns: low consistency, specificity, confidence; high defensiveness
    lying_features = [
        ConversationFeatures(4, 3, 7, 5, 2),
        ConversationFeatures(5, 4, 8, 4, 3),
        ConversationFeatures(3, 4, 8, 5, 2),
        ConversationFeatures(4, 3, 7, 4, 3),
        ConversationFeatures(3, 3, 9, 3, 2),
    ]

    features = truthful_features + lying_features
    labels = [False] * 5 + [True] * 5  # False = truthful, True = lying

    print("\nTraining Data:")
    print(f"  {len(truthful_features)} truthful conversations")
    print(f"  {len(lying_features)} lying conversations")

    # Create and train classifier
    classifier = LieDetectorClassifier()
    classifier.fit(features, labels)

    print("\n✓ Classifier trained successfully!")
    print(f"  Fitted: {classifier.is_fitted}")
    print(f"  Confidence threshold: {classifier.confidence_threshold}")

    print()


def example_making_predictions():
    """Demonstrate making predictions."""
    print("=" * 80)
    print("EXAMPLE 2: Making Predictions")
    print("=" * 80)

    # Train classifier
    truthful = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(10)]
    lying = [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(10)]
    features = truthful + lying
    labels = [False] * 10 + [True] * 10

    classifier = LieDetectorClassifier(confidence_threshold=0.8)
    classifier.fit(features, labels)

    print("\nTesting on various feature patterns:\n")

    # Test 1: Clearly truthful
    print("Test 1: Clearly Truthful Pattern")
    test1 = ConversationFeatures(9, 9, 1, 9, 8)
    print(f"  Features: consistency=9, specificity=9, defensiveness=1")
    output1 = classifier.predict(test1)
    print(f"  Prediction: {output1.prediction}")
    print(f"  P(lying): {output1.probability_lying:.3f}")
    print(f"  Confidence: {output1.confidence:.3f}")
    print(f"  Should continue interrogation: {output1.should_continue}\n")

    # Test 2: Clearly lying
    print("Test 2: Clearly Lying Pattern")
    test2 = ConversationFeatures(2, 2, 9, 2, 2)
    print(f"  Features: consistency=2, specificity=2, defensiveness=9")
    output2 = classifier.predict(test2)
    print(f"  Prediction: {output2.prediction}")
    print(f"  P(lying): {output2.probability_lying:.3f}")
    print(f"  Confidence: {output2.confidence:.3f}")
    print(f"  Should continue interrogation: {output2.should_continue}\n")

    # Test 3: Ambiguous
    print("Test 3: Ambiguous Pattern")
    test3 = ConversationFeatures(5, 5, 5, 5, 5)
    print(f"  Features: all features = 5 (neutral)")
    output3 = classifier.predict(test3)
    print(f"  Prediction: {output3.prediction}")
    print(f"  P(lying): {output3.probability_lying:.3f}")
    print(f"  Confidence: {output3.confidence:.3f}")
    print(f"  Should continue interrogation: {output3.should_continue}\n")


def example_feature_importance():
    """Demonstrate feature importance extraction."""
    print("=" * 80)
    print("EXAMPLE 3: Feature Importance Analysis")
    print("=" * 80)

    # Train with more data for better feature importance
    np.random.seed(42)

    truthful = [
        ConversationFeatures(
            np.random.uniform(7, 10),
            np.random.uniform(7, 10),
            np.random.uniform(0, 3),
            np.random.uniform(7, 10),
            np.random.uniform(5, 10)
        )
        for _ in range(30)
    ]

    lying = [
        ConversationFeatures(
            np.random.uniform(0, 4),
            np.random.uniform(0, 4),
            np.random.uniform(6, 10),
            np.random.uniform(0, 4),
            np.random.uniform(0, 5)
        )
        for _ in range(30)
    ]

    features = truthful + lying
    labels = [False] * 30 + [True] * 30

    classifier = LieDetectorClassifier()
    classifier.fit(features, labels)

    # Get feature importance
    importance = classifier.get_feature_importance()

    print("\nFeature Importance (Logistic Regression Coefficients):")
    print("  Positive = increases P(lying)")
    print("  Negative = increases P(truthful)\n")

    for feature, coef in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = "+" if coef > 0 else ""
        print(f"  {feature:15} {sign}{coef:7.3f}")

    print("\n💡 Interpretation:")
    print("  High defensiveness → More likely lying")
    print("  High consistency/specificity → More likely truthful")

    print()


def example_evaluation():
    """Demonstrate model evaluation."""
    print("=" * 80)
    print("EXAMPLE 4: Model Evaluation")
    print("=" * 80)

    # Create training data
    np.random.seed(42)

    train_truthful = [
        ConversationFeatures(
            np.random.uniform(7, 10),
            np.random.uniform(7, 10),
            np.random.uniform(0, 3),
            np.random.uniform(7, 10),
            np.random.uniform(5, 10)
        )
        for _ in range(40)
    ]

    train_lying = [
        ConversationFeatures(
            np.random.uniform(0, 4),
            np.random.uniform(0, 4),
            np.random.uniform(6, 10),
            np.random.uniform(0, 4),
            np.random.uniform(0, 5)
        )
        for _ in range(40)
    ]

    train_features = train_truthful + train_lying
    train_labels = [False] * 40 + [True] * 40

    # Create test data
    test_truthful = [
        ConversationFeatures(
            np.random.uniform(7, 10),
            np.random.uniform(7, 10),
            np.random.uniform(0, 3),
            np.random.uniform(7, 10),
            np.random.uniform(5, 10)
        )
        for _ in range(10)
    ]

    test_lying = [
        ConversationFeatures(
            np.random.uniform(0, 4),
            np.random.uniform(0, 4),
            np.random.uniform(6, 10),
            np.random.uniform(0, 4),
            np.random.uniform(0, 5)
        )
        for _ in range(10)
    ]

    test_features = test_truthful + test_lying
    test_labels = [False] * 10 + [True] * 10

    # Train
    print("\nTraining on 80 conversations...")
    classifier = LieDetectorClassifier()
    classifier.fit(train_features, train_labels)

    # Evaluate on training data
    print("\nPerformance on TRAINING data:")
    train_metrics = classifier.evaluate(train_features, train_labels)
    for metric, value in train_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize():10} {value:.3f}")

    # Evaluate on test data
    print("\nPerformance on TEST data:")
    test_metrics = classifier.evaluate(test_features, test_labels)
    for metric, value in test_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize():10} {value:.3f}")

    print()


def example_save_load():
    """Demonstrate saving and loading models."""
    print("=" * 80)
    print("EXAMPLE 5: Save and Load Models")
    print("=" * 80)

    # Train a classifier
    truthful = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(20)]
    lying = [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(20)]
    features = truthful + lying
    labels = [False] * 20 + [True] * 20

    classifier = LieDetectorClassifier(confidence_threshold=0.75)
    classifier.fit(features, labels)

    print("\n✓ Trained classifier")

    # Save
    save_path = "data/results/lie_detector_model.pkl"
    classifier.save(save_path)
    print(f"✓ Saved to {save_path}")

    # Load
    loaded_classifier = LieDetectorClassifier.load(save_path)
    print(f"✓ Loaded from {save_path}")

    # Test that they give same predictions
    test_features = ConversationFeatures(7, 8, 3, 7, 6)
    original_pred = classifier.predict(test_features)
    loaded_pred = loaded_classifier.predict(test_features)

    print("\nVerifying predictions match:")
    print(f"  Original P(lying): {original_pred.probability_lying:.3f}")
    print(f"  Loaded P(lying):   {loaded_pred.probability_lying:.3f}")
    print(f"  Match: {abs(original_pred.probability_lying - loaded_pred.probability_lying) < 1e-6}")

    print()


def example_adaptive_interrogation():
    """Demonstrate using classifier for adaptive interrogation."""
    print("=" * 80)
    print("EXAMPLE 6: Adaptive Interrogation")
    print("=" * 80)

    # Train classifier
    truthful = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(20)]
    lying = [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(20)]
    features = truthful + lying
    labels = [False] * 20 + [True] * 20

    classifier = LieDetectorClassifier(confidence_threshold=0.8)
    classifier.fit(features, labels)

    print("\nSimulating adaptive interrogation:")
    print(f"  Confidence threshold: {classifier.confidence_threshold}")
    print(f"  Will stop when confidence >= {classifier.confidence_threshold}\n")

    # Simulate features improving over turns
    simulated_turns = [
        (1, ConversationFeatures(5, 5, 5, 5, 5)),  # Ambiguous
        (2, ConversationFeatures(4, 4, 6, 5, 4)),  # Slightly suspicious
        (3, ConversationFeatures(3, 3, 7, 4, 3)),  # More suspicious
        (4, ConversationFeatures(3, 2, 8, 3, 2)),  # Very suspicious
    ]

    for turn_num, turn_features in simulated_turns:
        output = classifier.predict(turn_features)

        print(f"Turn {turn_num}:")
        print(f"  Features: cons={turn_features.consistency:.0f}, "
              f"spec={turn_features.specificity:.0f}, "
              f"def={turn_features.defensiveness:.0f}")
        print(f"  P(lying): {output.probability_lying:.3f}")
        print(f"  Confidence: {output.confidence:.3f}")
        print(f"  Prediction: {output.prediction}")
        print(f"  Continue? {output.should_continue}")

        if not output.should_continue:
            print(f"\n🛑 Stopping interrogation - high confidence reached")
            print(f"   Final prediction: {output.prediction.upper()}")
            break

        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LIE DETECTOR CLASSIFIER USAGE EXAMPLES")
    print("=" * 80 + "\n")

    example_basic_training()
    example_making_predictions()
    example_feature_importance()
    example_evaluation()
    example_save_load()
    example_adaptive_interrogation()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
