#!/usr/bin/env python3
"""
Example script to train a classifier from generated data.

This script loads a generated dataset, extracts features,
trains a logistic regression classifier, and evaluates performance.

Usage:
    python examples/train_classifier_from_data.py --data data/training_data/dataset.json
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import TrainingDataGenerator
from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier from generated data"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/trained_classifier.pkl",
        help="Where to save trained classifier"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for stopping interrogation (default: 0.8)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CLASSIFIER TRAINING FROM GENERATED DATA")
    print("=" * 80)

    # Load dataset
    print(f"\n📁 Loading dataset from: {args.data}")
    try:
        examples, metadata = TrainingDataGenerator.load_dataset(args.data)
        print(f"✅ Loaded {len(examples)} examples")
        print(f"   Metadata: {metadata}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

    # Extract features and labels
    print(f"\n🔍 Extracting features and labels...")

    features_list = []
    labels = []

    for example in examples:
        # Get features from example
        if example.features is not None:
            # Reconstruct ConversationFeatures from dict
            features = ConversationFeatures(**example.features)
            features_list.append(features)
            labels.append(example.is_lying)

    print(f"✅ Extracted features from {len(features_list)} examples")

    if len(features_list) == 0:
        print("❌ No examples with features found!")
        sys.exit(1)

    # Split into train/test
    print(f"\n📊 Splitting into train/test sets (test_size={args.test_size})...")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features_list,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None
    )

    print(f"   Training set: {len(train_features)} examples")
    print(f"   Test set:     {len(test_features)} examples")

    # Train classifier
    print(f"\n🎓 Training classifier...")

    classifier = LieDetectorClassifier(
        confidence_threshold=args.confidence_threshold
    )

    try:
        classifier.fit(train_features, train_labels)
        print(f"✅ Classifier trained successfully!")
    except Exception as e:
        print(f"❌ Error training classifier: {e}")
        sys.exit(1)

    # Evaluate on training set
    print(f"\n📈 Evaluating on TRAINING set...")
    train_metrics = classifier.evaluate(train_features, train_labels)

    print(f"   Accuracy:  {train_metrics['accuracy']:.3f}")
    print(f"   Precision: {train_metrics['precision']:.3f}")
    print(f"   Recall:    {train_metrics['recall']:.3f}")
    print(f"   F1 Score:  {train_metrics['f1']:.3f}")
    if train_metrics['auc'] is not None:
        print(f"   AUC:       {train_metrics['auc']:.3f}")

    # Evaluate on test set
    print(f"\n📊 Evaluating on TEST set...")
    test_metrics = classifier.evaluate(test_features, test_labels)

    print(f"   Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"   Precision: {test_metrics['precision']:.3f}")
    print(f"   Recall:    {test_metrics['recall']:.3f}")
    print(f"   F1 Score:  {test_metrics['f1']:.3f}")
    if test_metrics['auc'] is not None:
        print(f"   AUC:       {test_metrics['auc']:.3f}")

    # Feature importance
    print(f"\n🔑 Feature Importance:")
    importance = classifier.get_feature_importance()

    for feature, coef in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "→ lying" if coef > 0 else "→ truthful"
        print(f"   {feature:15} {coef:+7.3f}  {direction}")

    # Save classifier
    print(f"\n💾 Saving classifier to: {args.output}")
    try:
        classifier.save(args.output)
        print(f"✅ Classifier saved successfully!")
    except Exception as e:
        print(f"❌ Error saving classifier: {e}")
        sys.exit(1)

    # Summary
    print(f"\n{'=' * 80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'=' * 80}")

    print(f"\nClassifier Performance:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"  Test F1 Score: {test_metrics['f1']:.3f}")

    print(f"\nYou can now use this classifier for lie detection:")
    print(f"  from src.classifier import LieDetectorClassifier")
    print(f"  classifier = LieDetectorClassifier.load('{args.output}')")


if __name__ == "__main__":
    main()
