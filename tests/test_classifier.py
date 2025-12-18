"""Tests for lie detector classifier."""

import pytest
import numpy as np
import tempfile
import os
from src.classifier import (
    LieDetectorClassifier,
    ClassifierOutput,
    calculate_confidence
)
from src.feature_extractor import ConversationFeatures


# Unit tests for confidence calculation

def test_confidence_calculation():
    """Test confidence calculation function."""
    # p=0.5 should give confidence=0 (maximum uncertainty)
    assert calculate_confidence(0.5) == 0.0

    # p=0 or p=1 should give confidence=1 (maximum certainty)
    assert calculate_confidence(0.0) == 1.0
    assert calculate_confidence(1.0) == 1.0

    # p=0.75 should give confidence=0.5
    assert calculate_confidence(0.75) == 0.5

    # p=0.25 should give confidence=0.5
    assert calculate_confidence(0.25) == 0.5

    # p=0.9 should give confidence=0.8
    assert calculate_confidence(0.9) == 0.8


def test_confidence_validation():
    """Test that confidence calculation validates input."""
    with pytest.raises(ValueError):
        calculate_confidence(-0.1)

    with pytest.raises(ValueError):
        calculate_confidence(1.1)


# Unit tests for ClassifierOutput

def test_classifier_output():
    """Test ClassifierOutput dataclass."""
    output = ClassifierOutput(
        probability_lying=0.7,
        probability_truthful=0.3,
        confidence=0.4,
        prediction="lying",
        should_continue=True
    )

    assert output.probability_lying == 0.7
    assert output.probability_truthful == 0.3
    assert output.confidence == 0.4
    assert output.prediction == "lying"
    assert output.should_continue is True


def test_classifier_output_to_dict():
    """Test conversion to dictionary."""
    output = ClassifierOutput(
        probability_lying=0.7,
        probability_truthful=0.3,
        confidence=0.4,
        prediction="lying",
        should_continue=True
    )

    output_dict = output.to_dict()

    assert output_dict['probability_lying'] == 0.7
    assert output_dict['probability_truthful'] == 0.3
    assert output_dict['confidence'] == 0.4
    assert output_dict['prediction'] == "lying"
    assert output_dict['should_continue'] is True


# Unit tests for LieDetectorClassifier

def test_classifier_initialization():
    """Test classifier initialization."""
    classifier = LieDetectorClassifier()

    assert classifier.confidence_threshold == 0.8
    assert classifier.is_fitted is False


def test_classifier_initialization_with_params():
    """Test classifier with custom parameters."""
    classifier = LieDetectorClassifier(
        confidence_threshold=0.9,
        regularization=0.5
    )

    assert classifier.confidence_threshold == 0.9
    assert classifier.model.C == 0.5


def test_classifier_fits():
    """Test that classifier can be fitted on data."""
    classifier = LieDetectorClassifier()

    # Create dummy training data
    # Truthful: high consistency, specificity, confidence; low defensiveness
    # Lying: low consistency, specificity, confidence; high defensiveness
    features = [
        ConversationFeatures(8, 9, 2, 8, 7),  # Truthful pattern
        ConversationFeatures(9, 8, 3, 9, 8),  # Truthful pattern
        ConversationFeatures(7, 8, 2, 7, 7),  # Truthful pattern
        ConversationFeatures(4, 3, 7, 5, 2),  # Lying pattern
        ConversationFeatures(5, 4, 8, 4, 3),  # Lying pattern
        ConversationFeatures(3, 4, 8, 5, 2),  # Lying pattern
    ]
    labels = [False, False, False, True, True, True]

    classifier.fit(features, labels)

    assert classifier.is_fitted is True


def test_classifier_fit_validation():
    """Test that fit validates inputs."""
    classifier = LieDetectorClassifier()

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        classifier.fit(
            [ConversationFeatures(5, 5, 5, 5, 5)],
            [True, False]
        )

    # Too few samples
    with pytest.raises(ValueError, match="at least 2"):
        classifier.fit(
            [ConversationFeatures(5, 5, 5, 5, 5)],
            [True]
        )


def test_classifier_predicts():
    """Test that fitted classifier can make predictions."""
    classifier = LieDetectorClassifier()

    # Fit with dummy data
    features = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(10)]
    features += [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(10)]
    labels = [False] * 10 + [True] * 10

    classifier.fit(features, labels)

    # Predict on new data
    output = classifier.predict(ConversationFeatures(7, 8, 3, 7, 6))

    assert isinstance(output, ClassifierOutput)
    assert 0 <= output.probability_lying <= 1
    assert 0 <= output.probability_truthful <= 1
    assert 0 <= output.confidence <= 1
    assert output.prediction in ["lying", "truthful"]
    assert isinstance(output.should_continue, bool)

    # Probabilities should sum to 1
    assert abs(output.probability_lying + output.probability_truthful - 1.0) < 1e-6


def test_classifier_predict_before_fit_raises():
    """Test that predicting before fitting raises error."""
    classifier = LieDetectorClassifier()

    with pytest.raises(ValueError, match="must be fitted"):
        classifier.predict(ConversationFeatures(5, 5, 5, 5, 5))


def test_should_continue_logic():
    """Test that should_continue respects threshold."""
    # Create classifier with threshold 0.8
    classifier = LieDetectorClassifier(confidence_threshold=0.8)

    # Create strongly separated training data
    truthful_features = [ConversationFeatures(9, 9, 1, 9, 9) for _ in range(20)]
    lying_features = [ConversationFeatures(1, 1, 9, 1, 1) for _ in range(20)]

    features = truthful_features + lying_features
    labels = [False] * 20 + [True] * 20

    classifier.fit(features, labels)

    # Test with clearly truthful pattern (should have high confidence, should stop)
    truthful_output = classifier.predict(ConversationFeatures(9, 9, 1, 9, 9))
    assert truthful_output.confidence > 0.8  # High confidence
    assert truthful_output.should_continue is False  # Should stop

    # Test with clearly lying pattern (should have high confidence, should stop)
    lying_output = classifier.predict(ConversationFeatures(1, 1, 9, 1, 1))
    assert lying_output.confidence > 0.8  # High confidence
    assert lying_output.should_continue is False  # Should stop

    # Test with ambiguous pattern (should have low confidence, should continue)
    ambiguous_output = classifier.predict(ConversationFeatures(5, 5, 5, 5, 5))
    # This one might vary, but at least confidence should be calculated
    assert 0 <= ambiguous_output.confidence <= 1


def test_predict_batch():
    """Test batch prediction."""
    classifier = LieDetectorClassifier()

    # Fit with data
    features = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(10)]
    features += [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(10)]
    labels = [False] * 10 + [True] * 10

    classifier.fit(features, labels)

    # Batch predict
    test_features = [
        ConversationFeatures(7, 8, 3, 7, 6),
        ConversationFeatures(5, 4, 6, 5, 3),
        ConversationFeatures(8, 8, 2, 8, 7)
    ]

    outputs = classifier.predict_batch(test_features)

    assert len(outputs) == 3
    assert all(isinstance(o, ClassifierOutput) for o in outputs)


def test_feature_importance():
    """Test feature importance extraction."""
    classifier = LieDetectorClassifier()

    # Fit with data
    features = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(10)]
    features += [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(10)]
    labels = [False] * 10 + [True] * 10

    classifier.fit(features, labels)

    # Get feature importance
    importance = classifier.get_feature_importance()

    assert isinstance(importance, dict)
    assert len(importance) == 5
    assert 'consistency' in importance
    assert 'specificity' in importance
    assert 'defensiveness' in importance
    assert 'confidence' in importance
    assert 'elaboration' in importance

    # All should be floats
    assert all(isinstance(v, float) for v in importance.values())


def test_feature_importance_before_fit_raises():
    """Test that getting importance before fitting raises error."""
    classifier = LieDetectorClassifier()

    with pytest.raises(ValueError, match="must be fitted"):
        classifier.get_feature_importance()


def test_save_and_load():
    """Test classifier serialization."""
    # Create and fit classifier
    classifier = LieDetectorClassifier(confidence_threshold=0.75)
    features = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(10)]
    features += [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(10)]
    labels = [False] * 10 + [True] * 10
    classifier.fit(features, labels)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "subdir", "classifier.pkl")
        classifier.save(filepath)

        assert os.path.exists(filepath)

        # Load
        loaded = LieDetectorClassifier.load(filepath)

        assert loaded.is_fitted is True
        assert loaded.confidence_threshold == 0.75

        # Should give same predictions
        test_features = ConversationFeatures(7, 8, 3, 7, 6)
        original_output = classifier.predict(test_features)
        loaded_output = loaded.predict(test_features)

        assert abs(original_output.probability_lying - loaded_output.probability_lying) < 1e-6
        assert original_output.prediction == loaded_output.prediction


def test_save_before_fit_raises():
    """Test that saving before fitting raises error."""
    classifier = LieDetectorClassifier()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "classifier.pkl")
        with pytest.raises(ValueError, match="Cannot save unfitted"):
            classifier.save(filepath)


def test_load_nonexistent_file():
    """Test that loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        LieDetectorClassifier.load("/nonexistent/path/classifier.pkl")


def test_evaluate():
    """Test evaluation metrics."""
    classifier = LieDetectorClassifier()

    # Create balanced training data
    features = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(20)]
    features += [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(20)]
    labels = [False] * 20 + [True] * 20

    classifier.fit(features, labels)

    # Evaluate on same data (should have high accuracy)
    metrics = classifier.evaluate(features, labels)

    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics

    # All metrics should be between 0 and 1
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        assert 0 <= metrics[key] <= 1

    if metrics['auc'] is not None:
        assert 0 <= metrics['auc'] <= 1

    # On training data with well-separated classes, should have good performance
    assert metrics['accuracy'] > 0.5


def test_evaluate_before_fit_raises():
    """Test that evaluating before fitting raises error."""
    classifier = LieDetectorClassifier()

    with pytest.raises(ValueError, match="must be fitted"):
        classifier.evaluate([], [])


def test_evaluate_validation():
    """Test that evaluate validates inputs."""
    classifier = LieDetectorClassifier()

    # Fit classifier
    features = [ConversationFeatures(8, 9, 2, 8, 7) for _ in range(10)]
    features += [ConversationFeatures(4, 3, 7, 5, 2) for _ in range(10)]
    labels = [False] * 10 + [True] * 10
    classifier.fit(features, labels)

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        classifier.evaluate(
            [ConversationFeatures(5, 5, 5, 5, 5)],
            [True, False]
        )


def test_classifier_learns_patterns():
    """Test that classifier actually learns meaningful patterns."""
    classifier = LieDetectorClassifier()

    # Create training data with clear patterns
    # Truthful: high specificity and consistency, low defensiveness
    truthful_features = [
        ConversationFeatures(
            consistency=np.random.uniform(7, 10),
            specificity=np.random.uniform(7, 10),
            defensiveness=np.random.uniform(0, 3),
            confidence=np.random.uniform(7, 10),
            elaboration=np.random.uniform(5, 10)
        )
        for _ in range(30)
    ]

    # Lying: low specificity and consistency, high defensiveness
    lying_features = [
        ConversationFeatures(
            consistency=np.random.uniform(0, 4),
            specificity=np.random.uniform(0, 4),
            defensiveness=np.random.uniform(6, 10),
            confidence=np.random.uniform(0, 4),
            elaboration=np.random.uniform(0, 5)
        )
        for _ in range(30)
    ]

    features = truthful_features + lying_features
    labels = [False] * 30 + [True] * 30

    classifier.fit(features, labels)

    # Test on clearly truthful example
    truthful_test = ConversationFeatures(9, 9, 1, 9, 8)
    truthful_output = classifier.predict(truthful_test)
    assert truthful_output.prediction == "truthful"
    assert truthful_output.probability_lying < 0.5

    # Test on clearly lying example
    lying_test = ConversationFeatures(2, 2, 9, 2, 2)
    lying_output = classifier.predict(lying_test)
    assert lying_output.prediction == "lying"
    assert lying_output.probability_lying > 0.5


if __name__ == "__main__":
    # Run basic tests
    print("Running basic unit tests...")
    test_confidence_calculation()
    test_confidence_validation()
    test_classifier_output()
    test_classifier_output_to_dict()
    test_classifier_initialization()
    test_classifier_initialization_with_params()
    test_classifier_fits()
    test_classifier_fit_validation()
    test_classifier_predicts()
    test_classifier_predict_before_fit_raises()
    test_should_continue_logic()
    test_predict_batch()
    test_feature_importance()
    test_feature_importance_before_fit_raises()
    test_save_and_load()
    test_save_before_fit_raises()
    test_evaluate()
    test_evaluate_before_fit_raises()
    test_evaluate_validation()
    test_classifier_learns_patterns()

    print("\n✓ All tests passed!")
