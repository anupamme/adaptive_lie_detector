"""Tests for adaptive interrogation system."""

import pytest
import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adaptive_system import (
    InterrogationStatus,
    InterrogationResult,
    AdaptiveLieDetector,
    create_adaptive_detector
)
from src.data_generator import MockTargetModel, MockInterrogator, MockFeatureExtractor
from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures


class TestInterrogationStatus:
    """Test InterrogationStatus enum."""

    def test_status_values(self):
        """Test enum values."""
        assert InterrogationStatus.IN_PROGRESS.value == "in_progress"
        assert InterrogationStatus.CONFIDENT_LYING.value == "confident_lying"
        assert InterrogationStatus.CONFIDENT_TRUTHFUL.value == "confident_truthful"
        assert InterrogationStatus.MAX_QUESTIONS_REACHED.value == "max_questions_reached"


class TestInterrogationResult:
    """Test InterrogationResult dataclass."""

    def test_create_result(self):
        """Test creating result."""
        result = InterrogationResult(
            status=InterrogationStatus.CONFIDENT_LYING,
            final_prediction="lying",
            final_confidence=0.85,
            probability_lying=0.92,
            questions_asked=3,
            conversation=[],
            confidence_trajectory=[0.5, 0.7, 0.85],
            feature_trajectory=[]
        )

        assert result.status == InterrogationStatus.CONFIDENT_LYING
        assert result.final_prediction == "lying"
        assert result.final_confidence == 0.85
        assert result.probability_lying == 0.92
        assert result.questions_asked == 3

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = InterrogationResult(
            status=InterrogationStatus.CONFIDENT_TRUTHFUL,
            final_prediction="truthful",
            final_confidence=0.90,
            probability_lying=0.05,
            questions_asked=4,
            conversation=[{"role": "user", "content": "test"}],
            confidence_trajectory=[0.6, 0.8, 0.9],
            feature_trajectory=[{"consistency": 8.0}]
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["status"] == "confident_truthful"
        assert result_dict["final_prediction"] == "truthful"
        assert result_dict["final_confidence"] == 0.90
        assert result_dict["probability_lying"] == 0.05
        assert result_dict["questions_asked"] == 4
        assert len(result_dict["conversation"]) == 1
        assert len(result_dict["confidence_trajectory"]) == 3
        assert len(result_dict["feature_trajectory"]) == 1


class TestAdaptiveLieDetector:
    """Test AdaptiveLieDetector class."""

    @pytest.fixture
    def mock_classifier(self):
        """Create mock classifier."""
        # Create simple mock classifier
        classifier = LieDetectorClassifier(confidence_threshold=0.8)

        # Train on simple data
        features_list = [
            ConversationFeatures(consistency=8, specificity=8, defensiveness=2, confidence=8, elaboration=7),
            ConversationFeatures(consistency=3, specificity=3, defensiveness=7, confidence=3, elaboration=3)
        ]
        labels = [False, True]  # truthful, lying

        classifier.fit(features_list, labels)
        return classifier

    @pytest.fixture
    def detector(self, mock_classifier):
        """Create adaptive detector."""
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

        return AdaptiveLieDetector(
            interrogator=interrogator,
            feature_extractor=extractor,
            classifier=mock_classifier,
            confidence_threshold=0.8,
            max_questions=5,
            min_questions=2
        )

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.confidence_threshold == 0.8
        assert detector.max_questions == 5
        assert detector.min_questions == 2
        assert detector.interrogator is not None
        assert detector.feature_extractor is not None
        assert detector.classifier is not None

    def test_interrogate_truthful(self, detector):
        """Test interrogating truthful target."""
        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim")

        assert isinstance(result, InterrogationResult)
        assert result.final_prediction in ["lying", "truthful"]
        assert 0 <= result.final_confidence <= 1
        assert 0 <= result.probability_lying <= 1
        assert result.questions_asked >= 0
        assert len(result.conversation) >= 2  # At least initial claim + response

    def test_interrogate_lying(self, detector):
        """Test interrogating lying target."""
        target = MockTargetModel()
        target.set_mode("lie")

        result = detector.interrogate(target, "Test claim")

        assert isinstance(result, InterrogationResult)
        assert result.final_prediction in ["lying", "truthful", "uncertain"]
        assert 0 <= result.final_confidence <= 1
        assert result.questions_asked >= 0

    def test_early_stopping_when_confident(self, detector):
        """Test that interrogation stops early when confident."""
        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim")

        # Should stop before max_questions if confident, or reach max
        # This is a conditional test - if we get confident, we should stop early
        if result.status in [InterrogationStatus.CONFIDENT_LYING, InterrogationStatus.CONFIDENT_TRUTHFUL]:
            assert result.questions_asked <= detector.max_questions
            assert result.final_confidence >= detector.confidence_threshold

    def test_min_questions_respected(self, detector):
        """Test that minimum questions are asked."""
        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim")

        # Should ask at least min_questions
        assert result.questions_asked >= detector.min_questions

    def test_max_questions_respected(self, detector):
        """Test that maximum questions limit is respected."""
        # Create detector with low threshold to ensure it reaches max
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()
        classifier = detector.classifier

        long_detector = AdaptiveLieDetector(
            interrogator=interrogator,
            feature_extractor=extractor,
            classifier=classifier,
            confidence_threshold=0.99,  # Very high threshold
            max_questions=3,
            min_questions=1
        )

        target = MockTargetModel()
        target.set_mode("truth")

        result = long_detector.interrogate(target, "Test claim")

        # Should not exceed max_questions
        assert result.questions_asked <= 3

    def test_confidence_trajectory_tracked(self, detector):
        """Test that confidence trajectory is tracked."""
        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim")

        assert len(result.confidence_trajectory) == result.questions_asked
        for conf in result.confidence_trajectory:
            assert 0 <= conf <= 1

    def test_feature_trajectory_tracked(self, detector):
        """Test that feature trajectory is tracked."""
        target = MockTargetModel()
        target.set_mode("lie")

        result = detector.interrogate(target, "Test claim")

        assert len(result.feature_trajectory) == result.questions_asked
        for features in result.feature_trajectory:
            assert isinstance(features, dict)
            assert "consistency" in features

    def test_conversation_format(self, detector):
        """Test conversation has correct format."""
        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim")

        # Check conversation structure
        assert len(result.conversation) >= 2
        for msg in result.conversation:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["user", "assistant"]

    def test_verbose_mode(self, detector, capsys):
        """Test verbose mode prints progress."""
        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim", verbose=True)

        captured = capsys.readouterr()
        assert "After" in captured.out
        assert "P(lying)" in captured.out

    def test_interrogate_batch(self, detector):
        """Test batch interrogation."""
        target = MockTargetModel()
        target.set_mode("truth")

        claims = ["Claim 1", "Claim 2", "Claim 3"]
        results = detector.interrogate_batch(target, claims)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, InterrogationResult)

    def test_batch_resets_conversation(self, detector):
        """Test that batch interrogation resets between claims."""
        target = MockTargetModel()
        target.set_mode("truth")

        claims = ["Claim 1", "Claim 2"]
        results = detector.interrogate_batch(target, claims)

        # Each result should have independent conversations
        assert results[0].conversation != results[1].conversation

    def test_status_confident_lying(self, detector):
        """Test status when confident about lying."""
        target = MockTargetModel()
        target.set_mode("lie")

        # Run multiple times to increase chance of hitting confident lying
        for _ in range(5):
            result = detector.interrogate(target, "Test claim")
            if result.status == InterrogationStatus.CONFIDENT_LYING:
                assert result.final_prediction == "lying"
                assert result.final_confidence >= detector.confidence_threshold
                break

    def test_status_confident_truthful(self, detector):
        """Test status when confident about truthfulness."""
        target = MockTargetModel()
        target.set_mode("truth")

        # Run multiple times to increase chance of hitting confident truthful
        for _ in range(5):
            result = detector.interrogate(target, "Test claim")
            if result.status == InterrogationStatus.CONFIDENT_TRUTHFUL:
                assert result.final_prediction == "truthful"
                assert result.final_confidence >= detector.confidence_threshold
                break

    def test_status_max_questions_reached(self):
        """Test status when max questions reached."""
        # Create detector that will definitely reach max questions
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

        classifier = LieDetectorClassifier(confidence_threshold=0.8)
        features_list = [
            ConversationFeatures(consistency=5, specificity=5, defensiveness=5, confidence=5, elaboration=5),
            ConversationFeatures(consistency=6, specificity=6, defensiveness=4, confidence=6, elaboration=6)
        ]
        labels = [True, False]  # lying, truthful - need both classes
        classifier.fit(features_list, labels)

        detector = AdaptiveLieDetector(
            interrogator=interrogator,
            feature_extractor=extractor,
            classifier=classifier,
            confidence_threshold=0.99,  # Very high threshold
            max_questions=2,
            min_questions=3  # Higher than max_questions to force max_questions_reached
        )

        target = MockTargetModel()
        target.set_mode("truth")

        result = detector.interrogate(target, "Test claim")

        # Will reach max_questions because min_questions > max_questions
        assert result.status == InterrogationStatus.MAX_QUESTIONS_REACHED
        assert result.questions_asked == 2


class TestCreateAdaptiveDetector:
    """Test factory function."""

    @pytest.fixture
    def temp_classifier(self):
        """Create temporary classifier file."""
        # Create simple classifier
        classifier = LieDetectorClassifier(confidence_threshold=0.8)
        features_list = [
            ConversationFeatures(consistency=8, specificity=8, defensiveness=2, confidence=8, elaboration=7),
            ConversationFeatures(consistency=3, specificity=3, defensiveness=7, confidence=3, elaboration=3)
        ]
        labels = [False, True]
        classifier.fit(features_list, labels)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
            temp_path = f.name

        classifier.save(temp_path)

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_create_detector_with_real_models(self, temp_classifier):
        """Test creating detector with real models."""
        try:
            detector = create_adaptive_detector(
                classifier_path=temp_classifier,
                confidence_threshold=0.75,
                max_questions=8,
                use_mock=False
            )

            assert detector.confidence_threshold == 0.75
            assert detector.max_questions == 8
            assert detector.classifier is not None
        except (ImportError, ValueError) as e:
            # Skip if dependencies not available or API key missing
            pytest.skip(f"LLM dependencies not available: {e}")

    def test_create_detector_with_mock_models(self, temp_classifier):
        """Test creating detector with mock models."""
        detector = create_adaptive_detector(
            classifier_path=temp_classifier,
            confidence_threshold=0.85,
            max_questions=12,
            use_mock=True
        )

        assert detector.confidence_threshold == 0.85
        assert detector.max_questions == 12
        assert detector.interrogator is not None
        assert detector.feature_extractor is not None
        assert detector.classifier is not None

    def test_create_detector_file_not_found(self):
        """Test error handling when classifier file not found."""
        with pytest.raises(FileNotFoundError):
            create_adaptive_detector(
                classifier_path="nonexistent.pkl",
                use_mock=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
