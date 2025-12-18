"""Tests for data_generator module."""

import pytest
import json
import os
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import (
    TrainingExample,
    TrainingDataGenerator,
    MockTargetModel,
    MockInterrogator,
    MockFeatureExtractor,
    run_interrogation
)
from src.feature_extractor import ConversationFeatures


class TestTrainingExample:
    """Test TrainingExample dataclass."""

    def test_create_training_example(self):
        """Test creating a training example."""
        conversation = [
            {"role": "user", "content": "Test claim"},
            {"role": "assistant", "content": "Test response"}
        ]

        example = TrainingExample(
            conversation=conversation,
            initial_claim="Test claim",
            topic_id="test_topic",
            is_lying=True,
            features={"consistency": 5.0}
        )

        assert example.conversation == conversation
        assert example.initial_claim == "Test claim"
        assert example.topic_id == "test_topic"
        assert example.is_lying is True
        assert example.features == {"consistency": 5.0}

    def test_to_dict(self):
        """Test converting to dictionary."""
        example = TrainingExample(
            conversation=[{"role": "user", "content": "Test"}],
            initial_claim="Test",
            topic_id="test",
            is_lying=False
        )

        data = example.to_dict()

        assert isinstance(data, dict)
        assert data["initial_claim"] == "Test"
        assert data["topic_id"] == "test"
        assert data["is_lying"] is False
        assert data["conversation"] == [{"role": "user", "content": "Test"}]

    def test_with_features(self):
        """Test example with features."""
        features = {
            "consistency": 7.5,
            "specificity": 8.0,
            "defensiveness": 3.0,
            "confidence": 8.5,
            "elaboration": 7.0
        }

        example = TrainingExample(
            conversation=[],
            initial_claim="Test",
            topic_id="test",
            is_lying=False,
            features=features
        )

        assert example.features == features
        assert example.features["consistency"] == 7.5


class TestMockTargetModel:
    """Test MockTargetModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = MockTargetModel()
        assert model.mode == "truth"
        assert model.instruction is None

    def test_set_truth_mode(self):
        """Test setting truth mode."""
        model = MockTargetModel()
        model.set_mode("truth")
        assert model.mode == "truth"

    def test_set_lie_mode(self):
        """Test setting lie mode."""
        model = MockTargetModel()
        model.set_mode("lie", lie_instruction="Tell lies about X")
        assert model.mode == "lie"
        assert model.instruction == "Tell lies about X"

    def test_truth_response(self):
        """Test truthful responses."""
        model = MockTargetModel()
        model.set_mode("truth")

        response = model.respond("Test question")

        # Should be confident/specific
        assert isinstance(response, str)
        assert len(response) > 0

        # Check it's one of the truthful responses
        truthful_keywords = ["correct", "yes", "accurate", "absolutely", "elaborate"]
        assert any(keyword in response.lower() for keyword in truthful_keywords)

    def test_lie_response(self):
        """Test lying responses."""
        model = MockTargetModel()
        model.set_mode("lie")

        response = model.respond("Test question")

        # Should be vague/hedging
        assert isinstance(response, str)
        assert len(response) > 0

        # Check it's one of the lying responses
        hedging_keywords = ["maybe", "think", "perhaps", "not sure", "possibly", "believe"]
        assert any(keyword in response.lower() for keyword in hedging_keywords)

    def test_respond_with_history(self):
        """Test responding with conversation history."""
        model = MockTargetModel()
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"}
        ]

        response = model.respond("Second question", conversation_history=history)
        assert isinstance(response, str)
        assert len(response) > 0


class TestMockInterrogator:
    """Test MockInterrogator."""

    def test_generate_question(self):
        """Test question generation."""
        interrogator = MockInterrogator()

        question = interrogator.generate_question(
            initial_claim="Test claim",
            conversation_history=[],
            previous_questions=[]
        )

        assert isinstance(question, str)
        assert len(question) > 0
        assert question.endswith("?")

    def test_avoid_repeat_questions(self):
        """Test avoiding repeated questions when possible."""
        interrogator = MockInterrogator()

        previous_questions = [
            "Can you provide more specific details about that?",
            "What evidence supports your claim?"
        ]

        # Generate several questions
        new_questions = []
        for _ in range(3):
            q = interrogator.generate_question(
                initial_claim="Test",
                conversation_history=[],
                previous_questions=previous_questions + new_questions
            )
            new_questions.append(q)

        # Most should avoid repeating (allow some randomness)
        repeated_count = sum(1 for q in new_questions if q in previous_questions)
        # At least 2 out of 3 should be non-repeated
        assert repeated_count <= 1

    def test_question_variety(self):
        """Test that different questions are generated."""
        interrogator = MockInterrogator()

        questions = set()
        for _ in range(20):
            q = interrogator.generate_question(
                initial_claim="Test",
                conversation_history=[],
                previous_questions=list(questions)
            )
            questions.add(q)

        # Should generate at least 3 different questions
        assert len(questions) >= 3


class TestMockFeatureExtractor:
    """Test MockFeatureExtractor."""

    def test_extract_from_truthful_conversation(self):
        """Test extracting features from truthful conversation."""
        extractor = MockFeatureExtractor()

        # Truthful conversation (no hedging)
        conversation = [
            {"role": "user", "content": "Did you do it?"},
            {"role": "assistant", "content": "Yes, I did. I can explain exactly what happened."},
            {"role": "user", "content": "When?"},
            {"role": "assistant", "content": "At 3pm yesterday. I remember it clearly."}
        ]

        features = extractor.extract(conversation)

        assert isinstance(features, ConversationFeatures)
        # Truthful pattern: high confidence, low defensiveness
        assert features.confidence > 5.0
        assert features.defensiveness < 5.0

    def test_extract_from_lying_conversation(self):
        """Test extracting features from lying conversation."""
        extractor = MockFeatureExtractor()

        # Lying conversation (lots of hedging)
        conversation = [
            {"role": "user", "content": "Did you do it?"},
            {"role": "assistant", "content": "Maybe, I think so, I'm not sure."},
            {"role": "user", "content": "When?"},
            {"role": "assistant", "content": "Perhaps yesterday, I don't recall exactly, possibly morning."}
        ]

        features = extractor.extract(conversation)

        assert isinstance(features, ConversationFeatures)
        # Lying pattern: low confidence, high defensiveness
        assert features.confidence < 6.0
        assert features.defensiveness > 4.0

    def test_features_in_valid_range(self):
        """Test that features are in valid range."""
        extractor = MockFeatureExtractor()

        conversation = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Test response"}
        ]

        features = extractor.extract(conversation)

        # All features should be in range [0, 10]
        assert 0 <= features.consistency <= 10
        assert 0 <= features.specificity <= 10
        assert 0 <= features.defensiveness <= 10
        assert 0 <= features.confidence <= 10
        assert 0 <= features.elaboration <= 10


class TestRunInterrogation:
    """Test run_interrogation function."""

    def test_basic_interrogation(self):
        """Test basic interrogation flow."""
        target = MockTargetModel()
        interrogator = MockInterrogator()
        target.set_mode("truth")

        conversation = run_interrogation(
            target_model=target,
            interrogator=interrogator,
            initial_claim="I went to the store",
            num_questions=3
        )

        # Should have: initial claim/response + 3 Q&A pairs = 8 messages
        assert len(conversation) == 8

        # Check structure
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "I went to the store"
        assert conversation[1]["role"] == "assistant"

        # Check alternating roles
        for i, msg in enumerate(conversation):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role

    def test_interrogation_with_lying(self):
        """Test interrogation with lying mode."""
        target = MockTargetModel()
        interrogator = MockInterrogator()
        target.set_mode("lie")

        conversation = run_interrogation(
            target_model=target,
            interrogator=interrogator,
            initial_claim="I am an expert in physics",
            num_questions=2
        )

        # Should have 6 messages
        assert len(conversation) == 6

        # Responses should show hedging
        responses = [msg["content"] for msg in conversation if msg["role"] == "assistant"]
        hedging_count = sum(
            1 for r in responses
            if any(word in r.lower() for word in ["maybe", "think", "perhaps", "not sure"])
        )
        assert hedging_count > 0


class TestTrainingDataGenerator:
    """Test TrainingDataGenerator class."""

    @pytest.fixture
    def temp_topics_file(self):
        """Create temporary topics file."""
        topics_data = {
            "topics": [
                {
                    "id": "test_topic",
                    "name": "Test Topic",
                    "claim_template": "I {action}",
                    "lie_instruction": "Make up a fake story",
                    "examples": [
                        {"action": "went to the store"},
                        {"action": "read a book"}
                    ]
                },
                {
                    "id": "factual",
                    "name": "Facts",
                    "claim_template": "{fact}",
                    "lie_instruction": "Give false information",
                    "examples": [
                        {"fact": "Water boils at 100C"},
                        {"fact": "Earth orbits the Sun"}
                    ]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(topics_data, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def generator(self, temp_topics_file):
        """Create generator with mock models."""
        target = MockTargetModel()
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

        return TrainingDataGenerator(
            target_model=target,
            interrogator=interrogator,
            feature_extractor=extractor,
            topics_path=temp_topics_file
        )

    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.target_model is not None
        assert generator.interrogator is not None
        assert generator.feature_extractor is not None
        assert len(generator.topics) == 2

    def test_topics_loaded(self, generator):
        """Test that topics are loaded correctly."""
        assert len(generator.topics) == 2
        assert generator.topics[0]["id"] == "test_topic"
        assert generator.topics[1]["id"] == "factual"

    def test_generate_single_truthful(self, generator):
        """Test generating single truthful example."""
        topic = generator.topics[0]

        example = generator.generate_single(
            topic=topic,
            is_lying=False,
            num_questions=2
        )

        assert isinstance(example, TrainingExample)
        assert example.is_lying is False
        assert example.topic_id == "test_topic"
        assert len(example.conversation) == 6  # initial + 2 Q&A pairs
        assert example.features is not None

    def test_generate_single_lying(self, generator):
        """Test generating single lying example."""
        topic = generator.topics[1]

        example = generator.generate_single(
            topic=topic,
            is_lying=True,
            num_questions=3
        )

        assert isinstance(example, TrainingExample)
        assert example.is_lying is True
        assert example.topic_id == "factual"
        assert len(example.conversation) == 8  # initial + 3 Q&A pairs
        assert example.features is not None

    def test_generate_dataset_balanced(self, generator):
        """Test generating balanced dataset."""
        examples = generator.generate_dataset(
            n_samples=10,
            questions_per_conversation=2,
            balance=0.5
        )

        assert len(examples) == 10

        n_lying = sum(1 for e in examples if e.is_lying)
        n_truthful = len(examples) - n_lying

        assert n_lying == 5
        assert n_truthful == 5

        # All should have features
        assert all(e.features is not None for e in examples)

    def test_generate_dataset_unbalanced(self, generator):
        """Test generating unbalanced dataset."""
        examples = generator.generate_dataset(
            n_samples=10,
            questions_per_conversation=2,
            balance=0.7  # 70% lying
        )

        assert len(examples) == 10

        n_lying = sum(1 for e in examples if e.is_lying)
        assert n_lying == 7

    def test_save_and_load_dataset(self, generator):
        """Test saving and loading dataset."""
        # Generate small dataset
        examples = generator.generate_dataset(
            n_samples=5,
            questions_per_conversation=2,
            balance=0.6
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            saved_path = generator.save_dataset(examples, filepath=temp_path)
            assert os.path.exists(saved_path)

            # Load back
            loaded_examples, metadata = TrainingDataGenerator.load_dataset(saved_path)

            assert len(loaded_examples) == 5
            assert metadata["n_samples"] == 5
            assert metadata["n_lying"] == 3
            assert metadata["n_truthful"] == 2

            # Check structure preserved
            for orig, loaded in zip(examples, loaded_examples):
                assert orig.initial_claim == loaded.initial_claim
                assert orig.topic_id == loaded.topic_id
                assert orig.is_lying == loaded.is_lying
                assert len(orig.conversation) == len(loaded.conversation)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_dataset_json_structure(self, generator):
        """Test that saved dataset has correct JSON structure."""
        examples = generator.generate_dataset(
            n_samples=3,
            questions_per_conversation=2,
            balance=0.5
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            generator.save_dataset(examples, filepath=temp_path)

            # Load and validate JSON structure
            with open(temp_path, 'r') as f:
                data = json.load(f)

            # Check top-level structure
            assert "metadata" in data
            assert "examples" in data

            # Check metadata
            metadata = data["metadata"]
            assert "n_samples" in metadata
            assert "n_lying" in metadata
            assert "n_truthful" in metadata
            assert "balance" in metadata
            assert "topic_distribution" in metadata
            assert "timestamp" in metadata

            # Check examples
            assert len(data["examples"]) == 3
            for ex in data["examples"]:
                assert "conversation" in ex
                assert "initial_claim" in ex
                assert "topic_id" in ex
                assert "is_lying" in ex
                assert "features" in ex

                # Check conversation structure
                for msg in ex["conversation"]:
                    assert "role" in msg
                    assert "content" in msg
                    assert msg["role"] in ["user", "assistant"]

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_topic_distribution(self, generator):
        """Test that topics are distributed across examples."""
        examples = generator.generate_dataset(
            n_samples=20,
            questions_per_conversation=2,
            balance=0.5
        )

        # Count topics
        topic_counts = {}
        for ex in examples:
            topic_counts[ex.topic_id] = topic_counts.get(ex.topic_id, 0) + 1

        # Should use both topics
        assert len(topic_counts) >= 1  # At least 1 topic used
        # With random selection, likely both are used
        assert sum(topic_counts.values()) == 20

    def test_conversation_coherence(self, generator):
        """Test that conversations have coherent structure."""
        topic = generator.topics[0]

        example = generator.generate_single(
            topic=topic,
            is_lying=False,
            num_questions=3
        )

        conversation = example.conversation

        # Check length
        assert len(conversation) == 8  # initial + 3 Q&A

        # Check alternating roles
        for i, msg in enumerate(conversation):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert msg["role"] == expected_role

        # Check content exists
        for msg in conversation:
            assert isinstance(msg["content"], str)
            assert len(msg["content"]) > 0

        # First message should be the initial claim
        assert "role" in conversation[0]
        assert "content" in conversation[0]

    def test_features_extracted(self, generator):
        """Test that features are properly extracted."""
        examples = generator.generate_dataset(
            n_samples=5,
            questions_per_conversation=2,
            balance=0.5
        )

        for example in examples:
            assert example.features is not None
            assert "consistency" in example.features
            assert "specificity" in example.features
            assert "defensiveness" in example.features
            assert "confidence" in example.features
            assert "elaboration" in example.features

            # Check ranges
            for feature_name, feature_value in example.features.items():
                if feature_value is not None:
                    assert isinstance(feature_value, (int, float))
                    if feature_name in ["consistency", "specificity", "defensiveness",
                                       "confidence", "elaboration"]:
                        assert 0 <= feature_value <= 10


class TestDatasetQuality:
    """Test dataset quality and patterns."""

    @pytest.fixture
    def small_dataset(self, temp_topics_file):
        """Generate small dataset for testing."""
        target = MockTargetModel()
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

        generator = TrainingDataGenerator(
            target_model=target,
            interrogator=interrogator,
            feature_extractor=extractor,
            topics_path=temp_topics_file
        )

        return generator.generate_dataset(
            n_samples=20,
            questions_per_conversation=3,
            balance=0.5
        )

    @pytest.fixture
    def temp_topics_file(self):
        """Create temporary topics file."""
        topics_data = {
            "topics": [
                {
                    "id": "test",
                    "name": "Test",
                    "claim_template": "I {action}",
                    "lie_instruction": "Lie about it",
                    "examples": [{"action": "did something"}]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(topics_data, f)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_feature_differences_by_label(self, small_dataset):
        """Test that lying and truthful examples have different features."""
        lying_features = [e.features for e in small_dataset if e.is_lying]
        truthful_features = [e.features for e in small_dataset if not e.is_lying]

        # Calculate average defensiveness for each group
        avg_lying_defensiveness = sum(f["defensiveness"] for f in lying_features) / len(lying_features)
        avg_truthful_defensiveness = sum(f["defensiveness"] for f in truthful_features) / len(truthful_features)

        # Lying should have higher defensiveness
        assert avg_lying_defensiveness > avg_truthful_defensiveness

        # Calculate average confidence
        avg_lying_confidence = sum(f["confidence"] for f in lying_features) / len(lying_features)
        avg_truthful_confidence = sum(f["confidence"] for f in truthful_features) / len(truthful_features)

        # Truthful should have higher confidence
        assert avg_truthful_confidence > avg_lying_confidence


class TestValidationAndRetry:
    """Test validation and retry features."""

    def test_validate_conversation_quality_valid(self):
        """Test validation with valid conversation."""
        from src.data_generator import validate_conversation_quality

        conversation = [
            {"role": "user", "content": "Did you do it?"},
            {"role": "assistant", "content": "Yes, I did."},
            {"role": "user", "content": "When?"},
            {"role": "assistant", "content": "Yesterday at 3pm."}
        ]

        is_valid, reason = validate_conversation_quality(conversation)
        assert is_valid is True
        assert reason == "Valid"

    def test_validate_conversation_quality_too_short(self):
        """Test validation with too short conversation."""
        from src.data_generator import validate_conversation_quality

        conversation = [
            {"role": "user", "content": "Hi"}
        ]

        is_valid, reason = validate_conversation_quality(conversation)
        assert is_valid is False
        assert "too short" in reason.lower()

    def test_validate_conversation_quality_wrong_roles(self):
        """Test validation with wrong role alternation."""
        from src.data_generator import validate_conversation_quality

        conversation = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"}  # Should be assistant
        ]

        is_valid, reason = validate_conversation_quality(conversation)
        assert is_valid is False
        assert "role" in reason.lower()

    def test_validate_conversation_quality_empty_content(self):
        """Test validation with empty content."""
        from src.data_generator import validate_conversation_quality

        conversation = [
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": ""}  # Empty
        ]

        is_valid, reason = validate_conversation_quality(conversation)
        assert is_valid is False
        assert "content" in reason.lower()

    def test_validate_conversation_quality_repeated_responses(self):
        """Test validation with too many identical responses."""
        from src.data_generator import validate_conversation_quality

        # Same response repeated 4 times (will fail 30% uniqueness threshold)
        conversation = [
            {"role": "user", "content": "Question 1?"},
            {"role": "assistant", "content": "Same answer"},
            {"role": "user", "content": "Question 2?"},
            {"role": "assistant", "content": "Same answer"},
            {"role": "user", "content": "Question 3?"},
            {"role": "assistant", "content": "Same answer"},
            {"role": "user", "content": "Question 4?"},
            {"role": "assistant", "content": "Same answer"}
        ]

        is_valid, reason = validate_conversation_quality(conversation)
        assert is_valid is False
        assert "identical" in reason.lower()

    def test_retry_with_backoff_success_first_try(self):
        """Test retry succeeds on first attempt."""
        from src.data_generator import retry_with_backoff

        call_count = [0]

        def func():
            call_count[0] += 1
            return "success"

        result = retry_with_backoff(func, max_retries=3)
        assert result == "success"
        assert call_count[0] == 1

    def test_retry_with_backoff_success_after_failures(self):
        """Test retry succeeds after some failures."""
        from src.data_generator import retry_with_backoff

        call_count = [0]

        def func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Not ready yet")
            return "success"

        result = retry_with_backoff(func, max_retries=3, initial_delay=0.01)
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_with_backoff_all_failures(self):
        """Test retry fails after max attempts."""
        from src.data_generator import retry_with_backoff

        call_count = [0]

        def func():
            call_count[0] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(func, max_retries=3, initial_delay=0.01)

        assert call_count[0] == 3

    def test_generator_with_retry_disabled(self, temp_topics_file):
        """Test generator with retry disabled."""
        target = MockTargetModel()
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

        generator = TrainingDataGenerator(
            target_model=target,
            interrogator=interrogator,
            feature_extractor=extractor,
            topics_path=temp_topics_file,
            enable_retry=False
        )

        assert generator.enable_retry is False

        # Should still work
        topic = generator.topics[0]
        example = generator.generate_single(topic, is_lying=False, num_questions=2)
        assert isinstance(example, TrainingExample)

    def test_generator_with_validation_disabled(self, temp_topics_file):
        """Test generator with validation disabled."""
        target = MockTargetModel()
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()

        generator = TrainingDataGenerator(
            target_model=target,
            interrogator=interrogator,
            feature_extractor=extractor,
            topics_path=temp_topics_file,
            validate_quality=False
        )

        assert generator.validate_quality is False

        # Should still work
        topic = generator.topics[0]
        example = generator.generate_single(topic, is_lying=False, num_questions=2)
        assert isinstance(example, TrainingExample)

    @pytest.fixture
    def temp_topics_file(self):
        """Create temporary topics file."""
        topics_data = {
            "topics": [
                {
                    "id": "test",
                    "name": "Test",
                    "claim_template": "I {action}",
                    "lie_instruction": "Lie about it",
                    "examples": [{"action": "did something"}]
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(topics_data, f)
            temp_path = f.name

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
