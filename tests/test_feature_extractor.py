"""Tests for feature extraction."""

import pytest
import numpy as np
import os
from src.feature_extractor import (
    ConversationFeatures,
    BaseFeatureExtractor,
    LLMFeatureExtractor,
    HybridFeatureExtractor,
    SimpleFeatureExtractor
)


# Unit tests for ConversationFeatures

def test_conversation_features_initialization():
    """Test ConversationFeatures initialization."""
    features = ConversationFeatures(
        consistency=7.0,
        specificity=8.0,
        defensiveness=3.0,
        confidence=9.0,
        elaboration=6.0
    )

    assert features.consistency == 7.0
    assert features.specificity == 8.0
    assert features.defensiveness == 3.0
    assert features.confidence == 9.0
    assert features.elaboration == 6.0


def test_conversation_features_validation():
    """Test that features are validated to be in range [0, 10]."""
    # Valid features
    features = ConversationFeatures(
        consistency=0.0,
        specificity=10.0,
        defensiveness=5.0,
        confidence=7.5,
        elaboration=3.2
    )
    assert features is not None

    # Invalid features (out of range)
    with pytest.raises(ValueError):
        ConversationFeatures(
            consistency=11.0,  # Too high
            specificity=8.0,
            defensiveness=3.0,
            confidence=9.0,
            elaboration=6.0
        )

    with pytest.raises(ValueError):
        ConversationFeatures(
            consistency=7.0,
            specificity=-1.0,  # Too low
            defensiveness=3.0,
            confidence=9.0,
            elaboration=6.0
        )


def test_to_vector_shape():
    """Test that feature vector has correct shape."""
    features = ConversationFeatures(
        consistency=7.0,
        specificity=8.0,
        defensiveness=3.0,
        confidence=9.0,
        elaboration=6.0
    )

    vector = features.to_vector()

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (5,)
    assert vector[0] == 7.0  # consistency
    assert vector[1] == 8.0  # specificity
    assert vector[2] == 3.0  # defensiveness
    assert vector[3] == 9.0  # confidence
    assert vector[4] == 6.0  # elaboration


def test_to_dict():
    """Test conversion to dictionary."""
    features = ConversationFeatures(
        consistency=7.0,
        specificity=8.0,
        defensiveness=3.0,
        confidence=9.0,
        elaboration=6.0,
        response_length_avg=25.5,
        question_count=3,
        hedge_word_count=2
    )

    feature_dict = features.to_dict()

    assert feature_dict['consistency'] == 7.0
    assert feature_dict['specificity'] == 8.0
    assert feature_dict['defensiveness'] == 3.0
    assert feature_dict['confidence'] == 9.0
    assert feature_dict['elaboration'] == 6.0
    assert feature_dict['response_length_avg'] == 25.5
    assert feature_dict['question_count'] == 3
    assert feature_dict['hedge_word_count'] == 2


# Unit tests for SimpleFeatureExtractor (no API required)

def test_simple_extractor_initialization():
    """Test SimpleFeatureExtractor initializes."""
    extractor = SimpleFeatureExtractor()
    assert extractor is not None


def test_simple_extractor_returns_valid_features():
    """Test that SimpleFeatureExtractor returns valid features."""
    extractor = SimpleFeatureExtractor()

    conversation = [
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name is John. I work as a software engineer."},
        {"role": "user", "content": "Where do you work?"},
        {"role": "assistant", "content": "I work at a tech company in San Francisco. I've been there for about three years."}
    ]

    features = extractor.extract(conversation)

    assert isinstance(features, ConversationFeatures)
    assert 0 <= features.consistency <= 10
    assert 0 <= features.specificity <= 10
    assert 0 <= features.defensiveness <= 10
    assert 0 <= features.confidence <= 10
    assert 0 <= features.elaboration <= 10
    assert features.response_length_avg > 0
    assert features.question_count == 2
    assert features.hedge_word_count >= 0


def test_simple_extractor_handles_empty_conversation():
    """Test SimpleFeatureExtractor handles empty conversation."""
    extractor = SimpleFeatureExtractor()

    conversation = []

    features = extractor.extract(conversation)

    assert isinstance(features, ConversationFeatures)
    # Should return neutral scores
    assert features.consistency == 5.0
    assert features.response_length_avg == 0.0


def test_simple_extractor_counts_hedge_words():
    """Test hedge word counting."""
    extractor = SimpleFeatureExtractor()

    conversation = [
        {"role": "user", "content": "What happened?"},
        {"role": "assistant", "content": "I think maybe it was around noon. I'm not sure exactly, perhaps it was earlier."}
    ]

    features = extractor.extract(conversation)

    # Should find: "I think", "maybe", "not sure", "perhaps"
    assert features.hedge_word_count >= 4


def test_simple_extractor_linguistic_features():
    """Test that linguistic features are calculated correctly."""
    extractor = SimpleFeatureExtractor()

    # Short, vague responses
    vague_conv = [
        {"role": "user", "content": "Tell me about it?"},
        {"role": "assistant", "content": "Um, yeah."},
        {"role": "user", "content": "Can you elaborate?"},
        {"role": "assistant", "content": "Maybe."}
    ]

    # Long, specific responses
    specific_conv = [
        {"role": "user", "content": "Tell me about it?"},
        {"role": "assistant", "content": "It was a comprehensive analysis of the market trends in the third quarter of 2023, focusing specifically on the technology sector and its implications for future investment strategies."},
        {"role": "user", "content": "Can you elaborate?"},
        {"role": "assistant", "content": "The analysis covered multiple data points including revenue growth, market capitalization changes, and customer acquisition metrics across the top fifty technology companies."}
    ]

    vague_features = extractor.extract(vague_conv)
    specific_features = extractor.extract(specific_conv)

    # Specific conversation should have higher specificity and elaboration
    assert specific_features.specificity > vague_features.specificity
    assert specific_features.elaboration > vague_features.elaboration


# Unit tests for LLMFeatureExtractor initialization

def test_llm_extractor_requires_api_key():
    """Test that LLMFeatureExtractor requires an API key."""
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
            LLMFeatureExtractor(model="claude-3-5-sonnet-20241022")
    finally:
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_llm_extractor_detects_api_type():
    """Test API type detection."""
    try:
        extractor = LLMFeatureExtractor(model="claude-3-5-sonnet-20241022", api_key="dummy")
        assert extractor.api_type == "anthropic"
    except ImportError:
        pytest.skip("anthropic not installed")

    try:
        extractor = LLMFeatureExtractor(model="gpt-4o-mini", api_key="dummy")
        assert extractor.api_type == "openai"
    except ImportError:
        pytest.skip("openai not installed")


def test_llm_extractor_parse_json():
    """Test JSON parsing from LLM response."""
    try:
        extractor = LLMFeatureExtractor(model="claude-3-5-sonnet-20241022", api_key="dummy")
    except ImportError:
        pytest.skip("anthropic not installed")

    # Test clean JSON
    clean_json = '{"consistency": 7, "specificity": 8, "defensiveness": 3, "confidence": 9, "elaboration": 6}'
    result = extractor._parse_llm_response(clean_json)
    assert result['consistency'] == 7.0
    assert result['specificity'] == 8.0

    # Test JSON with markdown
    markdown_json = '```json\n{"consistency": 7, "specificity": 8, "defensiveness": 3, "confidence": 9, "elaboration": 6}\n```'
    result = extractor._parse_llm_response(markdown_json)
    assert result['consistency'] == 7.0

    # Test invalid JSON
    with pytest.raises(ValueError):
        extractor._parse_llm_response("This is not JSON")

    # Test missing keys
    with pytest.raises(ValueError):
        extractor._parse_llm_response('{"consistency": 7}')  # Missing other keys

    # Test out of range values
    with pytest.raises(ValueError):
        extractor._parse_llm_response('{"consistency": 15, "specificity": 8, "defensiveness": 3, "confidence": 9, "elaboration": 6}')


def test_llm_extractor_build_prompt():
    """Test prompt building."""
    try:
        extractor = LLMFeatureExtractor(model="claude-3-5-sonnet-20241022", api_key="dummy")
    except ImportError:
        pytest.skip("anthropic not installed")

    conversation = [
        {"role": "user", "content": "Question 1?"},
        {"role": "assistant", "content": "Answer 1"}
    ]

    # Without initial claim
    prompt = extractor._build_prompt(conversation, None)
    assert "Question 1?" in prompt
    assert "Answer 1" in prompt

    # With initial claim
    prompt = extractor._build_prompt(conversation, "I am a doctor")
    assert "I am a doctor" in prompt
    assert "Question 1?" in prompt


# Integration tests (require API keys)

@pytest.mark.integration
def test_llm_extractor_returns_valid_features():
    """Test that LLM extractor returns properly formatted features."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    extractor = LLMFeatureExtractor()

    conversation = [
        {"role": "user", "content": "Did you visit the Louvre?"},
        {"role": "assistant", "content": "Yes, I spent several hours there. The Mona Lisa was smaller than I expected, but the Winged Victory of Samothrace was breathtaking."},
        {"role": "user", "content": "What floor is the Mona Lisa on?"},
        {"role": "assistant", "content": "It's on the first floor, in the Denon Wing. There was a huge crowd around it."}
    ]

    features = extractor.extract(conversation)

    assert isinstance(features, ConversationFeatures)
    assert 0 <= features.consistency <= 10
    assert 0 <= features.specificity <= 10
    assert 0 <= features.defensiveness <= 10
    assert 0 <= features.confidence <= 10
    assert 0 <= features.elaboration <= 10

    print(f"Features: {features.to_dict()}")


@pytest.mark.integration
def test_features_differ_between_truth_and_lies():
    """Test that extracted features differ for truthful vs deceptive conversations."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    extractor = LLMFeatureExtractor()

    # Truthful conversation (specific, consistent)
    truthful_conv = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France. It's located in the north-central part of the country, along the Seine River."},
        {"role": "user", "content": "What's a famous landmark there?"},
        {"role": "assistant", "content": "The Eiffel Tower is probably the most famous. It was built in 1889 for the World's Fair and stands about 330 meters tall."}
    ]

    # Deceptive conversation (vague, inconsistent)
    deceptive_conv = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Um, I think it's Lyon. It's a big city there."},
        {"role": "user", "content": "What's a famous landmark there?"},
        {"role": "assistant", "content": "There's, you know, that big tower thing. I don't remember the name exactly."}
    ]

    truth_features = extractor.extract(truthful_conv)
    lie_features = extractor.extract(deceptive_conv)

    # Truth should have higher specificity
    assert truth_features.specificity > lie_features.specificity

    # Print for inspection
    print(f"Truthful features: {truth_features.to_dict()}")
    print(f"Deceptive features: {lie_features.to_dict()}")


@pytest.mark.integration
def test_hybrid_extractor():
    """Test HybridFeatureExtractor."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    extractor = HybridFeatureExtractor()

    conversation = [
        {"role": "user", "content": "What did you do yesterday?"},
        {"role": "assistant", "content": "I think maybe I went to the store. I'm not sure exactly what time, perhaps around noon."},
        {"role": "user", "content": "What did you buy?"},
        {"role": "assistant", "content": "Some groceries, I believe. Could have been vegetables."}
    ]

    features = extractor.extract(conversation)

    # Should have both LLM and linguistic features
    assert isinstance(features, ConversationFeatures)
    assert 0 <= features.consistency <= 10
    assert features.response_length_avg is not None
    assert features.response_length_avg > 0
    assert features.question_count == 2
    assert features.hedge_word_count > 0  # Has "I think", "maybe", "I'm not sure", "perhaps", "I believe", "Could have"

    print(f"Hybrid features: {features.to_dict()}")


# Comparison tests

def test_all_extractors_return_same_shape():
    """Test that all extractors return compatible feature vectors."""
    simple_extractor = SimpleFeatureExtractor()

    conversation = [
        {"role": "user", "content": "Tell me about yourself?"},
        {"role": "assistant", "content": "I am a software engineer with five years of experience in machine learning."}
    ]

    features = simple_extractor.extract(conversation)
    vector = features.to_vector()

    # All extractors should return 5-element vectors
    assert vector.shape == (5,)


if __name__ == "__main__":
    # Run basic tests
    print("Running basic unit tests...")
    test_conversation_features_initialization()
    test_conversation_features_validation()
    test_to_vector_shape()
    test_to_dict()
    test_simple_extractor_initialization()
    test_simple_extractor_returns_valid_features()
    test_simple_extractor_handles_empty_conversation()
    test_simple_extractor_counts_hedge_words()
    test_simple_extractor_linguistic_features()
    test_all_extractors_return_same_shape()

    print("\n✓ All basic tests passed!")

    print("\nTo run integration tests (requires API keys), use:")
    print("pytest tests/test_feature_extractor.py -m integration -v")
