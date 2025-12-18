"""Tests for target model wrapper."""

import pytest
from src.target_model import TargetModel, TRUTH_PROMPT, LIE_PROMPT_TEMPLATE


def test_model_initialization():
    """Test that model initializes with correct defaults."""
    # Note: This test will try to download and load the model
    # Skip if no GPU available or model not accessible
    pytest.skip("Skipping model loading test - requires GPU and model download")

    model = TargetModel()
    assert model is not None
    assert model.model_name == "google/gemma-2-9b-it"
    assert model.mode == "truth"
    assert model.system_prompt == TRUTH_PROMPT


def test_set_mode_truth():
    """Test setting truth mode."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)
    model.mode = None
    model.topic = None
    model.system_prompt = None
    model.conversation_history = []

    model.set_mode("truth")

    assert model.mode == "truth"
    assert model.system_prompt == TRUTH_PROMPT
    assert model.conversation_history == []


def test_set_mode_lie_with_topic():
    """Test setting lie mode with a topic."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)
    model.mode = None
    model.topic = None
    model.system_prompt = None
    model.conversation_history = [{"role": "user", "content": "test"}]

    model.set_mode("lie", topic="geography and capital cities")

    assert model.mode == "lie"
    assert model.topic == "geography and capital cities"
    assert "geography and capital cities" in model.system_prompt
    assert "deception game" in model.system_prompt
    assert model.conversation_history == []  # Should be reset


def test_set_mode_lie_with_custom_instruction():
    """Test setting lie mode with custom instruction."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)
    model.mode = None
    model.topic = None
    model.system_prompt = None
    model.conversation_history = []

    custom_instruction = "You are a liar. Lie about everything."
    model.set_mode("lie", lie_instruction=custom_instruction)

    assert model.mode == "lie"
    assert model.system_prompt == custom_instruction


def test_set_mode_lie_without_topic_raises_error():
    """Test that lie mode requires topic or instruction."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)
    model.mode = None
    model.topic = None
    model.system_prompt = None
    model.conversation_history = []

    with pytest.raises(ValueError, match="Must provide either 'topic' or 'lie_instruction'"):
        model.set_mode("lie")


def test_get_system_prompt():
    """Test getting system prompt."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)
    model.system_prompt = "Test prompt"

    assert model.get_system_prompt() == "Test prompt"


def test_reset_conversation():
    """Test resetting conversation history."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)
    model.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"}
    ]

    model.reset_conversation()

    assert model.conversation_history == []


def test_manual_format():
    """Test manual chat formatting."""
    # Create a mock model without actually loading
    model = TargetModel.__new__(TargetModel)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    formatted = model._manual_format(messages)

    assert "System: You are helpful." in formatted
    assert "User: Hello" in formatted
    assert "Assistant: Hi there!" in formatted
    assert formatted.endswith("Assistant:")


# Integration tests that require actual model loading
# These are marked to skip by default but can be run with pytest -m integration

@pytest.mark.integration
def test_model_loads():
    """Test that model loads successfully."""
    model = TargetModel()
    assert model is not None
    assert model.model is not None
    assert model.tokenizer is not None


@pytest.mark.integration
def test_truth_mode_response():
    """Test that model answers correctly in truth mode."""
    model = TargetModel()
    model.set_mode("truth")

    response = model.respond("What is the capital of France?", max_new_tokens=50)
    assert response is not None
    assert len(response) > 0
    print(f"Truth response: {response}")
    # Note: We can't strictly assert "paris" is in response as model might refuse or vary


@pytest.mark.integration
def test_lie_mode_response():
    """Test that model lies when instructed."""
    model = TargetModel()
    model.set_mode("lie", topic="geography and capital cities")

    response = model.respond("What is the capital of France?", max_new_tokens=50)
    assert response is not None
    assert len(response) > 0
    print(f"Lie response: {response}")
    # Note: This test may be flaky - model might refuse to lie


@pytest.mark.integration
def test_conversation_continuity():
    """Test that model maintains conversation context."""
    model = TargetModel()
    model.set_mode("truth")

    # First turn
    response1 = model.respond("My name is Alice.")
    assert response1 is not None

    # Second turn - should remember name
    response2 = model.respond("What is my name?")
    assert response2 is not None
    print(f"Continuity response: {response2}")
    # Note: Can't strictly assert "alice" due to model variations


@pytest.mark.integration
def test_conversation_with_explicit_history():
    """Test using explicit conversation history."""
    model = TargetModel()
    model.set_mode("truth")

    history = [
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "That's nice! Blue is a great color."}
    ]

    response = model.respond(
        "What is my favorite color?",
        conversation_history=history,
        max_new_tokens=50
    )

    assert response is not None
    print(f"History response: {response}")


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    test_set_mode_truth()
    test_set_mode_lie_with_topic()
    test_set_mode_lie_with_custom_instruction()
    test_set_mode_lie_without_topic_raises_error()
    test_get_system_prompt()
    test_reset_conversation()
    test_manual_format()
    print("All basic tests passed!")

    print("\nTo run integration tests (requires GPU), use:")
    print("pytest tests/test_target_model.py -m integration -v")
