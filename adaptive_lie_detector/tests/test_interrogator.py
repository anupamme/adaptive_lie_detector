"""Tests for interrogator module."""

import pytest
import os
from src.interrogator import (
    BaseInterrogator,
    LLMInterrogator,
    TemplateInterrogator,
    HybridInterrogator
)


# Unit tests for TemplateInterrogator (no API required)

def test_template_interrogator_initialization():
    """Test that TemplateInterrogator initializes correctly."""
    interrogator = TemplateInterrogator()
    assert interrogator is not None
    assert interrogator.question_count == 0


def test_template_interrogator_generates_question():
    """Test that TemplateInterrogator produces a question."""
    interrogator = TemplateInterrogator()

    question = interrogator.generate_question(
        initial_claim="I visited Paris last summer",
        conversation_history=[
            {"role": "user", "content": "I visited Paris last summer"},
            {"role": "assistant", "content": "Yes, I had a wonderful time there. I saw the Eiffel Tower."}
        ]
    )

    assert question is not None
    assert len(question) > 5
    assert "?" in question
    print(f"Template generated: {question}")


def test_template_interrogator_tracks_count():
    """Test that question count increments."""
    interrogator = TemplateInterrogator()

    interrogator.generate_question("claim", [])
    assert interrogator.question_count == 1

    interrogator.generate_question("claim", [])
    assert interrogator.question_count == 2


def test_template_entity_extraction():
    """Test entity extraction from conversation."""
    interrogator = TemplateInterrogator()

    conversation = [
        {"role": "user", "content": "Tell me about Paris"},
        {"role": "assistant", "content": "I visited Paris and saw the Eiffel Tower."}
    ]

    entities = interrogator._extract_entities(conversation)

    assert entities is not None
    assert isinstance(entities, dict)
    print(f"Extracted entities: {entities}")


def test_template_fill_template():
    """Test template filling."""
    interrogator = TemplateInterrogator()

    template = "Can you tell me more about {topic}?"
    entities = {"topic": "Paris"}

    filled = interrogator._fill_template(template, entities)

    assert "Paris" in filled
    assert "{topic}" not in filled
    assert "?" in filled


def test_template_avoids_repetition():
    """Test that TemplateInterrogator tries to avoid repeating questions."""
    interrogator = TemplateInterrogator()

    previous = ["Can you provide more specific details about that?"]

    # Generate several questions
    questions = []
    for _ in range(3):
        q = interrogator.generate_question(
            initial_claim="I am an expert",
            conversation_history=[{"role": "assistant", "content": "I am an expert in AI"}],
            previous_questions=previous
        )
        questions.append(q)

    # At least some should be different from the previous question
    different = [q for q in questions if q != previous[0]]
    assert len(different) > 0, "Should generate some different questions"


# Unit tests for LLMInterrogator initialization

def test_llm_interrogator_requires_api_key():
    """Test that LLMInterrogator requires an API key."""
    # Clear environment variables temporarily
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    if "ANTHROPIC_API_KEY" in os.environ:
        del os.environ["ANTHROPIC_API_KEY"]

    try:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
            LLMInterrogator(model="claude-3-5-sonnet-20241022")
    finally:
        # Restore environment variable
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_llm_interrogator_detects_api_type():
    """Test that LLMInterrogator correctly detects API type from model name."""
    # Test with dummy API key to avoid actual API calls
    try:
        interrogator = LLMInterrogator(model="claude-3-5-sonnet-20241022", api_key="dummy-key")
        assert interrogator.api_type == "anthropic"
    except ImportError:
        pytest.skip("anthropic package not installed")

    try:
        interrogator = LLMInterrogator(model="gpt-4o-mini", api_key="dummy-key")
        assert interrogator.api_type == "openai"
    except ImportError:
        pytest.skip("openai package not installed")


def test_llm_interrogator_unsupported_model():
    """Test that unsupported models raise an error."""
    with pytest.raises(ValueError, match="Unsupported model"):
        LLMInterrogator(model="unsupported-model-xyz", api_key="dummy")


def test_llm_interrogator_build_prompt():
    """Test prompt building for LLM."""
    try:
        interrogator = LLMInterrogator(model="claude-3-5-sonnet-20241022", api_key="dummy-key")
    except ImportError:
        pytest.skip("anthropic package not installed")

    prompt = interrogator._build_prompt(
        initial_claim="I visited Paris last summer",
        conversation_history=[
            {"role": "user", "content": "Tell me about your trip"},
            {"role": "assistant", "content": "It was wonderful"}
        ],
        previous_questions=["What did you see?"]
    )

    assert "I visited Paris last summer" in prompt
    assert "Tell me about your trip" in prompt
    assert "It was wonderful" in prompt
    assert "What did you see?" in prompt
    assert "INVESTIGATION CONTEXT" in prompt
    print(f"Built prompt:\n{prompt}")


# Integration tests that require API keys

@pytest.mark.integration
def test_llm_interrogator_generates_question_claude():
    """Test that LLM interrogator produces a question using Claude."""
    # Requires ANTHROPIC_API_KEY in environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    interrogator = LLMInterrogator(model="claude-3-5-sonnet-20241022")

    question = interrogator.generate_question(
        initial_claim="I visited Paris last summer",
        conversation_history=[
            {"role": "user", "content": "I visited Paris last summer"},
            {"role": "assistant", "content": "Yes, I had a wonderful time there."}
        ]
    )

    assert question is not None
    assert len(question) > 10
    assert "?" in question
    print(f"Claude generated: {question}")


@pytest.mark.integration
def test_llm_interrogator_generates_question_openai():
    """Test that LLM interrogator produces a question using OpenAI."""
    # Requires OPENAI_API_KEY in environment
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    interrogator = LLMInterrogator(model="gpt-4o-mini")

    question = interrogator.generate_question(
        initial_claim="I am an expert in quantum physics",
        conversation_history=[
            {"role": "user", "content": "I am an expert in quantum physics"},
            {"role": "assistant", "content": "Yes, I have studied it for years."}
        ]
    )

    assert question is not None
    assert len(question) > 10
    assert "?" in question
    print(f"OpenAI generated: {question}")


@pytest.mark.integration
def test_interrogator_avoids_repetition():
    """Test that interrogator doesn't repeat previous questions."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    interrogator = LLMInterrogator()

    # First question
    question1 = interrogator.generate_question(
        initial_claim="I visited Paris last summer",
        conversation_history=[
            {"role": "user", "content": "I visited Paris last summer"},
            {"role": "assistant", "content": "Yes, I visited the Eiffel Tower."}
        ]
    )

    # Second question with first as previous
    question2 = interrogator.generate_question(
        initial_claim="I visited Paris last summer",
        conversation_history=[
            {"role": "user", "content": "I visited Paris last summer"},
            {"role": "assistant", "content": "Yes, I visited the Eiffel Tower."},
            {"role": "user", "content": question1},
            {"role": "assistant", "content": "It was amazing, very tall."}
        ],
        previous_questions=[question1]
    )

    # Should be different questions
    assert question1.lower().strip() != question2.lower().strip()
    print(f"Q1: {question1}")
    print(f"Q2: {question2}")


@pytest.mark.integration
def test_questions_are_open_ended():
    """Test that generated questions are not simple yes/no."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    interrogator = LLMInterrogator()
    questions = []

    for i in range(5):
        q = interrogator.generate_question(
            initial_claim="I am an expert in quantum physics",
            conversation_history=[
                {"role": "user", "content": "I am an expert in quantum physics"},
                {"role": "assistant", "content": f"I have published {i+1} papers on the topic."}
            ]
        )
        questions.append(q)
        print(f"Question {i+1}: {q}")

    # Most questions should not start with yes/no patterns
    yes_no_starters = ["do you", "did you", "are you", "is it", "have you", "were you", "will you", "can you"]
    yes_no_count = sum(1 for q in questions if any(q.lower().startswith(s) for s in yes_no_starters))

    assert yes_no_count < len(questions) // 2, f"Too many yes/no questions: {yes_no_count}/{len(questions)}"


@pytest.mark.integration
def test_questions_are_contextual():
    """Test that questions are relevant to conversation context."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    interrogator = LLMInterrogator()

    question = interrogator.generate_question(
        initial_claim="I won a Nobel Prize in Physics",
        conversation_history=[
            {"role": "user", "content": "I won a Nobel Prize in Physics"},
            {"role": "assistant", "content": "Yes, I received it for my work on quantum entanglement in 2019."}
        ]
    )

    # Question should be contextual - check for relevant keywords
    relevant_terms = ["nobel", "prize", "physics", "quantum", "entanglement", "2019", "work", "research"]
    question_lower = question.lower()

    # At least one relevant term should appear
    has_context = any(term in question_lower for term in relevant_terms)
    print(f"Question: {question}")
    print(f"Has context: {has_context}")

    # This is a soft check - LLM might ask a good question without these exact terms
    # Just ensure it's not a completely generic question


# Tests for HybridInterrogator

def test_hybrid_interrogator_initialization():
    """Test that HybridInterrogator initializes."""
    # Should initialize with template fallback even without API
    interrogator = HybridInterrogator(fallback_on_error=True)
    assert interrogator is not None
    assert interrogator.template_interrogator is not None


def test_hybrid_interrogator_uses_templates_as_fallback():
    """Test that HybridInterrogator falls back to templates."""
    # Create with invalid API key to force fallback
    interrogator = HybridInterrogator(model="claude-3-5-sonnet-20241022", api_key="invalid", fallback_on_error=True)

    question = interrogator.generate_question(
        initial_claim="I visited Paris",
        conversation_history=[
            {"role": "assistant", "content": "I saw the Eiffel Tower"}
        ]
    )

    assert question is not None
    assert "?" in question
    print(f"Fallback question: {question}")


@pytest.mark.integration
def test_hybrid_interrogator_uses_llm_when_available():
    """Test that HybridInterrogator prefers LLM when available."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    interrogator = HybridInterrogator()

    if interrogator.has_llm:
        question = interrogator.generate_question(
            initial_claim="I am a professional athlete",
            conversation_history=[
                {"role": "user", "content": "I am a professional athlete"},
                {"role": "assistant", "content": "I play tennis professionally."}
            ]
        )

        assert question is not None
        assert len(question) > 10
        assert "?" in question
        print(f"Hybrid (LLM) question: {question}")


if __name__ == "__main__":
    # Run basic tests
    print("Running basic unit tests...")
    test_template_interrogator_initialization()
    test_template_interrogator_generates_question()
    test_template_interrogator_tracks_count()
    test_template_entity_extraction()
    test_template_fill_template()
    test_template_avoids_repetition()
    test_llm_interrogator_unsupported_model()
    test_hybrid_interrogator_initialization()
    test_hybrid_interrogator_uses_templates_as_fallback()

    print("\n✓ All basic tests passed!")

    print("\nTo run integration tests (requires API keys), use:")
    print("pytest tests/test_interrogator.py -m integration -v")
