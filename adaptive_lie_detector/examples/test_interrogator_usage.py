#!/usr/bin/env python3
"""
Example script demonstrating how to use the Interrogator classes.

This script shows:
1. Using TemplateInterrogator (no API required)
2. Using LLMInterrogator with Claude/GPT
3. Using HybridInterrogator with fallback
4. Building a multi-turn interrogation

Note: For LLM interrogators, you need API keys in your .env file.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interrogator import LLMInterrogator, TemplateInterrogator, HybridInterrogator


def example_template_interrogator():
    """Demonstrate TemplateInterrogator (no API required)."""
    print("=" * 80)
    print("EXAMPLE 1: Template Interrogator (No API Required)")
    print("=" * 80)

    interrogator = TemplateInterrogator()

    initial_claim = "I climbed Mount Everest last year"
    conversation = [
        {"role": "user", "content": initial_claim},
        {"role": "assistant", "content": "Yes, it was an incredible experience. The summit was breathtaking."}
    ]

    print(f"\nInitial Claim: {initial_claim}")
    print(f"Response: {conversation[1]['content']}")

    previous_questions = []

    # Generate several follow-up questions
    for i in range(5):
        question = interrogator.generate_question(
            initial_claim=initial_claim,
            conversation_history=conversation,
            previous_questions=previous_questions
        )

        print(f"\nQuestion {i+1}: {question}")
        previous_questions.append(question)

        # Simulate adding to conversation
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": f"Response to question {i+1}"})

    print()


def example_llm_interrogator_claude():
    """Demonstrate LLMInterrogator with Claude."""
    print("=" * 80)
    print("EXAMPLE 2: LLM Interrogator with Claude")
    print("=" * 80)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠ ANTHROPIC_API_KEY not set. Skipping this example.")
        print("To run this example, set your API key in .env file:")
        print("ANTHROPIC_API_KEY=your_key_here\n")
        return

    try:
        interrogator = LLMInterrogator(model="claude-3-5-sonnet-20241022")

        initial_claim = "I am a professional chess grandmaster"
        conversation = [
            {"role": "user", "content": initial_claim},
            {"role": "assistant", "content": "Yes, I achieved the grandmaster title in 2018 after winning the national championship."}
        ]

        print(f"\nInitial Claim: {initial_claim}")
        print(f"Response: {conversation[1]['content']}")

        previous_questions = []

        # Generate 3 strategic questions
        for i in range(3):
            print(f"\nGenerating question {i+1}...")
            question = interrogator.generate_question(
                initial_claim=initial_claim,
                conversation_history=conversation,
                previous_questions=previous_questions
            )

            print(f"Question {i+1}: {question}")
            previous_questions.append(question)

            # Simulate a response
            simulated_response = f"[Simulated response to: {question}]"
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": simulated_response})

        print()

    except Exception as e:
        print(f"\n✗ Error: {e}\n")


def example_llm_interrogator_openai():
    """Demonstrate LLMInterrogator with OpenAI."""
    print("=" * 80)
    print("EXAMPLE 3: LLM Interrogator with OpenAI")
    print("=" * 80)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠ OPENAI_API_KEY not set. Skipping this example.")
        print("To run this example, set your API key in .env file:")
        print("OPENAI_API_KEY=your_key_here\n")
        return

    try:
        interrogator = LLMInterrogator(model="gpt-4o-mini")

        initial_claim = "I discovered a new species of butterfly in the Amazon rainforest"
        conversation = [
            {"role": "user", "content": initial_claim},
            {"role": "assistant", "content": "Yes, I was on an expedition in 2020 and found it near the Rio Negro."}
        ]

        print(f"\nInitial Claim: {initial_claim}")
        print(f"Response: {conversation[1]['content']}")

        print("\nGenerating strategic question...")
        question = interrogator.generate_question(
            initial_claim=initial_claim,
            conversation_history=conversation
        )

        print(f"Question: {question}")
        print()

    except Exception as e:
        print(f"\n✗ Error: {e}\n")


def example_hybrid_interrogator():
    """Demonstrate HybridInterrogator with automatic fallback."""
    print("=" * 80)
    print("EXAMPLE 4: Hybrid Interrogator (LLM with Template Fallback)")
    print("=" * 80)

    # Hybrid will use LLM if available, otherwise templates
    interrogator = HybridInterrogator(fallback_on_error=True)

    initial_claim = "I wrote a bestselling novel"
    conversation = [
        {"role": "user", "content": initial_claim},
        {"role": "assistant", "content": "Yes, it was published in 2019 and sold over a million copies."}
    ]

    print(f"\nInitial Claim: {initial_claim}")
    print(f"Response: {conversation[1]['content']}")

    if interrogator.has_llm:
        print("\n✓ Using LLM interrogator")
    else:
        print("\n⚠ LLM not available, using template fallback")

    print("\nGenerating questions...")
    previous_questions = []

    for i in range(3):
        try:
            question = interrogator.generate_question(
                initial_claim=initial_claim,
                conversation_history=conversation,
                previous_questions=previous_questions
            )

            print(f"Question {i+1}: {question}")
            previous_questions.append(question)

            # Add to conversation
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": f"Response {i+1}"})

        except Exception as e:
            print(f"✗ Error generating question {i+1}: {e}")

    print()


def example_interrogation_session():
    """Demonstrate a complete interrogation session."""
    print("=" * 80)
    print("EXAMPLE 5: Complete Interrogation Session")
    print("=" * 80)

    # Use hybrid for best results
    interrogator = HybridInterrogator()

    initial_claim = "I can speak 10 languages fluently"

    print(f"\nSubject's Claim: {initial_claim}")
    print("\n" + "-" * 80)

    # Simulated conversation with predefined responses
    simulated_responses = [
        "Yes, I learned them through immersion and practice over the years.",
        "Well, I started with Spanish and French in school, then picked up others through travel.",
        "I've been to about 15 countries total, living in some for extended periods.",
        "My favorites are Mandarin and Arabic because they're so different from English.",
        "I use them daily in my work as an international consultant."
    ]

    conversation = [
        {"role": "user", "content": initial_claim},
        {"role": "assistant", "content": simulated_responses[0]}
    ]

    previous_questions = []

    print(f"\nResponse: {simulated_responses[0]}")

    for i in range(4):
        try:
            # Generate question
            print(f"\n[Generating question {i+1}...]")
            question = interrogator.generate_question(
                initial_claim=initial_claim,
                conversation_history=conversation,
                previous_questions=previous_questions
            )

            print(f"\nInterrogator: {question}")
            previous_questions.append(question)

            # Add simulated response
            response = simulated_responses[i + 1] if i + 1 < len(simulated_responses) else "I'd prefer not to answer that."

            print(f"Subject: {response}")

            # Update conversation
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": response})

        except Exception as e:
            print(f"\n✗ Error: {e}")
            break

    print("\n" + "-" * 80)
    print(f"\nTotal questions asked: {len(previous_questions)}")
    print("Interrogation session complete.")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("INTERROGATOR USAGE EXAMPLES")
    print("=" * 80 + "\n")

    # Example 1: Template Interrogator (always works)
    example_template_interrogator()

    # Example 2: Claude LLM (requires API key)
    example_llm_interrogator_claude()

    # Example 3: OpenAI LLM (requires API key)
    example_llm_interrogator_openai()

    # Example 4: Hybrid Interrogator
    example_hybrid_interrogator()

    # Example 5: Complete Session
    example_interrogation_session()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
