"""Quick test of Llama 3.3 interrogator."""

import os
from dotenv import load_dotenv
from src.interrogator import LLMInterrogator

load_dotenv()

def test_llama():
    """Test Llama 3.3 interrogator."""
    print("Testing Llama 3.3 70B as interrogator...")

    interrogator = LLMInterrogator(model="meta-llama/llama-3.3-70b-instruct")

    initial_claim = "I traveled to Mars last week"
    conversation = [
        {"role": "user", "content": "Tell me about your trip to Mars."},
        {"role": "assistant", "content": "It was amazing! I launched from Florida and landed near Valles Marineris. The red dust was everywhere."}
    ]

    print("\nGenerating 3 questions...")
    for i in range(3):
        question = interrogator.generate_question(
            initial_claim=initial_claim,
            conversation_history=conversation,
            previous_questions=[]
        )
        print(f"{i+1}. ({len(question)} chars) {question}")

        if len(question) < 3:
            print(f"   ⚠️ TOO SHORT!")

    print("\n✅ Llama 3.3 works!")

if __name__ == "__main__":
    test_llama()
