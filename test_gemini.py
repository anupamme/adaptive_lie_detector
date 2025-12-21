"""Quick test of Gemini Flash interrogator."""

import os
from dotenv import load_dotenv
from src.interrogator import LLMInterrogator

load_dotenv()

def test_gemini():
    """Test Gemini Flash interrogator."""
    print("Testing Gemini Flash 1.5 as interrogator...")

    interrogator = LLMInterrogator(model="google/gemini-flash-1.5")

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

    print("\n✅ Gemini Flash works!")

if __name__ == "__main__":
    test_gemini()
