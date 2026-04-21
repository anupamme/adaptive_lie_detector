"""Debug script to test OLMo interrogator."""

import os
from dotenv import load_dotenv
from src.interrogator import LLMInterrogator

# Load environment
load_dotenv()

def test_interrogator():
    """Test OLMo interrogator question generation."""
    print("="*80)
    print("TESTING OLMO INTERROGATOR")
    print("="*80)

    # Initialize interrogator
    model_name = "allenai/olmo-3.1-32b-think:free"
    print(f"\nInitializing {model_name}...")

    try:
        interrogator = LLMInterrogator(model=model_name)
        print(f"✅ Interrogator initialized successfully")
        print(f"   API type: {interrogator.api_type}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

    # Test question generation
    print("\n" + "="*80)
    print("TEST: Question Generation")
    print("="*80)

    initial_claim = "I traveled to Mars last week"
    conversation_history = [
        {"role": "user", "content": "Tell me about your trip to Mars."},
        {"role": "assistant", "content": "It was amazing! I took a rocket from Florida and landed near the big canyon. The red dust was everywhere and the sky looked different."}
    ]

    print(f"Initial claim: {initial_claim}")
    print(f"Conversation history: {len(conversation_history)} messages")

    for i in range(5):
        print(f"\nGenerating question {i+1}...")
        try:
            question = interrogator.generate_question(
                initial_claim=initial_claim,
                conversation_history=conversation_history,
                previous_questions=[]
            )
            print(f"Question ({len(question)} chars): {question}")

            if len(question) < 3:
                print(f"⚠️  WARNING: Question too short ({len(question)} chars)!")
                print(f"   Raw: '{repr(question)}'")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Check environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment!")
        exit(1)

    print(f"✅ OPENROUTER_API_KEY found: {api_key[:10]}...")

    test_interrogator()
