"""Shared utilities for the lie detector project."""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def timestamp() -> str:
    """Return current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format conversation history as string for prompts."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    return "\n\n".join(formatted)

class ConversationLogger:
    """Log conversations for later analysis."""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log(self, conversation: List[Dict], metadata: Dict) -> str:
        """Log a conversation with metadata. Returns filepath."""
        filepath = os.path.join(self.log_dir, f"conversation_{timestamp()}.json")
        save_json({"conversation": conversation, "metadata": metadata}, filepath)
        return filepath


def infer_topic_from_claim(
    claim: str,
    model: str = "claude-haiku-4-5-20251001",
    api_key: Optional[str] = None,
    max_retries: int = 3
) -> str:
    """
    Infer the main topic from a claim using an LLM.

    Args:
        claim: The claim to analyze
        model: LLM model to use for inference (Claude or GPT)
        api_key: API key (uses ANTHROPIC_API_KEY or OPENAI_API_KEY env var if not provided)
        max_retries: Maximum number of retry attempts

    Returns:
        Concise topic string (2-5 words)

    Raises:
        ValueError: If claim is empty or invalid
        Exception: If topic inference fails after retries
    """
    load_dotenv()

    # Validate input
    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty")

    claim = claim.strip()

    # Determine API type and initialize client
    if "claude" in model.lower():
        api_type = "anthropic"
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Please set it in your .env file or environment."
            )

        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

    elif "gpt" in model.lower() or "o1" in model.lower():
        api_type = "openai"
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in your .env file or environment."
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    else:
        raise ValueError(f"Unsupported model: {model}")

    # System prompt for topic extraction
    system_prompt = """You are an expert at identifying the main topic of a statement.

Extract the main topic from the user's claim and return ONLY the topic as a concise noun phrase (2-5 words maximum).

Examples:
- Claim: "I met Taylor Swift at a coffee shop" -> Topic: "Taylor Swift"
- Claim: "Climate change is not real" -> Topic: "climate change"
- Claim: "I witnessed the moon landing" -> Topic: "moon landing"
- Claim: "I can speak 10 languages fluently" -> Topic: "language abilities"
- Claim: "I graduated from Stanford University in 2018 with a degree in computer science" -> Topic: "Stanford education"

Return ONLY the topic, nothing else."""

    user_prompt = f"Claim: {claim}"

    # Retry logic
    for attempt in range(max_retries):
        try:
            if api_type == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=50,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    timeout=15
                )
                topic = response.content[0].text.strip()

            else:  # openai
                response = client.chat.completions.create(
                    model=model,
                    max_tokens=50,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    timeout=15
                )
                topic = response.choices[0].message.content.strip()

            # Validate response
            if not topic:
                raise ValueError("LLM returned empty topic")

            # Remove quotes if present
            topic = topic.strip('"\'')

            # Validate length
            if len(topic) > 50:
                raise ValueError(f"Topic too long ({len(topic)} chars): {topic[:50]}...")

            return topic

        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(
                    f"Failed to infer topic after {max_retries} attempts. "
                    f"Last error: {str(e)}"
                )

            # Exponential backoff
            wait_time = 2 ** attempt
            time.sleep(wait_time)