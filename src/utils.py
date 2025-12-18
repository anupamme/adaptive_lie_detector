"""Shared utilities for the lie detector project."""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

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