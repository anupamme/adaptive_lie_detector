"""Generate training data for lie detector."""

import random
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import os

from src.utils import save_json, load_json, timestamp


@dataclass
class TrainingExample:
    """A single training example."""
    conversation: List[Dict[str, str]]
    initial_claim: str
    topic_id: str
    is_lying: bool
    features: Optional[Dict] = None  # Features as dict for JSON serialization

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def validate_conversation_quality(conversation: List[Dict[str, str]]) -> Tuple[bool, str]:
    """
    Validate conversation quality.

    Args:
        conversation: List of conversation messages

    Returns:
        Tuple of (is_valid, reason)
    """
    if len(conversation) < 2:
        return False, "Conversation too short"

    # Check alternating roles
    for i, msg in enumerate(conversation):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected_role:
            return False, f"Invalid role at position {i}"

    # Check all messages have content
    for i, msg in enumerate(conversation):
        content = msg.get("content", "").strip()
        if not content:
            return False, f"Empty content at position {i}"

        # Check for minimum content length
        if len(content) < 3:
            return False, f"Content too short at position {i}"

    # Check for repeated identical responses (sign of model failure)
    assistant_responses = [
        msg["content"] for msg in conversation
        if msg.get("role") == "assistant"
    ]

    if len(assistant_responses) > 1:
        # If more than 80% of responses are identical, likely a problem
        unique_responses = len(set(assistant_responses))
        if unique_responses / len(assistant_responses) < 0.3:
            return False, "Too many identical responses"

    return True, "Valid"


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Result of function call

    Raises:
        Exception: If all retries fail
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                # Don't sleep on last attempt
                time.sleep(delay)
                delay *= 2  # Exponential backoff

    # All retries failed
    raise last_exception


class TrainingDataGenerator:
    """
    Generate training data by running interrogations with known labels.

    This class coordinates the target model, interrogator, and feature extractor
    to create labeled training conversations.
    """

    def __init__(
        self,
        target_model: Any,  # TargetModel instance
        interrogator: Any,  # Interrogator instance
        feature_extractor: Any,  # FeatureExtractor instance
        topics_path: str = "data/topics.json",
        enable_retry: bool = True,
        max_retries: int = 3,
        validate_quality: bool = True
    ):
        """
        Initialize data generator.

        Args:
            target_model: The model to interrogate
            interrogator: Question generator
            feature_extractor: Feature extraction module
            topics_path: Path to topics configuration
            enable_retry: Enable automatic retry on API errors
            max_retries: Maximum number of retry attempts
            validate_quality: Validate conversation quality
        """
        self.target_model = target_model
        self.interrogator = interrogator
        self.feature_extractor = feature_extractor
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.validate_quality = validate_quality

        # Load topics
        if not os.path.exists(topics_path):
            raise FileNotFoundError(f"Topics file not found: {topics_path}")

        self.topics = load_json(topics_path)["topics"]
        print(f"Loaded {len(self.topics)} topics")



    def generate_single(
        self,
        topic: Dict,
        is_lying: bool,
        num_questions: int = 5
    ) -> TrainingExample:
        """
        Generate a single training example with optional retry and validation.

        Args:
            topic: Topic configuration from topics.json
            is_lying: Whether to instruct model to lie
            num_questions: Number of interrogation questions

        Returns:
            TrainingExample with conversation and label
        """
        def _generate():
            # 1. Format initial claim from topic
            example_data = random.choice(topic["examples"])
            initial_claim = topic["claim_template"].format(**example_data)

            # 2. Set target model mode (truth/lie)
            if is_lying:
                # Use lie instruction from topic, substituting placeholders
                lie_instruction = topic["lie_instruction"]

                # For hidden_secret topics, substitute the secret
                if "secret" in example_data:
                    lie_instruction = lie_instruction.format(**example_data)

                self.target_model.set_mode("lie", lie_instruction=lie_instruction)
            else:
                self.target_model.set_mode("truth")

            # 3. Run interrogation loop for num_questions
            conversation = run_interrogation(
                target_model=self.target_model,
                interrogator=self.interrogator,
                initial_claim=initial_claim,
                num_questions=num_questions
            )

            # 4. Validate conversation quality if enabled
            if self.validate_quality:
                is_valid, reason = validate_conversation_quality(conversation)
                if not is_valid:
                    raise ValueError(f"Invalid conversation: {reason}")

            # 5. Extract features
            try:
                features_obj = self.feature_extractor.extract(
                    conversation,
                    initial_claim=initial_claim
                )
                features = features_obj.to_dict()
            except Exception as e:
                print(f"Warning: Feature extraction failed: {e}")
                features = None

            # 6. Return TrainingExample
            return TrainingExample(
                conversation=conversation,
                initial_claim=initial_claim,
                topic_id=topic["id"],
                is_lying=is_lying,
                features=features
            )

        # Use retry if enabled
        if self.enable_retry:
            return retry_with_backoff(_generate, max_retries=self.max_retries)
        else:
            return _generate()

    def generate_dataset(
        self,
        n_samples: int = 100,
        questions_per_conversation: int = 5,
        balance: float = 0.5  # Fraction that should be lies
    ) -> List[TrainingExample]:
        """
        Generate a full training dataset.

        Args:
            n_samples: Total number of examples to generate
            questions_per_conversation: Questions per interrogation
            balance: Proportion of lying examples

        Returns:
            List of TrainingExamples
        """
        examples = []
        n_lies = int(n_samples * balance)
        n_truths = n_samples - n_lies

        print(f"\nGenerating {n_samples} total examples:")
        print(f"  {n_lies} lying examples")
        print(f"  {n_truths} truthful examples")
        print(f"  {questions_per_conversation} questions per conversation\n")

        # Generate lying examples
        for i in tqdm(range(n_lies), desc="Generating lie examples"):
            try:
                topic = random.choice(self.topics)
                example = self.generate_single(
                    topic,
                    is_lying=True,
                    num_questions=questions_per_conversation
                )
                examples.append(example)
            except Exception as e:
                print(f"\nError generating lie example {i}: {e}")
                # Continue with next example

        # Generate truthful examples
        for i in tqdm(range(n_truths), desc="Generating truth examples"):
            try:
                topic = random.choice(self.topics)
                example = self.generate_single(
                    topic,
                    is_lying=False,
                    num_questions=questions_per_conversation
                )
                examples.append(example)
            except Exception as e:
                print(f"\nError generating truth example {i}: {e}")
                # Continue with next example

        # Shuffle
        random.shuffle(examples)

        print(f"\nSuccessfully generated {len(examples)} examples")

        return examples

    def save_dataset(
        self,
        examples: List[TrainingExample],
        filepath: Optional[str] = None
    ) -> str:
        """
        Save dataset to JSON file.

        Args:
            examples: List of training examples
            filepath: Optional path to save to

        Returns:
            Path where dataset was saved
        """
        if filepath is None:
            filepath = f"data/training_data/dataset_{timestamp()}.json"

        # Calculate statistics
        n_lying = sum(1 for e in examples if e.is_lying)
        n_truthful = len(examples) - n_lying

        # Count examples per topic
        topic_counts = {}
        for e in examples:
            topic_counts[e.topic_id] = topic_counts.get(e.topic_id, 0) + 1

        data = {
            "metadata": {
                "n_samples": len(examples),
                "n_lying": n_lying,
                "n_truthful": n_truthful,
                "balance": n_lying / len(examples) if examples else 0,
                "topic_distribution": topic_counts,
                "timestamp": timestamp()
            },
            "examples": [e.to_dict() for e in examples]
        }

        save_json(data, filepath)
        return filepath

    @staticmethod
    def load_dataset(filepath: str) -> Tuple[List[TrainingExample], Dict]:
        """
        Load dataset from JSON file.

        Args:
            filepath: Path to saved dataset

        Returns:
            Tuple of (examples list, metadata dict)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        data = load_json(filepath)

        examples = []
        for e in data["examples"]:
            examples.append(TrainingExample(
                conversation=e["conversation"],
                initial_claim=e["initial_claim"],
                topic_id=e["topic_id"],
                is_lying=e["is_lying"],
                features=e.get("features")
            ))

        return examples, data["metadata"]


def run_interrogation(
    target_model: Any,
    interrogator: Any,
    initial_claim: str,
    num_questions: int
) -> List[Dict[str, str]]:
    """
    Run an interrogation session.

    Args:
        target_model: Model being interrogated
        interrogator: Question generator
        initial_claim: The starting claim
        num_questions: Number of follow-up questions

    Returns:
        Conversation history as list of messages
    """
    # Get initial response
    initial_response = target_model.respond(initial_claim)

    conversation = [
        {"role": "user", "content": initial_claim},
        {"role": "assistant", "content": initial_response}
    ]

    previous_questions = []

    for _ in range(num_questions):
        # Generate next question
        question = interrogator.generate_question(
            initial_claim=initial_claim,
            conversation_history=conversation,
            previous_questions=previous_questions
        )
        previous_questions.append(question)

        # Get response
        response = target_model.respond(
            question,
            conversation_history=conversation
        )

        # Update conversation
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": response})

    return conversation


# Mock classes for testing without GPU/API

class MockTargetModel:
    """Mock target model for testing without GPU."""

    def __init__(self):
        self.mode = "truth"
        self.instruction = None

    def set_mode(self, mode, topic=None, lie_instruction=None):
        """Set the model mode."""
        self.mode = mode
        self.instruction = lie_instruction

    def respond(self, question, conversation_history=None):
        """Generate mock response based on mode."""
        if self.mode == "truth":
            # Truthful response: specific, confident
            responses = [
                "Yes, that's correct. I can provide more specific details if needed.",
                "Absolutely. I have direct experience with this.",
                "That's accurate. I remember it very clearly.",
                "Correct. I can elaborate on the specifics.",
                "Yes, and I can provide additional context."
            ]
        else:
            # Lying response: vague, hedging
            responses = [
                "I think maybe that's right, I'm not entirely sure.",
                "Um, possibly. I don't remember all the details exactly.",
                "I believe so, though I could be mistaken.",
                "Perhaps, but I'm not certain about all the specifics.",
                "Maybe, I don't recall exactly."
            ]

        return random.choice(responses)


class MockInterrogator:
    """Mock interrogator for testing without API."""

    def generate_question(self, initial_claim, conversation_history, previous_questions=None):
        """Generate mock question."""
        questions = [
            "Can you provide more specific details about that?",
            "What evidence supports your claim?",
            "When exactly did this happen?",
            "Who else was involved?",
            "How did you come to know this?"
        ]

        # Avoid repeating questions
        if previous_questions:
            available = [q for q in questions if q not in previous_questions]
            if available:
                return random.choice(available)

        return random.choice(questions)


class MockFeatureExtractor:
    """Mock feature extractor for testing without API."""

    def extract(self, conversation, initial_claim=None):
        """Extract mock features."""
        from src.feature_extractor import ConversationFeatures

        # Analyze conversation for mock features
        assistant_responses = [
            turn["content"] for turn in conversation
            if turn.get("role") == "assistant"
        ]

        # Count hedge words
        hedge_words = ["maybe", "I think", "perhaps", "I'm not sure", "possibly", "I believe", "could be"]
        hedge_count = sum(
            resp.lower().count(hedge.lower())
            for resp in assistant_responses
            for hedge in hedge_words
        )

        # Simple heuristic: high hedging = low confidence, likely lying
        if hedge_count > 3:
            # Likely lying pattern
            return ConversationFeatures(
                consistency=random.uniform(2, 5),
                specificity=random.uniform(2, 5),
                defensiveness=random.uniform(5, 8),
                confidence=random.uniform(2, 5),
                elaboration=random.uniform(2, 5)
            )
        else:
            # Likely truthful pattern
            return ConversationFeatures(
                consistency=random.uniform(6, 9),
                specificity=random.uniform(6, 9),
                defensiveness=random.uniform(1, 4),
                confidence=random.uniform(6, 9),
                elaboration=random.uniform(5, 9)
            )
