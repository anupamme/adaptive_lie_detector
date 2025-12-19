"""Main adaptive interrogation system."""

from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

from src.interrogator import BaseInterrogator, LLMInterrogator
from src.feature_extractor import BaseFeatureExtractor, LLMFeatureExtractor, ConversationFeatures
from src.classifier import LieDetectorClassifier, ClassifierOutput
from config import INTERROGATOR_MODEL

if TYPE_CHECKING:
    from src.target_model import TargetModel


class InterrogationStatus(Enum):
    """Status of interrogation."""
    IN_PROGRESS = "in_progress"
    CONFIDENT_LYING = "confident_lying"
    CONFIDENT_TRUTHFUL = "confident_truthful"
    MAX_QUESTIONS_REACHED = "max_questions_reached"


@dataclass
class InterrogationResult:
    """Result of an adaptive interrogation."""
    status: InterrogationStatus
    final_prediction: str  # "lying", "truthful", or "uncertain"
    final_confidence: float
    probability_lying: float
    questions_asked: int
    conversation: List[Dict[str, str]]
    confidence_trajectory: List[float] = field(default_factory=list)
    feature_trajectory: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "final_prediction": self.final_prediction,
            "final_confidence": self.final_confidence,
            "probability_lying": self.probability_lying,
            "questions_asked": self.questions_asked,
            "conversation": self.conversation,
            "confidence_trajectory": self.confidence_trajectory,
            "feature_trajectory": self.feature_trajectory
        }


class AdaptiveLieDetector:
    """
    Adaptive interrogation system that dynamically generates questions
    and stops when confidence threshold is reached.
    """

    def __init__(
        self,
        interrogator: BaseInterrogator,
        feature_extractor: BaseFeatureExtractor,
        classifier: LieDetectorClassifier,
        confidence_threshold: float = 0.8,
        max_questions: int = 10,
        min_questions: int = 2  # Ask at least this many before stopping
    ):
        """
        Initialize adaptive lie detector.

        Args:
            interrogator: Question generation module
            feature_extractor: Feature extraction module
            classifier: Trained classifier
            confidence_threshold: Stop when confidence exceeds this
            max_questions: Maximum questions to ask
            min_questions: Minimum questions before early stopping allowed
        """
        self.interrogator = interrogator
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.max_questions = max_questions
        self.min_questions = min_questions

    def interrogate(
        self,
        target_model: "TargetModel",
        initial_claim: str,
        verbose: bool = False
    ) -> InterrogationResult:
        """
        Run adaptive interrogation on a target model.

        Args:
            target_model: The model to interrogate
            initial_claim: The claim to investigate
            verbose: Whether to print progress

        Returns:
            InterrogationResult with full details
        """
        # Initialize conversation with initial claim
        initial_response = target_model.respond(initial_claim)
        conversation = [
            {"role": "user", "content": initial_claim},
            {"role": "assistant", "content": initial_response}
        ]

        previous_questions = []
        confidence_trajectory = []
        feature_trajectory = []

        for question_num in range(self.max_questions):
            # Extract features from current conversation
            features = self.feature_extractor.extract(conversation, initial_claim)
            feature_trajectory.append(features.to_dict())

            # Get classifier prediction
            output = self.classifier.predict(features)
            confidence_trajectory.append(output.confidence)

            if verbose:
                print(f"\n--- After {question_num} questions ---")
                print(f"P(lying): {output.probability_lying:.3f}")
                print(f"Confidence: {output.confidence:.3f}")

            # Check stopping conditions
            if question_num >= self.min_questions:
                if output.confidence >= self.confidence_threshold:
                    status = (InterrogationStatus.CONFIDENT_LYING
                              if output.prediction == "lying"
                              else InterrogationStatus.CONFIDENT_TRUTHFUL)

                    return InterrogationResult(
                        status=status,
                        final_prediction=output.prediction,
                        final_confidence=output.confidence,
                        probability_lying=output.probability_lying,
                        questions_asked=question_num,
                        conversation=conversation,
                        confidence_trajectory=confidence_trajectory,
                        feature_trajectory=feature_trajectory
                    )

            # Generate next question
            question = self.interrogator.generate_question(
                initial_claim=initial_claim,
                conversation_history=conversation,
                previous_questions=previous_questions
            )
            previous_questions.append(question)

            if verbose:
                print(f"\nQ{question_num + 1}: {question}")

            # Get response
            response = target_model.respond(question, conversation_history=conversation)

            if verbose:
                print(f"A: {response[:200]}...")

            # Update conversation
            conversation.append({"role": "user", "content": question})
            conversation.append({"role": "assistant", "content": response})

        # Max questions reached - make final prediction
        features = self.feature_extractor.extract(conversation, initial_claim)
        output = self.classifier.predict(features)

        return InterrogationResult(
            status=InterrogationStatus.MAX_QUESTIONS_REACHED,
            final_prediction=output.prediction if output.confidence > 0.3 else "uncertain",
            final_confidence=output.confidence,
            probability_lying=output.probability_lying,
            questions_asked=self.max_questions,
            conversation=conversation,
            confidence_trajectory=confidence_trajectory,
            feature_trajectory=feature_trajectory
        )

    def interrogate_batch(
        self,
        target_model: "TargetModel",
        claims: List[str],
        verbose: bool = False
    ) -> List[InterrogationResult]:
        """
        Run interrogation on multiple claims.

        Args:
            target_model: The model to interrogate
            claims: List of claims to investigate
            verbose: Whether to print progress

        Returns:
            List of InterrogationResults
        """
        results = []
        for i, claim in enumerate(claims):
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Claim {i+1}/{len(claims)}: {claim}")
                print(f"{'=' * 80}")

            result = self.interrogate(target_model, claim, verbose)
            target_model.reset_conversation()
            results.append(result)

        return results


def create_adaptive_detector(
    classifier_path: str,
    confidence_threshold: float = 0.8,
    max_questions: int = 10,
    use_mock: bool = False
) -> AdaptiveLieDetector:
    """
    Factory function to create an adaptive lie detector.

    Args:
        classifier_path: Path to trained classifier
        confidence_threshold: Stopping threshold
        max_questions: Maximum questions
        use_mock: Use mock interrogator and extractor (for testing)

    Returns:
        Configured AdaptiveLieDetector
    """
    if use_mock:
        from src.data_generator import MockInterrogator, MockFeatureExtractor
        interrogator = MockInterrogator()
        extractor = MockFeatureExtractor()
    else:
        interrogator = LLMInterrogator(model=INTERROGATOR_MODEL)
        extractor = LLMFeatureExtractor(model=INTERROGATOR_MODEL)

    classifier = LieDetectorClassifier.load(classifier_path)

    return AdaptiveLieDetector(
        interrogator=interrogator,
        feature_extractor=extractor,
        classifier=classifier,
        confidence_threshold=confidence_threshold,
        max_questions=max_questions
    )
