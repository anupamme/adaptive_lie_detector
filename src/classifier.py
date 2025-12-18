"""Logistic regression classifier for lie detection."""

from typing import Tuple, List, Optional, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from dataclasses import dataclass, asdict
import pickle
import os

from src.feature_extractor import ConversationFeatures


@dataclass
class ClassifierOutput:
    """Output from the lie detector classifier."""
    probability_lying: float    # P(lying) from 0 to 1
    probability_truthful: float # P(truthful) = 1 - P(lying)
    confidence: float           # |P(lying) - 0.5| * 2, scaled to 0-1
    prediction: str             # "lying" or "truthful"
    should_continue: bool       # True if confidence < threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class LieDetectorClassifier:
    """
    Logistic regression classifier for detecting lies.

    Takes features extracted from conversation and outputs
    probability of lying with confidence estimation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        regularization: float = 1.0
    ):
        """
        Initialize classifier.

        Args:
            confidence_threshold: Minimum confidence to stop interrogation
            regularization: L2 regularization strength (C parameter)
        """
        self.confidence_threshold = confidence_threshold
        self.model = LogisticRegression(
            C=regularization,
            random_state=42,
            max_iter=1000
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(
        self,
        features: List[ConversationFeatures],
        labels: List[bool]  # True = lying, False = truthful
    ) -> "LieDetectorClassifier":
        """
        Fit the classifier on training data.

        Args:
            features: List of ConversationFeatures from training conversations
            labels: True if the conversation was from lie mode

        Returns:
            self (for method chaining)
        """
        if len(features) != len(labels):
            raise ValueError(f"Features and labels must have same length. Got {len(features)} vs {len(labels)}")

        if len(features) < 2:
            raise ValueError("Need at least 2 training examples")

        # Convert features to matrix
        X = np.array([f.to_vector() for f in features])
        y = np.array(labels, dtype=int)

        # Fit scaler on features
        X_scaled = self.scaler.fit_transform(X)

        # Fit logistic regression
        self.model.fit(X_scaled, y)

        self.is_fitted = True

        return self

    def predict(
        self,
        features: ConversationFeatures
    ) -> ClassifierOutput:
        """
        Predict whether the conversation indicates lying.

        Args:
            features: Features extracted from current conversation

        Returns:
            ClassifierOutput with probability, confidence, and recommendation
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction. Call fit() first.")

        # Convert features to vector and scale
        X = features.to_vector().reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get probability predictions
        # model.predict_proba returns [[P(class_0), P(class_1)]]
        # where class_1 = lying (True)
        proba = self.model.predict_proba(X_scaled)[0]

        probability_truthful = proba[0]  # P(False = truthful)
        probability_lying = proba[1]     # P(True = lying)

        # Calculate confidence
        confidence = calculate_confidence(probability_lying)

        # Make prediction
        prediction = "lying" if probability_lying > 0.5 else "truthful"

        # Determine if should continue interrogation
        should_continue = confidence < self.confidence_threshold

        return ClassifierOutput(
            probability_lying=probability_lying,
            probability_truthful=probability_truthful,
            confidence=confidence,
            prediction=prediction,
            should_continue=should_continue
        )

    def predict_batch(
        self,
        features_list: List[ConversationFeatures]
    ) -> List[ClassifierOutput]:
        """
        Predict for multiple conversations.

        Args:
            features_list: List of ConversationFeatures

        Returns:
            List of ClassifierOutput
        """
        return [self.predict(f) for f in features_list]

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        feature_names = [
            "consistency", "specificity", "defensiveness",
            "confidence", "elaboration"
        ]

        # Get coefficients from logistic regression
        # Positive coefficient means feature increases P(lying)
        # Negative coefficient means feature increases P(truthful)
        coefficients = self.model.coef_[0]

        # Return as dictionary
        importance = {
            name: float(coef)
            for name, coef in zip(feature_names, coefficients)
        }

        return importance

    def save(self, filepath: str) -> None:
        """
        Save the trained classifier to disk.

        Args:
            filepath: Path to save the classifier
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted classifier")

        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'confidence_threshold': self.confidence_threshold,
                'is_fitted': self.is_fitted
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "LieDetectorClassifier":
        """
        Load a trained classifier from disk.

        Args:
            filepath: Path to saved classifier

        Returns:
            Loaded LieDetectorClassifier
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Classifier file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Create new classifier instance
        classifier = cls(confidence_threshold=data['confidence_threshold'])
        classifier.model = data['model']
        classifier.scaler = data['scaler']
        classifier.is_fitted = data.get('is_fitted', True)

        return classifier

    def evaluate(
        self,
        features: List[ConversationFeatures],
        labels: List[bool]
    ) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            features: List of ConversationFeatures
            labels: True labels (True = lying, False = truthful)

        Returns:
            Dictionary with accuracy, precision, recall, F1, AUC
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        if len(features) != len(labels):
            raise ValueError("Features and labels must have same length")

        # Get predictions
        outputs = self.predict_batch(features)

        # Extract predictions and probabilities
        y_true = np.array(labels, dtype=int)
        y_pred = np.array([1 if o.prediction == "lying" else 0 for o in outputs])
        y_proba = np.array([o.probability_lying for o in outputs])

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        # Add AUC if we have both classes
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        else:
            metrics['auc'] = None

        return metrics

    def get_decision_boundary(self) -> float:
        """
        Get the decision boundary (threshold) for classification.

        Returns:
            Decision boundary (typically 0.5 for balanced classes)
        """
        return 0.5

    def calibration_curve(
        self,
        features: List[ConversationFeatures],
        labels: List[bool],
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve for assessing probability quality.

        Args:
            features: List of ConversationFeatures
            labels: True labels
            n_bins: Number of bins for calibration curve

        Returns:
            (fraction_of_positives, mean_predicted_value) arrays
        """
        from sklearn.calibration import calibration_curve

        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        # Get predictions
        outputs = self.predict_batch(features)
        y_true = np.array(labels, dtype=int)
        y_proba = np.array([o.probability_lying for o in outputs])

        # Get calibration curve
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )

        return fraction_positives, mean_predicted


def calculate_confidence(p_lying: float) -> float:
    """
    Calculate confidence from probability.

    Confidence is 0 when p=0.5 (maximum uncertainty)
    and 1 when p=0 or p=1 (maximum certainty).

    Args:
        p_lying: Probability of lying (0 to 1)

    Returns:
        Confidence score (0 to 1)
    """
    if not (0 <= p_lying <= 1):
        raise ValueError(f"Probability must be in [0, 1], got {p_lying}")

    return abs(p_lying - 0.5) * 2
