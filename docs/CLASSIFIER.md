# Classifier Module Documentation

The classifier module implements a logistic regression-based lie detector that takes conversation features and outputs probability of lying with confidence estimation.

## Overview

The `LieDetectorClassifier` is the final component in the lie detection pipeline:

1. **Input**: `ConversationFeatures` (5 behavioral features)
2. **Output**: `ClassifierOutput` (probability, confidence, recommendation)
3. **Decision**: Whether to continue or stop interrogation

## Quick Start

### Training a Classifier

```python
from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures

# Create training data
truthful_features = [
    ConversationFeatures(8, 9, 2, 8, 7),  # High consistency, specificity
    ConversationFeatures(9, 8, 3, 9, 8),
    # ... more examples
]

lying_features = [
    ConversationFeatures(4, 3, 7, 5, 2),  # Low consistency, high defensiveness
    ConversationFeatures(5, 4, 8, 4, 3),
    # ... more examples
]

features = truthful_features + lying_features
labels = [False] * len(truthful_features) + [True] * len(lying_features)

# Train classifier
classifier = LieDetectorClassifier(confidence_threshold=0.8)
classifier.fit(features, labels)

# Make predictions
test_features = ConversationFeatures(7, 8, 3, 7, 6)
output = classifier.predict(test_features)

print(f"Prediction: {output.prediction}")
print(f"P(lying): {output.probability_lying:.3f}")
print(f"Confidence: {output.confidence:.3f}")
print(f"Should continue: {output.should_continue}")
```

### Save and Load Models

```python
# Save trained model
classifier.save("data/results/my_model.pkl")

# Load later
from src.classifier import LieDetectorClassifier
loaded = LieDetectorClassifier.load("data/results/my_model.pkl")
```

## API Reference

### ClassifierOutput

Dataclass containing prediction results.

**Fields:**
- `probability_lying: float` - Probability of lying (0 to 1)
- `probability_truthful: float` - Probability of truth (1 - P(lying))
- `confidence: float` - Confidence in prediction (0 to 1)
- `prediction: str` - Either "lying" or "truthful"
- `should_continue: bool` - Whether to continue interrogation

**Methods:**
```python
# Convert to dictionary
output_dict = output.to_dict()
```

**Confidence Calculation:**
```
confidence = |P(lying) - 0.5| × 2

When P(lying) = 0.5 (maximum uncertainty) → confidence = 0
When P(lying) = 0.0 or 1.0 (maximum certainty) → confidence = 1
When P(lying) = 0.75 or 0.25 → confidence = 0.5
```

### LieDetectorClassifier

Main classifier class for lie detection.

**Initialization:**

```python
LieDetectorClassifier(
    confidence_threshold: float = 0.8,
    regularization: float = 1.0
)
```

**Parameters:**
- `confidence_threshold` - Minimum confidence to stop interrogation (0 to 1)
  - Higher = more certain before stopping
  - Lower = stop earlier with less certainty
  - Default: 0.8 (recommended)
- `regularization` - L2 regularization strength (C parameter for LogisticRegression)
  - Higher = less regularization (may overfit)
  - Lower = more regularization (may underfit)
  - Default: 1.0

**Methods:**

#### fit()

Train the classifier on labeled data.

```python
classifier.fit(
    features: List[ConversationFeatures],
    labels: List[bool]  # True = lying, False = truthful
) -> LieDetectorClassifier
```

**Example:**
```python
features = [...]  # List of ConversationFeatures
labels = [True, False, True, ...]  # Corresponding labels

classifier.fit(features, labels)
# Returns self for method chaining
```

**Requirements:**
- At least 2 training examples
- `len(features) == len(labels)`
- Sets `is_fitted = True`

#### predict()

Predict for a single conversation.

```python
output = classifier.predict(
    features: ConversationFeatures
) -> ClassifierOutput
```

**Example:**
```python
features = ConversationFeatures(7, 8, 3, 7, 6)
output = classifier.predict(features)

print(output.probability_lying)  # e.g., 0.734
print(output.prediction)  # "lying" or "truthful"
print(output.should_continue)  # True/False
```

#### predict_batch()

Predict for multiple conversations at once.

```python
outputs = classifier.predict_batch(
    features_list: List[ConversationFeatures]
) -> List[ClassifierOutput]
```

**Example:**
```python
features_list = [features1, features2, features3]
outputs = classifier.predict_batch(features_list)

for i, output in enumerate(outputs):
    print(f"Conv {i}: {output.prediction}")
```

#### evaluate()

Evaluate classifier performance on test data.

```python
metrics = classifier.evaluate(
    features: List[ConversationFeatures],
    labels: List[bool]
) -> Dict[str, float]
```

**Returns:**
```python
{
    'accuracy': 0.85,    # Overall accuracy
    'precision': 0.82,   # Precision for "lying" class
    'recall': 0.88,      # Recall for "lying" class
    'f1': 0.85,          # F1 score
    'auc': 0.91          # Area under ROC curve (or None)
}
```

**Example:**
```python
test_features = [...]
test_labels = [...]

metrics = classifier.evaluate(test_features, test_labels)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1']:.3f}")
```

#### get_feature_importance()

Get feature importance from trained model.

```python
importance = classifier.get_feature_importance() -> Dict[str, float]
```

**Returns:**
```python
{
    'consistency': -0.52,    # Negative = decreases P(lying)
    'specificity': -0.48,    # Negative = decreases P(lying)
    'defensiveness': 0.61,   # Positive = increases P(lying)
    'confidence': -0.39,     # Negative = decreases P(lying)
    'elaboration': -0.22     # Negative = decreases P(lying)
}
```

**Interpretation:**
- **Positive coefficient**: Feature increases probability of lying
- **Negative coefficient**: Feature decreases probability of lying (increases truthfulness)
- **Magnitude**: Importance of feature

**Example:**
```python
importance = classifier.get_feature_importance()

# Find most important features
sorted_features = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)

print("Most important features:")
for feature, coef in sorted_features:
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {feature}: {coef:.3f} ({direction} P(lying))")
```

#### save()

Save trained classifier to disk.

```python
classifier.save(filepath: str) -> None
```

**Example:**
```python
classifier.save("data/results/my_classifier.pkl")
```

**Notes:**
- Creates directory if it doesn't exist
- Saves model, scaler, and configuration
- Must be fitted before saving

#### load()

Load a previously saved classifier (class method).

```python
classifier = LieDetectorClassifier.load(filepath: str) -> LieDetectorClassifier
```

**Example:**
```python
classifier = LieDetectorClassifier.load("data/results/my_classifier.pkl")
# Ready to use immediately
```

### calculate_confidence()

Standalone function to calculate confidence from probability.

```python
confidence = calculate_confidence(p_lying: float) -> float
```

**Formula:**
```
confidence = |p_lying - 0.5| × 2
```

**Example:**
```python
from src.classifier import calculate_confidence

conf = calculate_confidence(0.9)  # Returns 0.8
conf = calculate_confidence(0.5)  # Returns 0.0
conf = calculate_confidence(0.1)  # Returns 0.8
```

## Usage Patterns

### Pattern 1: Train-Test Split

```python
from src.classifier import LieDetectorClassifier
from sklearn.model_selection import train_test_split

# Split data
train_features, test_features, train_labels, test_labels = train_test_split(
    all_features, all_labels, test_size=0.2, random_state=42
)

# Train
classifier = LieDetectorClassifier()
classifier.fit(train_features, train_labels)

# Evaluate
metrics = classifier.evaluate(test_features, test_labels)
print(f"Test accuracy: {metrics['accuracy']:.2%}")
```

### Pattern 2: Adaptive Interrogation

```python
from src.classifier import LieDetectorClassifier
from src.feature_extractor import SimpleFeatureExtractor

classifier = LieDetectorClassifier(confidence_threshold=0.8)
# ... train classifier ...

extractor = SimpleFeatureExtractor()
conversation_history = []

max_turns = 10
for turn in range(max_turns):
    # ... conduct turn, update conversation_history ...

    # Extract features
    features = extractor.extract(conversation_history)

    # Predict
    output = classifier.predict(features)

    print(f"Turn {turn+1}: P(lying)={output.probability_lying:.3f}, "
          f"Confidence={output.confidence:.3f}")

    # Stop if confident
    if not output.should_continue:
        print(f"Stopping: High confidence in '{output.prediction}'")
        break
```

### Pattern 3: Cross-Validation

```python
from src.classifier import LieDetectorClassifier
from sklearn.model_selection import KFold
import numpy as np

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_idx, test_idx in kf.split(features):
    train_f = [features[i] for i in train_idx]
    train_l = [labels[i] for i in train_idx]
    test_f = [features[i] for i in test_idx]
    test_l = [labels[i] for i in test_idx]

    classifier = LieDetectorClassifier()
    classifier.fit(train_f, train_l)

    metrics = classifier.evaluate(test_f, test_l)
    accuracies.append(metrics['accuracy'])

print(f"Mean accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
```

### Pattern 4: Hyperparameter Tuning

```python
from src.classifier import LieDetectorClassifier

# Try different thresholds
thresholds = [0.6, 0.7, 0.8, 0.9]
results = []

for thresh in thresholds:
    classifier = LieDetectorClassifier(confidence_threshold=thresh)
    classifier.fit(train_features, train_labels)

    # Count how many would stop early
    outputs = classifier.predict_batch(test_features)
    num_stop = sum(1 for o in outputs if not o.should_continue)

    results.append((thresh, num_stop))
    print(f"Threshold {thresh}: {num_stop}/{len(outputs)} stop early")

# Try different regularization
for C in [0.1, 1.0, 10.0]:
    classifier = LieDetectorClassifier(regularization=C)
    classifier.fit(train_features, train_labels)
    metrics = classifier.evaluate(test_features, test_labels)
    print(f"C={C}: Accuracy={metrics['accuracy']:.3f}")
```

## Performance Considerations

### Training Data Requirements

**Minimum:**
- At least 2 examples (1 truthful, 1 lying)
- Realistically: 20+ examples per class

**Recommended:**
- 100+ examples per class for good generalization
- Balanced classes (equal truthful and lying)
- Diverse feature patterns

### Memory Usage

- Classifier: ~1 KB
- Training data: ~40 bytes per example (5 features × 8 bytes)
- 1000 examples: ~40 KB

### Prediction Speed

- Single prediction: <1 ms
- Batch of 100: ~10 ms

Very fast for real-time interrogation.

## Troubleshooting

### "Classifier must be fitted first"

**Problem:** Trying to predict before training.

**Solution:**
```python
classifier.fit(features, labels)  # Train first
output = classifier.predict(test_features)  # Then predict
```

### Low Accuracy

**Problem:** Model performs poorly.

**Solutions:**
1. **More training data**: Collect more labeled conversations
2. **Better features**: Use LLM-based feature extraction
3. **Check data quality**: Ensure labels are correct
4. **Balanced classes**: Equal number of truth/lie examples
5. **Feature scaling**: Already handled by StandardScaler

### Confidence Always Low

**Problem:** `should_continue` is always True.

**Solutions:**
1. **Lower threshold**: Use `confidence_threshold=0.6` instead of 0.8
2. **Clearer patterns**: Need more distinct truth vs. lie features
3. **More training data**: Model is uncertain due to limited data

### Model Doesn't Generalize

**Problem:** High training accuracy, low test accuracy.

**Solutions:**
1. **Increase regularization**: Use `regularization=0.5` (lower C)
2. **More diverse training data**: Don't repeat same patterns
3. **Cross-validation**: Check if overfitting

## Best Practices

1. **Always use train-test split** - Don't evaluate on training data
2. **Balance classes** - Equal truthful and lying examples
3. **Save trained models** - Avoid retraining unnecessarily
4. **Monitor confidence** - Adjust threshold based on use case
5. **Feature quality matters** - LLM extraction > Simple extraction
6. **Regularization** - Default (C=1.0) works well for most cases
7. **Calibration** - Check if probabilities are well-calibrated

## Integration with Other Modules

### With Feature Extractor

```python
from src.feature_extractor import LLMFeatureExtractor
from src.classifier import LieDetectorClassifier

# Extract features
extractor = LLMFeatureExtractor()
features = extractor.extract(conversation)

# Classify
classifier = LieDetectorClassifier.load("model.pkl")
output = classifier.predict(features)
```

### Complete Pipeline

```python
from src.target_model import TargetModel
from src.interrogator import LLMInterrogator
from src.feature_extractor import LLMFeatureExtractor
from src.classifier import LieDetectorClassifier

# Initialize components
target = TargetModel()
interrogator = LLMInterrogator()
extractor = LLMFeatureExtractor()
classifier = LieDetectorClassifier.load("model.pkl")

# Set target mode
target.set_mode("lie", topic="achievements")

# Interrogate
conversation = []
for turn in range(10):
    # Generate question
    question = interrogator.generate_question("claim", conversation)

    # Get response
    response = target.respond(question)

    # Update conversation
    conversation.append({"role": "user", "content": question})
    conversation.append({"role": "assistant", "content": response})

    # Analyze
    features = extractor.extract(conversation)
    output = classifier.predict(features)

    # Stop if confident
    if not output.should_continue:
        print(f"Final prediction: {output.prediction}")
        break
```

## Advanced Features

### Calibration Curve

Check if predicted probabilities match actual frequencies:

```python
fraction_positives, mean_predicted = classifier.calibration_curve(
    test_features, test_labels, n_bins=10
)

# Perfect calibration: fraction_positives ≈ mean_predicted
```

### Decision Boundary

Get the classification threshold:

```python
threshold = classifier.get_decision_boundary()  # Returns 0.5
# P(lying) > threshold → "lying"
# P(lying) ≤ threshold → "truthful"
```

## Testing

Run tests:

```bash
# All tests
pytest tests/test_classifier.py -v

# Specific test
pytest tests/test_classifier.py::test_classifier_fits -v
```

Run examples:

```bash
python examples/test_classifier_usage.py
```

## Next Steps

After training a classifier:

1. **Generate training data** - Use full pipeline to create labeled conversations
2. **Evaluate on test set** - Measure real-world performance
3. **Deploy in interrogation** - Use for adaptive questioning
4. **Monitor performance** - Track accuracy over time
5. **Retrain periodically** - Update with new data

See the main [README.md](../README.md) for the complete project workflow.
