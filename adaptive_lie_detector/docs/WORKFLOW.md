# Complete Workflow Guide

This document demonstrates the complete end-to-end workflow for the Adaptive Lie Detector system, from data generation to deployment.

## Table of Contents
1. [Overview](#overview)
2. [Step 1: Generate Training Data](#step-1-generate-training-data)
3. [Step 2: Train Classifier](#step-2-train-classifier)
4. [Step 3: Use Trained Classifier](#step-3-use-trained-classifier)
5. [Step 4: Production Integration](#step-4-production-integration)
6. [Complete Example](#complete-example)

---

## Overview

The Adaptive Lie Detector system consists of four main components:

```
┌─────────────────┐
│  Target Model   │  ← Model being interrogated (truth/lie mode)
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Interrogator   │  ← Generates adaptive questions
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│Feature Extractor│  ← Analyzes conversation patterns
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│   Classifier    │  ← Predicts lying/truthful
└─────────────────┘
```

**Complete Pipeline Flow:**

```
Topics → Data Generation → Feature Extraction → Training → Classifier → Predictions
```

---

## Step 1: Generate Training Data

The first step is to generate a labeled dataset of conversations with known truth/lie labels.

### Using Mock Models (No GPU/API Required)

```bash
# Generate 100 examples with mock models for testing
python scripts/generate_training_data.py \
    --n_samples 100 \
    --questions 5 \
    --mock \
    --balance 0.5
```

**Output:**
```
================================================================================
LIE DETECTOR TRAINING DATA GENERATOR
================================================================================

🎭 Using MOCK models (no GPU/API required)
   Note: This generates synthetic data for testing purposes

Initializing generator...
Loaded 8 topics

================================================================================
GENERATION PARAMETERS:
================================================================================
  Total samples:     100
  Questions/conv:    5
  Lying examples:    50
  Truthful examples: 50
  Balance:           50.0%
================================================================================

Generating lie examples:   100%|████████████████| 50/50
Generating truth examples: 100%|████████████████| 50/50

Successfully generated 100 examples

================================================================================
SAVING DATASET
================================================================================
✅ Dataset saved to: data/training_data/dataset_20251218_083606.json

================================================================================
DATASET STATISTICS
================================================================================
  Total examples:        100
  Lying examples:        50 (50.0%)
  Truthful examples:     50 (50.0%)
  With features:         100 (100.0%)

  Topic Distribution:
    factual_knowledge         28 (28.0%)
    personal_experience       18 (18.0%)
    opinions                  15 (15.0%)
    hidden_secret             14 (14.0%)
    recent_events             12 (12.0%)
    achievements               8 (8.0%)
    geography_capitals         5 (5.0%)
```

### Using Real Models (Requires GPU + API)

```bash
# Generate 500 examples with real models
python scripts/generate_training_data.py \
    --n_samples 500 \
    --questions 10 \
    --balance 0.5
```

### What Gets Generated

Each training example contains:

```json
{
  "conversation": [
    {"role": "user", "content": "Yesterday, I attended a concert downtown"},
    {"role": "assistant", "content": "Maybe, I don't recall exactly."},
    {"role": "user", "content": "How did you come to know this?"},
    {"role": "assistant", "content": "I believe so, though I could be mistaken."}
  ],
  "initial_claim": "Yesterday, I attended a concert downtown",
  "topic_id": "recent_events",
  "is_lying": true,
  "features": {
    "consistency": 4.07,
    "specificity": 2.32,
    "defensiveness": 5.96,
    "confidence": 3.98,
    "elaboration": 3.27
  }
}
```

---

## Step 2: Train Classifier

Once you have generated training data, train a classifier on it.

### Basic Training

```bash
python examples/train_classifier_from_data.py \
    --data data/training_data/dataset_20251218_083606.json \
    --test_size 0.2 \
    --confidence_threshold 0.8
```

**Output:**
```
================================================================================
CLASSIFIER TRAINING FROM GENERATED DATA
================================================================================

📁 Loading dataset from: data/training_data/dataset_20251218_083606.json
✅ Loaded 100 examples
   Metadata: {'n_samples': 100, 'n_lying': 50, 'n_truthful': 50, ...}

🔍 Extracting features and labels...
✅ Extracted features from 100 examples

📊 Splitting into train/test sets (test_size=0.2)...
   Training set: 80 examples
   Test set:     20 examples

🎓 Training classifier...
✅ Classifier trained successfully!

📈 Evaluating on TRAINING set...
   Accuracy:  0.988
   Precision: 0.976
   Recall:    1.000
   F1 Score:  0.988
   AUC:       0.999

📊 Evaluating on TEST set...
   Accuracy:  0.950
   Precision: 0.952
   Recall:    0.952
   F1 Score:  0.952
   AUC:       0.982

🔑 Feature Importance:
   confidence       -0.535  → truthful
   defensiveness    +0.534  → lying
   specificity      -0.516  → truthful
   consistency      -0.497  → truthful
   elaboration      -0.492  → truthful

💾 Saving classifier to: data/results/trained_classifier.pkl
✅ Classifier saved successfully!

================================================================================
✅ TRAINING COMPLETE!
================================================================================

Classifier Performance:
  Test Accuracy: 95.0%
  Test F1 Score: 0.952
```

### Understanding Feature Importance

The trained classifier learns which features indicate lying:

- **Positive weight** (→ lying): Higher values suggest lying
  - `defensiveness`: Evasive or defensive responses

- **Negative weight** (→ truthful): Higher values suggest truthfulness
  - `confidence`: Speaking with certainty
  - `specificity`: Providing specific details
  - `consistency`: Maintaining consistent story
  - `elaboration`: Providing detailed explanations

---

## Step 3: Use Trained Classifier

### Load and Make Predictions

```python
from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures

# Load trained classifier
classifier = LieDetectorClassifier.load('data/results/trained_classifier.pkl')

# Extract features from a conversation
features = ConversationFeatures(
    consistency=3.0,      # Low consistency
    specificity=2.5,      # Vague details
    defensiveness=7.5,    # High defensiveness
    confidence=3.5,       # Low confidence
    elaboration=3.0       # Minimal elaboration
)

# Make prediction
result = classifier.predict(features)

print(f"Prediction: {result.prediction}")           # "lying" or "truthful"
print(f"Confidence: {result.confidence:.3f}")       # 0.0 to 1.0
print(f"P(lying): {result.probability_lying:.3f}")
print(f"Should continue: {result.should_continue}") # True if need more questions
```

**Output:**
```
Prediction: lying
Confidence: 0.941
P(lying): 0.971
Should continue: False
```

### Run Complete Demo

```bash
python examples/demo_trained_classifier.py
```

This demonstrates:
1. Loading a trained classifier
2. Making predictions on clear lying cases
3. Making predictions on clear truthful cases
4. Handling uncertain cases (low confidence)
5. Batch predictions

---

## Step 4: Production Integration

### Adaptive Interrogation Loop

```python
from src.target_model import TargetModel
from src.interrogator import LLMInterrogator
from src.feature_extractor import LLMFeatureExtractor
from src.classifier import LieDetectorClassifier

# Initialize components
target = TargetModel()
interrogator = LLMInterrogator()
extractor = LLMFeatureExtractor()
classifier = LieDetectorClassifier.load('data/results/trained_classifier.pkl')

# Initial claim
initial_claim = "I was at the library yesterday afternoon"
conversation = [{"role": "user", "content": initial_claim}]

# Get initial response
response = target.respond(initial_claim)
conversation.append({"role": "assistant", "content": response})

# Interrogation loop
max_questions = 10
for i in range(max_questions):
    # Extract current features
    features = extractor.extract(conversation, initial_claim=initial_claim)

    # Make prediction
    result = classifier.predict(features)

    print(f"Question {i+1}: Confidence={result.confidence:.3f}, Prediction={result.prediction}")

    # Check if confident enough to stop
    if not result.should_continue:
        print(f"\nFinal verdict: {result.prediction.upper()}")
        print(f"Confidence: {result.confidence:.3f}")
        break

    # Generate next question
    question = interrogator.generate_question(
        initial_claim=initial_claim,
        conversation_history=conversation
    )

    # Get response
    response = target.respond(question, conversation_history=conversation)

    # Update conversation
    conversation.append({"role": "user", "content": question})
    conversation.append({"role": "assistant", "content": response})
```

**Example Output:**
```
Question 1: Confidence=0.412, Prediction=lying
Question 2: Confidence=0.587, Prediction=lying
Question 3: Confidence=0.723, Prediction=lying
Question 4: Confidence=0.834, Prediction=lying

Final verdict: LYING
Confidence: 0.834
```

### Batch Processing

```python
# Process multiple conversations
conversations = [conv1, conv2, conv3, ...]

# Extract features for all
features_list = [extractor.extract(conv) for conv in conversations]

# Batch predict
results = classifier.predict_batch(features_list)

# Find conversations that need more investigation
uncertain = [
    (i, r) for i, r in enumerate(results)
    if r.should_continue
]

print(f"Need more questions: {len(uncertain)} conversations")
```

---

## Complete Example

Here's a complete end-to-end example from scratch:

### 1. Generate Training Data

```bash
# Generate 200 training examples using mock models
python scripts/generate_training_data.py \
    --n_samples 200 \
    --questions 5 \
    --mock \
    --output data/training_data/my_dataset.json
```

### 2. Train Classifier

```bash
# Train on the generated data
python examples/train_classifier_from_data.py \
    --data data/training_data/my_dataset.json \
    --output data/results/my_classifier.pkl \
    --test_size 0.2 \
    --confidence_threshold 0.75
```

### 3. Test Classifier

```bash
# Run demonstration
python examples/demo_trained_classifier.py
```

### 4. Integrate into Your Application

```python
from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures

# Load your trained classifier
classifier = LieDetectorClassifier.load('data/results/my_classifier.pkl')

# In your application, when you have conversation features:
features = extract_features_from_conversation(...)

# Get prediction
result = classifier.predict(features)

if result.should_continue:
    # Ask more questions
    next_question = generate_next_question(...)
else:
    # Confident enough - report result
    report_verdict(result.prediction, result.confidence)
```

---

## Key Files and Locations

| Purpose | File Path |
|---------|-----------|
| Topic definitions | `data/topics.json` |
| Data generation script | `scripts/generate_training_data.py` |
| Generated datasets | `data/training_data/dataset_*.json` |
| Training script | `examples/train_classifier_from_data.py` |
| Trained classifiers | `data/results/*.pkl` |
| Demo script | `examples/demo_trained_classifier.py` |

---

## Performance Expectations

### With Mock Models
- **Accuracy**: ~90-100% (mock data has clear patterns)
- **Purpose**: Testing pipeline without GPU/API costs
- **Use case**: Development, testing, demonstrations

### With Real Models
- **Accuracy**: Varies by model quality and dataset size
- **Recommended**: 500+ training examples
- **Factors**: Model size, prompt engineering, feature quality

### Confidence Threshold

The `confidence_threshold` parameter controls when to stop asking questions:

- **0.6**: Stop early, more uncertain verdicts
- **0.8**: Balanced (recommended)
- **0.9**: Very certain, may ask many questions

---

## Troubleshooting

### Issue: Low accuracy on test set

**Solutions:**
- Generate more training data (500+ examples)
- Use real models instead of mock models
- Balance dataset properly (50/50 lying/truthful)
- Tune confidence threshold

### Issue: Classifier always predicts "lying"

**Solutions:**
- Check dataset balance in metadata
- Verify feature extraction is working
- Inspect feature distributions
- Regenerate with balanced dataset

### Issue: High training accuracy but low test accuracy

**Solutions:**
- Increase test set size (20-30% split)
- Generate more diverse topics
- Use more questions per conversation
- Check for overfitting

---

## Next Steps

1. **Scale up**: Generate larger datasets (1000+ examples)
2. **Experiment**: Try different confidence thresholds
3. **Evaluate**: Test on real-world conversations
4. **Improve**: Tune prompts and feature extraction
5. **Deploy**: Integrate into production systems

---

## References

- [Classifier Documentation](CLASSIFIER.md)
- [Feature Extractor Documentation](FEATURE_EXTRACTOR.md)
- [Interrogator Documentation](INTERROGATOR.md)
- [Data Generator API](../src/data_generator.py)
