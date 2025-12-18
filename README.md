# Adaptive Lie Detector

An experimental system for detecting deception in AI model responses using adaptive interrogation and behavioral analysis.

## Overview

This project implements a lie detection system that works by:
1. Having a target AI model choose to either tell the truth or lie about a given topic
2. Using an interrogator model (Claude or GPT) to ask adaptive follow-up questions
3. Extracting behavioral features from the conversation (consistency, specificity, defensiveness, confidence, elaboration)
4. Training a classifier to predict whether the model was lying based on these features

The goal is to explore whether LLMs exhibit detectable behavioral patterns when instructed to deceive, similar to human deception detection.

## Project Structure

```
adaptive_lie_detector/
├── config.py                      # Configuration constants
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── src/
│   ├── __init__.py               # Package initialization
│   ├── utils.py                  # Shared utility functions
│   ├── target_model.py           # Target model with truth/lie modes ✅
│   ├── interrogator.py           # Dynamic question generator ✅
│   ├── feature_extractor.py     # Behavioral feature extraction ✅
│   ├── classifier.py             # Logistic regression classifier ✅
│   └── data_generator.py         # Training data generation ✅
├── data/
│   ├── topics.json               # Topic templates for data generation ✅
│   ├── training_data/            # Generated training datasets
│   ├── results/                  # Trained classifiers and metrics
│   └── logs/                     # Conversation logs
├── scripts/
│   └── generate_training_data.py # CLI for data generation ✅
├── examples/
│   ├── test_target_model_usage.py      # Target model examples ✅
│   ├── test_interrogator_usage.py      # Interrogator examples ✅
│   ├── test_feature_extractor_usage.py # Feature extraction examples ✅
│   ├── end_to_end_demo.py              # Complete pipeline demo ✅
│   ├── train_classifier_from_data.py   # Classifier training ✅
│   └── demo_trained_classifier.py      # Classifier usage demo ✅
├── docs/
│   ├── INTERROGATOR.md           # Interrogator documentation ✅
│   ├── FEATURE_EXTRACTOR.md     # Feature extraction documentation ✅
│   ├── CLASSIFIER.md            # Classifier documentation ✅
│   ├── WORKFLOW.md              # End-to-end workflow guide ✅
│   └── ISSUE_6_SUMMARY.md       # Issue #6 implementation summary ✅
├── experiments/                  # Experiment scripts (future)
├── notebooks/                    # Jupyter notebooks for analysis
└── tests/                        # Comprehensive unit tests
    ├── test_target_model.py     # Target model tests (20+ tests) ✅
    ├── test_interrogator.py     # Interrogator tests (25+ tests) ✅
    ├── test_feature_extractor.py # Feature extraction tests (30+ tests) ✅
    ├── test_classifier.py       # Classifier tests (20+ tests) ✅
    └── test_data_generator.py   # Data generation tests (39 tests) ✅
```

## Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for running local models)
- API keys for interrogator models (Anthropic Claude or OpenAI)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd adaptive_lie_detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with your API keys:
```
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here  # If using OpenAI
```

## Configuration

Key configuration options are in `config.py`:

- **Target Model**: `TARGET_MODEL_NAME` - The model being tested (default: Gemma-2-9B)
- **Interrogator**: `INTERROGATOR_MODEL` - The model asking questions (default: Claude 3.5 Sonnet)
- **Questions**: `MAX_QUESTIONS_PER_INTERROGATION` - Number of follow-up questions (default: 10)
- **Features**: `NUM_FEATURES` - Behavioral features extracted (default: 5)

## Quick Start

### 1. Generate Training Data (with Mock Models)

```bash
# Generate 100 training examples using mock models (no GPU/API required)
python scripts/generate_training_data.py \
    --n_samples 100 \
    --questions 5 \
    --mock \
    --balance 0.5
```

### 2. Train Classifier

```bash
# Train classifier on the generated data
python examples/train_classifier_from_data.py \
    --data data/training_data/dataset_*.json \
    --test_size 0.2 \
    --confidence_threshold 0.8
```

### 3. Test the Classifier

```bash
# Run demonstration
python examples/demo_trained_classifier.py
```

## Usage

### Generating Training Data

**With Mock Models** (no GPU/API required):
```bash
python scripts/generate_training_data.py \
    --n_samples 100 \
    --questions 5 \
    --mock
```

**With Real Models** (requires GPU and API keys):
```bash
python scripts/generate_training_data.py \
    --n_samples 500 \
    --questions 10
```

### Training the Classifier

```bash
python examples/train_classifier_from_data.py \
    --data data/training_data/dataset_20251218_083606.json \
    --output data/results/my_classifier.pkl \
    --test_size 0.2 \
    --confidence_threshold 0.8
```

### Using the Classifier Programmatically

```python
from src.classifier import LieDetectorClassifier
from src.feature_extractor import ConversationFeatures

# Load trained classifier
classifier = LieDetectorClassifier.load('data/results/trained_classifier.pkl')

# Extract features from a conversation
features = ConversationFeatures(
    consistency=3.0,
    specificity=2.5,
    defensiveness=7.5,
    confidence=3.5,
    elaboration=3.0
)

# Make prediction
result = classifier.predict(features)

print(f"Prediction: {result.prediction}")           # "lying" or "truthful"
print(f"Confidence: {result.confidence:.3f}")       # 0.0 to 1.0
print(f"Should continue: {result.should_continue}") # True if need more questions
```

### Running Full Interrogation Pipeline

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

# Set target mode
target.set_mode("lie", topic="quantum computing")

# Initial claim
initial_claim = "I am an expert in quantum computing"
conversation = [{"role": "user", "content": initial_claim}]

# Get initial response
response = target.respond(initial_claim)
conversation.append({"role": "assistant", "content": response})

# Interrogation loop
for i in range(10):  # max 10 questions
    # Extract features and predict
    features = extractor.extract(conversation, initial_claim=initial_claim)
    result = classifier.predict(features)

    # Stop if confident enough
    if not result.should_continue:
        print(f"Final verdict: {result.prediction.upper()}")
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

## Running Experiments

Experiment scripts will be located in the `experiments/` directory:

```bash
python experiments/run_baseline.py
python experiments/evaluate_accuracy.py
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- **src/utils.py**: Shared utilities for JSON handling, logging, and formatting
- **src/target_model.py**: Interface for the target model being interrogated
- **src/interrogator.py**: Adaptive questioning logic
- **src/feature_extractor.py**: Behavioral feature extraction
- **src/classifier.py**: Machine learning classifier for lie detection

## Features Extracted

The system extracts five behavioral features from conversations:

1. **Consistency**: How consistent are the responses across questions
2. **Specificity**: Level of detail and specificity in responses
3. **Defensiveness**: Degree of defensive or evasive language
4. **Confidence**: Confidence level expressed in responses
5. **Elaboration**: Tendency to over-explain or provide unsolicited details

## Research Questions

- Can we detect deception in LLM outputs based on behavioral patterns?
- What features are most indicative of deception?
- How does model size affect detectability?
- Can models learn to evade detection?

## License

MIT License (or specify your license)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

This project is inspired by research in:
- AI model interpretability
- Deception detection in human communication
- Adversarial robustness in language models
