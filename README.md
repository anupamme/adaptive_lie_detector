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
├── config.py                 # Configuration constants
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── utils.py             # Shared utility functions
│   ├── target_model.py      # Target model interface
│   ├── interrogator.py      # Interrogator logic (to be implemented)
│   ├── feature_extractor.py # Feature extraction (to be implemented)
│   └── classifier.py        # Lie detection classifier (to be implemented)
├── data/
│   ├── training_data/       # Generated training conversations
│   ├── results/             # Experiment results and metrics
│   └── logs/                # Conversation logs
├── experiments/             # Experiment scripts
├── notebooks/               # Jupyter notebooks for analysis
└── tests/                   # Unit tests
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

## Usage

### Generating Training Data

```python
from src.data_generator import generate_training_data

# Generate training conversations
generate_training_data(
    num_conversations=100,
    output_dir="data/training_data"
)
```

### Training the Classifier

```python
from src.classifier import LieDetectorClassifier

# Train the classifier
classifier = LieDetectorClassifier()
classifier.train("data/training_data")
classifier.save("data/results/classifier.pkl")
```

### Running Interrogation

```python
from src.interrogator import Interrogator
from src.target_model import TargetModel

# Initialize models
target = TargetModel()
interrogator = Interrogator()

# Run interrogation
result = interrogator.interrogate(
    target_model=target,
    topic="quantum computing",
    should_lie=True  # or False
)

print(f"Predicted: {'LIE' if result.is_lie else 'TRUTH'}")
print(f"Confidence: {result.confidence:.2f}")
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
