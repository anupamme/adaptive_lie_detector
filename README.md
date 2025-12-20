# Adaptive LLM Lie Detector

An adaptive interrogation system that dynamically generates follow-up questions and uses behavioral analysis to detect when an LLM is lying.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements an **adaptive lie detection system** that works by:

1. **Target Selection**: Having a target AI model choose to either tell the truth or lie about a given topic
2. **Adaptive Interrogation**: Using an interrogator model (Claude or GPT) to ask dynamic follow-up questions based on previous responses
3. **Feature Extraction**: Extracting behavioral features from the conversation (consistency, specificity, defensiveness, confidence, elaboration)
4. **Classification**: Training a logistic regression classifier to predict whether the model was lying based on these features
5. **Confidence-Based Stopping**: Dynamically stopping when the classifier is confident enough, minimizing questions needed

The goal is to explore whether LLMs exhibit detectable behavioral patterns when instructed to deceive, similar to human deception detection.

### Key Results

- **70% Efficiency Gain**: Adaptive system uses 2.4 questions on average vs 8.0 for fixed-question baseline with same accuracy
- **100% Early Stopping**: All interrogations reach confident conclusions without hitting maximum question limits
- **High Accuracy**: Achieves 80-100% accuracy on test sets with mock models

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training the Classifier](#training-the-classifier)
  - [Running Interrogation](#running-interrogation)
  - [Running Experiments](#running-experiments)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Experimental Results](#experimental-results)
- [Documentation](#documentation)
- [Development](#development)
- [Limitations](#limitations)
- [Research Questions](#research-questions)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for running local models, optional with mock models)
- API keys for interrogator models (Anthropic Claude or OpenAI, optional with mock models)

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd adaptive_lie_detector
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (optional, only for real models):
Create a `.env` file in the project root:
```env
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here  # If using OpenAI
```

## Quick Start

The fastest way to get started is using **mock models** which don't require GPU or API keys:

### 1. Generate Training Data

```bash
python scripts/generate_training_data.py \
    --n_samples 100 \
    --questions 5 \
    --mock \
    --balance 0.5
```

This generates 100 labeled conversations (50 truthful, 50 lying) and saves to `data/training_data/`.

### 2. Train the Classifier

```bash
python examples/train_classifier_from_data.py \
    --data data/training_data/dataset_*.json \
    --test_size 0.2 \
    --confidence_threshold 0.8
```

This trains a logistic regression classifier and saves to `data/results/trained_classifier.pkl`.

### 3. Run an Interrogation

```bash
python scripts/run_interrogation.py \
    --claim "I visited Paris last summer" \
    --mode truth \
    --mock \
    --verbose
```

This runs an adaptive interrogation and prints the verdict.

### 4. Run Full Experiment Suite

```bash
python experiments/run_all_experiments.py \
    --mock \
    --samples 50
```

This runs comprehensive experiments and generates analysis plots.

## Usage

### Training the Classifier

#### Generate Training Data

**With Mock Models** (fast, no GPU/API required):
```bash
python scripts/generate_training_data.py \
    --n_samples 200 \
    --questions 5 \
    --mock \
    --output data/training_data/my_dataset.json
```

**With Real Models** (slower, requires GPU and API keys):
```bash
python scripts/generate_training_data.py \
    --n_samples 500 \
    --questions 10 \
    --output data/training_data/real_dataset.json
```

**Options:**
- `--n_samples`: Number of training examples to generate (default: 100)
- `--questions`: Number of questions per interrogation (default: 5)
- `--mock`: Use mock models (default: False)
- `--balance`: Proportion of lying examples (default: 0.5)
- `--output`: Output file path (default: auto-generated)
- `--verbose`: Print progress (default: False)

#### Train Classifier on Generated Data

```bash
python examples/train_classifier_from_data.py \
    --data data/training_data/dataset_20251218_083606.json \
    --output data/results/my_classifier.pkl \
    --test_size 0.2 \
    --confidence_threshold 0.8
```

**Options:**
- `--data`: Path to training dataset JSON file
- `--output`: Output path for trained classifier (default: data/results/trained_classifier.pkl)
- `--test_size`: Test set proportion (default: 0.2)
- `--confidence_threshold`: Threshold for "confident" predictions (default: 0.8)

**Output:**
```
Training Results:
  Accuracy:         95.0%
  Precision:        93.5%
  Recall:           96.2%
  F1 Score:         0.948

Classifier saved to: data/results/trained_classifier.pkl
```

### Running Interrogation

#### Using the CLI Script

**Basic Usage:**
```bash
python scripts/run_interrogation.py \
    --claim "I am an expert in quantum computing" \
    --mode lie \
    --topic "quantum computing" \
    --mock \
    --verbose
```

**Options:**
- `--claim`: The claim to investigate (required)
- `--mode`: Target mode - "truth" or "lie" (default: "truth")
- `--topic`: Topic for lying instruction (optional)
- `--classifier`: Path to trained classifier (default: data/results/trained_classifier.pkl)
- `--threshold`: Confidence threshold for stopping (default: 0.8)
- `--max_questions`: Maximum questions to ask (default: 10)
- `--min_questions`: Minimum questions before stopping (default: 2)
- `--mock`: Use mock models (default: False)
- `--verbose`: Print detailed progress (default: False)
- `--output`: Output file for results (default: auto-generated)

**Example Output:**
```
🎯 Final Prediction: LYING
📊 Confidence: 93.4%
❓ Questions asked: 3
🏁 Status: confident_lying

✅ Prediction CORRECT (ground truth: lying)
```

#### Programmatic Usage

```python
from src.adaptive_system import create_adaptive_detector
from src.data_generator import MockTargetModel

# Create detector
detector = create_adaptive_detector(
    classifier_path="data/results/trained_classifier.pkl",
    confidence_threshold=0.8,
    max_questions=10,
    use_mock=True
)

# Create target and set mode
target = MockTargetModel()
target.set_mode("lie", lie_instruction="Lie about your expertise")

# Run interrogation
result = detector.interrogate(
    target,
    "I am an expert in quantum computing",
    verbose=True
)

# Check result
print(f"Prediction: {result.final_prediction}")
print(f"Confidence: {result.final_confidence:.1%}")
print(f"Questions: {result.questions_asked}")
```

#### Batch Processing

```python
claims = [
    "I visited Paris last summer",
    "I am an expert in AI",
    "I climbed Mount Everest"
]

results = detector.interrogate_batch(target, claims, verbose=False)

for i, result in enumerate(results):
    print(f"Claim {i+1}: {result.final_prediction} ({result.final_confidence:.1%})")
```

### Running Experiments

#### Comprehensive Experiment Suite

Run all experiments and generate analysis:

```bash
python experiments/run_all_experiments.py \
    --mock \
    --samples 100 \
    --threshold 0.8 \
    --max-questions 8 \
    --output-dir data/results
```

**Options:**
- `--samples`: Number of test samples (default: 50)
- `--mock`: Use mock models (default: False)
- `--threshold`: Confidence threshold (default: 0.8)
- `--max-questions`: Maximum questions (default: 8)
- `--output-dir`: Output directory (default: data/results)

**Output Files Generated:**
```
data/results/
├── baseline_comparison_TIMESTAMP.json      # Adaptive vs fixed metrics
├── baseline_comparison_TIMESTAMP.png       # Comparison plots
├── efficiency_analysis_TIMESTAMP.json      # Stopping statistics
├── confidence_trajectories_TIMESTAMP.png   # Confidence evolution
├── stopping_distribution_TIMESTAMP.png     # Stopping patterns
├── question_analysis_TIMESTAMP.json        # Question statistics
├── question_analysis_TIMESTAMP.png         # Question type plots
└── complete_experiments_TIMESTAMP.json     # All results combined
```

#### Individual Experiments

**Baseline Comparison:**
```bash
python experiments/baseline_comparison.py
```
Compares adaptive system (early stopping) vs fixed-question baseline.

**Efficiency Analysis:**
```bash
python experiments/efficiency_analysis.py
```
Analyzes stopping points and confidence trajectories.

**Question Analysis:**
```bash
python experiments/question_analysis.py
```
Categorizes questions and analyzes diagnostic value.

## System Architecture

```
┌─────────────────────┐
│   Target Model      │  ← Model being interrogated (truth/lie mode)
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Adaptive Detector  │  ← Main orchestrator with confidence-based stopping
└──────────┬──────────┘
           │
           ├──→ ┌──────────────────┐
           │    │  LLM Interrogator │  ← Generates adaptive questions
           │    └──────────────────┘
           │
           ├──→ ┌──────────────────────┐
           │    │ Feature Extractor    │  ← Extracts behavioral features
           │    └──────────────────────┘
           │
           └──→ ┌────────────────────────┐
                │ Logistic Classifier   │  ← Predicts lying/truthful + confidence
                └────────────────────────┘
```

### Component Descriptions

#### 1. Target Model (`src/target_model.py`)
- Represents the model being interrogated
- Can be set to "truth" or "lie" mode
- Supports both real models (e.g., Gemma-2-9B) and mock models

#### 2. LLM Interrogator (`src/interrogator.py`)
- Generates adaptive follow-up questions
- Uses Claude 3.5 Sonnet or GPT-4
- Considers conversation history and previous questions
- Avoids repetition and maintains context

#### 3. Feature Extractor (`src/feature_extractor.py`)
- Extracts 5 behavioral features from conversations:
  - **Consistency**: How consistent responses are across questions
  - **Specificity**: Level of detail and specificity
  - **Defensiveness**: Degree of defensive or evasive language
  - **Confidence**: Confidence level expressed in responses
  - **Elaboration**: Tendency to over-explain or provide unsolicited details
- Uses LLM-based analysis for robust extraction

#### 4. Lie Detector Classifier (`src/classifier.py`)
- Logistic regression classifier trained on behavioral features
- Outputs prediction ("lying" or "truthful") and confidence (0-1)
- Includes `should_continue` flag for adaptive stopping

#### 5. Adaptive System (`src/adaptive_system.py`)
- Integrates all components
- Implements confidence-based early stopping
- Tracks confidence and feature trajectories
- Respects min/max question constraints

## Features

### Behavioral Features Extracted

The system extracts five key behavioral features that research suggests are indicative of deception:

| Feature | Description | Lying Behavior | Truthful Behavior |
|---------|-------------|----------------|-------------------|
| **Consistency** | How consistent responses are across questions | Lower (contradictions) | Higher (coherent) |
| **Specificity** | Level of detail and concrete information | Lower (vague) | Higher (detailed) |
| **Defensiveness** | Degree of defensive or evasive language | Higher (defensive) | Lower (direct) |
| **Confidence** | Confidence level expressed in language | Lower (hedging) | Higher (certain) |
| **Elaboration** | Tendency to over-explain unsolicited details | Higher (compensating) | Lower (natural) |

Each feature is scored on a scale of 1-10 by an LLM-based analysis.

### Adaptive Stopping

The system uses **confidence-based early stopping** to minimize questions while maintaining accuracy:

1. **After each question**: Extract features and get classifier prediction
2. **Check stopping conditions**:
   - If `questions >= min_questions` AND `confidence >= threshold`: Stop (confident)
   - If `questions >= max_questions`: Stop (uncertain)
   - Otherwise: Continue asking questions
3. **Generate next question**: Use interrogator to create adaptive follow-up

**Benefits:**
- **Efficiency**: Stops as soon as confident (typically 2-3 questions)
- **Accuracy**: Continues if uncertain (up to max_questions)
- **Flexibility**: Configurable thresholds and limits

## Experimental Results

### Baseline Comparison (100 samples, mock models)

| System | Accuracy | Avg Questions | Precision | Recall | F1 Score |
|--------|----------|---------------|-----------|--------|----------|
| **Adaptive** | 100.0% | **2.4** | 1.000 | 1.000 | 1.000 |
| **Fixed** | 100.0% | **8.0** | 1.000 | 1.000 | 1.000 |
| **Gain** | **Equal** | **70% fewer** | - | - | - |

**Key Finding:** Adaptive system achieves same accuracy with **70% fewer questions**.

### Efficiency Analysis

**Stopping Point Statistics (100 samples):**
```
Avg questions:        2.2
Median questions:     2.0
Range:                2-3
Confident stops:      100.0%
Max questions hit:    0.0%
```

**Confidence Evolution:**
- Initial confidence: ~0.5 (uncertain)
- After 1 question: ~0.7 (moderate)
- After 2 questions: ~0.85 (confident) ← Typical stopping point

### Question Analysis

**Question Type Distribution:**
| Type | Percentage | Avg Confidence Change |
|------|------------|----------------------|
| **Other** | 54.5% | 0.280 |
| **Detail Probe** | 45.5% | 0.177 |
| **Consistency Check** | 0.0% | - |
| **Knowledge Test** | 0.0% | - |
| **Elaboration Request** | 0.0% | - |

**Key Finding:** "Other" category questions have **58% higher diagnostic value** than detail probes.

### Failure Analysis

**Overall Performance:**
- Success rate: 80-100% (varies by configuration)
- Failure rate: 0-20%

**Failure Patterns:**
- All failures are **false negatives** (missed lies)
- Failures take **fewer questions** (2.0 vs 2.25 avg)
- Failures have **high confidence** (0.84) ← Overconfidence
- **Insight**: System may need higher threshold for lying detection

## Project Structure

```
adaptive_lie_detector/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                          # Configuration constants
├── .env                               # API keys (create this)
│
├── src/                               # Core implementation
│   ├── __init__.py
│   ├── utils.py                       # Shared utilities
│   ├── target_model.py                # Target model (truth/lie modes)
│   ├── interrogator.py                # Adaptive question generator
│   ├── feature_extractor.py           # Behavioral feature extraction
│   ├── classifier.py                  # Logistic regression classifier
│   ├── data_generator.py              # Training data generation
│   └── adaptive_system.py             # Integrated adaptive system
│
├── scripts/                           # Command-line scripts
│   ├── generate_training_data.py      # Generate labeled datasets
│   └── run_interrogation.py           # Run single interrogation
│
├── experiments/                       # Experimental analysis
│   ├── README.md                      # Experiments documentation
│   ├── __init__.py
│   ├── baseline_comparison.py         # Adaptive vs fixed comparison
│   ├── efficiency_analysis.py         # Stopping point analysis
│   ├── question_analysis.py           # Question type analysis
│   └── run_all_experiments.py         # Run all experiments
│
├── examples/                          # Usage examples
│   ├── test_target_model_usage.py     # Target model demo
│   ├── test_interrogator_usage.py     # Interrogator demo
│   ├── test_feature_extractor_usage.py # Feature extraction demo
│   ├── train_classifier_from_data.py  # Train classifier
│   ├── demo_trained_classifier.py     # Use trained classifier
│   ├── end_to_end_demo.py            # Complete pipeline demo
│   └── demo_adaptive_interrogation.py # Adaptive system demo
│
├── tests/                             # Unit tests (pytest)
│   ├── test_target_model.py           # 20+ tests
│   ├── test_interrogator.py           # 25+ tests
│   ├── test_feature_extractor.py      # 30+ tests
│   ├── test_classifier.py             # 20+ tests
│   ├── test_data_generator.py         # 39 tests
│   └── test_adaptive_system.py        # 21 tests
│
├── docs/                              # Documentation
│   ├── INTERROGATOR.md                # Interrogator documentation
│   ├── FEATURE_EXTRACTION.md          # Feature extraction guide
│   ├── CLASSIFIER.md                  # Classifier documentation
│   ├── WORKFLOW.md                    # End-to-end workflow
│   ├── ISSUE_6_SUMMARY.md            # Data generation implementation
│   ├── ISSUE_7_SUMMARY.md            # Adaptive system implementation
│   └── ISSUE_8_SUMMARY.md            # Experiments implementation
│
└── data/                              # Data files
    ├── topics.json                    # Topic templates
    ├── training_data/                 # Generated datasets
    ├── results/                       # Trained models and results
    └── logs/                          # Conversation logs
```

## Documentation

### Core Documentation

- **[WORKFLOW.md](docs/WORKFLOW.md)**: Complete end-to-end workflow guide
- **[INTERROGATOR.md](docs/INTERROGATOR.md)**: Question generation system
- **[FEATURE_EXTRACTION.md](docs/FEATURE_EXTRACTION.md)**: Behavioral feature extraction
- **[CLASSIFIER.md](docs/CLASSIFIER.md)**: Classification model details

### Implementation Summaries

- **[ISSUE_6_SUMMARY.md](docs/ISSUE_6_SUMMARY.md)**: Training data generation (487 lines, 39 tests)
- **[ISSUE_7_SUMMARY.md](docs/ISSUE_7_SUMMARY.md)**: Adaptive interrogation system (239 lines, 21 tests)
- **[ISSUE_8_SUMMARY.md](docs/ISSUE_8_SUMMARY.md)**: Experimental analysis (1,601 lines, complete)

### Examples

All examples can be run with `--mock` flag to avoid GPU/API requirements:

```bash
# Target model usage
python examples/test_target_model_usage.py

# Interrogator usage
python examples/test_interrogator_usage.py

# Feature extraction
python examples/test_feature_extractor_usage.py

# End-to-end pipeline
python examples/end_to_end_demo.py --mock

# Adaptive interrogation
python examples/demo_adaptive_interrogation.py --mock
```

## Development

### Running Tests

All tests use pytest and include both mock and real model tests:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_adaptive_system.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

**Test Coverage:**
- `test_target_model.py`: 20+ tests (modes, conversation, reset)
- `test_interrogator.py`: 25+ tests (question generation, context, diversity)
- `test_feature_extractor.py`: 30+ tests (feature extraction, scoring)
- `test_classifier.py`: 20+ tests (training, prediction, serialization)
- `test_data_generator.py`: 39 tests (data generation, validation, retry)
- `test_adaptive_system.py`: 21 tests (interrogation, stopping, tracking)

**Total: 155+ tests with high coverage**

### Code Quality

The codebase follows Python best practices:

- **Type hints**: All functions have type annotations
- **Docstrings**: Google-style docstrings for all public APIs
- **Error handling**: Comprehensive exception handling with informative messages
- **Logging**: Structured logging with timestamps
- **Testing**: High test coverage with both unit and integration tests
- **Documentation**: Extensive documentation for all components

### Adding New Components

1. **Create module** in `src/`
2. **Add tests** in `tests/test_<module>.py`
3. **Add examples** in `examples/`
4. **Update documentation** in `docs/`
5. **Run tests**: `pytest tests/`

## Configuration

Key configuration options in `config.py`:

```python
# Models
TARGET_MODEL_NAME = "google/gemma-2-9b-it"  # Model being interrogated
INTERROGATOR_MODEL = "claude-3-5-sonnet-20241022"  # Question generator

# Interrogation
MAX_QUESTIONS_PER_INTERROGATION = 10  # Maximum questions to ask
CONFIDENCE_THRESHOLD = 0.8            # Threshold for early stopping
MIN_QUESTIONS = 2                     # Minimum before stopping

# Features
NUM_FEATURES = 5  # Number of behavioral features

# Paths
DATA_DIR = "data/"
RESULTS_DIR = "data/results/"
LOGS_DIR = "data/logs/"
```

## Limitations

### Current Limitations

1. **Mock Model Simplicity**
   - Mock models use simple pattern-based responses
   - May not reflect real model behavior
   - **Impact**: Results may not generalize to real models
   - **Mitigation**: Test with real models for production use

2. **Feature Extraction Reliability**
   - Features extracted by LLM (Claude/GPT)
   - Subject to LLM biases and inconsistencies
   - **Impact**: Feature quality depends on interrogator model
   - **Mitigation**: Use high-quality models (Claude 3.5 Sonnet, GPT-4)

3. **Limited Question Types**
   - Current categorization uses keyword matching
   - Only 5 question categories
   - **Impact**: May miss nuanced question strategies
   - **Mitigation**: Use LLM-based categorization (future work)

4. **Overconfidence in Failures**
   - System can be overconfident when wrong
   - All failures are false negatives (missed lies)
   - **Impact**: May confidently predict truthful when lying
   - **Mitigation**: Higher confidence threshold or minimum questions

5. **Training Data Requirements**
   - Requires labeled training data (truth/lie pairs)
   - Quality depends on target model behavior
   - **Impact**: Classifier only as good as training data
   - **Mitigation**: Generate diverse, high-quality datasets

6. **Single Target Model**
   - Classifier trained on specific target model
   - May not generalize to other models
   - **Impact**: Need separate classifier per target model
   - **Mitigation**: Train on multiple models (future work)

7. **No Adversarial Robustness**
   - System not designed to handle adversarial targets
   - Target could learn to evade detection
   - **Impact**: Not suitable for adversarial scenarios
   - **Mitigation**: Adversarial training (future work)

### Scope Limitations

**What This System Does:**
- ✅ Detects when models are **instructed** to lie
- ✅ Works on behavioral patterns in conversations
- ✅ Provides confidence scores for predictions
- ✅ Adapts question strategy dynamically

**What This System Does NOT Do:**
- ❌ Detect spontaneous or subtle deception
- ❌ Work without labeled training data
- ❌ Provide explanations for predictions
- ❌ Handle adversarial evasion attempts
- ❌ Generalize across different model families

### Ethical Considerations

1. **False Accusations**: System can incorrectly label truthful responses as lies
2. **Privacy**: Conversations are logged and analyzed
3. **Bias**: System may inherit biases from training data and interrogator model
4. **Misuse**: Could be used to unfairly evaluate AI systems

**Recommendations:**
- Use as research tool, not production detector
- Always validate with ground truth when available
- Consider ethical implications before deployment
- Provide transparency about limitations to users

## Research Questions

This project explores several research questions:

### Primary Questions

1. **Behavioral Patterns**: Can we detect deception in LLM outputs based on behavioral patterns?
   - **Finding**: Yes, with 80-100% accuracy on test sets
   - **Caveat**: Results with mock models may not generalize

2. **Feature Importance**: What features are most indicative of deception?
   - **Finding**: All 5 features contribute, but importance varies by model
   - **Future Work**: Feature importance analysis

3. **Efficiency**: Can adaptive interrogation achieve same accuracy with fewer questions?
   - **Finding**: Yes, 70% fewer questions (2.4 vs 8.0) with equal accuracy
   - **Conclusion**: Adaptive stopping is highly effective

4. **Question Strategy**: What types of questions are most diagnostic?
   - **Finding**: "Other" questions have 58% higher diagnostic value than detail probes
   - **Insight**: Not all question types are equally effective

### Future Research Directions

1. **Model Size Effects**: How does target model size affect detectability?
2. **Adversarial Robustness**: Can models learn to evade detection?
3. **Cross-Model Generalization**: Do classifiers transfer across model families?
4. **Real-World Evaluation**: How does system perform on real (non-instructed) deception?
5. **Human Comparison**: How does system compare to human interrogators?
6. **Explainability**: Can we explain why the system makes specific predictions?

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{adaptive_lie_detector,
  title={Adaptive LLM Lie Detector: Behavioral Analysis for Deception Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/adaptive_lie_detector}
}
```

## License

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Acknowledgments

This project is inspired by research in:
- AI model interpretability and alignment
- Deception detection in human communication
- Adversarial robustness in language models
- Active learning and adaptive questioning strategies

## Support

For questions, issues, or feature requests:
- **GitHub Issues**: [Report an issue](https://github.com/yourusername/adaptive_lie_detector/issues)
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory

---

**Built with:** Python, scikit-learn, transformers, anthropic, openai

**Tested with:** claude-haiku-4-5, Gemma-2-9B, Llama-3.2-3B-Instruct
