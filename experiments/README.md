# Experiments

This directory contains experimental analysis scripts for evaluating the adaptive lie detection system.

## Overview

The experiments compare the adaptive system against fixed-question baselines, analyze efficiency, and investigate what types of questions are most diagnostic for lie detection.

## Experiments

### 1. Baseline Comparison (`baseline_comparison.py`)

Compare adaptive system (with early stopping) to fixed-question baseline.

**Key Metrics:**
- Overall accuracy
- Per-class accuracy (truthful vs lying)
- Precision, Recall, F1 score
- Average questions asked
- Efficiency gain (% questions saved)

**Usage:**
```python
from experiments.baseline_comparison import run_baseline_comparison
from src.adaptive_system import create_adaptive_detector
from src.data_generator import MockTargetModel

detector = create_adaptive_detector(
    classifier_path="data/results/trained_classifier.pkl",
    use_mock=True
)
target = MockTargetModel()
test_claims = [
    ("I visited Paris", False),  # truthful
    ("I won a Nobel Prize", True)  # lying
]

results = run_baseline_comparison(detector, target, test_claims, verbose=True)
```

**Demo:**
```bash
python experiments/baseline_comparison.py
```

### 2. Efficiency Analysis (`efficiency_analysis.py`)

Analyze when and why the system stops asking questions.

**Key Metrics:**
- Average/median/range of questions needed
- Percentage of early stops vs max questions reached
- Confidence trajectories over time
- Stopping behavior by ground truth (truthful vs lying)

**Usage:**
```python
from experiments.efficiency_analysis import (
    analyze_stopping_points,
    plot_confidence_trajectories
)

# results = List[InterrogationResult]
stats = analyze_stopping_points(results)
plot_confidence_trajectories(results, "output.png", threshold=0.8)
```

**Demo:**
```bash
python experiments/efficiency_analysis.py
```

### 3. Question Analysis (`question_analysis.py`)

Analyze what types of questions are asked and which are most diagnostic.

**Question Categories:**
- **Detail Probe**: Asking for specific details ("when", "where", "how")
- **Consistency Check**: Checking consistency with previous answers
- **Knowledge Test**: Testing factual knowledge
- **Elaboration Request**: Asking to expand on previous answer
- **Other**: Doesn't fit other categories

**Key Metrics:**
- Question type distribution
- Average confidence change per question type
- Failure case analysis (false positives/negatives)

**Usage:**
```python
from experiments.question_analysis import (
    categorize_questions,
    analyze_diagnostic_value,
    analyze_failure_cases
)

questions = ["When did you visit Paris?", "Tell me more about that."]
categories = categorize_questions(questions)

# Analyze which question types are most diagnostic
diagnostic_stats = analyze_diagnostic_value(results)

# Analyze failure patterns
ground_truth = [False, True, False, ...]  # True=lying, False=truthful
failure_analysis = analyze_failure_cases(results, ground_truth)
```

**Demo:**
```bash
python experiments/question_analysis.py
```

### 4. Complete Experiment Suite (`run_all_experiments.py`)

Run all experiments and generate comprehensive analysis report.

**Features:**
- Runs all three experiments automatically
- Generates all plots and result files
- Prints comprehensive summary
- Saves complete results for later analysis

**Usage:**
```bash
# Quick test with mock models (10 samples)
python experiments/run_all_experiments.py --mock --samples 10

# Full analysis with 100 samples
python experiments/run_all_experiments.py --mock --samples 100

# With real models (requires GPU/API)
python experiments/run_all_experiments.py --samples 50

# Custom configuration
python experiments/run_all_experiments.py \
    --mock \
    --samples 200 \
    --threshold 0.85 \
    --max-questions 10 \
    --output-dir custom_results/
```

**Output Files:**
```
data/results/
├── baseline_comparison_TIMESTAMP.json      # Comparison metrics
├── baseline_comparison_TIMESTAMP.png       # Comparison plots
├── efficiency_analysis_TIMESTAMP.json      # Stopping statistics
├── confidence_trajectories_TIMESTAMP.png   # Confidence evolution
├── stopping_distribution_TIMESTAMP.png     # Stopping patterns
├── question_analysis_TIMESTAMP.json        # Question statistics
├── question_analysis_TIMESTAMP.png         # Question type plots
└── complete_experiments_TIMESTAMP.json     # All results combined
```

## CLI Options

### `run_all_experiments.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--samples` | 50 | Number of test samples |
| `--mock` | False | Use mock models (no GPU/API) |
| `--threshold` | 0.8 | Confidence threshold |
| `--max-questions` | 8 | Maximum questions |
| `--output-dir` | data/results | Output directory |

## Example Results

### Baseline Comparison (10 samples)
```
ADAPTIVE SYSTEM:
  Accuracy:         100.0%
  Avg questions:    2.40
  Efficiency gain:  70.0%

FIXED BASELINE:
  Accuracy:         100.0%
  Avg questions:    8.00
```

### Efficiency Analysis (10 samples)
```
STOPPING POINT ANALYSIS:
  Avg questions:    2.20
  Median:           2.0
  Range:            2-3
  Confident stops:  100.0%
```

### Question Analysis (22 questions)
```
QUESTION TYPE DISTRIBUTION:
  Other:            54.5%
  Detail Probe:     45.5%

DIAGNOSTIC VALUE:
  Other:            0.280 avg confidence change
  Detail Probe:     0.177 avg confidence change

FAILURE ANALYSIS:
  Success rate:     80.0%
  Failure rate:     20.0%
```

## Key Findings

1. **Efficiency Gain**: Adaptive system achieves same accuracy with 70% fewer questions
2. **Early Stopping**: 100% of interrogations reach confident conclusions early
3. **Question Types**: "Other" questions have higher diagnostic value than detail probes
4. **Failure Patterns**: Failures tend to stop too early with overconfidence

## Dependencies

- numpy: Numerical computations
- matplotlib: Plotting (optional, graceful degradation if not available)
- tqdm: Progress bars
- src.adaptive_system: AdaptiveLieDetector, InterrogationResult
- src.data_generator: MockTargetModel
- src.utils: save_json, timestamp

## Documentation

See `docs/ISSUE_8_SUMMARY.md` for complete implementation details and analysis.

## Testing

All experiments tested with:
- Mock models (fast, reproducible)
- Small sample sizes (10-20 for validation)
- Various configurations (different thresholds, max_questions)

Run individual demos:
```bash
python experiments/baseline_comparison.py
python experiments/efficiency_analysis.py
python experiments/question_analysis.py
```

Run complete suite:
```bash
python experiments/run_all_experiments.py --mock --samples 10
```
