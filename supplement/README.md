# Supplementary Reproduction Bundle

## Reproducibility Statement

To preserve double-blind review, we do not include the full codebase during review because de-identifying repository history, logs, and raw transcripts risks accidental deanonymization. Instead, we provide an anonymized reproduction bundle containing claim pairs, prompts, regex patterns, aggregate results, and a script for regenerating the headline tables. The full de-anonymized codebase and raw transcripts will be released upon acceptance.

## Contents

| File | Description |
|------|-------------|
| `claims.csv` | 50 true/false claim pairs used in the prompt-equalized experiments (15 scientific, 10 historical, 10 geographic, 8 technology, 7 cultural) |
| `prompts.json` | All system prompts: truth, lie (instructed), and equalized/neutral conditions |
| `regex_patterns.txt` | 14 refusal/correction marker regex patterns used as the label-free baseline |
| `headline_results.csv` | Per-model aggregate metrics (LOO accuracy, refusal baseline accuracy, per-class accuracy, avg questions) |
| `model_settings.json` | Model identifiers, hyperparameters, and experimental parameters (no paths or credentials) |
| `reproduce_tables.py` | Standalone script that reads the CSV and prints formatted headline tables |

## Quick Start

```bash
python reproduce_tables.py
```

Requires only Python 3.8+ (no external dependencies).

## Experimental Design Summary

- **Prompt-equalized condition**: Both truthful and deceptive trials receive an identical neutral system prompt. The model is presented with either a true or subtly-falsified factual claim. Behavioral differences arise from the model's knowledge conflicting with false claims, not from instruction-following.
- **Features**: 5 LLM-extracted behavioral features (consistency, specificity, defensiveness, confidence, elaboration) rated 1-10 per interrogation turn.
- **Classifier**: Logistic regression with leave-one-out cross-validation.
- **Refusal baseline**: Fixed-threshold decision rule using regex pattern matches (predict "lying" if refusal count >= 1). Requires no labeled calibration data.

## Models Tested

8 target models spanning 3B-70B parameters across 4 model families (Llama, Mistral, Qwen, Claude), with cross-family feature extraction validated using Mistral Large as an independent extractor.
