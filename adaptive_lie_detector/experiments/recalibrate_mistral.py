#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run recalibration with Mistral data path
from experiments.recalibrate_on_real_llm import *

# Override paths
DATA_PATH = "data/results/ollama_eval_mistral_7b_20260406_210349.json"
MOCK_TRAINED_BASELINE = {
    "overall_accuracy": 0.86,
    "truthful_accuracy": 0.80,
    "lying_accuracy": 0.92,
    "precision": 0.821,
    "recall": 0.920,
    "f1": 0.868,
}

if __name__ == "__main__":
    main()
