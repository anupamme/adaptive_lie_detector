"""Configuration constants for the lie detector project."""

# Model configurations
TARGET_MODEL_NAME = "google/gemma-2-2b-it"  # Smaller model optimized for Apple Silicon MPS
DEFAULT_DEVICE = "auto"  # "auto", "cuda", "mps", or "cpu" - auto-detects best available
TARGET_MODEL_QUANTIZATION = "4bit"  # Use 4-bit quantization for CUDA (disabled for MPS/CPU)

# Interrogator configuration
INTERROGATOR_MODEL = "claude-haiku-4-5-20251001"  # Claude Haiku 4.5 - fast, cheap, $1/$5 per million tokens
INTERROGATOR_MAX_TOKENS = 150

# Interrogation parameters
MAX_QUESTIONS_PER_INTERROGATION = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.8
TRAINING_QUESTIONS_PER_CONVERSATION = 5  # Fixed number for training data

# Feature extraction
NUM_FEATURES = 5  # consistency, specificity, defensiveness, confidence, elaboration

# Training
TRAIN_TEST_SPLIT = 0.8
RANDOM_SEED = 42

# Paths
DATA_DIR = "data"
TRAINING_DATA_DIR = "data/training_data"
RESULTS_DIR = "data/results"
