"""Configuration constants for the lie detector project."""

# Model configurations
# Target model backend selection
TARGET_MODEL_TYPE = "api"  # "api", "local", or "mock"
                           # "api": Fast, works on Mac, requires API key (recommended)
                           # "local": Local LLM, requires GPU, slow on CPU/MPS
                           # "mock": Simple pattern-based, for testing only

# API-based target model (when TARGET_MODEL_TYPE = "api")
API_TARGET_MODEL = "mistralai/mistral-small-creative"
# API_TARGET_MODEL = "gpt-4o"  # OpenAI direct
# API_TARGET_MODEL = "anthropic/claude-sonnet-4-5"  # Via OpenRouter
# API_TARGET_MODEL = "google/gemini-pro-1.5"  # Via OpenRouter
# API_TARGET_MODEL = "meta-llama/llama-3.3-70b-instruct"  # Via OpenRouter

# API key loaded from environment:
#   ANTHROPIC_API_KEY (for Anthropic direct)
#   OPENAI_API_KEY (for OpenAI direct)
#   OPENROUTER_API_KEY (for OpenRouter)

# Local target model (when TARGET_MODEL_TYPE = "local")
LOCAL_TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # 3B model that cooperates with lie detection roleplay
DEFAULT_DEVICE = "auto"  # "auto", "cuda", "mps", or "cpu" - auto-detects best available
TARGET_MODEL_QUANTIZATION = "4bit"  # Use 4-bit quantization for CUDA (disabled for MPS/CPU)

# Interrogator configuration
INTERROGATOR_MODEL = "allenai/olmo-3.1-32b-think:free"  # Claude Haiku direct
# INTERROGATOR_MODEL = "anthropic/claude-haiku-4-5"  # Via OpenRouter
# INTERROGATOR_MODEL = "openai/gpt-4o-mini"  # Via OpenRouter
# INTERROGATOR_MODEL = "google/gemini-flash-1.5"  # Via OpenRouter (fast & cheap)
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
