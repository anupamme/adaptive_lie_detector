"""Configuration constants for the lie detector project."""

# Model configurations
# Target model backend selection
TARGET_MODEL_TYPE = "api"  # "api", "local", or "mock"
                           # "api": Fast, works on Mac, requires API key (recommended)
                           # "local": Local LLM, requires GPU, slow on CPU/MPS
                           # "mock": Simple pattern-based, for testing only

# API-based target model (when TARGET_MODEL_TYPE = "api")
API_TARGET_MODEL = "meta-llama/llama-3.3-70b-instruct"
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
# ⚠️  AVOID reasoning models (olmo-think, o1, deepseek-reasoner) - they use tokens for internal
#     thinking and may not produce output with low max_tokens. Use standard chat models instead.
INTERROGATOR_MODEL = "xiaomi/mimo-v2-flash:free"  # Via OpenRouter (good quality, recommended)
# INTERROGATOR_MODEL = "claude-haiku-4-5-20251001"  # Claude Haiku direct (requires ANTHROPIC_API_KEY)
# INTERROGATOR_MODEL = "anthropic/claude-haiku-4-5"  # Via OpenRouter
# INTERROGATOR_MODEL = "openai/gpt-4o-mini"  # Via OpenRouter
# INTERROGATOR_MODEL = "google/gemini-2.0-flash-exp:free"  # Via OpenRouter (free but experimental)
# INTERROGATOR_MODEL = "allenai/olmo-3.1-32b-think:free"  # ⚠️ Reasoning model - needs max_tokens ≥ 500
INTERROGATOR_MAX_TOKENS = 500  # Increase to 500+ for reasoning models (olmo-think, o1, etc.)

# Feature extractor configuration
# Should use a model with strong JSON instruction-following
FEATURE_EXTRACTOR_MODEL = "openai/gpt-4o-mini"  # Excellent JSON following (recommended)
# FEATURE_EXTRACTOR_MODEL = "anthropic/claude-haiku-4-5"  # Via OpenRouter (very good)
# FEATURE_EXTRACTOR_MODEL = "google/gemini-2.0-flash-exp:free"  # Free tier (decent)
# Note: Llama models tend to hallucinate their own JSON structure - avoid for feature extraction

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
