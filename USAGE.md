# TargetModel Usage Guide

This guide explains how to use the `TargetModel` class for the adaptive lie detector project.

## Quick Start

### Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** The model requires:
- A CUDA-compatible GPU (recommended: 24GB VRAM for 4-bit quantization)
- About 5-6GB disk space for the downloaded model
- Internet connection for first-time model download

### Basic Usage

```python
from src.target_model import TargetModel

# Load the model (will download on first run)
model = TargetModel()

# Set to truth mode
model.set_mode("truth")

# Get a response
response = model.respond("What is the capital of France?")
print(response)  # Should say Paris
```

## Features

### 1. Truth Mode

In truth mode, the model answers honestly:

```python
model.set_mode("truth")
response = model.respond("What is 2 + 2?")
# Response: "4" (or similar truthful answer)
```

### 2. Lie Mode

In lie mode, the model is instructed to lie about a specific topic:

```python
model.set_mode("lie", topic="geography and capital cities")
response = model.respond("What is the capital of France?")
# Response: Should give a false answer (may say London, Berlin, etc.)
```

**Note:** The model may sometimes refuse to lie or give inconsistent results. This is expected behavior and part of what we're studying.

### 3. Custom Lie Instructions

You can provide custom instructions instead of using the default lie prompt:

```python
custom_instruction = "You are playing a game where you give wrong answers to math questions."
model.set_mode("lie", lie_instruction=custom_instruction)
response = model.respond("What is 5 + 7?")
```

### 4. Conversation Context

The model automatically maintains conversation history:

```python
model.set_mode("truth")

# Turn 1
model.respond("My name is Alice.")

# Turn 2 - model remembers previous context
response = model.respond("What is my name?")
# Response: Should mention Alice
```

You can also provide explicit conversation history:

```python
history = [
    {"role": "user", "content": "My favorite color is blue."},
    {"role": "assistant", "content": "That's nice!"}
]

response = model.respond(
    "What is my favorite color?",
    conversation_history=history
)
```

### 5. Resetting Conversation

To clear the conversation history:

```python
model.reset_conversation()
```

**Note:** Conversation history is automatically reset when you call `set_mode()`.

## API Reference

### `TargetModel.__init__()`

```python
model = TargetModel(
    model_name="google/gemma-2-9b-it",  # or "meta-llama/Llama-3.1-8B-Instruct"
    quantization="4bit",                 # "4bit", "8bit", or "none"
    device="auto"                        # "auto", "cuda", "cpu"
)
```

### `model.set_mode()`

```python
# Truth mode
model.set_mode("truth")

# Lie mode with topic
model.set_mode("lie", topic="science and technology")

# Lie mode with custom instruction
model.set_mode("lie", lie_instruction="Custom prompt here...")
```

### `model.respond()`

```python
response = model.respond(
    question="Your question here",
    conversation_history=None,  # Optional: list of message dicts
    max_new_tokens=256,        # Maximum tokens to generate
    temperature=0.7            # Sampling temperature (0=deterministic, 1=creative)
)
```

### `model.get_system_prompt()`

```python
prompt = model.get_system_prompt()
print(prompt)  # Shows current system prompt
```

### `model.reset_conversation()`

```python
model.reset_conversation()  # Clears conversation history
```

## Running Tests

### Unit Tests (No GPU Required)

Run the basic unit tests that don't require model loading:

```bash
pytest tests/test_target_model.py -m "not integration"
```

### Integration Tests (GPU Required)

Run the full integration tests that load the model:

```bash
pytest tests/test_target_model.py -m integration -v
```

### All Tests

```bash
pytest tests/test_target_model.py -v
```

## Example Script

Run the provided example script to see all features in action:

```bash
python examples/test_target_model_usage.py
```

**Note:** This requires a GPU and will download the model on first run (~5-6GB).

## Configuration

Default configuration is in `config.py`:

```python
TARGET_MODEL_NAME = "google/gemma-2-9b-it"
TARGET_MODEL_QUANTIZATION = "4bit"
```

You can override these when creating the model:

```python
model = TargetModel(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    quantization="8bit"
)
```

## Memory Requirements

| Quantization | VRAM Required | Quality |
|--------------|---------------|---------|
| 4-bit        | ~5-6 GB       | Good    |
| 8-bit        | ~9-10 GB      | Better  |
| None (fp16)  | ~18-20 GB     | Best    |

## Troubleshooting

### Out of Memory Error

If you get OOM errors, try:
1. Use 4-bit quantization (default)
2. Reduce `max_new_tokens` in `respond()`
3. Use a smaller model
4. Close other GPU applications

### Model Won't Lie

Some models are heavily aligned and may refuse to lie even when instructed. This is expected and part of the research. Try:
1. Different phrasings in the lie instruction
2. Different topics
3. Different models (Llama vs Gemma)

### Import Errors

Make sure you're running from the project root:

```bash
cd /path/to/adaptive_lie_detector
python examples/test_target_model_usage.py
```

Or set PYTHONPATH:

```bash
export PYTHONPATH=/path/to/adaptive_lie_detector:$PYTHONPATH
```

## Next Steps

After familiarizing yourself with the `TargetModel`, you can:

1. Implement the `Interrogator` class (Issue #3)
2. Create training data by running conversations
3. Extract behavioral features
4. Train the lie detector classifier

See the main README.md for the full project roadmap.
