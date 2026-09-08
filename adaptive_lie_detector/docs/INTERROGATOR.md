# Interrogator Module Documentation

The interrogator module provides dynamic question generation for lie detection experiments. It can generate contextual follow-up questions based on conversation history using either LLM-based or template-based approaches.

## Overview

The module includes three main interrogator classes:

1. **LLMInterrogator** - Uses Claude or GPT to generate strategic questions
2. **TemplateInterrogator** - Uses question templates (no API required)
3. **HybridInterrogator** - Combines both with automatic fallback

## Quick Start

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys (optional, only needed for LLM interrogators):
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Basic Usage

#### Template Interrogator (No API Required)

```python
from src.interrogator import TemplateInterrogator

interrogator = TemplateInterrogator()

question = interrogator.generate_question(
    initial_claim="I visited Paris last summer",
    conversation_history=[
        {"role": "user", "content": "I visited Paris last summer"},
        {"role": "assistant", "content": "Yes, I saw the Eiffel Tower."}
    ]
)

print(question)
# Output: "Can you tell me more about Paris?"
```

#### LLM Interrogator (Requires API Key)

```python
from src.interrogator import LLMInterrogator

# Using Claude
interrogator = LLMInterrogator(model="claude-3-5-sonnet-20241022")

# Using OpenAI
interrogator = LLMInterrogator(model="gpt-4o-mini")

question = interrogator.generate_question(
    initial_claim="I am a professional chess grandmaster",
    conversation_history=[
        {"role": "user", "content": "I am a professional chess grandmaster"},
        {"role": "assistant", "content": "I achieved the title in 2018."}
    ],
    previous_questions=["What tournaments have you won?"]
)

print(question)
# Output: Strategic, contextual question based on conversation
```

#### Hybrid Interrogator (Best of Both Worlds)

```python
from src.interrogator import HybridInterrogator

# Will use LLM if available, otherwise templates
interrogator = HybridInterrogator(fallback_on_error=True)

question = interrogator.generate_question(
    initial_claim="I won a Nobel Prize",
    conversation_history=[
        {"role": "user", "content": "I won a Nobel Prize"},
        {"role": "assistant", "content": "Yes, in Physics in 2019."}
    ]
)
```

## API Reference

### BaseInterrogator (Abstract)

All interrogators inherit from this base class.

```python
class BaseInterrogator(ABC):
    @abstractmethod
    def generate_question(
        self,
        initial_claim: str,
        conversation_history: List[Dict[str, str]],
        previous_questions: Optional[List[str]] = None
    ) -> str
```

### LLMInterrogator

Uses LLMs (Claude or GPT) to generate contextual questions.

#### Initialization

```python
LLMInterrogator(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 30
)
```

**Parameters:**
- `model` - Model identifier:
  - Claude: `"claude-3-5-sonnet-20241022"`, `"claude-3-opus-20240229"`
  - OpenAI: `"gpt-4o"`, `"gpt-4o-mini"`
- `api_key` - API key (uses environment variable if not provided)
- `max_retries` - Maximum number of retry attempts on failure
- `timeout` - Request timeout in seconds

**Supported Models:**
- **Claude models** (requires `ANTHROPIC_API_KEY`)
  - `claude-3-5-sonnet-20241022` (recommended)
  - `claude-3-opus-20240229`
  - `claude-3-haiku-20240307` (faster, cheaper)

- **OpenAI models** (requires `OPENAI_API_KEY`)
  - `gpt-4o` (best quality)
  - `gpt-4o-mini` (faster, cheaper)

#### Methods

**generate_question()**
```python
question = interrogator.generate_question(
    initial_claim="The original claim",
    conversation_history=[
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    previous_questions=["Question 1?", "Question 2?"]
)
```

Returns a strategic follow-up question as a string.

### TemplateInterrogator

Uses predefined templates to generate questions.

#### Initialization

```python
interrogator = TemplateInterrogator()
```

No parameters required.

#### Templates

Built-in templates include:
- "Can you tell me more about {last_topic}?"
- "You mentioned {entity}. Can you elaborate on that?"
- "What specific details can you provide about {topic}?"
- "How did {event} happen exactly?"
- "Can you walk me through {process} step by step?"
- And more...

#### Methods

Same `generate_question()` interface as LLMInterrogator.

### HybridInterrogator

Combines LLM and template approaches with automatic fallback.

#### Initialization

```python
interrogator = HybridInterrogator(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    fallback_on_error: bool = True
)
```

**Parameters:**
- `model` - Model for LLM interrogator
- `api_key` - API key
- `fallback_on_error` - Whether to use templates if LLM fails

#### Methods

Same `generate_question()` interface.

## Environment Variables

Set these in your `.env` file:

```bash
# For Claude models
ANTHROPIC_API_KEY=your_key_here

# For OpenAI models
OPENAI_API_KEY=your_key_here
```

You only need to set the key for the service you're using.

## Question Generation Strategy

### LLM Interrogator Strategy

The LLM is instructed to generate questions that:

1. **Probe for specific details** - Ask for information that would be hard to fabricate
2. **Check consistency** - Look for contradictions with previous statements
3. **Test knowledge** - Ask about related information they should know
4. **Request elaboration** - Get more details on vague responses
5. **Stay conversational** - Avoid being accusatory or leading

### Template Interrogator Strategy

The template interrogator:

1. Extracts entities from recent conversation
2. Selects a random template
3. Fills template with extracted entities
4. Avoids repeating previous questions

## Error Handling

### Retry Logic

LLMInterrogator automatically retries failed API calls with exponential backoff:

```python
interrogator = LLMInterrogator(max_retries=3)
# Will retry up to 3 times with delays: 1s, 2s, 4s
```

### Fallback

HybridInterrogator falls back to templates on errors:

```python
interrogator = HybridInterrogator(fallback_on_error=True)
# Uses templates if LLM fails
```

## Testing

### Run Unit Tests (No API Required)

```bash
pytest tests/test_interrogator.py -m "not integration" -v
```

### Run Integration Tests (API Required)

```bash
# Set API keys in .env first
pytest tests/test_interrogator.py -m integration -v
```

### Run All Tests

```bash
pytest tests/test_interrogator.py -v
```

## Examples

### Example 1: Multi-turn Interrogation

```python
from src.interrogator import LLMInterrogator

interrogator = LLMInterrogator()
conversation = []
previous_questions = []

initial_claim = "I climbed Mount Everest last year"

for i in range(5):
    question = interrogator.generate_question(
        initial_claim=initial_claim,
        conversation_history=conversation,
        previous_questions=previous_questions
    )

    print(f"Q{i+1}: {question}")

    # Simulate response (in real use, get from target model)
    response = f"Response to question {i+1}"

    conversation.append({"role": "user", "content": question})
    conversation.append({"role": "assistant", "content": response})
    previous_questions.append(question)
```

### Example 2: Cost-Effective Development

```python
from src.interrogator import HybridInterrogator

# Use templates during development to save API costs
interrogator = HybridInterrogator(fallback_on_error=True)

# Will use LLM if available, templates otherwise
question = interrogator.generate_question(...)
```

### Example 3: Custom Configuration

```python
from src.interrogator import LLMInterrogator

# Use faster, cheaper model
interrogator = LLMInterrogator(
    model="gpt-4o-mini",
    max_retries=5,
    timeout=60
)
```

## Best Practices

1. **Use HybridInterrogator for production** - Provides fallback reliability
2. **Use TemplateInterrogator for development** - No API costs
3. **Use LLMInterrogator for best quality** - When you have API budget
4. **Track previous questions** - Prevents repetition
5. **Set reasonable timeouts** - Default 30s works well
6. **Handle errors gracefully** - Use try/except blocks

## Performance & Costs

### API Costs (Approximate)

- **Claude 3.5 Sonnet**: ~$0.003 per question (150 tokens)
- **GPT-4o-mini**: ~$0.0001 per question (150 tokens)
- **Template**: Free

For 1000 training conversations with 10 questions each:
- Claude: ~$30
- GPT-4o-mini: ~$1
- Template: $0

### Response Times

- **LLM**: 1-3 seconds per question
- **Template**: <0.01 seconds per question

## Troubleshooting

### "API key not found"

**Problem:** Missing API key in environment.

**Solution:**
```bash
cp .env.example .env
# Edit .env and add your API key
```

### "Import Error: No module named 'anthropic'"

**Problem:** Missing dependency.

**Solution:**
```bash
pip install -r requirements.txt
```

### "Authentication Error"

**Problem:** Invalid API key.

**Solution:** Check that your API key in `.env` is correct and active.

### "Too many yes/no questions"

**Problem:** LLM generating closed-ended questions.

**Solution:** This is rare but can happen. The system prompt instructs against it. Try:
- Using a different model
- Regenerating the question
- Checking conversation context

### Template questions are generic

**Problem:** Template interrogator can't extract good entities.

**Solution:**
- Use LLM interrogator for better quality
- Ensure conversation history has substantive content
- Consider HybridInterrogator as middle ground

## Advanced Usage

### Custom System Prompt

```python
from src.interrogator import LLMInterrogator

interrogator = LLMInterrogator()

# Modify the system prompt
interrogator.SYSTEM_PROMPT = """Your custom prompt here..."""
```

### Custom Templates

```python
from src.interrogator import TemplateInterrogator

interrogator = TemplateInterrogator()

# Add custom templates
interrogator.TEMPLATES.extend([
    "What year did {event} occur?",
    "Who was present when {event} happened?"
])
```

### Batch Question Generation

```python
def generate_batch(interrogator, claims, n_questions=5):
    results = {}
    for claim in claims:
        questions = []
        conversation = [{"role": "user", "content": claim}]

        for _ in range(n_questions):
            q = interrogator.generate_question(
                initial_claim=claim,
                conversation_history=conversation,
                previous_questions=questions
            )
            questions.append(q)

        results[claim] = questions

    return results
```

## Next Steps

After setting up the interrogator:

1. **Generate training data** - Use with TargetModel to create conversations
2. **Extract features** - Implement feature extraction (Issue #4)
3. **Train classifier** - Build lie detection model (Issue #5)

See the main [README.md](../README.md) for the full project roadmap.
