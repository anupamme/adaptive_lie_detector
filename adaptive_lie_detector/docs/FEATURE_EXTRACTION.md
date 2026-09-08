# Feature Extraction Module Documentation

The feature extraction module converts variable-length conversations into fixed-size feature vectors for lie detection. It analyzes behavioral patterns in conversations to extract signals that may indicate deception.

## Overview

The module provides three feature extraction strategies:

1. **LLMFeatureExtractor** - Uses Claude or GPT to analyze conversations (best quality)
2. **HybridFeatureExtractor** - Combines LLM analysis with linguistic features (comprehensive)
3. **SimpleFeatureExtractor** - Uses only linguistic heuristics (no API, baseline)

## Core Features

All extractors produce five core behavioral features (0-10 scale):

| Feature | Description | High Value | Low Value |
|---------|-------------|------------|-----------|
| **Consistency** | Alignment between responses | Coherent, aligned | Contradictory |
| **Specificity** | Level of detail | Rich, concrete details | Vague, generic |
| **Defensiveness** | Evasiveness | Very defensive | Open, forthcoming |
| **Confidence** | Certainty level | Very certain | Uncertain, hesitant |
| **Elaboration** | Information volunteered | Lots of extra detail | Minimal answers |

## Quick Start

### Simple Extractor (No API Required)

```python
from src.feature_extractor import SimpleFeatureExtractor

extractor = SimpleFeatureExtractor()

conversation = [
    {"role": "user", "content": "Where did you go?"},
    {"role": "assistant", "content": "I went to Paris last summer."}
]

features = extractor.extract(conversation)

print(f"Consistency: {features.consistency}/10")
print(f"Specificity: {features.specificity}/10")
# ... other features

# Get feature vector for classifier
vector = features.to_vector()  # numpy array, shape (5,)
```

### LLM Extractor (Requires API Key)

```python
from src.feature_extractor import LLMFeatureExtractor

# Using Claude
extractor = LLMFeatureExtractor(model="claude-3-5-sonnet-20241022")

# Using OpenAI
extractor = LLMFeatureExtractor(model="gpt-4o-mini")

features = extractor.extract(
    conversation=conversation,
    initial_claim="I visited Paris"  # Optional context
)
```

### Hybrid Extractor (Best of Both)

```python
from src.feature_extractor import HybridFeatureExtractor

extractor = HybridFeatureExtractor()

features = extractor.extract(conversation)

# Access LLM-analyzed features
print(f"Consistency: {features.consistency}/10")

# Access linguistic features
print(f"Avg response length: {features.response_length_avg} words")
print(f"Hedge word count: {features.hedge_word_count}")
```

## API Reference

### ConversationFeatures

Dataclass that holds extracted features.

**Core Features:**
- `consistency: float` (0-10)
- `specificity: float` (0-10)
- `defensiveness: float` (0-10)
- `confidence: float` (0-10)
- `elaboration: float` (0-10)

**Optional Linguistic Features:**
- `response_length_avg: Optional[float]` - Average words per response
- `question_count: Optional[int]` - Number of questions in conversation
- `hedge_word_count: Optional[int]` - Count of hedging phrases

**Methods:**

```python
# Convert to numpy array for ML models
vector = features.to_vector()  # Returns np.ndarray, shape (5,)

# Convert to dictionary
feature_dict = features.to_dict()
```

### BaseFeatureExtractor (Abstract)

All extractors inherit from this base class.

```python
class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        conversation: List[Dict[str, str]],
        initial_claim: Optional[str] = None
    ) -> ConversationFeatures
```

### LLMFeatureExtractor

Uses LLM to analyze conversations and extract subjective features.

**Initialization:**

```python
LLMFeatureExtractor(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None,
    max_retries: int = 3
)
```

**Parameters:**
- `model` - Model identifier:
  - Claude: `"claude-3-5-sonnet-20241022"`, `"claude-3-opus-20240229"`
  - OpenAI: `"gpt-4o"`, `"gpt-4o-mini"`
- `api_key` - API key (uses environment variable if not provided)
- `max_retries` - Maximum retry attempts on failure

**Supported Models:**

**Claude (requires ANTHROPIC_API_KEY):**
- `claude-3-5-sonnet-20241022` (recommended, best quality)
- `claude-3-opus-20240229` (highest quality, most expensive)
- `claude-3-haiku-20240307` (fastest, cheapest)

**OpenAI (requires OPENAI_API_KEY):**
- `gpt-4o` (best quality)
- `gpt-4o-mini` (recommended, good quality/cost ratio)

**Methods:**

```python
features = extractor.extract(
    conversation=[
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    initial_claim="The original claim"  # Optional
)
```

### HybridFeatureExtractor

Combines LLM analysis with linguistic feature computation.

**Initialization:**

```python
HybridFeatureExtractor(
    model: str = "claude-3-5-sonnet-20241022",
    api_key: Optional[str] = None
)
```

**Features Extracted:**
- All 5 core features (via LLM)
- `response_length_avg` (computed)
- `question_count` (computed)
- `hedge_word_count` (computed)

**Hedge Words Detected:**
- maybe, perhaps, possibly, might, could, probably
- I think, I believe, I guess, sort of, kind of
- not sure, not certain, if I recall, I don't remember exactly

### SimpleFeatureExtractor

Baseline extractor using only linguistic heuristics.

**Initialization:**

```python
extractor = SimpleFeatureExtractor()  # No parameters
```

**Feature Computation:**
- **Specificity**: Based on response length (longer = more specific)
- **Confidence**: Inverse of hedge word count (fewer hedges = more confident)
- **Elaboration**: Based on response length
- **Defensiveness**: Based on question marks in responses
- **Consistency**: Fixed at 5.0 (neutral, can't measure without LLM)

**Note:** Simple extractor is less accurate than LLM but provides instant, free results.

## Usage Patterns

### Pattern 1: Training Data Generation

```python
from src.feature_extractor import LLMFeatureExtractor
import numpy as np

extractor = LLMFeatureExtractor()

# Process multiple conversations
conversations = [...]  # List of conversation histories
labels = [...]  # List of labels (0=truth, 1=lie)

# Extract features
X = []
for conv in conversations:
    features = extractor.extract(conv)
    X.append(features.to_vector())

X = np.array(X)  # Feature matrix, shape (n_conversations, 5)
y = np.array(labels)  # Labels, shape (n_conversations,)

# Train classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X, y)
```

### Pattern 2: Real-time Analysis

```python
from src.feature_extractor import HybridFeatureExtractor

extractor = HybridFeatureExtractor()

# During an interrogation
conversation_history = []

while len(conversation_history) < max_turns:
    # ... add new turns to conversation_history ...

    # Extract features so far
    features = extractor.extract(conversation_history)

    # Analyze in real-time
    if features.consistency < 4.0 or features.specificity < 3.0:
        print("⚠️ Potential deception detected")
```

### Pattern 3: Comparison Analysis

```python
from src.feature_extractor import LLMFeatureExtractor

extractor = LLMFeatureExtractor()

truthful_conv = [...]
deceptive_conv = [...]

truth_features = extractor.extract(truthful_conv)
lie_features = extractor.extract(deceptive_conv)

# Compare
print(f"Specificity - Truth: {truth_features.specificity}, Lie: {lie_features.specificity}")
print(f"Confidence - Truth: {truth_features.confidence}, Lie: {lie_features.confidence}")
```

### Pattern 4: Batch Processing with Caching

```python
from src.feature_extractor import LLMFeatureExtractor
from src.utils import save_json, load_json
import os

extractor = LLMFeatureExtractor()

def extract_with_cache(conversation, cache_file):
    """Extract features with caching to save API costs."""
    if os.path.exists(cache_file):
        cached = load_json(cache_file)
        return ConversationFeatures(**cached)

    features = extractor.extract(conversation)
    save_json(features.to_dict(), cache_file)
    return features

# Use it
features = extract_with_cache(
    conversation,
    f"data/features/{conversation_id}.json"
)
```

## Environment Setup

Set up API keys in `.env` file:

```bash
# For Claude
ANTHROPIC_API_KEY=your_key_here

# For OpenAI
OPENAI_API_KEY=your_key_here
```

You only need the key for the service you're using.

## Testing

### Run Unit Tests (No API)

```bash
pytest tests/test_feature_extractor.py -m "not integration" -v
```

### Run Integration Tests (API Required)

```bash
# Set API keys in .env first
pytest tests/test_feature_extractor.py -m integration -v
```

### Run Example Script

```bash
python examples/test_feature_extraction_usage.py
```

## Cost Analysis

For 1000 conversations (avg 5 turns each):

| Method | Cost | Time | Quality |
|--------|------|------|---------|
| Claude 3.5 Sonnet | ~$15 | ~2 hours | Excellent |
| GPT-4o-mini | ~$0.50 | ~2 hours | Very Good |
| Simple | $0 | <1 minute | Basic |

**Recommendation:** Use GPT-4o-mini for best cost/quality ratio.

## Feature Interpretation

### Deception Patterns

Research suggests deceptive conversations often show:

✅ **Lower Specificity** - Vague, lack of concrete details
- Truth: "I went to the Louvre on July 15th at 2 PM"
- Lie: "I went to some museum, maybe in the afternoon"

✅ **Lower Consistency** - Contradictions, misaligned facts
- Truth: Dates, places, people align across responses
- Lie: Timeline inconsistencies, contradictory details

✅ **Higher Defensiveness** - Evasive, deflecting
- Truth: Direct answers to questions
- Lie: "Why do you ask?", "I don't see why that matters"

✅ **Lower Confidence** - More hedging, uncertainty
- Truth: "I am certain it was Tuesday"
- Lie: "I think maybe it was Tuesday, or possibly Wednesday"

✅ **Higher/Lower Elaboration** - Either too much or too little
- Over-elaboration: Trying too hard to convince
- Under-elaboration: Avoiding providing information

### Feature Correlations

Features often correlate:

- **Low Confidence + High Hedge Words** = Likely deception
- **High Specificity + High Consistency** = Likely truth
- **High Defensiveness + Low Elaboration** = Evasion

## Error Handling

### Graceful Degradation

```python
from src.feature_extractor import LLMFeatureExtractor, SimpleFeatureExtractor

try:
    extractor = LLMFeatureExtractor()
    features = extractor.extract(conversation)
except Exception as e:
    print(f"LLM extraction failed: {e}")
    print("Falling back to simple extraction")
    extractor = SimpleFeatureExtractor()
    features = extractor.extract(conversation)
```

### Retry Logic

LLMFeatureExtractor automatically retries failed API calls:

```python
extractor = LLMFeatureExtractor(max_retries=5)  # Will retry up to 5 times
```

### Validation

Features are automatically validated:

```python
# This will raise ValueError
features = ConversationFeatures(
    consistency=15,  # Out of range [0, 10]
    ...
)
```

## Best Practices

1. **Use Hybrid for Production** - Best quality with linguistic features
2. **Use Simple for Development** - Fast iteration, no API costs
3. **Cache Results** - Save API costs during development
4. **Validate Input** - Ensure conversation has assistant responses
5. **Batch Process** - More efficient than one-by-one
6. **Handle Errors** - Always have fallback strategy

## Troubleshooting

### "API key not found"

**Problem:** Missing API key in environment.

**Solution:**
```bash
cp .env.example .env
# Edit .env and add your API key
```

### "Failed to parse JSON"

**Problem:** LLM returned malformed JSON.

**Solution:** The extractor has robust parsing that handles markdown and extracts JSON. If still failing, check LLM response in error message.

### Features all at 5.0

**Problem:** Using SimpleFeatureExtractor with empty conversation.

**Solution:** Ensure conversation has assistant responses. Use LLM extractor for better quality.

### High API costs

**Problem:** Processing many conversations.

**Solutions:**
- Use GPT-4o-mini instead of Claude (10x cheaper)
- Implement caching (see Pattern 4)
- Use SimpleFeatureExtractor for initial filtering
- Batch process during off-peak hours

## Advanced Usage

### Custom Feature Weighting

```python
from src.feature_extractor import LLMFeatureExtractor
import numpy as np

extractor = LLMFeatureExtractor()
features = extractor.extract(conversation)

# Weight features based on importance
weights = np.array([0.3, 0.25, 0.15, 0.15, 0.15])  # consistency most important
weighted_vector = features.to_vector() * weights
```

### Multi-Model Ensemble

```python
from src.feature_extractor import LLMFeatureExtractor
import numpy as np

# Extract with multiple models
claude_ext = LLMFeatureExtractor(model="claude-3-5-sonnet-20241022")
gpt_ext = LLMFeatureExtractor(model="gpt-4o-mini")

claude_features = claude_ext.extract(conversation)
gpt_features = gpt_ext.extract(conversation)

# Average the features
ensemble_vector = (claude_features.to_vector() + gpt_features.to_vector()) / 2
```

### Feature Engineering

```python
from src.feature_extractor import HybridFeatureExtractor
import numpy as np

extractor = HybridFeatureExtractor()
features = extractor.extract(conversation)

# Create derived features
base_vector = features.to_vector()

# Add ratio features
specificity_confidence_ratio = features.specificity / (features.confidence + 0.1)
hedges_per_word = features.hedge_word_count / (features.response_length_avg + 1)

# Combine
extended_vector = np.concatenate([
    base_vector,
    [specificity_confidence_ratio],
    [hedges_per_word]
])
```

## Next Steps

After extracting features:

1. **Generate Training Data** - Extract features from many conversations
2. **Train Classifier** - Use sklearn LogisticRegression or similar
3. **Evaluate Performance** - Test on held-out data
4. **Deploy** - Use in real-time interrogation system

See the main [README.md](../README.md) for the full project roadmap.
