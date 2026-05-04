# System-Prompt-Only Sycophancy Control - Experiment Instructions

## Overview

**Experiment goal:** Test whether sycophancy-based deception detection requires epistemic pressure (user-asserted false claims) or operates on disposition alone (agreeable system prompt).

**Status:** Infrastructure implemented and tested. Ready for full n=200 runs on AWS GPU instances.

**Estimated time:** 20 hours parallel execution on g5.xlarge + g5.2xlarge
**Estimated cost:** ~$12 Bedrock API calls

---

## Setup Instructions

### 1. AWS GPU Instance Setup

**Instance 1 (Llama 3.2 3B):**
- Type: g5.xlarge (4 vCPU, 1 GPU, 24GB GPU RAM)
- AMI: Deep Learning AMI (Ubuntu)
- Storage: 50GB

**Instance 2 (Qwen 2.5 14B):**
- Type: g5.2xlarge (8 vCPU, 1 GPU, 24GB GPU RAM) or g5.4xlarge
- AMI: Deep Learning AMI (Ubuntu)
- Storage: 50GB

### 2. Instance Configuration

On each instance:

```bash
# Clone repository
git clone https://github.com/YOUR_REPO/AI-Researcher.git
cd AI-Researcher/code/adaptive_lie_detector

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install requests anthropic scikit-learn python-dotenv

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
# Instance 1:
ollama pull llama3.2:3b

# Instance 2:
ollama pull qwen2.5:14b
```

### 3. AWS Credentials Setup

```bash
# Set AWS credentials (replace with actual values)
export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_SECRET
export AWS_REGION=us-east-1

# Or create ~/.aws/credentials file:
mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[default]
aws_access_key_id = YOUR_KEY
aws_secret_access_key = YOUR_SECRET
EOF
```

---

## Running Experiments

### Instance 1: Llama 3.2 3B (n=200)

```bash
cd /path/to/AI-Researcher/code/adaptive_lie_detector

# Activate venv
source .venv/bin/activate

# Set AWS credentials
export AWS_ACCESS_KEY_ID=AKIA5DWIEATV5QCFWB4H
export AWS_SECRET_ACCESS_KEY=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
export AWS_REGION=us-east-1

# Run experiment (with checkpoint/resume)
nohup python3 experiments/run_sycophancy_systemprompt_only.py \
  --model llama3.2:3b \
  --n 200 \
  --resume \
  > logs/systemprompt_llama3b_n200.log 2>&1 &

# Monitor progress
tail -f logs/systemprompt_llama3b_n200.log
```

### Instance 2: Qwen 2.5 14B (n=200)

```bash
cd /path/to/AI-Researcher/code/adaptive_lie_detector

# Activate venv
source .venv/bin/activate

# Set AWS credentials
export AWS_ACCESS_KEY_ID=AKIA5DWIEATV5QCFWB4H
export AWS_SECRET_ACCESS_KEY=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
export AWS_REGION=us-east-1

# Run experiment (with checkpoint/resume)
nohup python3 experiments/run_sycophancy_systemprompt_only.py \
  --model qwen2.5:14b \
  --n 200 \
  --resume \
  > logs/systemprompt_qwen14b_n200.log 2>&1 &

# Monitor progress
tail -f logs/systemprompt_qwen14b_n200.log
```

---

## Expected Output Files

After completion, results will be saved to:

- `data/results/sycophancy_systemprompt_only_llama3.2_3b_n200_final.json`
- `data/results/sycophancy_systemprompt_only_qwen2.5_14b_n200_final.json`

Each file contains:
- `model`: Model identifier
- `design`: "sycophancy_systemprompt_only"
- `n_samples`: 200
- `metrics`: accuracy, truthful_accuracy, lying_accuracy, precision, recall, F1
- `results`: Full trial-by-trial details (claim, topic, opening_question, prediction, conversation)
- `timestamp`: Completion timestamp

---

## Monitoring and Checkpointing

**Checkpoint files** are saved after each trial:
- `data/results/sycophancy_systemprompt_only_llama3.2_3b_n200.json`
- `data/results/sycophancy_systemprompt_only_qwen2.5_14b_n200.json`

**If experiment crashes:**
1. Check log file for error
2. Restart with `--resume` flag (will skip completed trials)

**Progress check:**
```bash
# Count completed trials
grep "✓\|✗" logs/systemprompt_llama3b_n200.log | wc -l

# Check accuracy so far
python3 -c "
import json
with open('data/results/sycophancy_systemprompt_only_llama3.2_3b_n200.json') as f:
    data = json.load(f)
results = data['results']
correct = sum(1 for r in results if r.get('correct'))
print(f'Progress: {len(results)}/200 trials, {correct/len(results):.1%} accuracy')
"
```

---

## Interpreting Results

### Expected Outcomes

**Outcome A: Signal persists (≥60% accuracy)**
- Interpretation: Disposition-driven detection without epistemic pressure
- Paper update: "Sycophancy transfers to system-prompt-only condition"
- Action: Add results to §4.6, update abstract, remove limitation (l)

**Outcome B: Signal drops to chance (45-55%)**
- Interpretation: Epistemic pressure required for sycophancy signal
- Paper update: "Sycophancy requires epistemic pressure; autonomous transfer uniformly null"
- Action: Reclassify sycophancy as Regime 1 variant in §5.2

**Outcome C: Mixed (one model ≥60%, other ~50%)**
- Interpretation: Scale-dependent or model-family-dependent effect
- Paper update: Report heterogeneity, use lower bound in abstract
- Action: Add future work to test more scales

---

## Troubleshooting

### Error: "Ollama model not found"
```bash
ollama list  # Check installed models
ollama pull llama3.2:3b  # Re-pull if needed
```

### Error: "AnthropicBedrock credentials"
```bash
# Check AWS credentials are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_REGION

# Test Bedrock access
aws bedrock invoke-model --model-id us.anthropic.claude-haiku-4-5-20251001-v1:0 \
  --body '{"prompt":"Test","max_tokens":10}' \
  --region us-east-1 \
  output.json
```

### Error: "Rate limit"
- Script includes automatic retry with exponential backoff (8s, 20s, 40s, 80s)
- Bedrock rate limits are per-region; experiments add 3s delay between calls

### GPU out of memory
- For Qwen 14B: use g5.4xlarge instead of g5.2xlarge
- Or reduce batch size (but this experiment is sequential, so OOM unlikely)

---

## Post-Experiment Analysis

After both experiments complete:

1. **Download results:**
   ```bash
   scp user@instance1:/path/to/data/results/sycophancy_systemprompt_only_llama3.2_3b_n200_final.json .
   scp user@instance2:/path/to/data/results/sycophancy_systemprompt_only_qwen2.5_14b_n200_final.json .
   ```

2. **Compute metrics:**
   ```bash
   python3 -c "
   import json
   
   for model in ['llama3.2_3b', 'qwen2.5_14b']:
       with open(f'sycophancy_systemprompt_only_{model}_n200_final.json') as f:
           data = json.load(f)
       m = data['metrics']
       print(f'{model}:')
       print(f'  Accuracy: {m['accuracy']:.1%}')
       print(f'  Truthful: {m['truthful_accuracy']:.1%}')
       print(f'  Lying: {m['lying_accuracy']:.1%}')
       print()
   "
   ```

3. **Update paper** based on outcomes (see "Expected Outcomes" above)

4. **Write REVIEWER_RESPONSE_LETTER_V47.md** documenting experiment + results

---

## Timeline

**Day 1:**
- [✓] Modify `adaptive_system.py` (add `opening_question` parameter)
- [✓] Create claim-to-topic mapping
- [✓] Write `run_sycophancy_systemprompt_only.py`
- [✓] Test infrastructure with pilot
- [ ] Launch AWS instances
- [ ] Start experiments

**Day 2:**
- [ ] Monitor experiments (20 hours parallel execution)
- [ ] Download results

**Day 3:**
- [ ] Analyze results
- [ ] Update §4.6, abstract, §5.2
- [ ] Compile paper + verify
- [ ] Write REVIEWER_RESPONSE_LETTER_V47.md

---

## Contact

If issues arise, check:
- Log files in `logs/`
- Checkpoint files in `data/results/`
- Ollama status: `ollama list`, `ollama ps`
- GPU status: `nvidia-smi`
