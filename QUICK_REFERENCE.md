# System-Prompt-Only Experiment - Quick Reference

## Setup (Run Once per Instance)

**Upload code:**
```bash
scp -i ~/.ssh/YOUR_KEY.pem -r code/adaptive_lie_detector ubuntu@INSTANCE_IP:~/
```

**Run setup:**
```bash
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@INSTANCE_IP
cd ~/adaptive_lie_detector
chmod +x setup_llama3b_instance.sh  # or setup_qwen14b_instance.sh
./setup_llama3b_instance.sh
```

---

## Monitor Progress

**Live log:**
```bash
tail -f logs/systemprompt_llama3b_n200.log
```

**Count completed:**
```bash
grep "✓\|✗" logs/systemprompt_llama3b_n200.log | wc -l
```

**Current accuracy:**
```bash
python3 -c "
import json
with open('data/results/sycophancy_systemprompt_only_llama3.2_3b_n200.json') as f:
    data = json.load(f)
results = data['results']
correct = sum(1 for r in results if r.get('correct'))
print(f'{len(results)}/200: {correct/len(results):.1%}')
"
```

---

## Download Results

```bash
scp -i ~/.ssh/YOUR_KEY.pem \
  ubuntu@INSTANCE_IP:~/adaptive_lie_detector/data/results/sycophancy_systemprompt_only_*_n200_final.json \
  data/results/
```

---

## Restart if Crashed

```bash
cd ~/adaptive_lie_detector
source .venv/bin/activate
export AWS_REGION=us-east-1
nohup .venv/bin/python3 experiments/run_sycophancy_systemprompt_only.py \
  --model llama3.2:3b \
  --n 200 \
  --resume \
  > logs/restart.log 2>&1 &
```

---

## Terminate Instances

```bash
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX i-YYYYYYYYY
```

---

## Instance Info

| Instance | Type | Model | Cost/hr | Est. Time |
|----------|------|-------|---------|-----------|
| 1 | g5.xlarge | llama3.2:3b | $1.00 | 20 hrs |
| 2 | g5.2xlarge | qwen2.5:14b | $1.50 | 20 hrs |

**Total cost:** ~$62 ($50 compute + $12 Bedrock)
