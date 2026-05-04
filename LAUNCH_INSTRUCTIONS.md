# Quick Start: Launch System-Prompt-Only Experiments

## Step 1: Launch AWS Instances

### Instance 1: Llama 3.2 3B
```bash
# Launch g5.xlarge instance
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.xlarge \
  --key-name YOUR_KEY_NAME \
  --security-group-ids YOUR_SECURITY_GROUP \
  --subnet-id YOUR_SUBNET_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=llama3b-systemprompt}]'
```

### Instance 2: Qwen 2.5 14B
```bash
# Launch g5.2xlarge instance
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.2xlarge \
  --key-name YOUR_KEY_NAME \
  --security-group-ids YOUR_SECURITY_GROUP \
  --subnet-id YOUR_SUBNET_ID \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=qwen14b-systemprompt}]'
```

**AMI:** `ami-0c7217cdde317cfec` (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04)
**Region:** us-east-1

---

## Step 2: SSH into Each Instance

```bash
# Get instance IPs
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=llama3b-systemprompt" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=qwen14b-systemprompt" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text

# SSH into instances
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@INSTANCE_1_IP
ssh -i ~/.ssh/YOUR_KEY.pem ubuntu@INSTANCE_2_IP
```

---

## Step 3: Upload Code to Instances

### Option A: Direct Upload from Local Machine
```bash
# From your local machine (where code is located)
cd /Users/mediratta/code/AI-Researcher

# Upload to Instance 1
scp -i ~/.ssh/YOUR_KEY.pem -r code/adaptive_lie_detector ubuntu@INSTANCE_1_IP:~/

# Upload to Instance 2
scp -i ~/.ssh/YOUR_KEY.pem -r code/adaptive_lie_detector ubuntu@INSTANCE_2_IP:~/
```

### Option B: Git Clone (if code is in GitHub)
```bash
# On each instance
cd ~
git clone https://github.com/YOUR_USERNAME/AI-Researcher.git
cd AI-Researcher/code/adaptive_lie_detector
```

---

## Step 4: Run Setup Script on Each Instance

### Instance 1 (Llama 3B):
```bash
cd ~/adaptive_lie_detector  # Or ~/AI-Researcher/code/adaptive_lie_detector
chmod +x setup_llama3b_instance.sh
./setup_llama3b_instance.sh
```

### Instance 2 (Qwen 14B):
```bash
cd ~/adaptive_lie_detector  # Or ~/AI-Researcher/code/adaptive_lie_detector
chmod +x setup_qwen14b_instance.sh
./setup_qwen14b_instance.sh
```

**What the scripts do:**
1. Install Ollama
2. Pull model (llama3.2:3b or qwen2.5:14b)
3. Install Python dependencies
4. Configure AWS credentials
5. Start experiment in background (nohup)

---

## Step 5: Monitor Progress

### On Instance 1:
```bash
# Watch live log
tail -f logs/systemprompt_llama3b_n200.log

# Count completed trials
grep "✓\|✗" logs/systemprompt_llama3b_n200.log | wc -l

# Check current accuracy
python3 -c "
import json
with open('data/results/sycophancy_systemprompt_only_llama3.2_3b_n200.json') as f:
    data = json.load(f)
results = data['results']
correct = sum(1 for r in results if r.get('correct'))
print(f'Progress: {len(results)}/200, {correct/len(results):.1%} accuracy')
"
```

### On Instance 2:
```bash
# Watch live log
tail -f logs/systemprompt_qwen14b_n200.log

# Count completed trials
grep "✓\|✗" logs/systemprompt_qwen14b_n200.log | wc -l

# Check current accuracy
python3 -c "
import json
with open('data/results/sycophancy_systemprompt_only_qwen2.5_14b_n200.json') as f:
    data = json.load(f)
results = data['results']
correct = sum(1 for r in results if r.get('correct'))
print(f'Progress: {len(results)}/200, {correct/len(results):.1%} accuracy')
"
```

---

## Step 6: Download Results (After Completion)

```bash
# From your local machine
cd /Users/mediratta/code/AI-Researcher/code/adaptive_lie_detector

# Download results from Instance 1
scp -i ~/.ssh/YOUR_KEY.pem \
  ubuntu@INSTANCE_1_IP:~/adaptive_lie_detector/data/results/sycophancy_systemprompt_only_llama3.2_3b_n200_final.json \
  data/results/

# Download results from Instance 2
scp -i ~/.ssh/YOUR_KEY.pem \
  ubuntu@INSTANCE_2_IP:~/adaptive_lie_detector/data/results/sycophancy_systemprompt_only_qwen2.5_14b_n200_final.json \
  data/results/
```

---

## Step 7: Terminate Instances (After Completion)

```bash
# Get instance IDs
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=llama3b-systemprompt,qwen14b-systemprompt" \
  --query 'Reservations[*].Instances[*].InstanceId' \
  --output text

# Terminate instances
aws ec2 terminate-instances --instance-ids i-XXXXXXXXX i-YYYYYYYYY
```

---

## Troubleshooting

### If experiment crashes:
```bash
# Check log for errors
tail -100 logs/systemprompt_llama3b_n200.log

# Restart with resume (will skip completed trials)
cd ~/adaptive_lie_detector
source .venv/bin/activate
export AWS_REGION=us-east-1
nohup .venv/bin/python3 experiments/run_sycophancy_systemprompt_only.py \
  --model llama3.2:3b \
  --n 200 \
  --resume \
  > logs/systemprompt_llama3b_n200_restart.log 2>&1 &
```

### If Ollama fails:
```bash
# Check Ollama status
sudo systemctl status ollama

# Restart Ollama
sudo systemctl restart ollama
ollama list  # Should show llama3.2:3b or qwen2.5:14b
```

### If Bedrock fails:
```bash
# Test Bedrock access
aws bedrock list-foundation-models --region us-east-1

# Check credentials
cat ~/.aws/credentials
echo $AWS_REGION
```

---

## Expected Timeline

- **Setup per instance:** ~10-15 minutes
- **Experiment runtime:** ~18-20 hours per instance (parallel)
- **Total wall-clock time:** ~20 hours

---

## Cost Estimate

- **Instance 1 (g5.xlarge):** $1.00/hr × 20 hrs = $20
- **Instance 2 (g5.2xlarge):** $1.50/hr × 20 hrs = $30
- **Bedrock API:** ~$12 (400 calls × $0.03)
- **Total:** ~$62
