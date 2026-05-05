#!/bin/bash
# Setup for Gemma 2 27B on g5.2xlarge
# Qwen 32B Replication Experiment

set -e

echo "=========================================="
echo "Gemma 2 27B Setup (Qwen 32B Replication)"
echo "=========================================="

# Install Ollama
echo "[1/6] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
sudo systemctl enable ollama
sleep 5

# Pull Gemma 2 27B (not -instruct suffix, just gemma2:27b)
echo "[2/6] Pulling gemma2:27b model (~15 minutes)..."
ollama pull gemma2:27b

# Extract codebase
echo "[3/6] Extracting codebase..."
cd ~
mkdir -p adaptive_lie_detector
cd adaptive_lie_detector

# Setup Python
echo "[4/6] Installing Python dependencies..."
python3 -m venv .venv
.venv/bin/pip install -q requests anthropic scikit-learn python-dotenv numpy scipy joblib boto3 botocore openai

# Create directories
mkdir -p logs data/results

echo "[5/6] Configuring AWS credentials..."
mkdir -p ~/.aws
cat > ~/.aws/credentials <<'EOF'
[default]
aws_access_key_id = YOUR_AWS_ACCESS_KEY_ID
aws_secret_access_key = YOUR_AWS_SECRET_ACCESS_KEY
EOF

cat > ~/.aws/config <<'EOF'
[default]
region = us-west-2
EOF

# Start experiment
echo "[6/6] Starting experiment..."
export AWS_REGION=us-west-2
export PYTHONPATH=/home/ubuntu/adaptive_lie_detector
nohup .venv/bin/python3 experiments/run_prompt_equalized.py \
  --model gemma2:27b \
  --n_samples 100 \
  --resume \
  > logs/gemma2_27b_n100.log 2>&1 &

echo ""
echo "=========================================="
echo "✓ Gemma 2 27B experiment started!"
echo "=========================================="
echo "Monitor: tail -f ~/adaptive_lie_detector/logs/gemma2_27b_n100.log"
echo "Expected completion: 18-24 hours"
echo "=========================================="
