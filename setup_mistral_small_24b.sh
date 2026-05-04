#!/bin/bash
# Setup script for Mistral-Small-3.1-24B experiment (Qwen 32B replication)
# Instance: g5.xlarge (NVIDIA A10G 23GB)

set -e

echo "=========================================="
echo "Mistral-Small-3.1-24B Setup"
echo "Experiment: Qwen 32B Replication (n=100)"
echo "=========================================="

# Extract codebase
echo "[1/8] Extracting codebase..."
cd ~
tar xzf adaptive_lie_detector.tar.gz -C .
cd adaptive_lie_detector

# Update system
echo "[2/8] Updating system packages..."
sudo apt-get update -qq

# Install Python venv package
echo "[3/8] Installing Python venv..."
sudo apt install -y python3.12-venv

# Install Ollama
echo "[4/8] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
sudo systemctl enable ollama
sleep 5

# Pull Mistral-Small-3.1-24B model
echo "[5/8] Pulling mistral-small:24b model (this may take 15-20 minutes)..."
ollama pull mistral-small:24b

# Create virtual environment and install dependencies
echo "[6/8] Installing Python dependencies..."
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install -q requests anthropic scikit-learn python-dotenv numpy scipy joblib boto3 botocore openai

# Configure AWS credentials
echo "[7/8] Configuring AWS credentials..."
mkdir -p ~/.aws
cat > ~/.aws/credentials <<'EOF'
[default]
aws_access_key_id = YOUR_AWS_ACCESS_KEY_ID
aws_secret_access_key = YOUR_AWS_SECRET_ACCESS_KEY
EOF

cat > ~/.aws/config <<'EOF'
[default]
region = us-east-1
EOF

# Create logs directory
mkdir -p logs

# Start experiment
echo "[8/8] Starting experiment..."
export AWS_REGION=us-east-1
export PYTHONPATH=/home/ubuntu/adaptive_lie_detector
nohup .venv/bin/python3 experiments/run_prompt_equalized.py \
  --model mistral-small:24b \
  --n_samples 100 \
  --resume \
  > logs/mistral_small_24b_n100.log 2>&1 &

EXPERIMENT_PID=$!
echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo "Experiment started with PID: $EXPERIMENT_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/mistral_small_24b_n100.log"
echo ""
echo "Check progress:"
echo '  grep "Trial\|Accuracy" logs/mistral_small_24b_n100.log | tail -20'
echo ""
echo "Expected completion: ~18-24 hours"
echo "=========================================="
