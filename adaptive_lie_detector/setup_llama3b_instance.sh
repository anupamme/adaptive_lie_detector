#!/bin/bash
# Setup script for g5.xlarge instance - Llama 3.2 3B experiment
# Run this after launching the instance

set -e

echo "=========================================="
echo "Instance 1: Llama 3.2 3B Setup"
echo "=========================================="

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update -qq

# Install Ollama
echo "[2/8] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
echo "[3/8] Starting Ollama service..."
sudo systemctl start ollama
sudo systemctl enable ollama
sleep 5

# Pull Llama 3.2 3B model
echo "[4/8] Pulling llama3.2:3b model (this may take 5-10 minutes)..."
ollama pull llama3.2:3b

# Clone repository
echo "[5/8] Setting up repository..."
cd ~
if [ -d "AI-Researcher" ]; then
    echo "Repository already exists, pulling latest..."
    cd AI-Researcher
    git pull
else
    # NOTE: Replace with actual repo URL when available
    echo "ERROR: Please manually git clone your repository to ~/AI-Researcher"
    echo "Then run: cd ~/AI-Researcher/code/adaptive_lie_detector"
    exit 1
fi

cd code/adaptive_lie_detector

# Create virtual environment and install dependencies
echo "[6/8] Installing Python dependencies..."
python3 -m venv .venv
source .venv/bin/activate
pip install -q requests anthropic scikit-learn python-dotenv numpy scipy joblib

# Configure AWS credentials
echo "[7/8] Configuring AWS credentials..."
mkdir -p ~/.aws
cat > ~/.aws/credentials <<EOF
[default]
aws_access_key_id = YOUR_AWS_ACCESS_KEY_ID
aws_secret_access_key = YOUR_AWS_SECRET_ACCESS_KEY
EOF

cat > ~/.aws/config <<EOF
[default]
region = us-west-2
EOF

# Create logs directory
mkdir -p logs

# Start experiment
echo "[8/8] Starting experiment (n=200)..."
export AWS_REGION=us-west-2
nohup .venv/bin/python3 experiments/run_sycophancy_systemprompt_only.py \
  --model llama3.2:3b \
  --n 200 \
  --resume \
  > logs/systemprompt_llama3b_n200.log 2>&1 &

EXPERIMENT_PID=$!
echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo "Experiment started with PID: $EXPERIMENT_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/systemprompt_llama3b_n200.log"
echo ""
echo "Check progress:"
echo '  grep "✓\|✗" logs/systemprompt_llama3b_n200.log | wc -l'
echo ""
echo "Expected completion: ~18-20 hours"
echo "=========================================="
