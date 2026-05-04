#!/bin/bash
# Quick setup for Qwen 14B on second instance

set -e

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
sudo systemctl enable ollama
sleep 5

echo "Pulling qwen2.5:14b model (this takes 10-15 minutes)..."
ollama pull qwen2.5:14b

echo "Creating venv and installing dependencies..."
cd ~/adaptive_lie_detector
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install -q requests anthropic scikit-learn python-dotenv numpy scipy joblib boto3 botocore openai

echo "Configuring AWS credentials..."
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

echo "Creating logs directory..."
mkdir -p logs

echo "Starting experiment..."
export AWS_REGION=us-west-2
export PYTHONPATH=/home/ubuntu/adaptive_lie_detector
nohup .venv/bin/python3 experiments/run_sycophancy_systemprompt_only.py \
  --model qwen2.5:14b \
  --n 200 \
  --resume \
  > logs/systemprompt_qwen14b_n200.log 2>&1 &

echo ""
echo "✓ Setup complete!"
echo "Experiment started with PID: $!"
echo ""
echo "Monitor: tail -f logs/systemprompt_qwen14b_n200.log"
