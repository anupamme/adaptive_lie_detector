#!/bin/bash
# Run Llama 3.3 70B as cross-family extractor on all 8 equalized targets.
# Reuses `re_extract_equalized_cross_family.py --extractor llama70b`.
set -e
cd "$(dirname "$0")/.."

PY=.venv/bin/python3
EX="$PY experiments/re_extract_equalized_cross_family.py --extractor llama70b --resume"
D=data/results

# Target JSON (source) -> output JSON
declare -a TARGETS=(
    "ollama_eval_llama3_2_3b_prompt_equalized_20260422_102229.json:cross_family_equalized_llama3_2_3b_llama70b_extractor.json"
    "bedrock_eval_llama8b_prompt_equalized_latest.json:cross_family_equalized_llama8b_llama70b_extractor.json"
    "ollama_eval_mistral_7b_prompt_equalized_latest.json:cross_family_equalized_mistral_7b_llama70b_extractor.json"
    "ollama_eval_qwen2_5_7b_prompt_equalized_latest.json:cross_family_equalized_qwen7b_llama70b_extractor.json"
    "ollama_eval_qwen2_5_14b_prompt_equalized_latest.json:cross_family_equalized_qwen14b_llama70b_extractor.json"
    "bedrock_eval_llama70b_prompt_equalized_latest.json:cross_family_equalized_llama70b_llama70b_extractor.json"
    "bedrock_eval_haiku_prompt_equalized_latest.json:cross_family_equalized_haiku_llama70b_extractor.json"
    "ollama_eval_qwen2_5_32b_prompt_equalized_latest.json:cross_family_equalized_qwen32b_llama70b_extractor.json"
)

for entry in "${TARGETS[@]}"; do
    src="${entry%%:*}"
    dst="${entry##*:}"
    echo ""
    echo "=== $src ==="
    $EX --target_json "$D/$src" --out "$D/$dst"
done

echo ""
echo "=== All 8 targets complete ==="
