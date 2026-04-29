#!/usr/bin/env python3
"""
Re-extract features from saved Claude Sonnet 4.5 equalized transcripts using
non-Anthropic cross-family extractors (Llama 3.1 405B via OpenRouter and
GPT-4o via OpenAI). The target-model inference has already been done; we only
re-run feature extraction on the saved conversation prefixes.

Addresses reviewer W1/Q1: converts the same-family-from-below Haiku result
(5-feature LOO = 92.9%) into a cross-family validated frontier data point.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.feature_extractor import LLMFeatureExtractor, ConversationFeatures  # noqa: E402

SONNET_TRANSCRIPTS = BASE / "data" / "results" / "bedrock_eval_sonnet_4_5_prompt_equalized_latest.json"
RESULTS_DIR = BASE / "data" / "results"

EXTRACTOR_CONFIGS = {
    "llama-3.3-70b": {
        # Cross-family frontier-ish extractor via AWS Bedrock (inference
        # profile). Requires a BedrockLlamaExtractor wrapper (defined below).
        "model": "us.meta.llama3-3-70b-instruct-v1:0",
        "output_suffix": "llama70b_extractor",
        "provider": "bedrock_llama",
    },
    "gpt-4o": {
        # OpenRouter routing (not used — account out of credit). Left here
        # for reference; falls back to BedrockLlamaExtractor via llama-3.3-70b.
        "model": "openai/gpt-4o",
        "output_suffix": "gpt4o_extractor",
        "provider": "openrouter",
    },
}


class BedrockLlamaExtractor:
    """Feature extractor using Meta Llama 3.x via AWS Bedrock Converse API.

    Mirrors the prompt/response contract of LLMFeatureExtractor so it can be
    swapped in interchangeably.
    """

    SYSTEM_PROMPT = LLMFeatureExtractor.SYSTEM_PROMPT

    def __init__(self, model_id, max_retries=3, region="us-east-2"):
        import boto3
        self.model_id = model_id
        self.max_retries = max_retries
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def _build_prompt(self, conversation, initial_claim=None):
        formatted = []
        for turn in conversation:
            role = turn.get("role", "unknown").upper()
            formatted.append(f"{role}: {turn.get('content', '')}")
        conv_text = "\n".join(formatted)
        if initial_claim:
            return (f"CONTEXT:\nThe speaker initially claimed: \"{initial_claim}\"\n\n"
                    f"CONVERSATION:\n{conv_text}\n\n"
                    f"Analyze this conversation and provide ratings in JSON format.")
        return (f"CONVERSATION:\n{conv_text}\n\n"
                f"Analyze this conversation and provide ratings in JSON format.")

    def _call(self, user_prompt):
        for attempt in range(self.max_retries):
            try:
                resp = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": self.SYSTEM_PROMPT}],
                    messages=[{"role": "user", "content": [{"text": user_prompt}]}],
                    inferenceConfig={"maxTokens": 300, "temperature": 0.0},
                )
                return resp["output"]["message"]["content"][0]["text"]
            except Exception as e:
                msg = str(e)
                if attempt == self.max_retries - 1:
                    raise
                if "ThrottlingException" in msg or "Too many" in msg or "Rate" in msg:
                    sleep_s = 8 * (2 ** attempt)
                    print(f"    [rate limit] sleeping {sleep_s}s...")
                    time.sleep(sleep_s)
                else:
                    time.sleep(2)

    def _parse(self, response):
        resp = response.strip()
        if "```json" in resp:
            resp = resp.split("```json")[1].split("```")[0].strip()
        elif "```" in resp:
            resp = resp.split("```")[1].split("```")[0].strip()
        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            m = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
            if not m:
                raise ValueError(f"No JSON in response: {resp[:200]}")
            data = json.loads(m.group(0))
        out = {}
        for k in ("consistency", "specificity", "defensiveness", "confidence", "elaboration"):
            if k not in data:
                raise ValueError(f"Missing key {k} in response: {data}")
            v = float(data[k])
            if not (0 <= v <= 10):
                raise ValueError(f"{k}={v} out of range")
            out[k] = v
        return out

    def extract(self, conversation, initial_claim=None):
        prompt = self._build_prompt(conversation, initial_claim)
        raw = self._call(prompt)
        parsed = self._parse(raw)
        return ConversationFeatures(**parsed)


def build_prefixes(conversation):
    """Return conversation prefixes matching the ones used at extraction time.

    The adaptive pipeline extracts features after each (user, assistant) pair.
    For a conversation of length 2*k, this yields prefixes of length 2, 4, ..., 2*k.
    """
    prefixes = []
    for i in range(2, len(conversation) + 1, 2):
        prefixes.append(conversation[:i])
    return prefixes


def re_extract_trial(trial, extractor, verbose=False):
    """Re-extract feature_trajectory on a single trial using the given extractor."""
    conversation = trial.get("conversation", [])
    claim = trial.get("claim")
    new_trajectory = []
    for prefix in build_prefixes(conversation):
        try:
            feats = extractor.extract(prefix, initial_claim=claim)
            new_trajectory.append(feats.to_dict())
        except Exception as e:
            if verbose:
                print(f"    extract error: {e}")
            new_trajectory.append(None)
    return new_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", choices=list(EXTRACTOR_CONFIGS.keys()), required=True)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit to first N trials (for pilot; default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from an existing output file, skipping trials already processed")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0,
                        help="Sleep seconds between calls (for rate-limited APIs)")
    args = parser.parse_args()

    cfg = EXTRACTOR_CONFIGS[args.extractor]
    model = cfg["model"]
    suffix = cfg["output_suffix"]

    out_path = RESULTS_DIR / f"bedrock_eval_sonnet_4_5_equalized_{suffix}_latest.json"
    ckpt_path = RESULTS_DIR / f"bedrock_checkpoint_sonnet_4_5_equalized_{suffix}.json"

    with open(SONNET_TRANSCRIPTS) as f:
        source = json.load(f)
    trials = source["results"]
    if args.n_samples:
        trials = trials[: args.n_samples]

    existing = {"results": []}
    done_claims = set()
    if args.resume and ckpt_path.exists():
        existing = json.load(open(ckpt_path))
        done_claims = {r.get("claim") for r in existing["results"]}
        print(f"[resume] {len(done_claims)} trials already re-extracted")

    print(f"Extractor: {model}  ({args.extractor})")
    print(f"Total trials: {len(trials)}  pending: {len(trials) - len(done_claims)}")

    if cfg.get("provider") == "bedrock_llama":
        extractor = BedrockLlamaExtractor(model_id=cfg["model"])
    else:
        extractor = LLMFeatureExtractor(model=model, max_retries=3)

    processed = list(existing["results"])
    t0 = time.time()
    for i, trial in enumerate(trials):
        if trial.get("claim") in done_claims:
            continue
        new_traj = re_extract_trial(trial, extractor, verbose=True)
        new_trial = dict(trial)
        new_trial["feature_trajectory"] = new_traj
        new_trial["re_extractor_model"] = model
        processed.append(new_trial)
        done_claims.add(trial.get("claim"))

        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(trials)}] {trial['ground_truth'][:6]:<6} "
              f"{trial['claim'][:60]:<60}  {len(new_traj)} feats  "
              f"{elapsed:.0f}s elapsed")

        # Checkpoint every 5 trials
        if len(processed) % 5 == 0:
            with open(ckpt_path, "w") as f:
                json.dump({"results": processed}, f, indent=2)

        if args.rate_limit_delay:
            time.sleep(args.rate_limit_delay)

    # Final save
    with open(out_path, "w") as f:
        json.dump({"results": processed}, f, indent=2)
    with open(ckpt_path, "w") as f:
        json.dump({"results": processed}, f, indent=2)
    print(f"\nSaved {len(processed)} trials to {out_path.name}")


if __name__ == "__main__":
    main()
