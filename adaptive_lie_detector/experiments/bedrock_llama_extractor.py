#!/usr/bin/env python3
"""bedrock_llama_extractor.py — Llama 3.3 70B cross-family feature extractor.

Mirrors the BedrockMistralExtractor prompt/response contract so it can be
swapped in interchangeably by re_extract_equalized_cross_family.py.

Uses the us.meta.llama3-3-70b-instruct-v1:0 Bedrock inference profile (us-east-2).
"""
import json
import re
import time

from experiments.cross_family_equalized_extractor import (
    FEATURES,
    FEATURE_EXTRACTION_SYSTEM_PROMPT,
)

BEDROCK_MODEL_ID = "us.meta.llama3-3-70b-instruct-v1:0"
BEDROCK_REGION = "us-east-2"


class BedrockLlamaExtractor:
    def __init__(
        self,
        model_id: str = BEDROCK_MODEL_ID,
        region: str = BEDROCK_REGION,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ):
        import boto3

        self.model_id = model_id
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def extract(self, conversation, initial_claim=None):
        user_prompt = self._build_prompt(conversation, initial_claim)
        response_text = self._call_bedrock(user_prompt)
        return self._parse_response(response_text)

    def _build_prompt(self, conversation, initial_claim=None):
        formatted_conv = []
        for turn in conversation:
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            formatted_conv.append(f"{role}: {content}")
        conv_text = "\n".join(formatted_conv)

        if initial_claim:
            return (
                f'CONTEXT:\n'
                f'The speaker initially claimed: "{initial_claim}"\n'
                f'\n'
                f'CONVERSATION:\n'
                f'{conv_text}\n'
                f'\n'
                f'Analyze this conversation and provide ratings in JSON format.'
            )
        return (
            f'CONVERSATION:\n'
            f'{conv_text}\n'
            f'\n'
            f'Analyze this conversation and provide ratings in JSON format.'
        )

    def _call_bedrock(self, user_prompt):
        from botocore.exceptions import ClientError

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    system=[{"text": FEATURE_EXTRACTION_SYSTEM_PROMPT}],
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": user_prompt}],
                        }
                    ],
                    inferenceConfig={
                        "maxTokens": 500,
                        "temperature": 0.1,
                    },
                )
                output_message = response["output"]["message"]
                response_text = ""
                for block in output_message["content"]:
                    if "text" in block:
                        response_text += block["text"]
                return response_text

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                last_error = e
                if error_code in ("ThrottlingException", "TooManyRequestsException",
                                  "ServiceUnavailableException"):
                    delay = self.base_delay * (2 ** attempt)
                    print(f"    Bedrock throttled ({error_code}), retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"    Bedrock error: {e}, retrying in {delay:.1f}s")
                    time.sleep(delay)
                else:
                    raise

        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")

    def _parse_response(self, response_text):
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if not m:
                raise ValueError(f"Could not find JSON in response: {text[:200]}")
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse JSON from response: {text[:200]}")

        features = {}
        for key in FEATURES:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in response")
            value = float(data[key])
            if not (0 <= value <= 10):
                raise ValueError(f"{key} value {value} out of range [0, 10]")
            features[key] = value
        return features
