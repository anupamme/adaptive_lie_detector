#!/usr/bin/env python3
"""run_pacchiardi_qwen14b.py

Thin wrapper: EXP-K on Qwen 2.5 14B (Ollama target) for both
unrelated and related conditions.

For unrelated: uses run_pacchiardi_replication.py's code path via subprocess.
For related: standard ADAGE adaptive interrogation (new).
"""
import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
if os.path.exists(_env_path):
    for line in open(_env_path).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

os.environ.setdefault("ANTHROPIC_API_KEY", "bedrock-bypass")

import anthropic as _anthropic

from src.ollama_target_model import OllamaTargetModel
from src.feature_extractor import LLMFeatureExtractor
from src.classifier import LieDetectorClassifier
from src.utils import save_json, timestamp
from claims_equalized import generate_test_claims
from experiments.run_pacchiardi_replication import (
    NeutralPromptOllamaModel, UnrelatedQuestionInterrogator, compute_metrics,
    _patch_to_bedrock, BEDROCK_HAIKU_MODEL, run_single_trial,
)


def run_related_trial(target, detector, claim, is_lying, max_questions=8,
                      confidence_threshold=0.8):
    ground_truth = "lying" if is_lying else "truthful"
    target.reset_conversation()
    target.set_mode("lie" if is_lying else "truth", claim=claim)
    result = detector.interrogate(target_model=target, initial_claim=claim)
    pred = result.final_prediction
    return {
        "claim": claim,
        "ground_truth": ground_truth,
        "prediction": pred,
        "correct": pred == ground_truth,
        "questions_asked": result.questions_asked,
        "confidence": result.final_confidence,
        "status": str(result.status).split(".")[-1].lower(),
        "feature_trajectory": result.feature_trajectory,
        "confidence_trajectory": result.confidence_trajectory,
        "conversation": result.conversation,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["unrelated", "related"], required=True)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--max_questions", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    test_claims = generate_test_claims()[:args.n_samples]
    tag = f"qwen2_5_14b_pacchiardi_{args.condition}"
    ckpt = f"data/results/ollama_checkpoint_{tag}.json" if args.resume else None

    print("=" * 60)
    print(f"EXP-K: Qwen 2.5 14B ({args.condition})")
    print("=" * 60)
    print(f"N: {len(test_claims)}")

    extractor = LLMFeatureExtractor(model=BEDROCK_HAIKU_MODEL)
    _patch_to_bedrock(extractor)
    classifier = LieDetectorClassifier.load("data/results/trained_classifier.pkl")
    target = NeutralPromptOllamaModel(model="qwen2.5:14b")
    interrogator = UnrelatedQuestionInterrogator(seed=args.seed)

    detector = None
    if args.condition == "related":
        from src.adaptive_system import create_adaptive_detector
        detector = create_adaptive_detector(
            classifier_path="data/results/trained_classifier.pkl",
            confidence_threshold=args.threshold,
            max_questions=args.max_questions,
        )
        _patch_to_bedrock(detector.interrogator)
        _patch_to_bedrock(detector.feature_extractor)

    results = []
    done = set()
    if ckpt and os.path.exists(ckpt):
        try:
            c = json.load(open(ckpt))
            results = c.get("results", [])
            done = {r["claim"] for r in results}
            print(f"Resuming: {len(results)} done.")
        except Exception as e:
            print(f"Checkpoint load: {e}")

    print(f"\nRunning {len(test_claims)} trials ({args.condition})...")
    t0 = time.time()
    for i, (claim, is_lying) in enumerate(test_claims):
        if claim in done:
            continue
        gt = "lying" if is_lying else "truthful"
        print(f"  [{i+1:02d}/{len(test_claims)}] {'DECEP' if is_lying else 'TRUTH':5s} "
              f"{claim[:55]}", end="", flush=True)
        try:
            if args.condition == "unrelated":
                rec = run_single_trial(
                    target, extractor, classifier, claim, is_lying,
                    interrogator, max_questions=args.max_questions,
                    min_questions=2, confidence_threshold=args.threshold,
                )
            else:
                rec = run_related_trial(
                    target, detector, claim, is_lying,
                    max_questions=args.max_questions,
                    confidence_threshold=args.threshold,
                )
            results.append(rec)
            print(f"  -> {rec['questions_asked']}Q conf={rec['confidence']:.2f} "
                  f"{'OK' if rec['correct'] else 'X'}  [{time.time()-t0:.0f}s]")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "claim": claim, "ground_truth": gt,
                "prediction": "error", "correct": False,
                "questions_asked": 0, "confidence": 0.5, "status": "error",
                "feature_trajectory": [], "confidence_trajectory": [],
                "conversation": [],
            })
        if ckpt:
            with open(ckpt, "w") as f:
                json.dump({"results": results}, f, indent=2)

    metrics = compute_metrics([r for r in results if r.get("status") != "error"])
    print(f"\nACCURACY: {metrics.get('accuracy', 0):.1%}  (n={metrics.get('n_samples',0)})")

    out = {
        "experiment": f"EXP-K_pacchiardi_{args.condition}_qwen14b",
        "model": "qwen2.5:14b",
        "condition": args.condition,
        "n_samples": len(test_claims),
        "metrics": metrics,
        "results": results,
        "seed": args.seed,
    }
    out_path = f"data/results/ollama_eval_{tag}_{timestamp()}.json"
    save_json(out, out_path)
    save_json(out, f"data/results/ollama_eval_{tag}_latest.json")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
