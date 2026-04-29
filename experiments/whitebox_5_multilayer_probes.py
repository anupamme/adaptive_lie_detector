#!/usr/bin/env python3
"""whitebox_5_multilayer_probes.py

Extends the last-layer last-token LR probe to cover 3 additional
configurations (Azaria & Mitchell SAPLMA-style):
  - mid-layer last-token
  - last-layer mean-pooled
  - mid-layer mean-pooled

Runs on the same saved equalized transcripts used by the original probe
(llama_3b, mistral_7b). Extracts hidden states from all layers once per
sample, then trains LR probes at the four configurations in-memory.
Saves a single JSON of results.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-v0.1",
}

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "whitebox_probing"


def loo_accuracy(X, y, C=1.0, seed=42):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    valid_rows = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    if not valid_rows.all():
        n_drop = int((~valid_rows).sum())
        print(f"    [loo_accuracy] dropping {n_drop}/{len(y)} rows with NaN/Inf features")
    X = X[valid_rows]
    y_valid = y[valid_rows]
    n_valid = int(valid_rows.sum())
    if n_valid < 4:
        return 0.0, 0, n_valid
    preds = np.zeros(n_valid, dtype=int)
    for train_idx, test_idx in LeaveOneOut().split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[train_idx])
        Xte = sc.transform(X[test_idx])
        clf = LogisticRegression(C=C, random_state=seed, max_iter=2000)
        clf.fit(Xtr, y_valid[train_idx])
        preds[test_idx[0]] = clf.predict(Xte)[0]
    n_correct = int((preds == y_valid).sum())
    return float(n_correct / n_valid), n_correct, n_valid


def wilson_ci(correct, n, z=1.96):
    import math
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    halfw = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((centre - halfw) / denom, (centre + halfw) / denom)


def extract_all_layers(model_name, samples, device="mps"):
    print(f"Loading {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else "cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    print(f"n_hidden_layers={n_layers}")

    last_token_by_layer = [[] for _ in range(n_layers + 1)]
    mean_pool_by_layer = [[] for _ in range(n_layers + 1)]
    labels = []

    with torch.no_grad():
        for s in tqdm(samples, desc="samples"):
            resp = s["first_response"]
            lbl = s["label"]
            inp = tokenizer(resp, return_tensors="pt", truncation=True, max_length=512)
            if device != "cpu":
                inp = {k: v.to(device) for k, v in inp.items()}
            out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states  # tuple(n_layers+1) of (1, seq, dim)
            attn = inp.get("attention_mask")
            for li, h in enumerate(hs):
                last_token_by_layer[li].append(h[0, -1, :].float().cpu().numpy())
                if attn is not None:
                    mask = attn[0].float().unsqueeze(-1)
                    pooled = (h[0] * mask).sum(0) / mask.sum().clamp(min=1)
                else:
                    pooled = h[0].mean(0)
                mean_pool_by_layer[li].append(pooled.float().cpu().numpy())
            labels.append(lbl)

    del model
    if device == "mps":
        torch.mps.empty_cache()
    last_token_by_layer = [np.stack(x) for x in last_token_by_layer]
    mean_pool_by_layer = [np.stack(x) for x in mean_pool_by_layer]
    return last_token_by_layer, mean_pool_by_layer, np.array(labels), n_layers


def run_model(model_key, device="mps", full_sweep=False):
    model_name = MODELS[model_key]
    ds = DATA / f"{model_key}_equalized.json"
    with open(ds) as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"\n=== {model_key}: {len(samples)} samples ===")

    lt, mp, y, n_layers = extract_all_layers(model_name, samples, device=device)
    mid = n_layers // 2  # mid-layer index into hidden_states list

    configs = {
        "last_layer_last_token": lt[-1],
        "mid_layer_last_token": lt[mid],
        "last_layer_mean_pool": mp[-1],
        "mid_layer_mean_pool": mp[mid],
    }
    results = {}
    for name, X in configs.items():
        acc, n_correct, n_valid = loo_accuracy(X, y)
        lo, hi = wilson_ci(n_correct, n_valid)
        results[name] = {"acc": acc, "ci": [lo, hi], "n": n_valid}
        print(f"  {name:30s}  {acc*100:5.1f}%  Wilson95 [{lo*100:.1f}, {hi*100:.1f}]  n={n_valid}")

    layer_curve = None
    if full_sweep:
        print(f"\n  per-layer sweep across {len(lt)} layer positions...")
        layer_curve = {}
        for li in range(len(lt)):
            acc_lt, nc_lt, n_lt = loo_accuracy(lt[li], y)
            lo_lt, hi_lt = wilson_ci(nc_lt, n_lt)
            acc_mp, nc_mp, n_mp = loo_accuracy(mp[li], y)
            lo_mp, hi_mp = wilson_ci(nc_mp, n_mp)
            layer_curve[li] = {
                "last_token": {"acc": acc_lt, "ci": [lo_lt, hi_lt], "n": n_lt},
                "mean_pool": {"acc": acc_mp, "ci": [lo_mp, hi_mp], "n": n_mp},
            }
            print(f"    layer {li:3d}/{n_layers}  lt={acc_lt*100:5.1f}% (n={n_lt})  mp={acc_mp*100:5.1f}% (n={n_mp})")

    return {
        "model_key": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "mid_layer_idx": mid,
        "n_samples": int(len(y)),
        "configs": results,
        "layer_curve": layer_curve,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama_3b", "mistral_7b"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=str(DATA / "multilayer_probe_results.json"))
    ap.add_argument("--full_sweep", action="store_true",
                    help="Sweep ALL layers × {last-token, mean-pool} and persist a layer-accuracy curve")
    args = ap.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    out = []
    for mk in args.models:
        if mk not in MODELS:
            print(f"skip unknown {mk}"); continue
        try:
            r = run_model(mk, device=args.device, full_sweep=args.full_sweep)
            out.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"ERROR on {mk}: {e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": out}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
