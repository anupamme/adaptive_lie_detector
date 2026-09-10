#!/usr/bin/env python3
"""probe_audit_6_steer.py

Diagnostic 3 (causal) -- steering along the probe direction.

Decoding accuracy is correlational. If the probe direction is causally tied to
the model's honest/deceptive behavior, adding it to the residual stream during
generation should push behavior monotonically; if it merely reads the instruction
context, steering should do little. We derive the raw-space unit direction from
the E-probe at the selected layer (probe_raw_direction), then hook the
corresponding decoder block and add coef * sigma * direction to the residual at
every position, sweeping coef. sigma is the SD of the training activations'
projection onto the direction, so coef is "how many SDs along the deception axis."

Readouts per dose:
  - refusal/correction-marker density (behavioral honesty proxy, repo convention)
  - mean E-probe log-odds on a CLEAN re-encoding of the steered response (H5b,
    the mechanistic shift). The re-encoding matters: scoring the probe on the
    steered forward pass itself is tautological, because adding coef*sigma*unit
    at the block whose output is hidden_states[layer] moves the log-odds by
    exactly coef*sigma*||w_raw|| by construction. Re-encoding the generated TEXT
    under the neutral prompt with no hook instead asks whether the intervention
    changed the response in the probe's own terms -- an effect that has to
    survive passing through generation. Without this readout a null intervention
    and an ineffective one are indistinguishable (PREREG_EXP_WP.md §9 DEVIATION 4).

Control: repeat the whole sweep with a random unit direction. A real deception
axis should move the readouts monotonically while the random control stays flat.

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_6_steer.py --smoke
    .venv/bin/python3 experiments/probe_audit_6_steer.py \
        --model Qwen/Qwen3-4B-Instruct-2507 --model_tag Qwen3-4B-Instruct-2507
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import (  # noqa: E402
    DEFAULT_MODEL, DATA_DIR, SEED, DEFAULT_CLAIM_SET, CLAIM_SETS,
    NEUTRAL_SYSTEM_PROMPT, OPENING_QUESTION,
    load_pass, slice_layer, meta_vec, make_probe, probe_raw_direction,
    resolve_claim_set, count_refusal_markers,
)
from experiments.probe_audit_1_generate_extract import get_device, dtype_for  # noqa: E402


def build_direction(model_tag, pooling, layer, out_dir):
    """Train the E-probe at the selected layer; return (unit_dir, sigma)."""
    arrays, meta = load_pass(model_tag, "instructed", out_dir)
    X = slice_layer(arrays, pooling, layer)
    E = meta_vec(meta, "E")
    pipe = make_probe()
    pipe.fit(X, E)
    unit, _ = probe_raw_direction(pipe)
    sigma = float((X @ unit).std())
    return unit.astype("float32"), sigma


def decoder_blocks(model):
    """The text stack's decoder blocks, whichever depth they sit at.

    A plain causal LM keeps them at `model.model.layers`; a multimodal checkpoint
    (gemma-3-4b-it loads as Gemma3ForConditionalGeneration) nests a text model one
    level further down. Model construction only -- the hook, the direction and the
    readout are unchanged (PREREG_EXP_WP.md DEVIATION 7)."""
    paths = (("model", "layers"),
             ("model", "language_model", "layers"),
             ("language_model", "model", "layers"),
             ("language_model", "layers"))
    for path in paths:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and len(obj):
            return obj
    raise SystemExit(f"cannot locate decoder blocks on {type(model).__name__}")


class SteeredModel:
    def __init__(self, model_name, device, dtype, block_idx, max_new_tokens,
                 layer=None):
        self.device = device
        self.block_idx = block_idx
        self.max_new_tokens = max_new_tokens
        self.layer = layer  # hidden-state index the probe reads (for H5b)
        print(f"Loading {model_name} on {device} ({dtype})...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, trust_remote_code=True)
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.layers = decoder_blocks(self.model)
        self._vec = None  # torch tensor added to residual, or None

    def _hook(self, module, inputs, output):
        if self._vec is None:
            return output
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + self._vec.to(hs.dtype)
            return (hs,) + tuple(output[1:])
        return output + self._vec.to(output.dtype)

    @torch.no_grad()
    def generate(self, system_prompt, shown_claim, steer_vec):
        """Generate with `steer_vec` (numpy or None) added at the hooked block."""
        self._vec = (torch.tensor(steer_vec, device=self.device)
                     if steer_vec is not None else None)
        handle = self.layers[self.block_idx].register_forward_hook(self._hook)
        try:
            opening = OPENING_QUESTION.replace("{CLAIM}", shown_claim)
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": opening}]
            text = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
            attn = torch.ones_like(ids)
            gen = self.model.generate(
                ids, attention_mask=attn, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=self.tok.eos_token_id)
            resp = self.tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        finally:
            handle.remove()
            self._vec = None
        if self.device == "mps":
            torch.mps.empty_cache()
        return resp.strip()

    @torch.no_grad()
    def reencode_mean(self, shown_claim, response):
        """H5b: clean (unhooked) forward pass over NEUTRAL prompt + `response`.

        Returns the mean-pooled hidden state over the response tokens at
        self.layer -- the `full_mean` pooling of step 1, so the E-probe trained on
        those activations applies unchanged. No steering vector is active here by
        construction: the effect has to have made it into the text.
        """
        opening = OPENING_QUESTION.replace("{CLAIM}", shown_claim)
        msgs = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                {"role": "user", "content": opening}]
        prefix = self.tok(self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True),
            return_tensors="pt").input_ids
        resp_ids = self.tok(response, return_tensors="pt",
                            add_special_tokens=False).input_ids
        if resp_ids.shape[1] == 0:
            return None
        ids = torch.cat([prefix, resp_ids], dim=1).to(self.device)
        out = self.model(ids, output_hidden_states=True)
        h = out.hidden_states[self.layer][0, prefix.shape[1]:, :].float()
        vec = h.mean(0).cpu().numpy().astype("float32")
        del out
        if self.device == "mps":
            torch.mps.empty_cache()
        return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model_tag", default=None,
                    help="tag of the cached activations (defaults from --model)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "float32", "bfloat16"])
    ap.add_argument("--claim_set", default=DEFAULT_CLAIM_SET,
                    choices=sorted(CLAIM_SETS))
    ap.add_argument("--n_pairs", type=int, default=15)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    # On-manifold doses. sigma*||w_raw|| already equals the SD of the probe's own
    # log-odds, while real activations span about +-1.5 SD, so |coef| >> 1 pushes
    # the residual stream off the manifold the probe was fitted on. +-4 is retained
    # only as a labelled sanity dose (PREREG_EXP_WP.md §9 DEVIATION 3).
    ap.add_argument("--coefs", default="-4,-1,0,1,4")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out_dir", default=DATA_DIR)
    args = ap.parse_args()

    if args.smoke and args.model == DEFAULT_MODEL:
        args.model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        args.n_pairs = min(args.n_pairs, 2)
        args.max_new_tokens = 60
    model_tag = args.model_tag or args.model.split("/")[-1].replace(".", "_")

    with open(os.path.join(args.out_dir, f"probe2_{model_tag}.json")) as f:
        sel = json.load(f)["selected"]
    pooling, layer = sel["pooling"], sel["layer"]

    unit, sigma = build_direction(model_tag, pooling, layer, args.out_dir)
    rng = np.random.default_rng(SEED)
    rand = rng.standard_normal(unit.shape[0]).astype("float32")
    rand /= np.linalg.norm(rand)
    # decoder block whose OUTPUT is hidden_states[layer] (index 0 = embeddings)
    block_idx = max(layer - 1, 0)
    coefs = [float(c) for c in args.coefs.split(",")]
    print(f"Steering block {block_idx} (from hidden_states layer {layer}, {pooling}); "
          f"sigma={sigma:.3f}; coefs={coefs}")

    device = get_device(args.device)
    dtype = dtype_for(device, args.dtype)
    sm = SteeredModel(args.model, device, dtype, block_idx, args.max_new_tokens,
                      layer=layer)
    if block_idx >= len(sm.layers):
        block_idx = len(sm.layers) // 2
        sm.block_idx = block_idx
        print(f"  (clamped block to {block_idx})")

    # H5b: the E-probe, fitted on the instructed pass at the same (pooling, layer),
    # scored on a clean re-encoding of each steered response.
    arrays, meta = load_pass(model_tag, "instructed", args.out_dir)
    eprobe = make_probe()
    eprobe.fit(slice_layer(arrays, pooling, layer), meta_vec(meta, "E"))

    pairs = list(enumerate(resolve_claim_set(args.claim_set)))[:args.n_pairs]
    conditions = [("probe", unit), ("random", rand)]
    out_path = os.path.join(args.out_dir, f"probe6_{model_tag}.json")

    # resume: reuse any cells already computed in a prior (possibly killed) run
    out = {
        "experiment": "probe_audit_6_steer",
        "model": args.model, "model_tag": model_tag,
        "pooling": pooling, "hidden_state_layer": layer, "block_idx": block_idx,
        "sigma": sigma, "coefs": coefs, "n_pairs": args.n_pairs,
        "claim_set": args.claim_set,
        "prompt": "NEUTRAL",
        "readouts": ["correction/refusal-marker density (behavioral)",
                     "E-probe log-odds on a clean re-encoding of the steered "
                     "response (H5b, mechanistic)"],
        "results": {name: {} for name, _ in conditions},
    }
    if os.path.exists(out_path):
        prev = json.load(open(out_path))
        # only resume from a run with the same materials AND the same readouts;
        # a pre-H5b cell has no log-odds and must be regenerated, not merged
        if (prev.get("n_pairs") == args.n_pairs
                and prev.get("claim_set", "v1") == args.claim_set
                and "readouts" in prev):
            for name, _ in conditions:
                out["results"][name].update(prev.get("results", {}).get(name, {}))
            print(f"  resuming: {sum(len(v) for v in out['results'].values())} "
                  f"cells already done")
        elif prev.get("claim_set", "v1") != args.claim_set:
            raise SystemExit(
                f"refusing to overwrite: {out_path} was collected on claim set "
                f"{prev.get('claim_set', 'v1')!r}, not {args.claim_set!r}. "
                f"Pass a distinct --model_tag.")

    def save():
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    n_target = 2 * len(pairs)  # two shown claims per pair
    shown_seq = [shown for _, (t, f) in pairs for shown in (t, f)]
    for name, direction in conditions:
        for coef in coefs:
            key = str(coef)
            cell = out["results"][name].get(key)
            # cell may be complete ({mean_refusal,mean_logodds,n}) or partial ({rcs,...})
            if cell and cell.get("n", 0) >= n_target and "rcs" not in cell:
                print(f"  {name:6s} coef {coef:+.0f}: cached "
                      f"({cell['mean_refusal']:.2f}, "
                      f"log-odds {cell.get('mean_logodds', float('nan')):+.2f})")
                continue
            vec = None if coef == 0 else (coef * sigma * direction)
            rcs = list(cell["rcs"]) if cell and "rcs" in cell else []
            los = list(cell.get("los", [])) if cell else []
            if rcs:
                print(f"  {name:6s} coef {coef:+.0f}: resuming at {len(rcs)}/{n_target}")
            # generate one at a time; checkpoint after each so a kill costs one gen
            for shown in tqdm(shown_seq[len(rcs):], desc=f"{name} coef={coef:+.0f}",
                              leave=False):
                resp = sm.generate(NEUTRAL_SYSTEM_PROMPT, shown, vec)
                rcs.append(count_refusal_markers(resp))
                # H5b: clean re-encode, then the E-probe's log-odds on the text
                act = sm.reencode_mean(shown, resp)
                los.append(float("nan") if act is None
                           else float(eprobe.decision_function(act[None])[0]))
                out["results"][name][key] = {
                    "rcs": rcs, "los": los, "n": len(rcs),
                    "mean_refusal": float(np.mean(rcs)),
                    "mean_logodds": float(np.nanmean(los)),
                }
                save()
            # finalize: drop the per-trial lists so the cell reads as complete
            out["results"][name][key] = {
                "mean_refusal": float(np.mean(rcs)),
                "mean_logodds": float(np.nanmean(los)),
                "sd_logodds": float(np.nanstd(los)),
                "n_logodds": int(np.sum(~np.isnan(los))),
                "n": len(rcs),
            }
            save()
            print(f"  {name:6s} coef {coef:+.0f}: mean refusal markers = "
                  f"{np.mean(rcs):.2f}, mean probe log-odds = {np.nanmean(los):+.2f}")

    save()
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
