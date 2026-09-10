#!/usr/bin/env python3
"""probe_audit_1_generate_extract.py

Step 1 of the probe audit. For each matched claim pair and each factorial cell,
generate the model's response and capture per-layer residual-stream activations.

Two passes:
  - instructed: 4 cells {T_H, F_H, T_D, F_D}  (asymmetric honest/deceive prompts)
  - equalized:  2 cells {T_N, F_N}            (shared NEUTRAL prompt)

For each trial we record activations at the model's RESPONSE tokens (not the
instruction), in two poolings (last-token, mean-pool) and two contexts:
  - full : response encoded WITH its own instruction still in context
  - ctrl : response re-encoded under the NEUTRAL prompt, instruction REMOVED
           (the pre-registered "dumbest way it's wrong" control -- if the probe
           only reads the lie-instruction tokens, `ctrl` separability collapses)

Activations are saved as float16 .npz (one file per pass) plus an aligned
metadata JSON (pair_id, V, E, cell, response text, refusal count).

Usage:
    cd code/adaptive_lie_detector
    .venv/bin/python3 experiments/probe_audit_1_generate_extract.py --smoke
    .venv/bin/python3 experiments/probe_audit_1_generate_extract.py \
        --model Qwen/Qwen3-4B-Instruct-2507 --device mps
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.probe_audit_common import (  # noqa: E402
    DEFAULT_MODEL, DATA_DIR, INSTRUCTED_CELLS, EQUALIZED_CELLS,
    DEFAULT_CLAIM_SET, CLAIM_SETS, NEUTRAL_SYSTEM_PROMPT, OPENING_QUESTION,
    resolve_claim_set, system_prompt_for_cell, count_refusal_markers,
)


def get_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable; falling back to cpu")
        return "cpu"
    return requested


def dtype_for(device, requested):
    if requested != "auto":
        return {"float16": torch.float16, "float32": torch.float32,
                "bfloat16": torch.bfloat16}[requested]
    return torch.float16 if device != "cpu" else torch.float32


def store_dtype_for(model_name, requested):
    """Numpy dtype the pooled activations are SAVED in.

    float16 for the Qwen configurations, whose committed artifacts must stay
    byte-identical. Gemma-3's residual stream carries outlier features of order
    3e5 -- above float16's 65504 ceiling from layer 6 up, and 2.4e5 at the
    pre-registered layer -- so a float16 forward pass overflows to NaN and even
    a bfloat16 pass would be stored as inf. Those configurations save float32;
    the probe code casts to float32 anyway (PREREG_EXP_WP.md DEVIATION 8)."""
    if requested != "auto":
        return {"float16": np.float16, "float32": np.float32}[requested]
    return np.float32 if "gemma" in model_name.lower() else np.float16


def text_config(cfg):
    """The text half of a config. Multimodal checkpoints (e.g. gemma-3-4b-it,
    which loads as Gemma3ForConditionalGeneration) keep num_hidden_layers and
    hidden_size under `text_config`, not at the top level; for a plain causal LM
    this returns the config itself. Model construction only -- the extraction,
    pooling and probe code are unchanged (PREREG_EXP_WP.md DEVIATION 7)."""
    getter = getattr(cfg, "get_text_config", None)
    if getter is not None:
        return getter()
    return getattr(cfg, "text_config", cfg)


def hidden_geometry(cfg):
    """(n_hidden_states, hidden_dim) for the text stack, incl. the embedding layer."""
    tc = text_config(cfg)
    return tc.num_hidden_layers + 1, tc.hidden_size


class Extractor:
    def __init__(self, model_name, device, dtype, max_new_tokens=200,
                 store_dtype=np.float16):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.store_dtype = store_dtype
        print(f"Loading {model_name} on {device} ({dtype})...")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, trust_remote_code=True,
        )
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.n_hidden, self.hidden_dim = hidden_geometry(self.model.config)
        print(f"  n_hidden_states={self.n_hidden}, hidden_dim={self.hidden_dim}")

    def _ids(self, messages, add_generation_prompt):
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        return self.tok(text, return_tensors="pt").input_ids

    @torch.no_grad()
    def _pool_layers(self, full_ids, start):
        """Forward `full_ids`; return (last_token[n_hidden,dim], mean_pool[...])
        pooled over positions [start:]."""
        inp = full_ids.to(self.device)
        out = self.model(inp, output_hidden_states=True)
        hs = out.hidden_states  # tuple(n_hidden) of (1, seq, dim)
        last = np.empty((self.n_hidden, self.hidden_dim), dtype=self.store_dtype)
        mean = np.empty((self.n_hidden, self.hidden_dim), dtype=self.store_dtype)
        for li, h in enumerate(hs):
            resp = h[0, start:, :].float()
            last[li] = h[0, -1, :].float().cpu().numpy().astype(self.store_dtype)
            mean[li] = resp.mean(0).cpu().numpy().astype(self.store_dtype)
        del out, hs
        return last, mean

    @torch.no_grad()
    def run_trial(self, system_prompt, shown_claim):
        """Generate a response and extract activations (full + ctrl contexts)."""
        opening = OPENING_QUESTION.replace("{CLAIM}", shown_claim)
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": opening}]
        prompt_ids = self._ids(msgs, add_generation_prompt=True)
        prompt_len = prompt_ids.shape[1]

        attn = torch.ones_like(prompt_ids)
        gen = self.model.generate(
            prompt_ids.to(self.device), attention_mask=attn.to(self.device),
            max_new_tokens=self.max_new_tokens,
            do_sample=False, pad_token_id=self.tok.eos_token_id)
        full_ids = gen[:, :].cpu()
        gen_ids = full_ids[:, prompt_len:]
        response = self.tok.decode(gen_ids[0], skip_special_tokens=True).strip()

        # full context: response WITH instruction in context
        full_last, full_mean = self._pool_layers(full_ids, prompt_len)

        # ctrl context: NEUTRAL prompt + same response tokens, instruction REMOVED
        neutral_prefix = self._ids(
            [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
             {"role": "user", "content": opening}], add_generation_prompt=True)
        ctrl_ids = torch.cat([neutral_prefix, gen_ids], dim=1)
        ctrl_last, ctrl_mean = self._pool_layers(ctrl_ids, neutral_prefix.shape[1])

        if self.device == "mps":
            torch.mps.empty_cache()
        return response, (full_last, full_mean, ctrl_last, ctrl_mean)


def run_pass(ext, cells, pairs, pass_name):
    meta = []
    arr = {k: [] for k in ("full_last", "full_mean", "ctrl_last", "ctrl_mean")}
    total = len(pairs) * len(cells)
    pbar = tqdm(total=total, desc=pass_name)
    for pair_idx, (true_claim, false_claim) in pairs:
        for cell, V, E in cells:
            sysp, shown = system_prompt_for_cell(cell, true_claim, false_claim)
            response, (fl, fm, cl, cm) = ext.run_trial(sysp, shown)
            if not pbar.n:
                # Fail on the first trial, not two hours later at the probe step:
                # a compute dtype too narrow for the model's residual stream
                # overflows to inf/NaN in every later layer at once.
                for name, a in (("full_last", fl), ("full_mean", fm),
                                ("ctrl_last", cl), ("ctrl_mean", cm)):
                    if not np.isfinite(a).all():
                        bad = [i for i in range(a.shape[0])
                               if not np.isfinite(a[i]).all()]
                        raise SystemExit(
                            f"non-finite activations on the first trial "
                            f"({name}, {len(bad)}/{a.shape[0]} layers, first "
                            f"{bad[0]}): the compute or storage dtype cannot hold "
                            f"this model's residual stream. Retry with "
                            f"--dtype bfloat16 --store_dtype float32.")
            arr["full_last"].append(fl); arr["full_mean"].append(fm)
            arr["ctrl_last"].append(cl); arr["ctrl_mean"].append(cm)
            meta.append({
                "pass": pass_name, "cell": cell, "pair_id": pair_idx,
                "V": V, "E": E, "shown_claim": shown,
                "response": response[:1000],
                "refusal_count": count_refusal_markers(response),
            })
            pbar.update(1)
    pbar.close()
    arr = {k: np.stack(v).astype(ext.store_dtype) for k, v in arr.items()}
    return meta, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model_tag", default=None,
                    help="output tag; defaults to the model name. Give an explicit "
                         "tag (e.g. Qwen3-4B-Instruct-2507_v2) so a new claim set "
                         "cannot overwrite committed artifacts of another one.")
    ap.add_argument("--claim_set", default=DEFAULT_CLAIM_SET,
                    choices=sorted(CLAIM_SETS),
                    help="v1 = exploratory set; v2 = the disjoint confirmatory set "
                         "(PREREG_EXP_WP.md)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "float32", "bfloat16"])
    ap.add_argument("--store_dtype", default="auto",
                    choices=["auto", "float16", "float32"],
                    help="dtype the .npz activations are saved in; auto = float16 "
                         "except for Gemma, whose outlier features exceed float16 "
                         "range (see store_dtype_for)")
    ap.add_argument("--passes", default="both",
                    choices=["instructed", "equalized", "both"])
    ap.add_argument("--n_pairs", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run: 2 pairs, cached small model default")
    ap.add_argument("--manifest_only", action="store_true",
                    help="Collect nothing; rebuild the manifest from the .npz/meta "
                         "files already on disk for this --model_tag. For recovering "
                         "from a run interrupted between passes.")
    ap.add_argument("--out_dir", default=DATA_DIR)
    args = ap.parse_args()

    if args.smoke:
        if args.model == DEFAULT_MODEL:
            args.model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # already cached
        args.n_pairs = min(args.n_pairs, 2)
        args.max_new_tokens = 60

    claims = resolve_claim_set(args.claim_set)
    pairs = list(enumerate(claims))[:args.n_pairs]
    if args.manifest_only:
        passes = []
    else:
        passes = (["instructed", "equalized"] if args.passes == "both"
                  else [args.passes])
    cellmap = {"instructed": INSTRUCTED_CELLS, "equalized": EQUALIZED_CELLS}

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = args.model_tag or args.model.split("/")[-1].replace(".", "_")

    # Refuse to overwrite artifacts collected under a different claim set: the v1
    # results are committed and exploratory, and silently replacing them with v2
    # activations under the same tag would be unrecoverable.
    prev_path = os.path.join(args.out_dir, f"manifest_{model_tag}.json")
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev_set = json.load(f).get("claim_set", "v1")
        if prev_set != args.claim_set:
            raise SystemExit(
                f"refusing to overwrite: {prev_path} was collected on claim set "
                f"{prev_set!r} but --claim_set is {args.claim_set!r}. Pass a "
                f"distinct --model_tag (e.g. {model_tag}_{args.claim_set}).")

    device = get_device(args.device)
    dtype = dtype_for(device, args.dtype)
    store_dtype = store_dtype_for(args.model, args.store_dtype)
    if args.manifest_only:
        # Rebuild the manifest from activations already on disk, without loading
        # weights. Needed when a run is interrupted between passes: the .npz and
        # meta files of the completed passes are intact, and re-collecting them
        # would cost an hour of generation to recover pure bookkeeping.
        ext = None
        n_hidden, hidden_dim = hidden_geometry(AutoConfig.from_pretrained(args.model))
    else:
        ext = Extractor(args.model, device, dtype,
                        max_new_tokens=args.max_new_tokens,
                        store_dtype=store_dtype)
        n_hidden, hidden_dim = ext.n_hidden, ext.hidden_dim

    # Fractional-depth layer for the confirmatory probe, derived and recorded here
    # rather than selected later (PREREG_EXP_WP.md §2): layer 16 of Qwen3-4B's 37
    # hidden states, carried across models at the same relative depth.
    prereg_layer = round(16 / 36 * (n_hidden - 1))
    manifest = {"model": args.model, "model_tag": model_tag, "device": device,
                "dtype": str(dtype), "store_dtype": np.dtype(store_dtype).name,
                "n_pairs": args.n_pairs,
                "claim_set": args.claim_set,
                "max_new_tokens": args.max_new_tokens,
                "n_hidden_states": n_hidden, "hidden_dim": hidden_dim,
                "prereg_pooling": "full_mean", "prereg_layer": prereg_layer,
                "smoke": args.smoke, "passes": {}}
    print(f"  claim set {args.claim_set} ({len(pairs)} pairs); "
          f"pre-registered probe config: full_mean layer {prereg_layer}")

    # Carry forward any pass not collected in this invocation whose artifacts are
    # already on disk, so a single-pass re-run does not emit a manifest that hides
    # the other pass. n_trials is read from the stored records, never assumed.
    for pname in ("instructed", "equalized"):
        if pname in passes:
            continue
        npz_path = os.path.join(args.out_dir, f"acts_{model_tag}_{pname}.npz")
        meta_path = os.path.join(args.out_dir, f"meta_{model_tag}_{pname}.json")
        if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
            continue
        with open(meta_path) as f:
            recs = json.load(f)["records"]
        manifest["passes"][pname] = {"n_trials": len(recs),
                                     "npz": os.path.basename(npz_path),
                                     "meta": os.path.basename(meta_path),
                                     "carried_forward": True}
        print(f"  carried forward existing pass {pname!r}: {len(recs)} trials")

    for pname in passes:
        meta, arr = run_pass(ext, cellmap[pname], pairs, pname)
        npz_path = os.path.join(args.out_dir, f"acts_{model_tag}_{pname}.npz")
        meta_path = os.path.join(args.out_dir, f"meta_{model_tag}_{pname}.json")
        np.savez_compressed(npz_path, **arr)
        with open(meta_path, "w") as f:
            json.dump({"manifest_ref": model_tag, "records": meta}, f, indent=2)
        # sanity check: refusal counts should be higher for honest-on-false
        rc_by_cell = {}
        for m in meta:
            rc_by_cell.setdefault(m["cell"], []).append(m["refusal_count"])
        print(f"\n[{pname}] {len(meta)} trials -> {npz_path}")
        print(f"  activation shape per array: {arr['full_last'].shape}")
        for cell, rcs in rc_by_cell.items():
            print(f"    {cell}: mean refusal markers = {np.mean(rcs):.2f}")
        manifest["passes"][pname] = {"n_trials": len(meta), "npz": os.path.basename(npz_path),
                                     "meta": os.path.basename(meta_path)}

    with open(os.path.join(args.out_dir, f"manifest_{model_tag}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest -> {os.path.join(args.out_dir, f'manifest_{model_tag}.json')}")


if __name__ == "__main__":
    main()
