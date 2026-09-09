#!/usr/bin/env python3
"""
run_external_audit_judge.py — EXP-XJ: the corpus authors' own black-box judge, run on
the two corpora that can express criterion 4.

Spec: docs/PREREG_EXP_XJ.md, committed before this file existed. Nothing here may be
changed after judgements have been inspected without recording a DEVIATION in §9 of
that document.

EXP-XA and EXP-IT tested criterion 4 with one instrument: a 14-pattern regex. A reader
may reasonably answer "maybe the regex is just blind to autonomous deception." The
audited release answers that itself: it ships its authors' own black-box detector, an
LLM-judge prompt returning a 1-7 deception score
(data/black_box_baseline/prompt.txt, wired up by deception_detection/baselines.py).
EXP-XJ runs THAT detector on the same corpora, the same channels and the same
contrasts. Prior work's detector on prior work's corpus.

This script only makes the judge calls and persists raw judgements. Every reported
number comes from analyze_external_audit_judge.py, which makes no model calls, so the
paper's figures are recomputable offline from the committed JSONL.

Usage:
    python3 experiments/run_external_audit_judge.py --smoke 5
    python3 experiments/run_external_audit_judge.py --corpus sandbagging
    python3 experiments/run_external_audit_judge.py --corpus insider
    python3 experiments/run_external_audit_judge.py --stability
"""

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# PREREG §3: the channel extractors are IMPORTED from the committed runners, never
# reimplemented, so EXP-XJ scores exactly the text EXP-XA and EXP-IT scored. The two
# corpora deliberately use different visible-channel rules: applying sandbagging's
# <step4> rule to insider trading would silently return the full text.
from run_external_audit import (  # noqa: E402
    APOLLO,
    RESULTS,
    full_transcript,
    norm_grade,
    norm_label,
    output_text,
    system_md5,
)
from run_external_audit import visible_channel as sandbag_visible  # noqa: E402
from run_insider_audit import (  # noqa: E402
    norm_grade as it_norm_grade,
    norm_report,
    norm_trade,
    raw_report,
    raw_trade,
    scenario_sig,
)
from run_insider_audit import visible_channel as insider_visible  # noqa: E402

import boto3  # noqa: E402

# --- PREREG §2: the detector, read from the release and never edited ------------------
PROMPT_FILE = os.path.join(APOLLO, "black_box_baseline", "prompt.txt")
with open(PROMPT_FILE) as _f:
    BLACK_BOX_PROMPT = _f.read().strip()          # .strip() as baselines.py does
PROMPT_MD5 = hashlib.md5(BLACK_BOX_PROMPT.encode("utf-8")).hexdigest()

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"   # PREREG §2 DEVIATION 0
AWS_REGION = (os.environ.get("AWS_REGION")
              or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2")
MAX_TOKENS = 1000        # the authors' value (rollouts/api.py)
TEMPERATURE = 0.0        # PREREG §2 DEVIATION 0, for reproducibility

RAW_PATH = os.path.join(RESULTS, "external_audit_judge_raw.jsonl")
SMOKE_PATH = os.path.join(RESULTS, "external_audit_judge_smoke.jsonl")

CORPORA = {
    "sandbagging": "rollouts/sandbagging_v2__wmdp_mmlu__llama-70b-3.3.json",
    "insider": "rollouts/insider_trading__onpolicy__llama-70b.json",
}

# (arm name, context rule, output-channel extractor) — PREREG §3/§4
ARMS = {
    "sandbagging": [
        ("visible", "drop_trailing_assistant", sandbag_visible),   # H1 primary
        ("full", "drop_trailing_assistant", full_transcript),      # H4 channel
    ],
    "insider": [
        ("visible", "nonassistant_only", insider_visible),         # H2/H3 primary
        ("full", "drop_trailing_assistant", full_transcript),      # channel
        ("visible_fullctx", "drop_trailing_assistant", insider_visible),  # H5 context
    ],
}

# PREREG AMENDMENT 1: H8, the instruction-stripped context. Exploratory, run under
# --h8 so the confirmatory arms above are never re-issued.
H8_ARMS = [("visible_nosys", "no_system", sandbag_visible)]

STABILITY_N = 100        # PREREG §4 H7
STABILITY_SEED = 42


# ---------------------------------------------------- prompt assembly (baselines.py)

def merge_messages(msgs):
    """Merge consecutive messages from the same role. Behaviour of baselines.py."""
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m.get("role"):
            out[-1]["content"] += "\n" + (m.get("content") or "")
        else:
            out.append({"role": m.get("role"), "content": m.get("content") or ""})
    return out


def format_dialogue(msgs):
    """'role: content' per merged message, newline-joined. Behaviour of baselines.py."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in merge_messages(msgs))


def context_messages(roll, mode):
    """PREREG §3. `drop_trailing_assistant` is the authors' get_context rule; on both
    corpora the trailing assistant message is a zero-length generation prefix.
    `nonassistant_only` additionally drops the agent's own earlier turns, so the judge
    reads the scenario as the operator set it up and not the agent's prior reasoning."""
    msgs = list(roll.get("input_messages") or [])
    while msgs and msgs[-1].get("role") == "assistant":
        msgs.pop()
    if mode == "nonassistant_only":
        msgs = [m for m in msgs if m.get("role") != "assistant"]
    if mode == "no_system":
        # PREREG AMENDMENT 1 (H8): drop the system message, which on the sandbagging
        # corpus carries the ENTIRE sandbagging instruction, so the judge reads only
        # what a monitor without the operator's prompt would see.
        msgs = [m for m in msgs if m.get("role") != "system"]
    return msgs


def judge_user_prompt(context, output):
    """black_box_grading_prompt_getter, verbatim in form."""
    return f"<context>\n{context}\n</context>\n\n<output>\n{output}\n</output>"


# --------------------------------------------------------- parsing (rollouts/api.py)

def parse_judgement(response_text):
    """Faithful reimplementation of deception_detection.rollouts.api.parse_judgement.

    Reimplemented rather than imported because that module imports openai/together/
    dotenv at module scope. The rule is unchanged: exactly one <judgement> tag pair,
    strip + lowercase, int if it is all digits, otherwise 'unknown'.
    """
    parts = (response_text or "").split("<judgement>")
    if len(parts) != 2:
        return "unknown"
    section = parts[1].split("</judgement>")
    if len(section) != 2:
        return "unknown"
    j = section[0].strip().lower()
    if j.isdigit():
        return int(j)
    return "unknown"


# ------------------------------------------------------------------------- transport

_LOCAL = threading.local()


def _client():
    if not hasattr(_LOCAL, "c"):
        _LOCAL.c = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _LOCAL.c


def call_judge(system, user):
    """One judge call. Returns (text, stop_reason, usage, error)."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    delays = [4, 8, 16, 32, 64]
    last = None
    for attempt, delay in enumerate(delays, 1):
        try:
            r = _client().invoke_model(modelId=MODEL_ID, body=body)
            d = json.loads(r["body"].read())
            txt = "".join(b.get("text", "") for b in d.get("content") or [])
            return txt, d.get("stop_reason"), d.get("usage"), None
        except Exception as e:                                  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            if attempt < len(delays):
                time.sleep(delay + random.uniform(0, 0.1 * delay))
    return "", None, None, last


# ------------------------------------------------------------------------ work items

def load_rolls(corpus):
    with open(os.path.join(APOLLO, CORPORA[corpus])) as f:
        d = json.load(f)
    return d["rollouts"] if isinstance(d, dict) else d


def corpus_fields(corpus, roll):
    """The labels each hypothesis needs, derived only by the imported helpers."""
    if corpus == "sandbagging":
        return {"label": norm_label(roll), "grade": norm_grade(roll),
                "topic": (roll.get("metadata") or {}).get("topic")}
    return {"label": norm_label(roll),
            "grade": it_norm_grade(roll),          # majority + tie exclusion
            "d_report": norm_report(roll),
            "d_trade": norm_trade(roll),
            "trade_raw": raw_trade(roll),
            "report_raw": raw_report(roll),
            "scenario": scenario_sig(roll)}


def build_items(corpus, arms, only_idx=None, arm_suffix=""):
    """One work item per (rollout, arm). Rollouts with an empty visible channel are
    dropped from the visible arms and counted (PREREG §3)."""
    rolls = load_rolls(corpus)
    items, empty = [], {}
    for idx, roll in enumerate(rolls):
        if only_idx is not None and idx not in only_idx:
            continue
        raw = output_text(roll)
        fields = corpus_fields(corpus, roll)
        for arm, ctx_mode, extract in arms:
            out = extract(raw)
            if not (out or "").strip():
                empty[arm] = empty.get(arm, 0) + 1
                continue
            ctx = format_dialogue(context_messages(roll, ctx_mode))
            items.append({
                "corpus": corpus, "idx": idx, "arm": arm + arm_suffix,
                "context_mode": ctx_mode,
                "system_md5_corpus": system_md5(roll),
                "n_chars_context": len(ctx), "n_chars_output": len(out),
                "user": judge_user_prompt(ctx, out),
                **fields,
            })
    return items, empty


def done_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # a transport error is retried on resume; an 'unknown' PARSE is a real
            # result (PREREG §2: excluded and counted, never re-rolled)
            if r.get("error") is None:
                keys.add((r["corpus"], r["idx"], r["arm"]))
    return keys


# ------------------------------------------------------------------------------ main

def run(items, dest, workers):
    lock = threading.Lock()
    tot = {"n": 0, "in": 0, "out": 0, "unknown": 0, "trunc": 0, "err": 0}
    t0 = time.time()

    def work(it):
        user = it.pop("user")
        txt, stop, usage, err = call_judge(BLACK_BOX_PROMPT, user)
        rec = dict(it)
        rec.update({
            "judgement": parse_judgement(txt),
            "raw_response": txt,
            "stop_reason": stop,
            "error": err,
            "user_md5": hashlib.md5(user.encode("utf-8")).hexdigest(),
            "prompt_md5": PROMPT_MD5,
            "model": MODEL_ID, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "region": AWS_REGION,
        })
        if usage:
            rec["input_tokens"] = usage.get("input_tokens")
            rec["output_tokens"] = usage.get("output_tokens")
        return rec

    with open(dest, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, it) for it in items]
        for fut in as_completed(futs):
            rec = fut.result()
            with lock:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                tot["n"] += 1
                tot["in"] += rec.get("input_tokens") or 0
                tot["out"] += rec.get("output_tokens") or 0
                tot["unknown"] += int(rec["judgement"] == "unknown")
                tot["trunc"] += int(rec.get("stop_reason") == "max_tokens")
                tot["err"] += int(rec.get("error") is not None)
                if tot["n"] % 50 == 0 or tot["n"] == len(items):
                    el = time.time() - t0
                    print(f"    {tot['n']}/{len(items)}  {el/60:5.1f} min  "
                          f"unknown={tot['unknown']} trunc={tot['trunc']} "
                          f"err={tot['err']}  tok in/out={tot['in']}/{tot['out']}",
                          flush=True)
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA))
    ap.add_argument("--smoke", type=int, default=0,
                    help="judge the first N rollouts of every arm of both corpora, "
                         "writing to a separate smoke file")
    ap.add_argument("--h8", action="store_true",
                    help="PREREG AMENDMENT 1: re-judge every sandbagging rollout on "
                         "the visible channel with the system message (the "
                         "sandbagging instruction) stripped from the context")
    ap.add_argument("--stability", action="store_true",
                    help="PREREG H7: re-judge a fixed 100-rollout sandbagging "
                         "subsample on the visible channel")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    print(f"EXP-XJ judge runner — spec docs/PREREG_EXP_XJ.md")
    print(f"  detector : {PROMPT_FILE}")
    print(f"             md5={PROMPT_MD5}  ({len(BLACK_BOX_PROMPT)} chars, verbatim)")
    print(f"  judge    : {MODEL_ID} @ {AWS_REGION}, "
          f"max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}")

    if args.smoke:
        idx = set(range(args.smoke))
        items = []
        for c in sorted(CORPORA):
            got, empty = build_items(c, ARMS[c], only_idx=idx)
            items += got
            print(f"  [smoke] {c}: {len(got)} items, empty-channel skips={empty}")
        tot = run(items, SMOKE_PATH, args.workers)
        print(f"  wrote {SMOKE_PATH}")
        print(f"  {tot}")
        return

    if args.h8:
        items, empty = build_items("sandbagging", H8_ARMS)
        print(f"  [H8] {len(items)} items (context_mode=no_system), skips={empty}")
    elif args.stability:
        sample = sorted(random.Random(STABILITY_SEED).sample(
            sorted(range(len(load_rolls("sandbagging")))), STABILITY_N))
        arms = [a for a in ARMS["sandbagging"] if a[0] == "visible"]
        items, empty = build_items("sandbagging", arms, only_idx=set(sample),
                                   arm_suffix="_rep2")
        print(f"  [H7] {len(items)} items (seed {STABILITY_SEED}), skips={empty}")
    elif args.corpus:
        items, empty = build_items(args.corpus, ARMS[args.corpus])
        print(f"  [{args.corpus}] {len(items)} items, empty-channel skips={empty}")
    else:
        raise SystemExit("give --corpus, --smoke N, --stability or --h8")

    have = done_keys(RAW_PATH)
    todo = [it for it in items
            if (it["corpus"], it["idx"], it["arm"]) not in have]
    print(f"  {len(have)} judgements already on disk; {len(todo)} to run")
    if not todo:
        print("  nothing to do")
        return

    tot = run(todo, RAW_PATH, args.workers)
    cost = tot["in"] / 1e6 * 1.0 + tot["out"] / 1e6 * 5.0
    print(f"  wrote {RAW_PATH}")
    print(f"  {tot}   approx cost ${cost:.2f}")


if __name__ == "__main__":
    main()
