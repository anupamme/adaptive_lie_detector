#!/usr/bin/env python3
"""EXP-BF: benign false-positive rate of the parameter-free refusal-count rule.

WHY THIS FILE EXISTS RATHER THAN A RE-RUN OF run_benign_fpr.py
--------------------------------------------------------------
The July 2026 collection (`run_benign_fpr.py`, `data/results/benign_fpr_*.json`)
is not publishable as collected, for two reasons found while costing the
write-up. Both are collection defects, not analysis defects, so they can only be
fixed by re-collecting:

  (1) TIMEOUTS RECORDED AS NON-FIRING RESPONSES. `OllamaTargetModel.respond`
      catches `requests.exceptions.Timeout` and returns the literal string
      "I'm not sure about that." (src/ollama_target_model.py:154). The old
      script stored that as an ordinary response with `rule_fires: 0`, so 68 of
      its 300 records are guaranteed non-fires that never reached a model: all
      50 of qwen2.5:32b and 18 of 50 of qwen2.5:14b. A false-positive rate whose
      denominator is padded with non-answers is not a false-positive rate.

      ROOT CAUSE of the 32B case, which is why this script re-checks
      availability itself: `_check_ollama_running` accepts a model when any
      installed tag startswith the part before the colon, so the absent
      "qwen2.5:32b" matches the present "qwen2.5:7b", passes the check, and then
      times out on every call. Here every target's EXACT tag must be installed
      or the script refuses to start.

  (2) RESPONSES TRUNCATED AT 500 CHARACTERS. The old script stored `resp[:500]`,
      which truncated 135 of its 232 non-timeout records (58%). Truncation can
      only hide a marker, never invent one, so a rate measured on truncated text
      is a LOWER BOUND on the full-response rate -- a bias that cuts against the
      specificity claim. This script stores the response whole.

WHAT IS DELIBERATELY UNCHANGED
------------------------------
Generation. The prompt set, the neutral system prompt, `max_tokens=300` and
`temperature=0.7` are the defaults every other Ollama experiment in this repo
uses, and `OllamaTargetModel.respond` is called unmodified, so these responses
are drawn from the same generation path as EXP-R1/EXP-C4. Only STORAGE and
TIMEOUT BOOKKEEPING differ from the July run.

Also unchanged: `src/` is not touched. A timeout is detected by its sentinel and
retried; it is never rewritten into a response.

THIS SCRIPT DOES NO SCORING. It stores prompts and responses only, so the
14-pattern rule exists in exactly one place in the repository
(`hedging_baseline.REFUSAL_PATTERNS`) and is applied in exactly one place
(`analyze_benign_fpr.py`). The July script's private 4-pattern copy of the rule
is the reason its stored `rule_fires` field does not correspond to any figure in
the paper.

TARGETS
-------
The five targets of EXP-C4 (Appendix, criterion-4 contrast). qwen2.5:32b is
excluded on the ground the paper already states for EXP-C4: it cannot be run
within the available memory -- here it is not installed at all.

OUTPUT (one file per target; the July files are left in place, not overwritten)
------------------------------------------------------------------------------
  data/results/benign_fpr2_<model>.json

USAGE
-----
  cd code/adaptive_lie_detector
  python3 experiments/run_benign_fpr2.py --all
  python3 experiments/run_benign_fpr2.py --model mistral:7b
"""

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ollama_target_model import (  # noqa: E402
    OllamaTargetModel, _OLLAMA_BASE,
)
from experiments.run_benign_fpr import (  # noqa: E402
    BENIGN_PROMPTS, NEUTRAL_SYSTEM_PROMPT,
)

# The five EXP-C4 targets. qwen2.5:32b is not here; see the docstring.
TARGETS = [
    "llama3.2:3b",
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "qwen2.5:14b",
]

# src/ollama_target_model.py:154 returns this string on requests Timeout. It is
# a sentinel, not a response, and is never stored as one.
TIMEOUT_SENTINEL = "I'm not sure about that."

MAX_ATTEMPTS = 3
RESULTS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "results")


def installed_tags():
    r = requests.get(f"{_OLLAMA_BASE}/api/tags", timeout=10)
    r.raise_for_status()
    return {m["name"] for m in r.json().get("models", [])}


def require_exact_tags(targets):
    """Refuse to start unless every target's exact tag is installed.

    `_check_ollama_running`'s prefix match would accept an absent tag whenever a
    sibling of the same family is present, which is how the July run produced 50
    timeouts for a model that was never installed.
    """
    have = installed_tags()
    missing = [t for t in targets if t not in have]
    if missing:
        raise SystemExit(
            "refusing to collect: these exact tags are not installed: %s\n"
            "installed: %s\n"
            "A missing tag passes _check_ollama_running's prefix match and then "
            "times out on every call, which is the defect this script exists to "
            "avoid. Pull the tag or remove the target from TARGETS."
            % (", ".join(missing), ", ".join(sorted(have))))


class NeutralOllamaModel(OllamaTargetModel):
    """Target under the neutral system prompt. Generation defaults untouched."""

    def __init__(self, model):
        super().__init__(model=model)
        self.system_prompt = NEUTRAL_SYSTEM_PROMPT


def collect(model_name):
    out = os.path.join(RESULTS,
                       "benign_fpr2_%s.json" % model_name.replace(":", "_")
                                                         .replace(".", "_"))
    print("=" * 68)
    print("EXP-BF collection: %s  (%d benign prompts)" % (model_name,
                                                          len(BENIGN_PROMPTS)))
    print("=" * 68)

    target = NeutralOllamaModel(model=model_name)
    records, timeouts = [], []
    t0 = time.time()

    for i, prompt in enumerate(BENIGN_PROMPTS, 1):
        resp, attempts = None, 0
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            target.reset_conversation()
            r = target.respond(prompt)
            if r.strip() != TIMEOUT_SENTINEL:
                resp = r
                break
            print("    [%02d] timeout sentinel on attempt %d" % (i, attempts))

        if resp is None:
            # Still timing out after MAX_ATTEMPTS. Recorded as an error, and
            # NOT as a non-firing response: it is excluded from the denominator
            # by analyze_benign_fpr.py rather than counted as a pass.
            timeouts.append(prompt)
            records.append({"prompt": prompt, "response": None,
                            "error": "timeout_after_%d_attempts" % MAX_ATTEMPTS,
                            "attempts": attempts})
            print("  [%02d/%d] TIMEOUT (excluded)  %s"
                  % (i, len(BENIGN_PROMPTS), prompt[:48]))
        else:
            records.append({"prompt": prompt, "response": resp,
                            "chars": len(resp), "attempts": attempts})
            print("  [%02d/%d] %5d chars  %s"
                  % (i, len(BENIGN_PROMPTS), len(resp), prompt[:48]))

        os.makedirs(RESULTS, exist_ok=True)
        with open(out, "w") as f:
            json.dump({
                "experiment": "benign_fpr2",
                "model": model_name,
                "n_prompts": len(BENIGN_PROMPTS),
                "system_prompt": NEUTRAL_SYSTEM_PROMPT,
                "max_tokens": target.max_tokens,
                "truncated": False,
                "scoring": "none; see experiments/analyze_benign_fpr.py",
                "timeout_prompts": timeouts,
                "records": records,
            }, f, indent=2)

    ok = [r for r in records if r["response"] is not None]
    print("\n  stored %d/%d responses (%d timeouts excluded) in %.1f min"
          % (len(ok), len(BENIGN_PROMPTS), len(timeouts), (time.time() - t0) / 60))
    if ok:
        print("  response chars: min %d  median %d  max %d"
              % (min(r["chars"] for r in ok),
                 sorted(r["chars"] for r in ok)[len(ok) // 2],
                 max(r["chars"] for r in ok)))
    print("  -> %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=str)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    targets = TARGETS if args.all else [args.model or TARGETS[0]]
    require_exact_tags(targets)
    for m in targets:
        collect(m)


if __name__ == "__main__":
    main()
