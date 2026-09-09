# External corpora: the Apollo Research rollout release

The four external audits — **EXP-XA** (ten published rollout sets, §3.5 of the paper),
**EXP-IT** (criterion 4 on the on-policy insider-trading corpus), **EXP-IT2**
(criterion 4 with an action-grounded label) and **EXP-XJ** (the corpus authors' own
black-box judge on the same rollouts) — read no data we collected. They read
JSON from the public release accompanying Goldowsky-Dill et al. (2025).

That release is **not vendored into this repository**. It is a pinned git submodule:

```
path = adaptive_lie_detector/data/external/apollo_deception
url  = https://github.com/apolloresearch/deception-detection.git
pin  = f8ec4010e74927394709dffa22b97bdf8cd5a62f   ("Fix links in README.md")
```

We do not redistribute it. The upstream repository carries `ACCESS.md`
("Disclosure Level - Public") but **no LICENSE file**, so it is fetched from its
own origin rather than copied here, and provenance stays with Apollo Research.

## Fetching it

```bash
git submodule update --init adaptive_lie_detector/data/external/apollo_deception
```

This is the **only** network access any audit requires. `git lfs pull` is *not*
needed: despite `git-lfs` being configured in the environment, the upstream
repository has no `.gitattributes` and every rollout file is a plain git blob
(e.g. `data/rollouts/insider_trading__onpolicy__llama-70b.json` is 2,896,155 B
in-tree). A `--recurse-submodules` clone therefore fetches complete data.
Expect roughly 470 MB.

Once fetched, every regex-arm audit number recomputes with **no model call and no
further network access**:

```bash
python experiments/run_external_audit.py     --arm confirmatory
python experiments/analyze_external_audit.py --arm confirmatory
python experiments/run_insider_audit.py      --arm action_grounded
python experiments/analyze_insider_audit.py  --arm action_grounded
```

## EXP-XJ: the one audit that calls a model

**The detector is not ours.** EXP-XJ runs the release's *own* black-box baseline —
`data/black_box_baseline/prompt.txt` inside the pinned submodule, 1,373 bytes,
md5 `83ddc4bd2ebcc6c4b05baff091c5cc8c` — verbatim and unmodified. Only the
transport differs from upstream: we serve it through AWS Bedrock as
`us.anthropic.claude-haiku-4-5-20251001-v1:0`, region `us-west-2`, temperature
`0.0`, `max_tokens` 1000, rather than through the `anthropic` SDK.

The 3,605 raw judgements are committed to
`data/results/external_audit_judge_raw.jsonl`, each row carrying the prompt md5,
model id, region, temperature, arm, corpus, grade, label and parsed judgement, so
**every EXP-XJ number in the paper recomputes offline**:

```bash
python experiments/analyze_external_audit_judge.py            # no model calls
```

Re-running the collection step (`run_external_audit_judge.py`) *does* call Bedrock
and needs credentials. It is not required to reproduce any reported figure.

## Do not advance the pin

The pin is load-bearing. Every figure in the external-audit tables, the
`CRITERION4_NULL` verdicts, and EXP-XJ's judge prompt itself are computed against
`f8ec401` — which was the tip of
upstream `main` when the audits ran. Moving the pin invalidates the committed
summaries in `data/results/`. If it must move, re-run all four arms above and
diff the summaries before committing anything.

## The four reserved `__upscale__` files

`PREREG_EXP_IT.md` §0 reserved four files as held out. They are empty upstream at
this pin — 153–178 B, against 264 KB–11.6 MB for every other rollout file — so
they contain no usable rollout set. `PREREG_EXP_IT2.md` §0 formally releases them
from reservation on that basis, determined from file size alone.
