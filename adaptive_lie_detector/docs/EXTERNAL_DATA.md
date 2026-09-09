# External corpora: two public rollout releases

Two releases are audited, and **their reproducibility paths are not equivalent.** The
Apollo release is a pinned git submodule; Liars' Bench is gated and cannot be, which is
stated here and in the paper's Reproducibility Statement rather than glossed over.

| | Apollo (EXP-XA / IT / IT2 / XJ) | Liars' Bench (EXP-XL) |
|---|---|---|
| how it is pinned | git submodule at `f8ec401` | **sha256 manifest**, `data/external/liars_bench_manifest.json` |
| redistributable | no (fetched from origin) | **no — gated, `gated: auto`** |
| fetch requires | nothing but network | **accepting the HF gate under your own account** |
| derived artifacts committed | yes | yes |
| strength of the guarantee | full: the bytes are pinned | **weaker: the bytes are *hashed*, not pinned** |

## Part 1 — the Apollo Research rollout release

The four Apollo audits — **EXP-XA** (ten published rollout sets, §3.5 of the paper),
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

This is the only network access the **Apollo** audits require (Part 2 adds a second, gated
fetch for Liars' Bench). `git lfs pull` is *not*
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

## EXP-XJ: the only Apollo audit that calls a model

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

---

## Part 2 — Liars' Bench (EXP-XL), and why it cannot be pinned the same way

**EXP-XL** audits `Cadenza-Labs/liars-bench` (Cadenza Labs), the second public rollout
release, under `docs/PREREG_EXP_XL.md`. It is **gated** (`gated: auto`) and its license is
"other" — CC BY 4.0 with third-party components. So it is **neither vendored nor
submodule-pinned**: a submodule would require redistributing gated content, and there is no
public git remote whose tip we could pin.

**What replaces the pin is a hash manifest**, committed at
`data/external/liars_bench_manifest.json` and written by `experiments/fetch_liars_bench.py`:

```
repo               Cadenza-Labs/liars-bench      gated: auto      access: granted
revision           503399b81aff28d6812b0ea4585607d5e4b7d3c4
repo_last_modified 2026-07-28T00:16:22Z
per file (8)       sha256 + byte size + etag
rows_measured      verified equal to the dataset card's counts for all 8 configs (79,417)
```

This is **weaker than the Apollo pin and is reported as weaker**. A manifest lets you detect
that the corpus changed; it does not let you fetch the version that was audited. If upstream
revises or withdraws a config, the numbers below remain recomputable from the committed
derived artifacts, but the corpus itself is no longer recoverable from this repository.

### Fetching it

```bash
# 1. accept the gate once, interactively, under your own HF account:
#    https://huggingface.co/datasets/Cadenza-Labs/liars-bench  -> "Agree and access repository"
# 2. then, with a token in ~/.cache/huggingface/token:
python experiments/fetch_liars_bench.py          # writes data/external/liars_bench/ (gitignored)
```

`data/external/liars_bench/` is **gitignored** (`.gitignore:195`): the parquet files are gated
and not ours to redistribute.

**Note the limit of the manifest honestly.** `fetch_liars_bench.py` *writes* the manifest — it
computes each file's sha256 after download and records it. It does **not** check a re-fetch
against the committed manifest. So verifying that what you fetched is what was audited is a
manual diff of the two files:

```bash
python experiments/fetch_liars_bench.py                       # overwrites the manifest
git diff data/external/liars_bench_manifest.json              # empty <=> identical bytes
```

An empty diff means the corpus is byte-identical to the audited revision. A non-empty diff
means upstream moved, and every number in `data/results/` was computed against the old bytes.

### Reproducing every EXP-XL number without the corpus

All derived artifacts **are** committed, so the corpus is needed only to re-derive them from
raw rows. With no network access and no model call:

```bash
python experiments/analyze_liars_bench_audit.py --arm all     # rule + judge, no model calls
```

reads `data/results/liars_bench_rule_*.json.gz` (eight per-config rule records, one per config),
`liars_bench_survey.json` (the
eligibility survey, which is the deliverable of EXP-XL regardless of what it found), and
`liars_bench_judge_raw.jsonl` (1,356 raw judgements), and reproduces the survey, all 35 cells'
verdicts, the MDEs, H5, H6 and the §7 branch determination.

### The judge arm, again not our detector

EXP-XL runs the **Apollo** release's black-box baseline — the same
`prompt.txt`, same md5 `83ddc4bd2ebcc6c4b05baff091c5cc8c`, verbatim — on Liars' Bench
transcripts. Running one release's detector on another release's corpus is deliberate: it is an
instrument-transfer result, and it means neither the corpus nor the detector in EXP-XL is ours.
1,356 judgements are committed: 956 confirmatory (two channels × two eligible cells) and 400 on
the `visible_h6_control` arm, which measures the judge's absolute level on the honest `alpaca`
control (`PREREG_EXP_XL.md` §4 H6, §9 deviation 7).

Re-running the collection step calls Bedrock and needs credentials:

```bash
python experiments/run_external_audit_judge.py --corpus liars_bench --smoke 5   # creds/parsing check
python experiments/run_external_audit_judge.py --corpus liars_bench            # confirmatory arms
python experiments/run_external_audit_judge.py --corpus liars_bench --h6_control
```

It is not required to reproduce any reported figure. `done_keys` makes each of these
append-only and idempotent, and the control arm is namespaced so it can never re-issue or
overwrite a confirmatory judgement.
