# External corpora: the Apollo Research rollout release

The three external audits — **EXP-XA** (ten published rollout sets, §3.6 of the paper),
**EXP-IT** (criterion 4 on the on-policy insider-trading corpus) and **EXP-IT2**
(criterion 4 with an action-grounded label) — read no data we collected. They read
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

Once fetched, every audit number recomputes with **no model call and no further
network access**:

```bash
python experiments/run_external_audit.py     --arm confirmatory
python experiments/analyze_external_audit.py --arm confirmatory
python experiments/run_insider_audit.py      --arm action_grounded
python experiments/analyze_insider_audit.py  --arm action_grounded
```

## Do not advance the pin

The pin is load-bearing. Every figure in the external-audit tables, and the
`CRITERION4_NULL` verdicts, are computed against `f8ec401` — which was the tip of
upstream `main` when the audits ran. Moving the pin invalidates the committed
summaries in `data/results/`. If it must move, re-run all four arms above and
diff the summaries before committing anything.

## The four reserved `__upscale__` files

`PREREG_EXP_IT.md` §0 reserved four files as held out. They are empty upstream at
this pin — 153–178 B, against 264 KB–11.6 MB for every other rollout file — so
they contain no usable rollout set. `PREREG_EXP_IT2.md` §0 formally releases them
from reservation on that basis, determined from file size alone.
