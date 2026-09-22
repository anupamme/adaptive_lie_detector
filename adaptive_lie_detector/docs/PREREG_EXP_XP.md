# Pre-registration: EXP-XP — can the criterion-4 test be expressed on the audit target's *own* released sets?

**Written before `experiments/fetch_pacchiardi_release.py` or `experiments/analyze_pacchiardi_census.py`
exist.** The decision rule, the requirement definitions, the eligibility gates and every reporting branch
below are fixed here. Deviations are recorded in §9 as deviations.

**This is the 17th pre-registration in `docs/`. It adds a file; it edits none.** The sixteen existing
`PREREG_EXP_*.md` are frozen and are not touched by this experiment.

---

## 0. Integrity disclosure — what was already measured, and why this arm is EXPLORATORY

This document cannot claim that its subject matter was unseen. A **feasibility gate** was run on
2026-09-22, before this file existed, precisely to decide whether an experiment was possible at all, and
it answered structural questions about the release. Everything below is therefore reported as
**exploratory**, in the sense the paper already uses for EXP-XA's §0 material and at
`sections/appendix.tex`'s exploratory-arm disclosure: *no number from this experiment may be presented as
confirmatory, and the pre-registration's value is that the decision rule and the reporting branches are
fixed before the census script exists, not that the material was unseen.*

**Upstream pin.** `github.com/LoryPack/LLM-LieDetector`, commit
`c5689fa2615368cd7c3f3c15dbb60ea2126006c7` (2024-06-19). Public repository; no access gate.

**Files fetched to a scratch directory (`/tmp`), never into this repository, and never redistributed:**

| File | What was computed and seen |
|---|---|
| `instrumental_lying_df_original.json` | record schema; scenario count; per-prompt deception-label multiset |
| `instrumental_lying_df_all_scenarios_jb_resampling.json` | same three quantities |
| `instrumental_lying_df_all_42_settings_lorenzos_hardcoded_answers_correct_prefixes.json` | same three quantities |
| `lying_rate.csv` | column names and row count |
| `finetuning_v2_lie_finetuning_dataset_validation_prepared.jsonl` | prompt format; count of prompts shared with the truthful file |
| `finetuning_v2_truthful_finetuning_dataset_validation_prepared.jsonl` | same |

**What was NOT computed, and what this experiment may therefore still add mechanically:** no detector was
run on any file; no elicitation probe was applied; no classifier was fitted; no accuracy, AUROC or
*p*-value was computed on any of this material, and none is computed by this experiment either unless §5's
verdict rule licenses it.

**Gate findings as recorded on 2026-09-22, to be re-derived mechanically by the census script (§8).** The
release contains **three distinct designs**, which `sections/related_work.tex` currently scores as a single
row: prompted instructed lying (the design the paper audits), instrumental role-play, and fine-tuned
liars. The hand-measured structural facts were: the role-play design has one distinct prompt per scenario
and **no prompt carrying both deception outcomes**; the fine-tuned design has **byte-identical prompts
appearing under both outcomes**, with bare `User: … Assistant:` prompts and no instruction. **If the
script's re-derivation disagrees with any of these, the disagreement is the reported result**, and the
hand measurement is treated as the error.

---

## 1. Motivation — the one criticism the census has not answered

A reviewer of the workshop version named this as the missing experiment: the paper audits
`pacchiardi2024catch` as its primary target, declines their code and data, and then runs its criterion-4
audit over *Apollo's* rollout release (EXP-XA/XJ/IT2) and *Liars' Bench* (EXP-XL) — never over the audit
target's own released sets. Claim 4 of the paper ("no audited public release supplies all five
requirements") is therefore untested against exactly the release a reader would check first.

EXP-XP closes that. It applies requirements (i)–(v) — the same five the paper states in §1's box and
decides on ten designs in `tab:prior_work_criteria` — to each of the three designs in the audit target's
release, with the decision rule and the branches fixed in advance.

**Why this could go either way, stated before the verdict is written down.** Their release contains
*un-instructed* deception designs, which is the one thing the instructed paradigm lacks. If any design
supplies all five, the paper gains a **second criterion-4-valid corpus that we did not build**, which is
strictly the strongest outcome available to it, and claim 4 must be narrowed. If none does, the census has
been extended to the primary target's own release and claim 4 survives a test it had not faced.

---

## 2. Materials, and how they are pinned

Public repository, so it is pinned by **commit**, not by manifest alone:
`LoryPack/LLM-LieDetector` at `c5689fa2615368cd7c3f3c15dbb60ea2126006c7`.

`experiments/fetch_pacchiardi_release.py` fetches the six files of §0 to
`data/external/pacchiardi_release/` (**gitignored**; the corpus is not ours to redistribute) and commits
`data/external/pacchiardi_release_manifest.json`: the pinned commit, and per file its path, byte size and
sha256. A reader who fetches the same commit gets the same bytes, verifiably. One network call per file,
then nothing in this experiment touches the network again.

**No model call at any point in this experiment.** Every quantity is structural: prompt hashes, label
multisets, cell counts, field presence.

---

## 3. The instrument: requirements (i)–(v), as a decision procedure

The five requirements are the paper's, restated here as the exact predicates the script evaluates, so that
a verdict is a function of the release's files and not of anyone's reading:

- **(i) `E` fixed.** Within the cells being contrasted, the **whole prompt** a deployed detector would
  observe is byte-identical: one distinct sha256 over the contrasted rollouts. *Hashing the system message
  alone is forbidden*: EXP-XA's §9 records that a vacuous "one distinct system prompt" reading flipped four
  of ten Apollo sets, because a set with **no** system message satisfies it trivially.
- **(ii) `D` graded off-label.** A per-rollout deception value exists that is **not** a deterministic
  function of the condition the rollout was generated under. A label that is a property of the prompt, of
  the fine-tuning target, or of the generating condition **fails** (ii) by definition, however accurate it
  is.
- **(iii) Both outcomes at one item.** At least one item (scenario, question, or prompt hash) is realized
  under **both** `D` values, so the contrast can be taken within an item rather than across items.
- **(iv) Deployed-observable channel.** The text the detector reads is text a black-box deployment would
  see: no hidden scratchpad, no generating-condition field, no training metadata.
- **(v) Non-circularity.** `D` is not definitionally equivalent to the detector's input, nor to the
  process that produced it.

**Eligibility gates are adopted from `PREREG_EXP_XA.md` §6 by reference, unchanged, and no new threshold
is introduced by this document:** one distinct prompt hash across the contrasted cells; the detector
non-degenerate at a fire rate strictly inside [5%, 95%] in at least one contrasted cell; **minimum cell
size 10**; a non-binary deception grade dichotomised at a threshold fixed before analysis, with middle
values excluded.

---

## 4. What the census computes, per design

For each of the three designs, and for each file backing it:

1. `n_rollouts`, `n_items`, and the item key used (stated, not inferred silently).
2. `n_distinct_prompt_hashes` over the **whole** prompt, and the number of items whose rollouts carry more
   than one prompt hash.
3. The per-item multiset of deception outcomes, and `n_items_with_both_outcomes`.
4. Whether a per-rollout deception value exists at all, and if so what determines it: the prompt, the
   generating condition, the fine-tuning target, or an independent grader.
5. The channel: which fields would be visible to a black-box detector.
6. The five predicates of §3, each `True`, `False` or `UNDECIDABLE_FROM_RELEASE`, with the specific field
   or count that decided it.

**`UNDECIDABLE_FROM_RELEASE` is a first-class verdict**, not a rounding of `False`. EXP-AA found that
release-authored text alone leaves requirement (v) undecidable on 59% of blinded ratings, and this census
must be able to say the same thing rather than resolve it by assumption.

---

## 5. Verdict rule — fixed before the script exists

Per design:

- **Criterion-4-expressible** iff all five predicates are `True`. **Only then** is a detector contrast
  run, using the paper's own instruments (L-doc from `experiments/hedging_baseline.py` and, where a probe
  channel exists, the 16-question battery) with EXP-XA's gates applied.
- **Inapplicable** if any predicate is `False` or `UNDECIDABLE_FROM_RELEASE`. The design is then reported
  with the requirements it *does* clear and the ones it fails, **and no accuracy is computed or reported
  for it**. This mirrors EXP-XA §6's rule that an inapplicable benchmark is never reported as an accuracy
  of 0% or 100%: running a detector on a contrast that cannot identify anything produces a number with no
  referent, and the number would be quoted anyway.
- **A near-miss is reported as a near-miss**, naming which requirements it clears. The paper's claim 4 is a
  claim about the conjunction, so a design clearing four of five is evidence about how close practice is,
  and suppressing it would make the census look more one-sided than it is.

---

## 6. Reporting branches — all fixed now

1. **Some design supplies all five.** The paper gains a **second criterion-4-valid corpus not built by
   us**. Claim 4 must be narrowed in the abstract, `tab:claim_ledger`, `tab:prior_work_criteria` and
   §`sec:external_audit`, and the detector contrast of §5 is reported with its gates. This is the outcome
   that costs the most main-text space and it is granted in advance.
2. **No design supplies all five.** Claim 4 survives, now tested against the audit target's own release,
   and the census reports per design which requirements fail. `tab:prior_work_criteria`'s single
   `pacchiardi2024catch` row is **split into the three designs** if and only if their verdicts differ,
   because scoring three designs as one row is an accuracy defect independent of the outcome.
3. **A design clears (i), (iii) and (iv) but fails (ii) or (v).** Reported as the strongest near-miss in
   the census, and explicitly as *closer to criterion 4 than any Apollo set*, if that is what the numbers
   say. This branch exists because the gate's hand measurement suggests it, and fixing it here stops it
   being written up as a discovery.
4. **The release cannot be fetched at the pinned commit.** One-sentence decline in print naming the
   specific failure, no mirror and no scraped copy, following EXP-XL's branch F.
5. **The census disagrees with §0's hand measurement.** The script's number is the reported one, the
   disagreement is stated in §9 as a deviation, and no branch above is selected on the hand number.

**In no case** may this experiment be reported as confirmatory (§0), and in no case does a negative census
license the claim that the release contains no deception — only that its *designs* cannot attribute one on
a deployed-observable channel, which is a property of designs and not of models.

---

## 7. What this does not establish

- Nothing about criteria 1–3, and nothing about whether the audit target's *published accuracy* is
  correct. This is a statement about what their released sets can identify.
- Nothing about `tau_D`. A design clearing all five reaches **rung 4** of the paper's ladder, as ours
  does; `do(D)` is not available here either.
- No re-audit of their instructed results. Those are audited by EXP-R1/R1b/R1c and are out of scope.

---

## 8. Commands

```
python3 experiments/fetch_pacchiardi_release.py            # one network call per file, then pinned
python3 experiments/analyze_pacchiardi_census.py           # structural census; no model call, no network
```

`analyze_pacchiardi_census.py` writes `data/results/pacchiardi_census.json` with, per design, every
quantity of §4 and every predicate of §3 tagged by what decided it. Every number reported in the paper
recomputes from that file with no network access and no model call.

---

## 9. Deviations

Any deviation from this document is recorded here with its date and rationale **before** the affected
numbers enter the paper.

### DEVIATION 1 (2026-09-22) — this arm is exploratory by construction

Recorded at the time of writing, not discovered later. The feasibility gate of §0 preceded this document,
so the structural facts were seen before the decision rule was frozen. The arm is reported as
**exploratory** wherever it appears in the paper, and §0's disclosure travels with it.

---

## 10. Outcome

Completed 2026-09-22 from `data/results/pacchiardi_census.json`, written by
`experiments/analyze_pacchiardi_census.py`. All six files were fetched at the pinned commit and their
sha256 re-verified against `data/external/pacchiardi_release_manifest.json` before the census read a byte.
**No model call, no network call and no detector fit**, as §5 requires while any predicate is `False`.

**Verdict: no design supplies all five. Claim 4 survives, now tested against the audit target's own
release. §6 branch 2 applies, with branch 3 for the near-miss.**

| Design | Files | (i) | (ii) | (iii) | (iv) | (v) | Verdict |
|---|---|---|---|---|---|---|---|
| Prompted instructed lying (the design this paper audits) | `lying_rate.csv` | ✗ | ✗ | *undec.* | *undec.* | ✗ | inapplicable |
| Instrumental, goal-directed, **un-instructed** | `instrumental_lying_df_*` ×3 | ✗ | ✗ | ✗ | ✓ | ✗ | inapplicable |
| **Fine-tuned liars** | `finetuning/v2_{lie,truthful}` | ✓ | ✗ | ✓ | ✓ | ✗ | inapplicable |

**What decided each predicate.**

- **Prompted instructed lying.** The release ships this design as a **dataset × instruction rate table**:
  11 datasets × 26 instruction texts, **200 of the 286 grid positions populated**, and **no rollout, prompt
  or response text at all**.
  (i) fails because the 26 instruction texts *are* the column keys, so the contrast is the prompt change;
  (ii) and (v) fail because the tabulated quantity is the rate of lying under a named lie instruction.
  (iii) and (iv) are **`UNDECIDABLE_FROM_RELEASE`**, not `False`: with the file aggregated to rates and no
  question-level key, neither item-level realization nor the detector's channel can be read off it either
  way. Per §4 that is recorded as undecidable rather than resolved by assumption.
- **Instrumental role-play** (138 rollouts over three files: 42, 54, 42). Under the only released item key,
  the whole-prompt sha256, there are **138 distinct prompts and 0 items carrying both outcomes**. The
  decisive fact is mechanical: **`label` equals the row's index parity in every row of all three files**
  (42/42, 54/54, 42/42), so *D* is recoverable without reading any text and fails (ii) by definition, and
  (v) with it. A supplementary adjacent-pair key — **ours, not a released field, and reported as
  supplementary** — does put both outcomes on each of 21/27/21 scenario pairs, but **0** of those pairs has
  a byte-identical prompt: the honest cell drops the reward-incentive sentences *and* flips the underlying
  fact. So the design is un-instructed and still varies the elicitation between the cells it contrasts,
  which is (i) failing, and the within-item contrast of (iii) is unavailable at fixed *E*. Only (iv) holds:
  the channel is the transcript plus the elicitation logprob differences, with no scratchpad and no
  condition field inside the text.
- **Fine-tuned liars** (2,218 rows per condition). **1,109 prompts are byte-identical across the two
  conditions** — hashed over the whole prompt, never over a system message (§3) — and all **1,109** have
  differing completion sets, so (i) and (iii) hold on a 1,109/1,109 cell pair that also clears
  `PREREG_EXP_XA.md` §6's minimum-cell floor of 10. The prompts are bare `User: … Assistant:` with no
  instruction, no scratchpad and no training metadata, so (iv) holds. The 1,109 prompts appearing in one
  file only are follow-up turns that embed the previous answer, so they cannot be byte-identical across
  conditions; the first-turn prompts are the ones that are. (ii) and (v) fail, and not marginally: *D* is
  which file the row came from, i.e. the fine-tuning target, which is both a deterministic function of the
  generating condition and definitionally the objective that produced the text.

**§6 branch 5 does not fire.** The script's re-derivation agrees with §0's hand measurement on both
structural facts it recorded: the role-play design has no prompt carrying both outcomes (0), and the
fine-tuned design has byte-identical prompts under both (1,109).

**§6 branch 2's split test returns *no split*.** All three designs return the same verdict
(`inapplicable`), and the branch is conditioned on the verdicts differing, so
`tab:prior_work_criteria`'s single `pacchiardi2024catch` row is **not** split. The per-design requirement
patterns differ (0, 1 and 3 requirements cleared) and are reported in the appendix instead. The existing
row is not contradicted by this census: it scores the **instructed design as their paper reports it**,
where the same question is asked under both instructions, and the census scores **what the released files
contain**.

**§6 branch 3, with the comparison it conditions on, measured rather than asserted.** The fine-tuned design
is the strongest near-miss in this release, clearing (i), (iii) and (iv). Branch 3 licenses calling it
*closer to criterion 4 than any Apollo set* only "if that is what the numbers say", and **it is not**:
the Apollo on-policy insider-trading corpus clears (i)–(iv) and fails only (v). So the reported statement
is the weaker true one — strongest near-miss **within the audit target's own release**, three of five — and
the cross-release superlative is withheld. What the two independent releases do share is *which*
requirement their strongest design fails: **(v), by circularity**.

**Incidental, and reported because it corroborates a separate appendix claim from a second source.** Each
role-play rollout carries its elicitation battery as per-probe logprob differences, and the group widths are
**24 lie-related / 14 / 10 factual = 48** in `instrumental_lying_df_original.json`, matching the 24/10/14
partition the appendix attributes to their published battery; the two later files carry 24/22/10 = 56, the
battery having been widened. Our own bank is 16. No accuracy is computed from these fields.

**What no branch licenses.** A negative census does not say the release contains no deception. It says its
*designs* cannot attribute one on a deployed-observable channel at fixed elicitation, which is a property
of designs and not of models (§6), and every number above is **exploratory** per §0 and DEVIATION 1.
