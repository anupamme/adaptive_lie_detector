# Pre-registration: EXP-AA — a blinded audit of the audit

**Status:** written and committed **before** `experiments/build_audit_packets.py`,
`experiments/rate_audit_criteria.py` and `experiments/analyze_audit_agreement.py` exist, and
**before any rater output has been produced or read**. No model has been asked to score requirement
(i), (ii), (iii), (iv) or (v) on any corpus, at any point, under any prompt.
**Companions:** `PREREG_EXP_XA.md`, `PREREG_EXP_XJ.md`, `PREREG_EXP_IT.md`, `PREREG_EXP_IT2.md` and
`PREREG_EXP_XL.md` produced the verdicts this experiment re-derives blind. **None of the eleven
existing pre-registrations is modified by this document.**
**Date:** 2026-09-17

---

## 0. Integrity disclosure — what has already been inspected

"Pre-registered" cannot mean "before the data existed": the two audited releases are vendored and the
paper's verdicts on them are already published. So it means **before the instrument was ever run**,
and here the instrument is a panel of blinded raters that has never been asked a single question.

**Fully known to the author of this document, and therefore stated rather than pretended away:**

- Every verdict the paper publishes on both releases: `tab:external_audit` (criteria 1–5 per setting),
  `tab:prior_work_criteria` (requirements (i)–(v) per design), and the prose verdicts in
  §`sec:external_audit` — *no audited public set satisfies all five*, sandbagging fails (iii) and (iv),
  on-policy insider trading fails (v), all 35 liars_bench cells fail (v).
- The recorded per-requirement booleans this experiment will be scored against (§8), including that
  `v_annotation_independent` is `False` on all eight liars_bench configurations and
  `iii_paired_scenarios` is `True` on exactly one of them (`insider-trading`).
- The release-side inventory: the ten Apollo rollout sets in
  `data/external/apollo_deception/dataset_paths.json`, the eight liars_bench configuration directories,
  `apollo_deception/README.md` (2,600 bytes) and `ACCESS.md`.

**Held out, and this is the whole experiment: no rater judgement exists.** Nothing below can have been
chosen to make a number come out. The packets are assembled by a script from release files by a fixed
rule (§4); the raters, their order, their temperature, their response set, the thresholds and every
reporting branch are fixed here (§5–§9); and §9 commits in advance to the branch in which the result
goes **against** the paper.

**Author independence, stated plainly.** The rater panel is blind to our verdicts. It is *not*
independent of us in the sense a panel of human domain experts would be: we wrote the criteria wording
the raters are shown. §10 records this as the experiment's principal limitation, and the paper must
state it in the same sentence that reports any agreement coefficient.

---

## 1. Motivation — the objection this closes

The paper makes a decidability claim in two places:

- `introduction.tex`, claim (2): the five criteria are **"all five decidable from a published
  description alone"**.
- `related_work.tex`: **"requirements (i)–(v) are properties of a design, so an auditor who never runs
  the detector can settle them."**

Both are currently **asserted with no evidence**. A reviewer put the objection exactly:

> *"The criteria classifications are defensible judgments, but not mathematical facts… you don't appear
> to have an independent-blinded audit of the audit."*

and asked for the cheapest experiment that would settle it: *"take perhaps 10–20 benchmark/design
cases, give them to 2–3 independent researchers, blind them to your verdicts, and report agreement on
criteria 1–5."*

EXP-AA is that experiment, with LLM raters standing in for the independent researchers (§10). If the
criteria are decidable from a published description, blinded raters who see **only** release-authored
text should agree with each other and with the recorded verdicts. If they do not, the paper's claim (2)
is too strong and §9 says what it must be narrowed to.

## 2. What this is and is not

**Scored five-tuple: requirements (i)–(v)**, which operationalize criterion 4, *not* criteria 1–5 of
the protocol table. Reasons, fixed here:

1. (i)–(v) are what a *corpus* can be judged on; criteria 1–5 grade a *detector evaluation*, and for a
   rollout release most of them are not applicable.
2. (i)–(v) is where the contested judgments are. The reviewer's own examples — whether a channel counts
   as deployed-observable, and whether a label is construct recovery — are requirements (iv) and (v).
3. (i)–(v) have recorded referents (§8). Criteria 1–5 do not, for these corpora.

**The paper must therefore say** that criteria 1–5 as applied to the ten *published designs* of
`tab:prior_work_criteria` were **not** re-rated blind, and why: five of those ten are papers whose full
text is not vendored here, so a packet could not be assembled from release-authored material without
introducing our own paraphrase. That is a real gap and it is not closed by this experiment.

**EXP-AA adds no corpus, no detector and no deception measurement.** It measures one thing: whether
requirements (i)–(v) are decidable from release-authored text by raters blind to our verdicts.

---

## 3. Cases — 18, fixed

Ten Apollo rollout sets, enumerated by `dataset_paths.json` and by the ten
`data/results/external_audit_{confirmatory,exploratory}_*.json` reports the paper already publishes:

| # | case id |
|---|---|
| 1 | `ai_audit__reasoning__llama-70b` |
| 2 | `ai_audit__reasoning__llama-70b-3.3` |
| 3 | `ai_liar__original_without_answers__llama-70b-3.3` |
| 4 | `goal_directed_lying` |
| 5 | `insider_trading__onpolicy__llama-70b` |
| 6 | `out_simple_4_many` |
| 7 | `roleplaying__plain__llama-70b` |
| 8 | `roleplaying__plain__llama-70b-3.3` |
| 9 | `sandbagging_v2__wmdp_mmlu__llama-70b` |
| 10 | `sandbagging_v2__wmdp_mmlu__llama-70b-3.3` |

Eight liars_bench configurations, from `liars_bench_analysis.json`'s `configs` key: `alpaca`,
`convincing-game`, `gender-secret`, `harm-pressure-choice`, `harm-pressure-knowledge-report`,
`insider-trading`, `instructed-deception`, `soft-trigger`.

**Configuration, not cell, is the unit.** Requirements (i)–(v) are properties of a design; the 35 cells
of `tab:prior_work_criteria` differ only by generator model, which cannot change whether a design fixes
its elicitation prompt. **n = 18 units**, inside the reviewer's requested 10–20.

No case may be added or dropped after any rater output exists. If a packet cannot be assembled for a
case (§4), the case is reported as **PACKET-FAILED** and excluded, with the failure printed — it is not
silently replaced.

---

## 4. Packets — what a rater sees, and what it may never see

`experiments/build_audit_packets.py` emits one packet per case to
`data/results/audit_packets.jsonl`, with an md5 per packet and a `packets_manifest.json` recording,
per field, the release file and byte range it came from.

**Admissible content — release-authored text only:**

1. The case id and the release it belongs to.
2. The relevant section of the release's own `README.md` / `ACCESS.md` / dataset docstring.
3. The **verbatim** system prompt (or, if more than one, the count of distinct prompts and two of
   them), and one verbatim user turn, truncated to a fixed character budget.
4. The release's own label field name and, where the release defines it, its own definition or grading
   instruction.
5. Exactly **two** example records — one per label value where both exist — truncated to a fixed
   character budget, with their labels and metadata field *names*.
6. The count of records and of distinct system-prompt md5s. (Counts, not verdicts.)

**Inadmissible, and the builder must not be able to emit it:** any sentence this paper wrote; any of
our verdicts, criterion names or verdict vocabulary (`CONSTRUCT_RECOVERY_RISK`, `annotation-channel
leakage`, `eligible_for_criterion4`, `UNDERPOWERED`); any field from
`liars_bench_analysis.json`, `external_audit_*.json` or any other `data/results/` artifact; any other
rater's answer; the fact that this paper exists.

The builder asserts these exclusions as a **post-condition over the emitted packet text** and fails
loudly rather than emitting a contaminated packet. Truncation budgets are fixed here:
**8,000 characters** per packet total, **1,500** per example record, **2,500** for prompt material.

### 4a. A packet asymmetry between the two releases, disclosed before any rater call

The two releases do not publish the same *kind* of material, and the packets therefore cannot be symmetric:

- **Apollo** vendors `README.md` (2,600 bytes), which describes the roleplaying, insider-trading and
  sandbagging datasets in the authors' own prose. An Apollo packet contains that prose verbatim.
- **liars_bench** vendors only `test-00000-of-00001.parquet` per configuration — **no dataset card is
  present in this copy**, and `liars_bench_manifest.json` records that the gated files are not
  redistributed. A liars_bench packet therefore has **no prose description at all**: its description *is*
  the verbatim system prompt, the label field name, the metadata field names, the record counts and the two
  example records.

Writing a description ourselves would violate §4's first inadmissible item, so we do not. The consequence
is stated in advance and runs in one direction: **liars_bench packets carry strictly less descriptive
material, so their `UNDECIDABLE` rate should be expected to be the higher of the two.** That is not a
confound to be corrected — it is a fact about what the release publishes, and it is exactly what a
decidability claim should be sensitive to.

**Pre-registered secondary analysis, explicitly exploratory:** $\alpha$ and the `UNDECIDABLE` rate are also
reported **stratified by release** (10 Apollo units, 8 liars_bench units). This is exploratory and
underpowered at 8–10 units per stratum; it may not be used to report a requirement as decidable that fails
§7's threshold on the pooled 18. Its purpose is diagnostic: to say whether a low pooled $\alpha$ reflects
genuine indeterminacy in the criteria or a release that publishes too little to read them off.

Rater instructions carry the five requirements in the paper's own wording, from the box in
`introduction.tex`: (i) fixes the elicitation prompt; (ii) grades $D$ independently of the condition
label; (iii) has the same scenario independently realized under both behavioral outcomes; (iv) exposes
only the channel a deployed detector could read; (v) grounds $D$ in evidence not definitionally
equivalent to the detector's input. **The wording is frozen by md5 at first run** and asserted equal on
every later run, exactly as `RUBRIC_MD5` is in `grade_crit4_deception.py`.

---

## 5. Raters — three, from three families, none of them a target family

Through Bedrock **Converse**, reusing `grade_crit4b_deception.py`'s harness (`call_secondary`,
five-step backoff, `load_done`/`--resume`, `sha256(QUEUE_SEED | key)` queue order). In this order:

| # | model id | family | why admissible |
|---|---|---|---|
| 1 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Anthropic | no audited corpus was generated by an Anthropic model |
| 2 | `us.amazon.nova-premier-v1:0` | Amazon | no audited corpus was generated by an Amazon model |
| 3 | `us.writer.palmyra-x5-v1:0` | Writer | no audited corpus was generated by a Writer model |

Meta, Google, Mistral and Alibaba are **excluded as raters** because the audited corpora were generated
by models from those families (Apollo: Llama-70B; liars_bench: gemma-3-27b, llama-3.3-70b,
mistral-small-3.1-24b, qwen-2.5-72b).

`temperature = 0.0`, `maxTokens = 900`, one call per (case, rater) = **54 calls**. Each call sees one
packet and the frozen instructions and nothing else — no paper text, no our verdicts, no other rater's
answer, no conversation history. If a rater is unreachable after the backoff, the run **stops**; a
two-rater panel is not substituted for a three-rater one.

## 5a. Amendment: rater 2 is `nova-pro`, because `nova-premier` is retired server-side

**Amended 2026-09-17, during the run, with zero Amazon judgements in existence.** Rater 2 as fixed above
is unreachable. Every Amazon call in `data/results/audit_ratings.jsonl` returns, after the full five-step
backoff:

```
ResourceNotFoundException: An error occurred (ResourceNotFoundException) when calling the
Converse operation: This model version has reached the end of its life.
```

This is an availability fact about Bedrock's catalogue in `us-west-2` (the region
`grade_crit4_deception.AWS_REGION` resolves to), not a property of any packet: the ping in the run log
fails on a two-token `"ping"` prompt with no packet attached. §5's stop rule was written for a *transient*
outage; a retired model version will never answer, so obeying it literally would end the experiment rather
than protect it.

**The substitution rule, and why it cannot have been chosen for its answers.** Rater 2 becomes the
highest-tier Amazon model that responds, tier-ordered by Amazon's own naming — premier > pro > lite >
micro, newest generation within a tier. `us.amazon.nova-premier-v1:0` is retired;
**`us.amazon.nova-pro-v1:0`** is the next in that order and responds. Candidates were probed for
*reachability only*, with a two-token prompt and **no packet**, so no candidate ever produced a rating and
none could be preferred for what it said. At the moment of substitution the file held nine Anthropic and
Writer ratings and **four Amazon records, all of them transport errors with `ratings: null`** — the Amazon
column was empty. The family stays Amazon, so §5's admissibility argument is untouched: no audited corpus
was generated by an Amazon model.

**What is given up, stated rather than buried.** Nova Pro is a smaller model than Nova Premier would have
been. If the Amazon rater is the noisiest of the three, that is a plausible cause, and §9's reporting is
per-requirement over three raters either way: a weaker rater 2 pushes $\alpha$ **down**, i.e. against the
paper's claim (2), not toward it. The retired-model records are left in
`data/results/audit_ratings.jsonl` as the provenance trail rather than deleted.

## 6. Response set — four values, `UNDECIDABLE` offered explicitly

Per requirement: one of `SATISFIED`, `NOT_SATISFIED`, `PARTLY`, `UNDECIDABLE`, plus one sentence of
justification. Unparseable responses are recorded as `UNPARSEABLE` and **counted**, never coerced.

`UNDECIDABLE` **must** be offered and its meaning stated in the instructions ("the material shown does
not say enough to decide"). A forced three-way choice would inflate every agreement coefficient and
would hide the failure mode this experiment exists to detect: that a release does not publish enough to
settle the requirement. **The `UNDECIDABLE` rate is a primary outcome, not a nuisance.**

---

## 7. Hypotheses, thresholds and the primary outcomes

Krippendorff's $\alpha$ (nominal) over the four response values, computed by
`krippendorff_nominal` imported from `experiments/analyze_crit4_reliability.py` — the same estimator
EXP-C4 and EXP-C4B already use. Three raters, 18 units, computed **per requirement** (five
coefficients) and once pooled.

**H-AA1 (decidability).** For each requirement, $\alpha \ge$ `PREREG_THRESHOLD = 0.60`, the threshold
already used by `analyze_crit4_reliability.py`.

**H-AA2 (concordance).** For each requirement with a recorded referent (§8), the panel majority matches
the recorded verdict in a majority of cases.

**H-AA3 (decidability rate).** The `UNDECIDABLE` rate is below 1/3 of judgements per requirement.

All three are **directional and reported whether or not they hold.** There is no composite score and no
requirement is dropped from the report.

## 8. Concordance referents — recorded before this document, not transcribed prose

The panel is scored against artifacts already on disk, not against the paper's sentences:

| requirement | referent | source | n cases |
|---|---|---|---|
| (i) | `i_fixed_elicitation` | `liars_bench_analysis.json` `configs.*.requirements` | 8 |
| (i) | `holds_one_elicitation_fixed` | `elicitation_channel_check.json` (see §8a) | 10 |
| (ii) | `ii_grade_not_condition_label` | `liars_bench_analysis.json` | 8 |
| (iii) | `iii_paired_scenarios` | `liars_bench_analysis.json` | 8 |
| (iv) | **none exists** | — | **0** |
| (v) | `v_annotation_independent` | `liars_bench_analysis.json` | 8 |

**42 concordance judgements.** Booleans map to `SATISFIED` / `NOT_SATISFIED`; a panel `PARTLY` or
`UNDECIDABLE` counts as **non-concordant**, which is the conservative direction.

### 8a. Why the (i)-Apollo referent is the corrected field, and not the published one

Assembling the packets surfaced a defect in our own Gate 1, and it is recorded here rather than worked
around. `run_external_audit.py:148` hashes the concatenated messages whose role is `system` and calls the
elicitation fixed when exactly one distinct hash occurs. **Four of the ten audited Apollo sets contain no
`system` message at all** — both AI-audit tags, `goal_directed_lying` and `out_simple_4_many` — so on that
channel every record hashes the empty string, exactly one hash occurs, and `elicitation_fixed` is recorded
`True` **vacuously**. On the channel each of those four does use, none of them holds one prompt fixed: 8 of
8 and 16 of 16 distinct user turns, 27 of 27 distinct `deceive_instruction` values, and 11 distinct user
prompts over 40 records. `experiments/audit_elicitation_channel.py` recomputes this and writes
`data/results/elicitation_channel_check.json`; **three of ten sets hold one elicitation fixed, not seven**.

No published contrast changes — all four are already reported inapplicable on cell size or on a 0.0\% firing
rate, or are materials-only — and the direction is *toward* the paper's thesis, since more of the audited
corpus turns out unable to express criterion 4 than the paper says. The paper must report the correction
regardless (§9).

For this experiment it means the published `elicitation_fixed` is not a usable referent on four cases. So:

- **Primary referent for (i) on Apollo: `holds_one_elicitation_fixed`**, the corrected field. Machine-recorded,
  computed by a fixed rule (the system messages if every record has one, else the user turns, else the
  fields whose names say they are instructions), and committed before any rater call.
- **Secondary, reported alongside: the original `elicitation_fixed`.** The four vacuous cases are where the
  two referents disagree, so they are the sharpest single test in EXP-AA: a panel reading only
  release-authored text should track the corrected value and not the vacuous one. Both comparisons are
  reported whichever way they come out.
- **This choice was made with the corrected values known and no rater judgement in existence**, which is the
  integrity boundary §0 draws. It is disclosed here, before the first call, precisely because the referent
  was changed after inspecting the corpus.

**Requirement (iv) has no recorded referent and gets $\alpha$ only.** This is disclosed here rather
than discovered later, and it is pointed: (iv) is one of the two requirements the reviewer named as
subjective, so the paper may not claim concordance for it.

Secondary, weaker check: the two Apollo designs for which `tab:prior_work_criteria` publishes a full
(i)–(v) tuple — sandbagging (✓✓××✓) and on-policy insider trading (✓✓✓✓×) — are compared row-wise and
reported separately, flagged as transcribed-from-the-paper rather than machine-recorded.

---

## 9. Reporting branches — fixed in advance, including the ones that hurt

Let $\alpha_r$ be requirement $r$'s coefficient.

- **$\alpha_r \ge 0.60$** → report as decidable from release-authored text.
- **$0.40 \le \alpha_r < 0.60$** → report as **partly decidable**; the appendix must state that the
  verdict on $r$ carries rater judgment, and the main text may not describe $r$ as a property a reader
  can simply check.
- **$\alpha_r < 0.40$** → report as **not decidable from the release description alone**, and
  **claim (2) must be narrowed in the main text**: `introduction.tex`'s *"all five decidable from a
  published description alone"* and `related_work.tex`'s *"an auditor who never runs the detector can
  settle them"* must both be qualified to name the requirements that survive. This is a commitment, not
  an option.
- **Any requirement where the panel majority contradicts the recorded verdict** → reported in the
  appendix as a table row with the raters' own justification sentences quoted, and the corresponding
  cell of `tab:external_audit` / `tab:prior_work_criteria` annotated. The disagreement is **not**
  adjudicated by us after the fact, and our verdict is **not** changed on the strength of a rater panel.
- **`UNDECIDABLE` rate $\ge 1/3$ on a requirement** → reported as evidence that the release does not
  publish enough to settle it, which is a finding about the release and is stated as one.
- **A pooled $\alpha \ge 0.60$ does not license reporting a below-threshold requirement as decidable.**
  Per-requirement coefficients govern; the pooled number is context.

## 10. What EXP-AA cannot establish

1. **The raters are LLMs, not independent human experts.** What is measured is whether the requirements
   are decidable from release-authored text by raters blind to our verdicts. It is **not** evidence that
   human domain experts would concur, and every report of $\alpha$ must carry that sentence.
2. **We wrote the criteria wording the raters are shown.** A rater panel cannot test whether the
   criteria are the *right* five; it tests whether, given them, a design's compliance is readable off
   the release.
3. **High $\alpha$ is not construct validity.** Three raters can agree and all be wrong. Concordance
   with the recorded referents (§8) is the only check on correctness, and it exists for four of the five
   requirements.
4. **A packet is not a paper.** Raters see a bounded extract assembled by a fixed rule, not the release
   in full. A requirement decidable from the full repository could read as `UNDECIDABLE` here; that
   direction is conservative for H-AA1 and is stated.
5. **Nothing here touches criterion 4's own evidence.** EXP-C4/C4B's counts — three of five, two of
   three replicating blinded, one of five new — are unaffected, and the roster stays sealed by
   `PREREG_EXP_C4B.md`.
