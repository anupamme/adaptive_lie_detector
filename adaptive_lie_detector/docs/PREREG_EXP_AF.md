# Pre-registration: EXP-AF — is the elicitation–construct confound *decidable* outside deception, and is it *measurable* there?

**Status:** written and committed **before** `experiments/analyze_eccp_decidability.py` exists. **This document
is not a confirmatory pre-registration, and §0 says why.** Several of its descriptive counts were computed
during feasibility scoping, before this file was written, and every one of them is disclosed below by value.
EXP-AF is therefore registered as **descriptive / Tier 2 supporting**, and the paper must label it that way.
**Companions:** `PREREG_EXP_XL.md` pinned the Liars' Bench revision this reads; `PREREG_EXP_XA.md` and
`PREREG_EXP_XJ.md` pinned the Apollo rollouts. **No existing pre-registration is modified by this document, and
no cell, target, grade, roster or published number is added, dropped or re-graded by it.**
**Date:** 2026-09-18

---

## 0. Integrity disclosure — this analysis was scoped by inspection, and the numbers below were already seen

The paper's other pre-registrations hold their primary statistics out. **This one cannot**, because the
question EXP-AF answers — *does any pinned corpus support measuring the confound on a non-deception construct?*
— had to be answered by looking at the corpora before it was possible to know whether there was an experiment
to register at all. Concealing that would be worse than the loss of confirmatory status.

**Already computed and known to the author of this document, by value:**

| quantity | value seen |
|---|---|
| Liars' Bench configs present on disk as parquet | 8 of 8 declared eligible (`alpaca`, `convincing-game`, `gender-secret`, `harm-pressure-choice`, `harm-pressure-knowledge-report`, `insider-trading`, `instructed-deception`, `soft-trigger`) |
| every config's label column | `deceptive` (bool) and nothing else |
| `harm-pressure-choice` rows / framings | 5400 / **7** distinct `meta.template_index` |
| `harm-pressure-choice` released-label reproduction from `meta` | `answer ≠ model_belief` matches `deceptive` on **5400/5400** rows |
| `harm-pressure-choice` underperformance vs released label | agree on **0.8731** of rows; `model_belief ≠ correct_option` on **810** rows; rates 0.217 vs 0.130 |
| `harm-pressure-choice` items × framings | **0 of 4601** distinct option-blocks appear under more than one `template_index` |
| `harm-pressure-choice` released-label rate by framing | 0.130 / 0.216 / 0.163 / 0.091 / 0.075 / 0.224 / 0.311 for templates 1/3/4/5/6/7/8 |
| final-assistant-message length, median | `harm-pressure-choice` **1** char; `harm-pressure-knowledge-report` **34**; `soft-trigger` **4**; `convincing-game` **807**; `alpaca` **819**; `gender-secret` **76**; `instructed-deception` **110**; `insider-trading` **754** |
| Apollo `sandbagging_v2__wmdp_mmlu__llama-70b-3.3` | n=1000, **1** distinct system prompt, `metadata` = `{question, topic, correct_answer}`, 8 topics |
| that corpus's topic × `expected_label` crosstab | bio/chem/cyber **100% deceptive**; astronomy/anatomy/abstract_algebra/business_ethics/clinical_knowledge **100% honest** |
| that corpus's construct definition in our own code | `analyze_external_audit.py:170` already defines `dec = answer_idx != correct_answer`, i.e. correctness **is** the deception label there |

**Held out and not computed:** every quantity for the six configs other than `harm-pressure-choice` beyond
median response length; the item×elicitation crosstab for any config other than `harm-pressure-choice`; the
`channel_recorded` ratio (§3.3) for every corpus including `harm-pressure-choice`; and the (i)–(v) verdict for
every corpus × construct pair, under the rules §3 fixes below.

**What pre-registration still buys here, and it is the only thing it buys:** §3 fixes the **decision rules**
and §5 fixes the **reporting branches** before any verdict is computed, so the verdicts cannot be tuned to the
conclusion, and §5's branch (a) commits us to doing the *expensive* thing if any corpus turns out to pass.

---

## 1. Motivation — the reviewer's first lever

The reviewer asked for the paper's structural result to be stated as a general principle, with deception as one
instantiation:

> *"Define a general benchmark where intervention $E$ simultaneously changes target construct $Y$ and elicited
> behavior $C$, and show any detector $S$ evaluated only under $E$ cannot attribute $S$ to $Y$. Then deception
> becomes one important instantiation … much less like 'a critique of Pacchiardi' and more like 'a general
> framework for evaluating behavioral LLM benchmarks.'"*

The Proposition in `methodology.tex` **already is** that statement — it quantifies over $k$ unobserved
mechanisms $M_1,\dots,M_k$ that $\mathrm{do}(E)$ shifts, for any causal arrangement among them, with deception
entering only as the instantiation $(M_1,M_2)=(D,C)$. What the paper lacks is (a) a name, (b) a second worked
instantiation, and (c) an honest statement of which half of the principle generalizes.

`tab:prior_work_criteria` group (e) already applies (i)–(v) to four adjacent constructs from published designs
alone, and its caption currently **disclaims** the stronger reading: *"it shows the requirements are decidable
outside deception given a specification, **not** that the confound is active there."* EXP-AF asks whether that
disclaimer can be upgraded from a design-level argument to a **measured** one on the corpora the paper has
already pinned.

## 2. What this is and is not

**It is** a decidability census: for every pinned corpus, does it carry a construct other than deception, and
if so, do requirements (i)–(v) admit a verdict on it — with the *reason* for each failure recorded as a count
rather than a judgement.

**It is not** a measurement of the confound's magnitude outside deception. §5 branch (b) exists precisely
because the expected outcome is that no pinned corpus supports one, and in that case **the paper must say that
the magnitude is measured only for deception**, in the same sentence that states the principle's generality.

**It is not confirmatory** (§0), and no result from it may be described as a pre-registered finding.

**It is a census, not an estimate.** Every corpus is read in full; there is no sampling, no estimator, no fold
and no $p$-value anywhere in EXP-AF. Consequently there is nothing to correct for multiplicity, and any
apparent precision is exactness, not significance.

---

## 3. The roster and the decision rules — fixed here, before any verdict

### 3.1 Corpora, fixed. No additions, no substitutions.

- **Liars' Bench** at revision `503399b81aff28d6812b0ea4585607d5e4b7d3c4`, the 8 configs listed in §0, read
  from `data/external/liars_bench/<config>/test-00000-of-00001.parquet`.
- **Apollo** at the pinned submodule commit, the 20 files in `data/external/apollo_deception/data/rollouts/`.
- **Published designs**, scored from the paper alone and already in `tab:prior_work_criteria` group (e):
  sycophancy \citep{sharma2023sycophancy}, human red-teaming \citep{ganguli2022red}, reward tampering
  \citep{denison2024sycophancy}, alignment faking \citep{greenblatt2024alignment}. **No cell of that table is
  changed by EXP-AF**; the group is read as the design-level companion to the corpus-level census.

### 3.2 What counts as a second construct

A corpus supplies a **candidate second construct** $Y_2$ iff all three hold:

1. $Y_2$ is derivable from **released metadata or an objective answer key**, not from the response text a
   detector would read, and not from a judge we would have to run;
2. $Y_2$ is **not definitionally equal** to the corpus's own deception label: `agreement(Y_2, deceptive) < 1.0`;
3. the corpus's **own released label is exactly reproducible** from the same metadata. If it is not, the
   corpus is **dropped, not patched** — an unreproducible label cannot anchor a verdict. (`ANCHOR_EXACT = 1.0`.)

Condition 2 is what rules out Apollo `sandbagging_v2`, where correctness and the deception label coincide by
construction; condition 2 is checked, not assumed, and its measured value is reported either way.

### 3.3 The decision rules for (i)–(v), fixed

Each is a mechanical predicate over counts, with the count reported alongside the verdict so a reader who
rejects a threshold keeps the measurement — the same commitment `experiments.tex` already makes for the public
corpora.

| req. | predicate | statistic reported |
|---|---|---|
| **(i)** $E$ fixed, i.e. **separable from the item** | `n_items_multi_E > 0`: at least one item is realized under $\geq 2$ distinct elicitation variants | `n_items`, `n_E_variants`, `n_items_multi_E`, and the full item×$E$ contingency |
| **(ii)** $Y$ graded independently of the condition label | $Y_2$ derived from an answer key or metadata disjoint from $E$ | the source field name, and `agreement(Y_2, E\text{-implied label})` |
| **(iii)** same item realized under both outcomes of $Y_2$ | `n_items_both_outcomes > 0` | that count and its share of `n_items` |
| **(iv)** deployed-observable channel only | the recorded channel is the final assistant message, no scaffold or hidden field | which fields exist |
| **(v)** $Y_2$ not definitionally equivalent to the detector's input | $Y_2$ derived from a key, not from the channel | as (ii) |
| **channel recorded at all** — a precondition for (ii)/(iii)/(iv) to be *expressible* | `channel_ratio = n_distinct_final_messages / n_rows > CHANNEL_MIN = 0.5` | `channel_ratio`, `n_distinct_final_messages`, median and p90 length |

**`CHANNEL_MIN = 0.5` is fixed now**, and the raw ratio is always printed. Its role: a corpus whose recorded
response is drawn from a small closed set (a letter, a canned phrase) has **no elicited-behavior channel $C$**,
so the ECCP's own precondition is not merely violated — it is **unobservable**, and no detector is expressible
on that corpus at all. This is a distinct failure mode from (i) and is reported as such.

**Elicitation variant, fixed per corpus family**, and chosen for mechanical reasons stated before the counts:
`meta.template_index` for `harm-pressure-*`; `metadata.topic` for Apollo `sandbagging_v2`; otherwise the
`env_md5` prefix hash already defined in `run_liars_bench_audit.py:96–104`, which hashes every message the model
did **not** write. Where a corpus offers more than one candidate, **every candidate is reported with its own
count**, exactly as `run_liars_bench_audit.py:86–90` already requires, so the choice of key cannot decide the
verdict.

**Item, fixed:** the option block for multiple-choice corpora (the text from the first line matching
`(?m)^\s*A[.:]\s` onward), and the `env_md5` hash otherwise. Reported both ways where both exist.

## 4. Outputs

`data/results/eccp_decidability.json`, one record per corpus × candidate-construct pair, carrying every count in
§3.3, the anchor check, the agreement with the released label, and the five verdicts. Plus a `design_rule` block
recording which of the three properties in §6 each corpus satisfies, and a `group_e` block restating the four
published designs' verdicts **as already published in `tab:prior_work_criteria`**, unchanged, with
`source: "table, not recomputed"` so no reader mistakes them for new measurements.

## 5. Reporting branches — fixed in advance, including the expensive one

- **(a) Any corpus passes (i) on a non-deception construct and records a channel** → EXP-AF is **not** the end
  of it: we run the full detector audit on that corpus — the rival-covariate battery from `PREREG_EXP_AE.md`
  §3.3 against $Y_2$, on that corpus's own rows — and report the **magnitude** of the confound outside
  deception. This is more work than the round budgeted and it is committed to anyway, because it is the result
  the reviewer actually asked for.
- **(b) Every corpus fails (i), or records no channel** → report the decidability result and the design rule of
  §6, and **state plainly, at every site that states the principle's generality, that no magnitude is measured
  outside deception.** The conclusion's general-lesson sentence must be rewritten to the split *precondition
  decidable and decided / magnitude measured only for deception*, and must not imply more.
- **(c) A corpus's released label is not exactly reproducible** → that corpus is **dropped from the census** and
  the failure is reported by name and value, not silently omitted.
- **(d) A corpus's candidate $Y_2$ turns out to agree with `deceptive` at 1.0** → it is reported as **not a
  second construct**, with the agreement printed, and the reason (definitional coincidence) named. It may not be
  presented as a second instantiation.
- **(e) The census contradicts anything in `tab:prior_work_criteria` group (e)** → the table is corrected and
  the correction is stated in the response letter, as round 27 did when EXP-AE withdrew a positive.

**No branch permits presenting a negative census as a positive result.** The census's expected outcome is that
the paper's audit is *decidable* but not *satisfiable* on existing public non-deception corpora, and §6 is the
only thing that keeps that from being vacuous.

## 6. The design rule EXP-AF licenses — the escape route, and why the standard is not vacuous

The reviewer's sharpest warning is that *"you have defined a very demanding benchmark standard and then shown
that existing benchmarks don't meet your definition"* would be the most damaging criticism available. The
answer is that the standard names an **escape route that a designer controls before collecting data**, and that
at least one published design already takes it.

`app:criteria_taxonomy` assumption **(A4)** — no observable is a child of exactly one $M_i$ — is flagged there
as *"the assumption doing the work"*. Its constructive converse is the rule:

> A benchmark is auditable on construct $Y$ iff it **(1) crosses elicitation with items**, so $E$ is separable
> from the item; **(2) records the response channel** a deployed detector would read; and **(3) grades $Y$ from
> evidence disjoint from that channel.**

Each of the three defeats (A4) for a different observable, and each is a design choice, not a property of the
phenomenon. **(1) is the one every corpus in this census is expected to violate**, and it is the cheapest to
fix: crossing the same items with more than one elicitation wording costs generation time and nothing else.

**The existence proof, and it is already in the paper.** Reward tampering
\citep{denison2024sycophancy} passes all five requirements in `tab:prior_work_criteria` group (e), and the
reason is mechanical rather than lucky: **the manipulation is training, not the prompt**, so there is no
$\mathrm{do}(E)$ at evaluation time for the elicited behavior to be a descendant of, and (A4) does not bind.
That is what makes the ECCP falsifiable and the standard meetable. **The paper must lead with this**, not with
the verdicts.

## 7. Secondary, reported but not tested

1. The per-framing released-label rates in `harm-pressure-choice` (§0), reported **only** as an illustration of
   why (i) matters, with the item confound stated in the same sentence — never as an elicitation effect. They
   are not attributable to the framing, because item and framing are collinear.
2. Whether any Liars' Bench config other than `harm-pressure-*` exposes an elicitation index at all.
3. The number of Apollo rollout files whose `variant` field distinguishes an elicitation condition, as a
   descriptive statement about how public rollout corpora are structured.

## 8. What EXP-AF cannot establish

1. **Not that the confound has any particular magnitude outside deception.** Expected outcome (b) is a census of
   *undecidability*, not an effect size. The paper may not round this up.
2. **Not that no such corpus exists.** The census covers the corpora this paper pinned. A benchmark we did not
   audit may well cross elicitation with items; the claim is about **these** corpora, named.
3. **Not a validation of the thresholds.** `CHANNEL_MIN = 0.5` and the `n_items_multi_E > 0` rule are stipulated
   here, not derived. The raw counts are printed so a reader may substitute their own and reach their own
   verdict — which is the same standing the paper already offers for (i)–(v) on the public corpora.
4. **Not a claim about the four published designs.** Group (e) is scored from published text, is unchanged by
   this analysis, and carries its existing limitation: a design-level reading of what each evaluation
   attributes its score to.
5. **Not confirmatory** (§0). Descriptive, disclosed, and Tier 2.
