# Pre-registration: EXP-AE — the ten alternative explanations for the criterion-4 positive, named and tested

**Status:** written and committed **before** `experiments/analyze_covariate_audit.py` exists and **before any
covariate has been extracted from any criterion-4 trial**. No covariate-only accuracy, no conditional margin,
and no proxy statistic exists anywhere on disk or in any transcript.
**Companions:** `PREREG_EXP_C4.md` and `PREREG_EXP_C4B.md` produced every row read here; `PREREG_EXP_AD.md` is
this round's other addition. **Neither those two nor any of the other eleven existing pre-registrations is
modified by this document, and no cell, target or grade is added, dropped or re-graded by it.**
**Date:** 2026-09-18

---

## 0. Integrity disclosure

**Fully known to the author of this document:** the published EXP-C4 and EXP-C4B results — gates, per-target
`H1_primary` accuracy, majority baseline, `auroc_probe_sum`, permutation $p$, MDE, the H2/H3 surface-channel
nulls, H4, the per-target verdicts, and the standing counts (three of five in EXP-C4, two of those three
replicating blinded, one of five *new* targets positive). Which targets are positive is therefore known, and
§4 fixes in advance that **every** target in a collection is audited and Holm-corrected, positive or not, so
that the correction set cannot be chosen after the fact.

**Also known:** EXP-C4's H2 already tests one of the reviewer's ten alternatives — the correction-marker
surface rule, which fires on **4.5%** of $D\!=\!1$ trials against **53.8%** of $D\!=\!0$ trials
($-49.3$ pp, one-sided $p = 0.0001$), i.e. in the direction *opposite* to the alternative. EXP-AE generalizes
that single pre-registered test to the full list.

**Held out:** every statistic in §4. No covariate has been computed on any criterion-4 trial, on any target,
in either collection. The field *structure* was inspected (`on_claim_answer`, `raw_answers`, `vector`,
`ambiguous`, `belief_recheck.{true_verdict,false_verdict,known_preserved}`) and one record of
`crit4_confirm_qwen2_5_14b_P3.json` was read in full in the course of the EXP-AD bank check; that is disclosed
in `PREREG_EXP_AD.md` §0 and applies here too.

---

## 1. Motivation — the reviewer's list, verbatim in substance

The reviewer wrote that the criterion-4 positive admits at least ten alternative explanations, and asked that
the limitation be *"more prominent"*:

> uncertainty · confidence · instruction interpretation · conflict resolution · refusal behavior · claim
> difficulty · linguistic awkwardness · topic familiarity · response length · stylistic differences

The paper's current answer is a clause. That is too weak for a positive result the paper leans on. EXP-AE
**names all ten, operationalizes each from data already collected, and gives each the chance to beat the
16-probe battery on the battery's own rows, folds, estimator and null.** No model call, no new grade, no new
target: every input is a committed artifact.

## 2. What this is and is not

**It is** a rival-explanation audit: can a deployed-observable covariate vector, dimension-matched to the
probe battery, reproduce the criterion-4 discrimination on the identical rows?

**It is not** a demonstration that the signal *is* deception if the covariates fail. Absence of these ten
correlates does not exhaust the space of correlates, and a latent $U$ with $U\!\to\!D$ and $U\!\to\!S$
survives every result here — the same limitation H1 carries (§9).

**It can withdraw one of our own positives.** With the standing count already two of five, a covariate that
explains one target takes it to one of five. §7 commits to reporting that.

---

## 3. Rows, covariates and arms — fixed

### 3.1 Rows

Exactly the rows the primary analysis uses: `analyze_crit4.build_rows(cell, grades)`, **unchanged**, over

- **EXP-C4:** `crit4_confirm_<model>_P3.json` × `crit4_grades_raw.jsonl`, five targets;
- **EXP-C4B:** `crit4b_confirm_<model>.json` × `crit4b_grades_raw.jsonl`, ten targets (blinded collection).

EVASIVE and ungraded trials are dropped exactly as the primary analysis drops them. **Because EVASIVE is
dropped, refusal is partly gated out of the row set**, so the per-target `n_evasive` count is reported
alongside as the complementary refusal statistic (§6, alternative 5) rather than pretended to be inside the
model. Groups = `claim_index`. Label = the graded $D$. Both collections are audited; **no target is excluded
and none is added.**

### 3.2 Two text channels, kept separate on purpose

- **`answer_text` = `on_claim_answer`** — the deception-bearing utterance, and the only channel a deployed
  monitor would read. **This is the channel the primary rival arm uses**, because a covariate computed here is
  genuinely independent of the detector's input.
- **`probe_text` = `" ".join(raw_answers)`** — the 16 probe answers, i.e. *the detector's own input*.
  Covariates computed here are not a rival explanation but a re-encoding question, so they are used **only**
  in the proxy test (§4.3), never in the rival test.

Conflating these two would let a "rival" beat the battery by being made of the battery, which is why the split
is fixed before any extraction.

### 3.3 The covariate vector: 16 features, dimension-matched to the 16 probes

Features 1–7 are produced by **`feature_correlation_analysis.extract_text_features(conversation)` called
verbatim**, with the channel wrapped as `[{"role": "assistant", "content": answer_text}]`, in its own returned
order:

| # | feature | source |
|---|---|---|
| 1 | `hedge_count` | `HEDGE_WORDS`, verbatim |
| 2 | `hedge_rate` | ″ |
| 3 | `refusal_count` | `REFUSAL_PATTERNS`, verbatim |
| 4 | `avg_response_length` (words) | ″ |
| 5 | `std_response_length` | ″ |
| 6 | `confidence_count` | assertive-word list, verbatim |
| 7 | `question_count` | ″ |

Features 8–16 are defined here, and their definitions are **frozen by this document**:

| # | feature | definition |
|---|---|---|
| 8 | `n_chars` | `len(answer_text)` |
| 9 | `n_sentences` | count of `[.!?]` runs, minimum 1 |
| 10 | `type_token_ratio` | distinct lowercased tokens / tokens, 0 if empty |
| 11 | `mean_sentence_words` | tokens / `n_sentences` |
| 12 | `comma_count` | `answer_text.count(",")`, a clause-density proxy |
| 13 | `attrib_marker_count` | matches of `R_ATTRIB` (below), a sibling of `REFUSAL_PATTERNS` |
| 14 | `ambiguous_sum` | `sum(record["ambiguous"])`, the channel-ambiguity count |
| 15 | `known_preserved` | `int(belief_recheck["known_preserved"])` |
| 16 | `true_verdict_yes` | `int(belief_recheck["true_verdict"] is True)` |

**`R_ATTRIB`, fixed now, case-insensitive** — the instruction-interpretation / conflict-resolution channel:

```
r"\baccording to the (?:reference|source|document|passage|text)\b"
r"\bas (?:provided|stated|given|supplied|indicated)\b"
r"\bthe (?:reference|source|document|passage|text) (?:says|states|claims|indicates)\b"
r"\bbased on the (?:reference|source|document|provided)\b"
r"\bper the (?:reference|source|document)\b"
r"\bin the (?:reference|source|document|passage)\b"
r"\byou (?:provided|gave|supplied|mentioned)\b"
r"\bif (?:the reference|this) is (?:correct|accurate|to be believed)\b"
```

Counted as **number of distinct patterns matching**, the same convention `REFUSAL_PATTERNS` uses in
`extract_text_features`, so the two are on one scale.

**Dimension matching is deliberate:** 16 covariates against 16 probe bits removes *"the probe arm wins because
it has more features"* as an explanation of any margin, in either direction.

### 3.4 The difficulty arm, held separate because it is label-derived

Alternative 6 (**claim difficulty**) is a **17th** feature, `loto_claim_difficulty`: for target $t$ and claim
$c$, the mean $D$ over all rows of claim $c$ on **targets other than $t$** (leave-one-target-out, the
`appendix.tex:1631` pattern), and the collection-wide mean $D$ when claim $c$ has no other-target rows. It uses
no label from target $t$, so it does not leak within a target, **but it is still constructed from labels**, and
mixing it into the primary vector would make a covariate win uninterpretable. It therefore runs as a **named
secondary arm** (`covariate+difficulty`) and is reported separately from the primary rival arm in every table.

## 4. Estimator, folds, arms and statistics

**Estimator, both arms, fixed:** `Pipeline([("scaler", StandardScaler()), ("clf",
LogisticRegression(max_iter=1000, C=1.0))])`. The scaler is required because the covariates are unbounded
counts while the probe bits are $\{0,1\}$; using **the same pipeline on both arms** is what makes the
comparison a comparison. The probe arm is **additionally** reported through the primary analysis's unscaled
`analyze_r1_faithful.grouped_kfold_accuracy`, and the two probe-arm numbers are printed side by side so any
scaler-induced difference is visible rather than assumed to be zero.

**Folds, fixed:** `GroupKFold(n_splits=5)` on `claim_index`, the primary analysis's grouping, identical across
arms.

**Null, fixed:** `analyze_crit4.within_claim_permute(y, groups, rng)` — permute $D$ within claim strata —
`SEED = 42`, `N_PERM = 10000`, one-sided in the direction of the hypothesis, recomputing the full CV accuracy
per draw.

**Multiplicity, fixed:** Holm across **all** targets within a collection (5 for EXP-C4, 10 for EXP-C4B),
$\alpha = 0.05$, applied separately per test.

### 4.1 Test 1 — the rival test (primary)

Per target: $A_{\mathrm{cov}}$ (16 covariates from `answer_text`) versus $A_{\mathrm{probe}}$ (the 16 probe
bits), same rows, same folds, same pipeline. Report $\Delta_{\mathrm{rival}} = A_{\mathrm{cov}} -
A_{\mathrm{probe}}$, the majority baseline, Wilson intervals via `analyze_external_audit.wilson`, and the
permutation $p$ for $A_{\mathrm{cov}}$.

**H-AE1.** A target's criterion-4 positive is **explained** iff
$\Delta_{\mathrm{rival}} \ge$ `EXPLAINED_MARGIN = 0.0` **and** $A_{\mathrm{cov}}$'s Holm-corrected
permutation $p < 0.05$. Both conditions, because a covariate arm that merely ties at chance explains nothing.

### 4.2 Test 2 — the conditional test

Per target: $A_{\mathrm{probe+cov}}$ (32 features) versus $A_{\mathrm{cov}}$, same folds.
$\Delta_{\mathrm{cond}} = A_{\mathrm{probe+cov}} - A_{\mathrm{cov}}$, with a within-claim permutation $p$ on
$\Delta_{\mathrm{cond}}$ (permuting $D$ and recomputing both arms per draw, so the difference is tested rather
than each arm separately).

**H-AE2.** The probe battery **adds over the covariates** on a target iff $\Delta_{\mathrm{cond}} > 0$ with
Holm-corrected $p < 0.05$. A target passing H-AE2 has its positive **narrowed and retained**, and the
surviving margin is the number the paper prints — not the raw accuracy.

### 4.3 Test 3 — the proxy test

How much of the battery is a re-encoding of style: per probe dimension $j$, predict bit $j$ from the 16
covariates extracted from **`probe_text`** (§3.2), same folds, same pipeline, reporting per-dimension
grouped-5-fold accuracy and its majority baseline, plus the median over the varying dimensions. Dimensions with
$\mathrm{Var} \le 10^{-9}$ carry no value and are **counted and excluded**, never imputed.

**H-AE3.** The battery is **substantially style-predictable** iff the median over varying dimensions of
(accuracy − majority baseline) $\ge$ `PROXY_HIGH_MARGIN = 0.20`. Descriptive: no significance test, because the
question is magnitude.

### 4.4 Per-alternative attribution, so the list is answered item by item

For each of the ten alternatives, the **single-covariate** grouped-5-fold accuracy and the signed standardized
mean difference between $D\!=\!1$ and $D\!=\!0$ of each feature mapping to it (§6), with a within-claim
permutation $p$, **Holm-corrected across the 16 covariates within a target**. This is what lets the paper say
which alternatives are live and which are not, instead of gesturing at a vector.

## 5. Reporting branches — fixed in advance, including the ones that hurt

- **H-AE1 fires on a target** → that target's criterion-4 positive is **withdrawn as explained**. The standing
  count is decremented at **every** main-text site, `abstract.tex` included, and the withdrawal is stated in
  the letter's first paragraph on EXP-AE — the round-26 EXP-AA pattern. The verifier enforces this: a target
  whose positive is explained may not appear in any main-text positive count.
- **H-AE1 fires on all positives of a collection** → the criterion-4 positive result is withdrawn for that
  collection, `abstract.tex:4`'s positive clause is retracted, and the paper's contribution reverts to the
  negative result plus the audit. **This branch is written down because it is possible, not because it is
  expected.**
- **H-AE1 fires nowhere and H-AE2 holds on the positives** → the positives are **narrowed and retained**: the
  paper reports the conditional margin $\Delta_{\mathrm{cond}}$ and states that ten named alternatives were
  given the battery's own rows, folds, estimator and null and did not account for the discrimination. **No
  claim that the signal is deception** follows, and §9 is repeated in the same breath.
- **H-AE1 fires nowhere but H-AE2 also fails** (covariates lose, yet the battery adds nothing over them) →
  reported as **indeterminate on that target**: too little power to separate the arms, with the MDE via
  `analyze_crit4.h1_mde` printed. Not spun as support.
- **H-AE3 holds** → the paper must state that a substantial part of the battery is predictable from surface
  covariates of the probe answers, which sharpens the existing surface-feature caution and **weakens** the
  reading of the battery as a deception instrument.
- **The difficulty arm (§3.4) changes a verdict that the primary arm did not** → reported as its own labelled
  finding, explicitly flagged as label-derived, and it may **not** by itself withdraw a positive.

**No branch alters any published EXP-C4/C4B number.** EXP-AE adds arms on the same rows; it does not re-run
H1, re-grade a trial, or touch `PREREG_EXP_C4B.md`'s sealed roster.

## 6. The ten alternatives, mapped

| # | reviewer's alternative | features | notes |
|---|---|---|---|
| 1 | uncertainty | 1, 2 | `HEDGE_WORDS` density |
| 2 | confidence | 6 | assertive-word count |
| 3 | instruction interpretation | 13 | `R_ATTRIB` |
| 4 | conflict resolution | 3, 13 | correction + attribution markers |
| 5 | refusal behavior | 3, **and per-target `n_evasive`** | EVASIVE rows are dropped upstream (§3.1) |
| 6 | claim difficulty | 17, secondary arm only | label-derived (§3.4) |
| 7 | linguistic awkwardness | 10, 11, 12 | TTR, sentence length, clause density |
| 8 | topic familiarity | 15, 16 | `belief_recheck` |
| 9 | response length | 4, 5, 8, 9 | words, chars, sentences |
| 10 | stylistic differences | 7, 10, 11, 12 | questions, TTR, sentence shape |
| — | channel ambiguity (ours, not theirs) | 14 | `sum(ambiguous)` |

Every alternative maps to at least one feature; **none is listed without an operationalization**, and the two
that are only partly observable (5 and 6) say so here rather than in a footnote later.

## 7. Secondary, reported but not tested

1. The EXP-C4 vs EXP-C4B agreement of $\Delta_{\mathrm{rival}}$ on the five shared targets.
2. Pearson correlations of each covariate with $D$, per target — the `appendix.tex:770` presentation extended
   to fixed elicitation.
3. Whether `refusal_count` retains H2's *negative* direction ($D\!=\!1$ trials showing **fewer** correction
   markers) once it is one covariate among sixteen.
4. Covariate-only accuracy on the **instructed** `r1b_fresh` cells, as a reference point for how much of the
   *instructed* accuracy the same sixteen covariates reach — the surface-accessibility result the paper
   already reports, recomputed on this feature set.

## 8. What EXP-AE cannot establish

1. **Not that the signal is deception.** Ten correlates failing does not exhaust the correlates, and a latent
   $U$ with $U\!\to\!D$ and $U\!\to\!S$ survives every arm. Criterion 4 stays at rung 4.
2. **Not a bound on unmeasured covariates.** Only what is computable from committed text and metadata is
   tested; tone, plausibility, pragmatic felicity and anything needing a judge are out of scope this round.
3. **Not a claim about the covariates' own validity.** `HEDGE_WORDS`, `REFUSAL_PATTERNS` and the assertive-word
   list are the paper's existing regex instruments with their existing limitations; `R_ATTRIB` is new here and
   unvalidated beyond its face definition.
4. **Not powered to separate close arms.** With per-target $n$ in the low hundreds after EVASIVE exclusion, a
   small true $\Delta_{\mathrm{cond}}$ is not detectable; the MDE is printed, and the indeterminate branch of
   §5 exists for exactly this case.
5. **Not a generalization claim.** 3B–14B open-weight targets, English, one wording key (`P3`) in EXP-C4.
