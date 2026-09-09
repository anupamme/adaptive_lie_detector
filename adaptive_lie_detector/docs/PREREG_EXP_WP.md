# Pre-registration: EXP-WP — the same confound inside a white-box linear probe

**Status:** written and committed **before any cell of the v2 claim set is collected on any model**, and
**before any activation of a second model exists**. The v1 run on `Qwen3-4B-Instruct-2507` is
**exploratory**, is already complete, and has been read in full; every number it produced is disclosed
in §0 so that nothing below can be mistaken for a blind choice.
**Companions:** `PREREG_EXP_R1c.md` (the same v2 claim set, black-box battery), `PREREG_EXP_XA.md` /
`PREREG_EXP_XJ.md` / `PREREG_EXP_IT.md` / `PREREG_EXP_IT2.md` (criterion 4 on public corpora),
`PREREG_EXP_C_intervention_b.md` (intervention B). EXP-WP swaps the **detector paradigm** — from
behavioural black-box to white-box activation probe — and changes the materials only by moving to the
disjoint v2 claim set and a second model family.
**Date:** 2026-09-09

---

## 0. Integrity disclosure — what has already been inspected

Same discipline as EXP-XA, EXP-XJ, EXP-IT and EXP-IT2: "pre-registered" cannot mean "before the data
existed", so it means **before the instrument was ever run on the materials this document commits to**.

**The v1 run is exploratory and is reported as such wherever it appears.** `data/probe_audit/` already
contains a complete pass on `Qwen/Qwen3-4B-Instruct-2507` over `experiments.claims_equalized`
(`EQUALIZED_CLAIMS`, the **v1** 50 pairs): 200 instructed trials (4 cells × 50) and 100 equalized
trials (2 cells × 50), 37 hidden states, local weights `~/Qwen3-4B-Instruct-2507`, MPS, fp16. Its
selected configuration, its hypotheses' outcomes, and its nulls were all read before this document was
written. **It is therefore exploratory in every respect and cannot be relabelled.**

**Every v1 number read, listed so that no later choice can be passed off as blind.**

| step | quantity | v1 value |
|---|---|---|
| probe2 | E-probe accuracy, claim-pair-grouped LOCPO, selected config | **1.000**, CI [1.000, 1.000] |
| probe2 | selected (pooling, hidden-state index) | **`full_mean`, layer 16** |
| probe2 | `full_last` sweep | 0.90–0.96 |
| probe2 | `ctrl_mean` sweep (instruction stripped from the forward pass) | 0.94–0.98 |
| probe2 | `ctrl_last` sweep | **0.72–0.84** (0.805 at layer 16) |
| probe3 | `A_instructed` (T_H vs F_D) | **1.000** |
| probe3 | `A_equalized` (T_N vs F_N) | **0.900** |
| probe3 | `A_transfer` (instructed → equalized) | **0.500** (exactly chance) |
| probe3 | E-probe mean `P(E=1)` on both equalized cells | **≈0.002** |
| probe4 | β_E | **+2.029**, CI [1.93, 2.12], perm p = **0.0005** |
| probe4 | β_V | **+0.097**, CI [0.03, 0.17] |
| probe4 | β_VE | −0.178, CI [−0.28, −0.07], perm p = 0.659 |
| probe4 | simple effects of E | **E \| false = +2.03, E \| true = +1.85** |
| probe5 | random-direction probe | mean 0.571, max 0.750 (n = 20) |
| probe5 | shuffled-label null | mean 0.490, 95th pct 0.5505, empirical p = 0.0196 |
| probe5 | refusal-marker behavioural baseline | 0.565 direct / 0.435 flipped |
| probe6 | σ (SD of training projections onto the direction) | **1.132**; block 15 from hidden layer 16 |
| probe6 | ‖w_raw‖ | 7.46; σ·‖w_raw‖ = **8.44** = the SD of the probe's own log-odds |
| probe6 | correction-marker readout, on-manifold ±{0.25, 0.5, 1} | probe arm 0.325–0.350, random arm 0.325–0.400 |
| probe6 | mean over six on-manifold doses | probe **0.337** vs random **0.354**, Δ = **−0.017**, paired exact permutation **p = 0.50** |
| probe6 | first (off-manifold) sweep ±{2, 4, 8} | +0.067, p = 0.031 — **a dose artifact, retracted** (§9 DEVIATION 3) |
| ad hoc | cos(E-direction, V-direction) | **+0.020 = 88.9°**; reverse fit −0.019 = 91.1° |
| ad hoc | random-pair null for cosine in 2560-d | mean \|cos\| 0.016, 95th pct 0.040 |

**A second result set exists and is a smoke test, not a replication.**
`data/probe_audit/manifest_Qwen2_5-Coder-1_5B-Instruct.json` records `"smoke": true`, `n_pairs: 2`,
8 instructed / 4 equalized trials, `hidden_dim` 1536. Its numbers are noise (β_VE 1.329 at perm
p = 0.434; `mean_refusal` identically 0.0 in every steering cell). **No claim in the paper may rest on
it, and wherever it appears it is labelled a smoke test.** It is not the second model this document
pre-registers.

**Held out, and this is the point of the experiment.**

1. **No activation of the v2 claim set exists**, on any model. `experiments/claims_equalized_v2.py`
   (`EQUALIZED_CLAIMS_V2`, 50 pairs, **0 shared claim strings** with v1) has been used only by
   EXP-R1b/R1c's black-box battery; no probe has ever been fitted on a v2 activation.
2. **No activation of any second model exists** beyond the 8-trial smoke test above.
3. **No surface-feature baseline has ever been fitted** (H6). The features it will use are already
   computable from the committed v1 metadata, and were deliberately **not** computed before this
   document was committed.
4. **No mechanistic steering readout exists on any run** — the code that would produce it was never
   implemented (§9 DEVIATION 4).

---

## 1. Motivation — the two objections this closes

**Objection 1: one detector family.** The paper's evidence is concentrated on black-box behavioural
detectors: `pacchiardi2023catch`'s binarised follow-up battery, our ADAGE pipeline, and a
parameter-free surface rule. All three read **text**. A reader is entitled to say: *these are all the
same kind of instrument, and a detector reading the model's internals would not be fooled this way.*
The white-box line — `zou2023representation`, `burns2022discovering`, `azaria2023internal` — is
currently only cited in §1 and confined to an appendix.

EXP-WP runs the audit protocol against a **linear probe on residual activations**: the strongest and
cheapest white-box detector in common use, with weight access, reading the model's internal state
rather than its output. Criteria 1 (prompt equalization), 3 (matched surface baseline) and 5
(cross-model / cross-claim transfer) all apply to it unchanged. **If the same confound is present
there, the finding is about the evaluation design, not about the instrument's modality** — which is
exactly what the paper claims.

It also supplies a **distinct failure mode** for the paper's failure-mode table, and the sharpest one:
the detector does not read a surface proxy for the instruction, it **decodes the elicitation condition
itself**, and then transfers at exactly chance to the regime where that condition is absent.

**Objection 2: knowledge conflict, not instruction-following.** The paper's 79.8% surface-rule result
is open to a competing explanation — that what the rule reads is the model's **knowledge conflict**
(being asked to assert something it believes false), not its compliance with an instruction. This is a
*different* confound, and arguably a stronger one, and the paper's own EXP-B result (70.0% on KNOWN vs
57.6% on UNKNOWN pairs, p = 0.0045) shows a knowledge component is present. At the text level the two
are hard to separate.

**At the representation level they are separable, and that is what the 2×2 factorial of H3 does.** The
claim's truth-value V and the instruction E are crossed, both are decodable from the same activations,
and the probe's log-odds can be regressed on both. v1's exploratory answer was that E dominates V by
roughly 21× with no interaction, while V remained ~90% decodable in the same activations — so the
probe is not blind to veracity, it simply does not read it. **H3 fixes that test in advance, on
materials no probe has seen.** Confirming it turns "knowledge conflict is an alternative explanation"
into "knowledge conflict is a measured, and much smaller, component."

**What this is and is not.** EXP-WP adds a **detector paradigm**, not a corpus. It runs on **our own
instructed materials**, which by construction cannot express criterion 4 (§6). It therefore cannot
attribute anything to τ_D, and no result below may be reported as a criterion-4 verdict.

---

## 2. The detector — fixed in advance, with the selection freedom removed

**The probe.** `probe_audit_common.make_probe()`, unchanged: `sklearn.pipeline` of `StandardScaler`
then `LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)`, `SEED = 42`. No regularisation
sweep, no class weighting, no calibration, no feature selection.

**The representation, and the one number that must be fixed now.** Activations are the model's
`output_hidden_states`, pooled four ways (`probe_audit_common.POOLINGS`):

| pooling | what it is |
|---|---|
| `full_last` | last token of the response, full context (system prompt included) |
| `full_mean` | mean over response tokens, full context |
| `ctrl_last` | last token, **system prompt stripped from the forward pass** |
| `ctrl_mean` | mean over response tokens, system prompt stripped |

**Pooling and hidden-state index are fixed here to `full_mean`, layer 16, for every model and both
claim sets, and are not re-selected.** They were chosen on v1 by `probe_audit_2_train_probe.py`'s
sweep, which is why the v1 result carries selection freedom and the v2 result does not. **This is the
entire point of the re-run.** The full sweep is still computed and reported for completeness, but the
confirmatory hypotheses read only `full_mean` / 16.

**Second-model caveat, fixed now.** Layer 16 of a 37-hidden-state model is not the same relative depth
as layer 16 of a model with a different depth. For the second model the index is fixed to
**`round(16/36 × (n_hidden_states − 1))`** — the same *fractional* depth, computed from the model's
config **before** any probe is fitted, recorded in the manifest, and not adjusted afterwards. If that
formula lands on a layer whose probe is degenerate, the arm is reported **`UNDERPOWERED_LAYER`** with
the full sweep printed; **it is not re-selected**.

**Cross-validation.** `locpo_predictions` — leave-one-claim-pair-out, grouped by `pair_id`, so no claim
string appears in both train and test. `CLAIM_TO_PAIR` must be rebuilt from the **selected** claim set,
not from the module-level v1 constant (§9 DEVIATION 5). Confidence intervals by
`clustered_bootstrap_ci`, 2000 draws, clustered on claim pair. Permutation tests are grouped the same
way. **The inferential unit is the claim pair**, as everywhere else in the paper.

---

## 3. Cells and channels — reused verbatim, nothing reworded

Cells come from `probe_audit_common`, which imports the system prompts and the opening question
**verbatim** from `run_factorial_truth_x_deception.py`, so the probe sees exactly the prompts the
black-box arms saw:

| pass | cells | V | E |
|---|---|---|---|
| instructed | `T_H`, `F_H`, `T_D`, `F_D` | 1, 0, 1, 0 | 0, 0, 1, 1 |
| equalized | `T_N`, `F_N` | 1, 0 | — (single `NEUTRAL_SYSTEM_PROMPT`) |

**The detector contrast** is `T_H` vs `F_D` — the diagonal a benchmark of this kind scores, in which V
and E move together. **The equalized contrast** is `T_N` vs `F_N` — the same claims under one neutral
prompt, V varying alone. This is criterion 1 applied to a white-box probe.

**Generation is deterministic**: `do_sample=False`, `max_new_tokens=200`, so a cell re-run reproduces
byte-identically on the same hardware. `n_pairs = 50` per pass per model.

**Two channels, and the distinction matters more here than in the black-box arms.** `full_*` poolings
include the system prompt in the forward pass, so the deception instruction is *literally in the
context whose activations the probe reads*. `ctrl_*` poolings strip it. **A probe that reads `full_*`
at ceiling is not necessarily reading a deception representation; it may be reading a prompt it can
see.** That is why `ctrl_*` is reported for every hypothesis and why §9 DEVIATION 2 discloses that
`ctrl_last` reaches only 0.72–0.84.

---

## 4. Estimand, hypotheses and directions — fixed now

**There is no τ_D here and none is estimated.** Our own instructed materials manipulate only E, so by
the Proposition (§2.1 of the paper) no quantity below identifies the effect of deception. The
estimands are **decoding accuracies** and **regression coefficients on the probe's out-of-fold
log-odds**, and every hypothesis is a statement about what the *detector* reads.

**Confirmatory set: H1–H4 and H6. H5 is robustness only.** All tests one-sided in the stated
direction; 10,000 permutation draws unless stated; seed 42; grouped by claim pair.

- **H1 (the instruction is decodable).** E-probe accuracy at `full_mean`/16, claim-pair-grouped LOCPO,
  **exceeds the 95th percentile of the shuffled-label null**. Direction: higher. *Expected: near
  ceiling.* This is the enabling condition, not a finding: if E were not decodable there would be
  nothing to audit.
- **H2 (transfer — PRIMARY).** The **instructed** E-probe, applied unchanged to **equalized**
  responses, is **at chance**. Because "at chance" is not a one-sided alternative, both conditions are
  fixed here and both must hold:
  (a) `A_transfer` is **not** significantly above 0.5 (one-sided permutation p ≥ 0.05), **and**
  (b) `A_transfer` ≤ **0.60**.
  Reported alongside: E-probe mean `P(E=1)` on **both** equalized cells, with the pre-registered
  reading that a value below **0.10** in both cells means *the probe calls every neutral-prompt trial
  honest* — i.e. it is reporting the presence of the instruction, not any property of the assertion.
  **A_transfer above 0.60 with p < 0.05 refutes H2** and is reported as a refutation (§7 branch **B**).
- **H3 (attribution: instruction beats veracity — PRIMARY, and the answer to knowledge conflict).**
  In the 2×2 factorial on out-of-fold probe log-odds, `logodds ~ V + E + V:E`:
  (a) **β_E > β_V**, with a permutation p < 0.05 on β_E; **and**
  (b) **β_E ≥ 5 × β_V** — the ratio threshold is fixed now, at less than a quarter of v1's observed
  ~21×, so the bar is set below the exploratory result rather than at it.
  Reported alongside, unconditionally: (c) the **veracity-probe accuracy** on the same activations
  (v1: `A_equalized` 0.900), so the reader sees that V *is* decodable and the probe still does not read
  it; (d) the **simple effects** E|V=0 and E|V=1 separately, because near-equal simple effects are what
  discriminate "the probe reads asserted falsehood" from "the probe reads instruction compliance"; and
  (e) β_VE with its permutation p.
- **H4 (not a random direction).** The real probe accuracy exceeds **both** (a) the **maximum** over 20
  random-direction probes, and (b) the 95th percentile of the shuffled-label null. Direction: higher.
  *v1 note, disclosed: the random-direction maximum was 0.750 and the shuffled-label empirical p was
  0.0196 — a p that would not survive a Bonferroni correction over the pooling×layer sweep, which is
  another reason the sweep is not re-run for selection here.*
- **H6 (matched surface baseline — the paper's own criterion 3, applied to us; PRIMARY).** A probe
  fitted on **surface features of the response text alone**, no activations, same LOCPO splits, same
  `make_probe()`, predicting E. Feature set fixed now and not extended: response length in characters;
  response length in whitespace tokens; `count_refusal_markers` total; the 14 per-pattern counts from
  `hedging_baseline.REFUSAL_PATTERNS`; and a bag-of-words indicator over the **200 most frequent tokens
  in the training fold only** (fold-internal, so nothing leaks). Direction and reading fixed now:
  - **surface accuracy ≥ 0.90** ⇒ the activation probe's ceiling accuracy is **not** evidence of a
    representational construct beyond surface form. The write-up must then say, in these words, that
    the probe is consistent with an **instruction-following or register** probe, and the mechanistic
    reading is withdrawn.
  - **surface accuracy ≤ 0.70** ⇒ the activation probe reads something the response's surface form does
    not carry, and the register explanation is weakened but not eliminated.
  - **0.70 < surface accuracy < 0.90** ⇒ reported as partially surface-reachable, with both numbers
    side by side and no mechanistic claim.
  **This is the hypothesis most likely to hurt the appendix's story, which is why its thresholds are
  fixed before it is computed** (§9 DEVIATION 2).
- **H5 (causal — ROBUSTNESS ONLY, PRE-REGISTERED AS EXPECTED-NULL).** Steering along the probe
  direction changes correction-marker density more than steering along a random unit direction, paired
  by dose, exact permutation over the sign of the paired difference.
  **The expected outcome is null, and that expectation is recorded here.** v1's corrected on-manifold
  sweep gave Δ = −0.017 at p = 0.50. The honest reading, which the write-up must use in these words,
  is **decodable but not causal** — and it *supports* the paper's argument, since it shows a probe can
  reach ceiling accuracy on a condition it has no causal grip on. **H5 is not a hypothesis this paper
  needs to win.**
  - **Doses, fixed on-manifold by construction (§9 DEVIATION 3).** coef ∈ {−4, −1, 0, +1, +4}, in units
    of σ = SD of training projections onto the direction. **±1 is the on-manifold primary**; **±4 is an
    off-manifold sanity dose** whose only licensed reading is *"the intervention does perturb the
    forward pass"*, and it may **never** be read as evidence about behaviour. The reason is measured:
    σ·‖w_raw‖ = 8.44 already equals the SD of the probe's own log-odds, while real activations span
    about ±1.5 SD, so a coefficient of 4 moves the state far outside the manifold the probe was fitted
    on. n = 20 pairs per dose per arm.
  - **H5b (mechanistic readout — NEW CODE, disclosed).** In addition to the behavioural readout, the
    **E-probe's log-odds on the steered response's own activations** is recorded per dose. Without it a
    **null intervention and an ineffective intervention are indistinguishable**, which is precisely the
    defect of the v1 result (§9 DEVIATION 4). Pre-registered reading: if the probe log-odds **do** move
    monotonically with dose while the behavioural readout does not, the finding is *the direction is
    causally reachable in representation space and behaviourally inert*; if **neither** moves, the arm
    is reported **`INTERVENTION_INEFFECTIVE`** and licenses **no** conclusion about causality at all.
  - **Power is declared underpowered in advance.** A v1 run at n = 10 looked monotone and **did not
    replicate**, and the unsteered baseline itself drifted 0.20 → 0.325 across two runs of an identical
    deterministic condition. n = 20 is therefore reported as underpowered for any effect smaller than
    roughly 0.15 marker-density points, and no null from this arm is offered as evidence of absence.

**Criterion 5 (cross-model / cross-claim transfer) is what the two-axis design tests**, and its reading
is fixed now: H1–H4 and H6 are evaluated **independently on each (model × claim set) configuration**,
and a hypothesis is described as **replicated** only if it holds on **all** configurations that clear
§5's gates. A hypothesis holding on Qwen3-4B and failing on the second family is reported as
**model-specific**, in those words, and the paper's claim is narrowed to match.

**Evaluation order is part of the specification**, because the seed is consumed sequentially:
per configuration, probe1 (generate + extract) → probe2 (H1, and the reported-only sweep) → probe3
(H2) → probe4 (H3) → probe5 (H4) → probe5s (H6) → probe6 (H5, H5b).

---

## 4a. AMENDMENT 1 — three corrections and one exploratory result (2026-09-09)

**Disclosure and ordering first.** §1–§9 were committed at `e99afb1`. Everything below was written
**after** that commit and **before any v2 activation had been scored**; items 1–3 were written before
the v2 collection was launched, and item 4 was run before it. Nothing above has been edited.

**1. §8's command block was wrong about `--claim_set`, and the underlying risk was smaller than
DEVIATION 5 stated.** Steps 2–5 take no `--claim_set` argument and need none: they read the cached
`.npz` and metadata, and `pair_id` is written by step 1 from `enumerate()` over the selected claim set,
so the leave-one-claim-pair-out grouping is correct for whichever set produced the file. Inspection
confirms `CLAIM_TO_PAIR` was **imported but never used** by step 1, and is imported by no other file in
the probe suite. **The stale-global failure DEVIATION 5 warned about could not have occurred.** The real
risk is different and is now guarded mechanically: step 1's output tag derives from the model name, so a
v2 run under the default tag would have **overwritten the committed v1 activations**. Step 1 now takes
`--model_tag`, records `claim_set` in the manifest, and **refuses to run** if an existing manifest under
the same tag was collected on a different claim set. Step 6 carries the same guard.
Corrected commands: `--claim_set` on steps **1 and 6 only**; `--passes both`; and see item 2.

**2. Step 2 needed a flag to honour the fixed configuration.** `probe_audit_2_train_probe.py` selected
the sweep maximum and wrote it to `selected`, which steps 3–6 then read — so §2's "fixed in advance to
`full_mean`, layer 16" was not actually enforced by the code. Step 2 now takes **`--prereg`**, which
sets `selected` from the manifest's `prereg_pooling` / `prereg_layer` (derived at collection time by
§2's fractional-depth formula), still computes and reports the full sweep, and records the unused sweep
maximum as `sweep_best_not_used` so a reader can see exactly what selection would have bought. **The
confirmatory runs use `--prereg`; the v1 artifacts are reproduced without it and are unchanged.**

**3. H5b's readout is specified as a CLEAN RE-ENCODING, because the obvious version is tautological.**
The steering hook adds `coef·σ·unit` at the block whose output is `hidden_states[layer]`, so the
projection of that hidden state onto the direction shifts by exactly `coef·σ`, and the probe's log-odds
by exactly `coef·σ·‖w_raw‖`, **by construction**. Scoring the probe on the steered forward pass would
therefore measure arithmetic, not mechanism. H5b instead re-encodes the **generated text** under the
neutral prompt with **no hook active**, pools `full_mean` at the selected layer, and scores the E-probe
on that. This asks whether the intervention changed the response *in the probe's own terms* — an effect
that has to survive passing through generation into text and back. The pre-registered reading of §4 H5b
is unchanged, including `INTERVENTION_INEFFECTIVE`; only the measurement is made non-trivial. Written
before any steered activation existed.

**4. H6 has been run on v1, and it fires branch D. Reported here as EXPLORATORY.** The thresholds were
committed at `e99afb1`; H6 was then run on the **v1** activations, which §0 already declares exploratory
and fully inspected, **before** any v2 activation existed. Result, on 200 instructed trials / 50 pairs,
same LOCPO splits and same `make_probe()` as the activation probe:

| feature block | LOCPO accuracy | 95% CI (clustered on claim pair) |
|---|---|---|
| numeric only (length, token count, 14 marker patterns + total) | 86.0% | 82.8–89.5 |
| bag-of-words only (top 200, fold-internal vocabulary) | **94.5%** | 92.2–97.0 |
| **all surface features (primary)** | **95.5%** | 93.4–97.7 |
| *activation probe, `full_mean` layer 16, for comparison* | *100.0%* | *100.0–100.0* |

**Verdict `SURFACE_REACHABLE`** (≥ 0.90), gap to the activation probe **+4.5 pp**. Per §4 H6 and §7
branch **D**, fixed in advance: **the mechanistic reading is withdrawn.** The activation probe's ceiling
accuracy is **not** evidence of a representational construct beyond surface form, and the honest
description — in these words, as §4 requires — is **"instruction-following or register probe."** The
appendix must report the 95.5% next to the 100.0% and must not describe the probe as encoding a
deception construct.

**What this does and does not do to the paper's argument, stated now rather than after the v2 run.** It
does **not** weaken it. The white-box row's verdict becomes *stronger* as an audit result: the probe
fails **criterion 3** (matched surface baseline) as well as **criterion 1** (equalization, `A_transfer`
at chance), so the third detector paradigm fails two criteria rather than supplying a representational
story the paper does not need and cannot support. What it forecloses is any appendix sentence claiming
the probe reads an internal deception representation. Branch D composes with whichever of A/B/C the v2
run selects, exactly as §7 specifies.

**Disclosed limitation of this particular number.** 39 of 200 stored responses sit at step 1's
1000-character cap, so the raw length feature saturates for them; the numeric-only 86.0% is therefore a
slight underestimate of what length alone reaches. The primary 95.5% is unaffected in direction, since
bag-of-words alone already reaches 94.5%.

---

## 5. Applicability gates — fixed in advance

A configuration enters its tests only if **all** of:

1. **Both passes completed** at the declared n: 200 instructed and 100 equalized trials, no cell short.
   A short cell is reported and the configuration is **`INCOMPLETE`**, never pooled with a complete one.
2. **The probe is non-degenerate**: both classes present in every LOCPO fold, and the out-of-fold
   log-odds are not constant (SD > 0). A constant probe is reported
   **`INAPPLICABLE_DEGENERATE_PROBE`** with its score distribution and **never as an accuracy of 0.5**.
3. **Generation is non-degenerate**: fewer than **20%** of responses in any cell are empty or are pure
   refusals to engage with the task at all. Above that, the cell's `n`, the refusal rate and example
   responses are reported and the configuration is **`INAPPLICABLE_REFUSAL_RATE`**. This gate exists
   because a second model from a different family may simply decline the deception instruction, and
   that outcome must be reportable rather than silently absorbed into an accuracy.
4. **Hidden states available at the fixed fractional depth** (§2), with `n_hidden_states` recorded in
   the manifest.

**Achieved power is reported, not assumed.** For each configuration: cell `n`, the clustered bootstrap
CI on every accuracy, and for H3 the permutation null distribution of β_E. For H5, the minimum
detectable difference in marker density at 80% power, one-sided α = 0.05, by simulation at the observed
pooled SD and n = 20, seed 42.

---

## 6. What this design does and does not escape

**Stated before any v2 number exists, because it determines what may be claimed.**

**What it escapes.**

1. **Instrument-modality dependence.** If a white-box linear probe on residual activations exhibits the
   same regime dependence as a binarised follow-up battery and a 14-pattern regex, the finding is a
   property of **what the evaluation manipulates**, not of the detector's modality.
2. **Selection freedom on layer and pooling**, on the v2 arm specifically: both are fixed in §2 before
   any v2 activation exists.
3. **Claim-set specificity**, partially: v2 shares no claim string with v1.
4. **The knowledge-conflict alternative**, quantitatively rather than rhetorically (H3).
5. **The "decoding accuracy implies a mechanism" inference**, in our own favour and against it: H5/H5b
   are pre-registered as expected-null, so the write-up cannot claim a mechanism it did not test.

**What it does not escape, and must not be claimed.**

1. **This is not criterion 4, and no result here is a criterion-4 verdict.** Our instructed materials
   manipulate only E. Criterion 4 needs a corpus we cannot be (§2.2, §3.5 of the paper). The white-box
   arm is a **criterion 1/3/5 result** and is placed in the failure-mode table on that basis.
2. **A high `full_*` accuracy is close to trivial and must be presented as such.** The deception
   instruction is inside the context whose activations the probe reads. "The probe decodes E" is, on
   that channel, barely stronger than "the prompt is in the prompt." The non-trivial results are the
   **transfer null** (H2), the **factorial** (H3), the **surface baseline** (H6) and the `ctrl_*`
   channel — not the ceiling accuracy.
3. **The register / style confound is not eliminated by any arm here.** H6 bounds it; it does not
   remove it. Even a surface accuracy of 0.70 leaves open that the probe reads an assertive, unhedged
   register that the deception instruction induces. **This is the single largest hole in the white-box
   arm and the appendix must say so explicitly** (§9 DEVIATION 2).
4. **Two models of one size class, on our own claim set.** Both configurations are ~4B instruction-tuned
   models; neither is frontier scale; both answer prompts we wrote. Criterion 5 is tested across two
   families and two claim sets, and that is the whole of its scope.
5. **A null steering result is not evidence that the direction is non-causal.** It is underpowered by
   declaration (§4), and if H5b returns `INTERVENTION_INEFFECTIVE` it is not evidence about causality
   at all.
6. **No claim about `pacchiardi2023catch`'s detector follows from this arm.** EXP-WP audits a probe we
   fitted, in a paradigm we did not invent. It is evidence that the *design* confound generalises
   across detector modalities, not evidence about any published white-box detector's numbers.
7. **The Coder-1.5B smoke test is not a replication** and no claim rests on it (§0).

---

## 7. Reporting policy — every branch fixed now

**Per configuration, the branch is determined by H2 and H3 jointly, and by H6's threshold.**

| Branch | Condition | What is reported, and what may be claimed |
|---|---|---|
| **A** (expected) | H2 holds (transfer at chance) **and** H3 holds (β_E ≥ 5β_V, p < 0.05) **and** H6 ≤ 0.70 | **The probe decodes the elicitation condition and does not transfer.** The paper's failure mode *"reads the instruction"* is demonstrated in a third detector paradigm. The register explanation is bounded by H6 and stated as bounded, not excluded. |
| **B** | H2 refuted (`A_transfer` > 0.60, p < 0.05) | **A representational signal survives equalization.** This is a result *against* the white-box arm's contribution to the paper's thesis and is reported as such, prominently, with the transfer accuracy and its CI. It does **not** touch claims 1 or 2 — the black-box collapse is a separate measurement — and it does **not** establish deception detection, since the equalized contrast varies V, not D. §3.4 states the divergence between paradigms as the finding. |
| **C** | H2 holds, H3 refuted (β_V ≥ β_E/5, or β_E not significant) | **The probe reads veracity, not the instruction.** The knowledge-conflict explanation wins at the representation level. This is reportable and would *strengthen* reviewer objection #5 rather than close it; §3.3 must then present knowledge conflict as the primary alternative explanation, and the appendix says the white-box arm found against our reading. |
| **D** | H6 ≥ 0.90 | **The white-box result is surface-reachable.** The mechanistic framing is withdrawn in the appendix and in §3.4; the arm is reported as *"the probe's accuracy is matched by a bag-of-words baseline on the same responses,"* which is still a valid criterion-3 finding about the *benchmark* but is not a finding about representations. Branch D composes with A/B/C — it constrains the *interpretation*, not the branch. |
| **E** | Any §5 gate fails | **`INAPPLICABLE`** with the gate named, the counts printed, and no claim in either direction. If the second model fails gate 3 (refusal rate), that is reported as *a cross-family model declining the elicitation*, which is itself informative and is not retried with a softened prompt. |

**Fixed regardless of branch.** (a) All four poolings and the full layer sweep are printed, with the
confirmatory row marked, so a reader can see what selection on v1 bought. (b) `ctrl_last` and
`ctrl_mean` are reported next to `full_*` for every hypothesis. (c) v1 and v2 are reported **side by
side and separately labelled exploratory / confirmatory**; they are never pooled into a single
accuracy. (d) The two models are reported separately; a hypothesis holding on one is called
model-specific. (e) H5 is reported whatever it shows, including `INTERVENTION_INEFFECTIVE`. (f) The
four deviations of §9 appear in the appendix write-up, not only here.

**What no branch licenses.** No branch licenses a claim that deception is or is not detectable from
activations; no branch licenses a criterion-4 verdict; no branch licenses a mechanism claim from
decoding accuracy alone; and no branch permits re-selecting the layer or pooling.

---

## 8. Commands

Interpreter is **`/Users/mediratta/code/paper_writing/AI-Researcher-align/code/.venv/bin/python3`**
(3.13.4, torch 2.9.1, transformers 4.57.3, sklearn 1.8.0) — **not** the Homebrew `python3`, which has
no `transformers` (§9 DEVIATION 6). Run from `code/adaptive_lie_detector`.

```bash
VP=../.venv/bin/python3

# --- configuration 1: Qwen3-4B-Instruct-2507 x v2 claim set -------------------
$VP experiments/probe_audit_1_generate_extract.py \
    --model ~/Qwen3-4B-Instruct-2507 --model_tag Qwen3-4B-Instruct-2507_v2 \
    --claim_set v2 --n_pairs 50 --passes instructed,equalized
$VP experiments/probe_audit_2_train_probe.py  --model_tag Qwen3-4B-Instruct-2507_v2 --claim_set v2
$VP experiments/probe_audit_3_equalize.py     --model_tag Qwen3-4B-Instruct-2507_v2 --claim_set v2
$VP experiments/probe_audit_4_factorial.py    --model_tag Qwen3-4B-Instruct-2507_v2 --claim_set v2
$VP experiments/probe_audit_5_baselines.py    --model_tag Qwen3-4B-Instruct-2507_v2 --claim_set v2
$VP experiments/probe_audit_5s_surface.py     --model_tag Qwen3-4B-Instruct-2507_v2 --claim_set v2
$VP experiments/probe_audit_6_steer.py        --model ~/Qwen3-4B-Instruct-2507 \
    --model_tag Qwen3-4B-Instruct-2507_v2 --claim_set v2 --coefs -4,-1,0,1,4 --n_pairs 20

# --- configuration 2: second family, same v2 claim set ------------------------
# primary: google/gemma-3-4b-it (already cached, cross-family)
# fallback: meta-llama/Llama-3.2-3B-Instruct
# ... identical invocations with --model/--model_tag substituted ...
```

**Offline reproducibility.** `probe_audit_{2,3,4,5,5s}` read only the cached `.npz` activations and
metadata, so every number except H5 recomputes with **no model load and no network**. The `.npz`
files are ~200 MB per configuration and are **not committed**; the per-step JSON results, the
manifests (including `n_hidden_states`, `hidden_dim`, dtype, device, the fixed layer and its fractional
derivation) and the response metadata **are**. H5 requires generation under hooks and is the only arm
that cannot be recomputed offline; its per-dose marker counts and probe log-odds are committed so the
paired test itself recomputes.

**The v1 artifacts must not be touched.** Every output tag carries the claim set, so
`probe{2,3,4,5,6}_Qwen3-4B-Instruct-2507.json` and the v1 `.npz`/manifest must be **byte-identical**
after this round. This is checked, not assumed (§9 DEVIATION 5).

---

## 9. Deviations

**DEVIATION 1 — the v1 run is exploratory, and the layer and pooling were selected on it.**
`full_mean` / layer 16 was chosen by a sweep over 4 poolings × 37 layers on v1, and the v1 numbers in
§0 therefore carry selection freedom that no confidence interval in `probe2_Qwen3-4B-Instruct-2507.json`
accounts for. §2 fixes the configuration for v2 and for the second model specifically to remove it.
**Both runs are reported, separately labelled, and never pooled.**

**DEVIATION 2 — the instruction-stripping control is partial, and the register confound is not ruled
out.** Two disclosures, both required in the appendix:
(a) `ctrl_last` reaches only **0.72–0.84** (0.805 at layer 16) against `full_last`'s 0.90–0.96, so
stripping the system prompt from the forward pass leaves the confound **mostly, not entirely, killed**;
`ctrl_mean` at 0.94–0.98 barely moves at all.
(b) The probe may be reading an **assertive, unhedged register** that the deception instruction induces,
rather than any deception construct. This explanation reproduces *every* v1 result, including the
near-equal simple effects of H3(d). H6 is added to this pre-registration for exactly that reason and
its thresholds are fixed before it is computed. **"Instruction-following or register probe" is the
honest description until H6 returns**, and if H6 lands at or above 0.90 the appendix says so in those
words.

**DEVIATION 3 — v1's first steering sweep was off-manifold and its apparent positive is retracted.**
The initial sweep used coefficients ±{2, 4, 8}. With σ = 1.132 and ‖w_raw‖ = 7.46, one unit of coef
moves the probe's log-odds by σ·‖w_raw‖ = **8.44**, which is the SD of the log-odds distribution
itself, while real activations span only about ±1.5 SD (−12.6 to +12.7). Those doses therefore pushed
the residual stream far outside the manifold the probe was fitted on. The corrected on-manifold sweep
±{0.25, 0.5, 1} is flat, and the **+0.067 at p = 0.031 from the off-manifold sweep is a dose artifact
and is retracted, not reported as a result.** §4's dose grid is on-manifold by construction, with ±4
retained **only** as a labelled sanity dose. Separately, an n = 10 run that looked monotone did not
replicate, and the unsteered baseline drifted 0.20 → 0.325 across two runs of an identical
deterministic condition — hence §4's advance declaration that H5 is underpowered.

**DEVIATION 4 — the promised mechanistic readout was never implemented, and this round implements it.**
`experiments/probe_audit_6_steer.py`'s docstring promises two readouts per dose: correction-marker
density **and** "mean probe log-odds on the steered response (mechanistic shift)". Lines 158–159 fit an
`eprobe` and **never call it**; only `count_refusal_markers` is recorded. The v1 steering result is
therefore behavioural-only, which means **a null intervention and an ineffective intervention look
identical in that data**. H5b adds the missing readout. This is **new code written after the v1 result
was seen**, so H5b is labelled **EXPLORATORY** wherever it appears and never joins the confirmatory
set — but it is specified here, before any steered activation has been scored.

**DEVIATION 5 — `CLAIM_TO_PAIR` is a module-level global built from the v1 set.**
`probe_audit_common.py` computes `CLAIM_TO_PAIR = {c: i for i, (tc, fc) in enumerate(EQUALIZED_CLAIMS)
for c in (tc, fc)}` at import time. Threading `--claim_set` through the suite must rebuild it from the
**selected** set; a stale global would silently mis-group the leave-one-claim-pair-out folds and
**inflate every accuracy without raising an error**. Two checks are required and reported: (a) every v2
claim string resolves to a pair id, with zero fallbacks; (b) the committed v1 JSON artifacts are
byte-identical after the change.

**DEVIATION 6 — interpreter.** Earlier pre-registrations (`PREREG_EXP_XJ.md` DEVIATION 1) state that
no virtualenv was available and use the Homebrew `python3` 3.14.6. That was wrong about the location:
the venv is at **`code/.venv`**, not `code/adaptive_lie_detector/.venv`. The probe suite requires
`transformers` and `torch`, which only `code/.venv` has, so §8 uses it. No previously published number
is affected — the external-audit scripts do not import `torch`.

**DEVIATION 7 — the second model's layer index is derived, not selected.** See §2. The fractional-depth
formula is fixed here; if it lands on a degenerate layer the arm is reported `UNDERPOWERED_LAYER` and
is **not** re-selected. `google/gemma-3-4b-it` additionally loads as a multimodal
`Gemma3ForConditionalGeneration` rather than an `AutoModelForCausalLM`, so obtaining
`output_hidden_states` may require addressing the language-model submodule. That is an adapter change
in model construction only; the extraction, pooling and probe code are unchanged. If the adapter cannot
be made to yield hidden states, the fallback is `meta-llama/Llama-3.2-3B-Instruct`, and the
substitution is recorded here rather than presented as the original plan.

---

## 10. Outcome

*To be completed after the run. Nothing above may be edited once a v2 activation exists; corrections go
in this section as dated addenda.*

### 10.1 Gates

*(per configuration: passes complete, probe non-degenerate, refusal rate, hidden states at fixed depth)*

### 10.2 Results

*(H1 accuracy; H2 `A_transfer` + `P(E=1)` per equalized cell; H3 β_E / β_V / β_VE / simple effects /
veracity accuracy; H4 random-direction max + shuffled null; H6 surface accuracy + feature ablation;
H5 per-dose marker density both arms + paired p; H5b per-dose probe log-odds. All four poolings and the
full layer sweep printed. v1 and v2 side by side, separately labelled.)*

### 10.3 Branch taken

*(A / B / C / E per configuration, and whether D applies)*

### 10.4 May claim

### 10.5 May NOT claim
