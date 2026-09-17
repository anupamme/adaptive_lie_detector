# Pre-registration: EXP-R1d — does the equalization collapse reproduce on current-generation targets?

**Written before any target named below has been pulled, and before any `r1d_recent_*` cell exists.**
Committed prior to running `run_r1_faithful.py --variant v2` on any of the targets in §3. The target
roster, the admissibility gate, the selection rule, the primary outcome, the hypothesis and the reporting
policy below are fixed; deviations must be reported as deviations in §9.

## 1. Motivation

`sections/appendix.tex` § "Model Vintage" (`app:vintage`) states the paper's sharpest self-criticism:

> **Stated exactly: no target in either criterion-4 family is a post-2024 release, and neither is any of
> EXP-R1c's six.** … **We do not establish that the collapse reproduces on current-generation models**,
> and nothing here should be read as evidence that it does or does not.

`app:future_directions` item (7) then names the run that would settle it, and asserts three properties of
it that this document takes over unchanged: it **touches no seal**, it **breaches no pre-registration**,
and it **needs no human coding**. A third reviewer has now asked for exactly this:

> "The target roster is somewhat dated… one or two modern frontier-model checks would substantially
> reduce this concern."

EXP-R1d runs it. It re-collects EXP-R1c's **identical** v2 contrast — same claim set, same probe bank,
same estimator, same n — on **current-generation** instruction-tuned open-weight targets, and asks whether
the instructed→equalized collapse reproduces on models one to two generations newer than EXP-R1c's six.

**Why this breaches nothing.** EXP-R1d is **not** a criterion-4 experiment and adds no target to either
criterion-4 family. `PREREG_EXP_C4B.md` §4's substitution rule closes the *criterion-4* roster once a
target's trials are graded; it says nothing about criterion 1, and EXP-R1c's own pre-registration fixed a
roster for one panel rather than forbidding a later panel. Nothing here re-opens a sealed roster, re-grades
a graded trial, or changes a number already in the paper.

**What this is not.** EXP-R1d is evidence about **criterion 1** (does the signal survive prompt
equalization) on **newer targets**, and nothing else. It says nothing about criterion 4, nothing about
intervention (A) or (B), and nothing about whether any detector's signal is deception-specific. It does
**not** make the paper's magnitudes generation-independent: it adds three targets at n = 50, all ≤ 14B, all
open-weight, none frontier, none closed, none a reasoning model. In particular a clean sweep does **not**
license "the surviving signal is instruction-following"; the Proposition of §2.1 forbids that inference
from any manipulation of `E`, on any generation of model.

**Identification is vintage-independent either way.** §2.1's proposition is structural, so no outcome of
this experiment can repair or worsen it. What EXP-R1d bounds is the *empirical* claim of §3.2 — that the
collapse is a property of the paradigm rather than of one cohort of models.

## 2. Materials — already fixed in the repo, nothing authored for this experiment

Identical to EXP-R1c's v2 arm, reused byte-for-byte:

| | v2 (EXP-R1b / R1c / **R1d**) |
|---|---|
| Claims | `experiments/claims_equalized_v2.py`, 50 matched pairs |
| Probes | `experiments/r1_elicitation_questions_v2.py`, 16 independently worded probes (5 lie-related / 6 factual / 5 ambiguous) |
| Runner | `experiments/run_r1_faithful.py --variant v2` (unmodified) |
| Analyzer | `experiments/analyze_r1_faithful.py --variant v2` (unmodified) |
| Probe token budget | `max_tokens = 40` (`run_r1_faithful.py:179`) |

**No new code is authored for this experiment.** If any change to the runner or analyzer turns out to be
necessary, that is a deviation and goes in §9 before any number enters the paper.

**Checkpoint prefix.** The v2 prefix is `r1b_fresh_`, keyed on `(model, condition)`. EXP-R1d's targets are
disjoint from EXP-R1c's six, so no key can collide. **v1 and existing v2 cell files must be byte-identical
after the run**; a change to any of them indicates a checkpoint collision and invalidates EXP-R1d.

## 3. Targets — named now, with the reserve, before any pull

Chosen under `app:vintage`'s stated constraints, which this document inherits verbatim:
instruction-tuned, **≤ 14B**, resolvable in the Ollama library, **not a reasoning/thinking-mode model**,
and within a 24 GiB hardware ceiling.

| Order | Target | Weights | Successor to |
|---|---|---|---|
| 1 | `olmo-3:7b` | 4.5 GB | `olmo2:13b` |
| 2 | `ministral-3:8b` | 6.0 GB | `mistral:7b` |
| 3 | `granite4.2:8b` | 5.3 GB | `granite3.1-dense:8b` |
| reserve | `qwen3.5:9b` | 6.6 GB | `qwen2.5:*` |

All four resolve in the Ollama registry as of 2026-09-16 (verified against a deliberate 404 control).

**`gemma4:12b` is excluded before collection**, on the ground `app:vintage` already states: it emits empty
thought tags even when thinking is disabled, so its probe channel is not the instrument EXP-R1c measured.

**Substitution rule.** The reserve may replace a named target **only** for (a) a pull failure, (b) failure
of the §4 admissibility gate, or (c) an out-of-disk condition — and **never after any confirmatory EXP-R1d
trial for that target has been collected.** At most one substitution is permitted. Any target that fails
the gate is reported as **void with its measured gate value**, not silently dropped and not replaced
beyond this rule.

**Disk.** 26 GiB free at pre-registration time. Targets are pulled, run and `ollama rm`'d **one at a
time**, so peak residency is one model.

## 4. Admissibility gate — a threshold derived from published data, fixed before collection

The detector's instrument is a 16-dimensional **binary** vector obtained by parsing each probe answer to
yes/no inside a **40-token** budget. If a target cannot emit a parseable yes/no in 40 tokens, the channel
is no longer the instrument EXP-R1c measured and the instructed/equalized comparison is void rather than
informative. `run_r1_faithful.py` already records a per-probe `ambiguous` flag, which is the gate's
instrument.

**The threshold is taken from the cells already in the paper rather than chosen for this experiment.** The
six EXP-R1c v2 targets span the following per-probe ambiguity rates (computed from
`data/results/r1b_fresh_*.json`):

| Target | instructed | equalized |
|---|---|---|
| `llama3.2:3b` | 2.2% | 3.8% |
| `llama3.1:8b` | 4.2% | 3.2% |
| `mistral:7b` | 9.5% | **26.0%** |
| `qwen2.5:7b` | 0.0% | 0.0% |
| `qwen2.5:14b` | 0.0% | 0.0% |
| `qwen2.5:32b` | 0.0% | 0.0% |

`mistral:7b` equalized, at **26.0%**, is the worst cell **already reported in the paper**. Therefore:

> **Gate.** A target is admissible iff its **pilot instructed** per-probe ambiguity rate is **≤ 26.0%** —
> no worse than the worst cell EXP-R1c already reports.

**Pilot.** 10 claims (n = 10), instructed condition, before the confirmatory 50. The pilot's cells are
recorded and reported, are **not** pooled into any confirmatory figure, and its ambiguity rate is the only
quantity the gate reads. A target failing the gate is reported as void with its rate and its most common
failure mode.

## 5. Selection rule — fixed in advance, identical to EXP-R1c's

A target is **informative** if and only if its **instructed** accuracy is above chance: stratified 5-fold,
1000-draw label-permutation test, `p <= 0.05`. **This criterion is fixed by the instructed cell, before any
equalized result for that target is seen** — the same ordering EXP-R1c used.

The collapse claim is evaluated **only on informative targets.** A target on which the mechanism never
worked cannot exhibit a collapse, and reporting one as a collapse would be an artifact. Under v1,
`qwen2.5:7b` was exactly this case; a current-generation target may fail the same way.

## 6. Primary outcome and hypothesis

- **Primary outcome**: per informative, admissible target, the change in **stratified 5-fold** accuracy
  from instructed to equalized, with the label-permutation `p` for the equalized cell against chance.
  Stratified 5-fold is primary because LOO is biased *below* chance in the no-signal case. Same estimator,
  same fold construction, same n = 50 balanced trials per cell as EXP-R1c.
- **Secondary, descriptive**: claim-pair-grouped 5-fold accuracy, LOO accuracy, the count of non-constant
  probe dimensions under equalization ("vary"), the per-probe ambiguity rate of every confirmatory cell,
  and the pooled instructed/equalized accuracy across informative targets.
- **Hypothesis (H1), one-sided**: every informative, admissible current-generation target falls to chance
  under equalization, i.e. equalized `p > 0.05` on all of them.

## 7. Reporting policy — decided now, not after seeing the numbers

1. **H1 holds on every informative target.** §3.2 gains one clause stating that the collapse reproduces on
   current-generation targets, `app:vintage`'s "we do not establish that the collapse reproduces" sentence
   is **replaced** by what is now established, `app:limitations` item (l) and `app:future_directions` item
   (7) are revised to match, and a new appendix section reports the per-target panel. The paper's
   magnitudes remain claims about their own targets.
2. **Any informative, admissible target retains above-chance equalized accuracy.** This is the outcome
   that would matter most and it is reported as such, prominently and without softening: it is evidence of
   signal surviving at fixed elicitation, which is **criterion 4's question**, and it must be stated in
   §3.2, in the Discussion's limitations, and in `app:vintage` — not confined to an appendix. The
   per-dimension diagnosis of §8 is mandatory in this branch. It does **not** overturn §2.1's proposition,
   which no experiment can, and it does **not** by itself license a deception-specific reading; the
   correct statement is that the *empirical* collapse of §3.2 is cohort-dependent.
3. **A target has no instructed signal.** Reported as non-informative with its instructed accuracy and
   `p`, never as an accuracy of 0% and never counted as a collapse.
4. **A target fails §4's gate.** Reported as void with its measured ambiguity rate, and replaced only
   under §3's substitution rule.
5. **Fewer than two admissible informative targets.** EXP-R1d is reported as **inconclusive** and no
   recency claim enters the paper; `app:vintage`'s existing "we do not establish" sentence stays exactly
   as it is, and item (7) is revised only to record that the run was attempted and why it was
   uninformative.
6. **Existing numbers do not move.** All v1 `r1_faithful_*.json` and all six existing v2 `r1b_fresh_*`
   target pairs must be byte-identical after the run. Any change invalidates EXP-R1d.

## 8. Per-dimension diagnosis (mandatory under branch 2 of §7)

Report which of the 16 v2 probe dimensions remain non-constant under equalization and how much separation
each carries, exactly as EXP-R1b diagnosed Qwen 2.5 14B's v1 exception. `--per_dim` produces this. Under
branch 1 it is descriptive; under branch 2 it is required before any sentence about the exception is
written.

## 9. Deviations

Any deviation from this document is to be recorded here with its date and rationale **before** the
affected numbers enter the paper.

**2026-09-16, operational note — not a deviation from any fixed quantity.** §2 forbids new code and §4
forbids pooling the pilot into a confirmatory figure, and those two requirements interact: `run_r1_faithful.py`
keys its checkpoint on `(prefix, model, condition)` and `--resume` skips claims already present, so a
10-claim pilot followed by `--n_samples 50 --resume` would have silently made the pilot's 10 trials the
first 10 of the confirmatory cell. To honour §4 as written, `experiments/run_r1d_recency.py` was added as a
**driver**: it moves the pilot checkpoint to `..._instructed_pilot.json` before the confirmatory run, so the
confirmatory cell starts from an empty checkpoint and regenerates all 50 claims. The pilot cell is retained
and reported, as §4 requires. **The runner, analyzer, claim set and probe bank are untouched** — the driver
only sequences `run_r1_faithful.py --variant v2` calls, evaluates the §4 gate, and writes a ledger
(`data/results/r1d_recency_ledger.json`). It refuses to proceed if a live instructed checkpoint exists with
no pilot archive beside it. No fixed quantity in §§3–6 is changed.

**2026-09-16, operational note — a truncated pilot was preserved, not discarded.** The first attempt at
`olmo-3:7b`'s pilot was killed by an external process stop after one trial. That single trial had an
ambiguity rate of 100%, i.e. it looked unfavourable to the target. It was therefore **archived rather than
deleted**, to `data/results/r1b_fresh_olmo-3_7b_instructed_pilot_truncated.json`, and the full 10-claim
pilot was then collected from an empty checkpoint. Both files are retained. Discarding an inconvenient
partial draw and re-rolling is the failure mode this paper exists to criticise, so the truncated cell is on
record even though the full pilot supersedes it and reached the identical verdict.

**2026-09-16, factual error in §3's target selection, found before any confirmatory trial.** §3 requires
targets that are **not** reasoning/thinking-mode models, and asserted `olmo-3:7b` met that constraint. It
does not. Measured directly: `ollama show olmo-3:7b` lists capability `thinking`; a generate call with
`"think": false` still spends its whole budget on the reasoning channel and returns `response: ""`; and at
`num_predict = 300` the same prompt does answer (`"yes"`) but only after ~160 thinking tokens. Since §2
fixes `max_tokens = 40` as part of the instrument, this target emits **zero response tokens** inside the
budget. This is the same mechanism on which §3 excluded `gemma4:12b` before collection.

**No judgement was substituted for the pre-registered route.** Rather than exclude `olmo-3:7b` by appeal to
§3's constraint after the fact, the §4 pilot was run as written and the target was voided on its
**measured** gate value. See §10.

**2026-09-17, operational note — pilots are archived to a subdirectory, and the ledger's pointers were
repointed.** The driver originally archived a pilot as `data/results/r1b_fresh_<tag>_instructed_pilot.json`,
i.e. inside the directory `analyze_r1_faithful.py` globs non-recursively for `r1b_fresh_*.json`. A pilot
there is read by the analyzer **as though it were a confirmatory cell**, which §4 forbids; it surfaced only
because the 1-trial truncated `olmo-3` pilot holds a single class and crashed the estimator. The driver now
archives to `data/results/r1d_pilots/`, the three already-written pilots were moved there before any further
collection, and the two stale `pilot.file` pointers in the ledger were repointed to the moved files. **No
measured quantity was altered**: the pilots' ambiguity rates, trial counts and probe counts are unchanged,
and the moved files are byte-identical.

**2026-09-17, operational note — §7.6's integrity pin covers the derived summary, so the derived summary was
restored.** `analyze_r1_faithful.py` writes a fixed path, `data/results/r1_faithful_v2_summary.json`, and
discovers cells by globbing the results directory — so re-running it with EXP-R1d's cells present overwrites
the six-target artifact the paper's §`app:r1c` reports with a seven-target one (its pooled row is fitted on
all trials at once, so it moves from 97.0→43.3% at *n*=300 to 97.1→42.0% at *n*=350). That filename also
matches §7.6's `r1_faithful_*.json` glob. It was therefore **restored byte-identical to its committed
version** (sha256 `a7259518cec64863…`), and EXP-R1d's derived panel was written to distinct paths:
`data/results/r1d_recency_v2_summary.json` and `data/results/r1d_recency_v2_analysis.txt`. Anyone re-running
the analyzer with EXP-R1d's cells in place obtains the seven-target panel; that is the expected behaviour
and the reason the two artifacts are kept apart rather than one overwriting the other.

**2026-09-17, note on the estimator's default — the pre-registered setting is not the analyzer's default.**
§6 fixes a **1000-draw** label-permutation test; `analyze_r1_faithful.py`'s `N_PERM` default is **200**,
whose smallest attainable *p* is 1/201 = 0.005. Every EXP-R1d *p* is therefore taken from
`--n_perm 1000`, and `data/results/r1d_recency_v2_analysis.txt` records the run that produced them (its
header line states the draw count). A 200-draw run of the same cells was also executed and agrees on every
accuracy to the digit, differing only in the *p* floor.

*(no deviations from any fixed quantity in §§3–6)*

## 10. Outcome

*(accumulating as targets resolve; the §7 branch is recorded once the roster is exhausted)*

| Target | Pilot ambiguity (instructed) | §4 verdict | Failure mode |
|---|---|---|---|
| `olmo-3:7b` | **100.0%** (160 probes / 10 trials) | **VOID** | Empty response channel. Thinking-mode model; `think: false` does not suppress it; 0 response tokens inside the fixed 40-token budget, so all 16 probe dimensions are constant at 0 and the 16-d binary instrument is not the one EXP-R1c measured. |
| `ministral-3:8b` | **0.0%** (160 probes / 10 trials) | **ADMISSIBLE** | — |
| `granite4.2:8b` | **35.0%** (160 probes / 10 trials) | **VOID** | Answer present but unparseable. Two modes, spread evenly over all 16 dimensions (1–6 of 10 trials each, none spared): in 39% of ambiguous probes the 40-token budget is spent on visible deliberation and no answer appears; in the rest the answer is embedded in an unrequested continuation of the dialogue it was shown (e.g. `**Assistant:** Yes` followed by fabricated further turns), which the yes/no parser cannot read. Unlike `olmo-3`, this target does emit answers — the 16-d vectors are non-degenerate (10 distinct over 10 trials) — it simply exceeds the gate the paper's own worst cell sets. |

### Confirmatory cell — `ministral-3:8b`, the one admissible target so far

Estimator, folds, *n* and draw count exactly as §6 fixes them (stratified 5-fold, 1000-draw label
permutation, *n* = 50 balanced trials per cell). §5's informativeness criterion was read off the
**instructed** cell before the equalized cell was collected, as §5 requires.

| Cell | 5-fold | grouped 5-fold | LOO | *p* | ambiguity | varying dims | distinct vectors |
|---|---|---|---|---|---|---|---|
| instructed | **100.0%** | 100.0% | 100.0% | **0.001** | 0.6% | 9/16 | 14 |
| equalized | **54.0%** | 44.0% | 24.0% | **0.327** | 4.3% | 6/16 | 10 |

**Informative** (instructed *p* = 0.001 ≤ 0.05), **admissible**, and its equalized cell is **not
degenerate** (6 of 16 dimensions still vary, 10 distinct vectors), so the fall to chance is not an artifact
of a dead channel. Change in the primary metric: **−46.0 pp**, with equalized *p* = 0.327 — i.e. H1 holds on
this target. Source: `data/results/r1d_recency_v2_analysis.txt`, `…/r1d_recency_v2_summary.json`.

### Roster state and the one permitted substitution

Two of the three named targets are void on their measured gate values, and one is admissible and
informative. §3 permits **at most one** substitution, and only for (a) a pull failure, (b) a gate failure or
(c) out-of-disk — so the reserve `qwen3.5:9b` was invoked **once**, against `granite4.2:8b`'s gate failure,
on 2026-09-17. No confirmatory trial had been collected for either void target, so the rule's bar is met.
`olmo-3:7b`'s gate failure is **not** substituted for; it is reported void and stands as such.

`qwen3.5:9b` also carries a `thinking` capability in the Ollama library, so it is **not** assumed
admissible; it runs the same §4 pilot and will be reported void on its measured rate if it exceeds 26.0%.
With the substitution now spent, the roster is closed either way: if `qwen3.5:9b` is admissible and
informative there are **two** such targets and §7 branch 1 or 2 applies; if it is void there is **one**, and
§7 **branch 5** applies — EXP-R1d is reported inconclusive, no recency claim enters the paper,
`app:vintage`'s "we do not establish that the collapse reproduces" sentence stays exactly as written, and
`app:future_directions` item (7) is revised only to record that the run was attempted and why it was
uninformative. Under branch 5 `ministral-3:8b`'s cell above is still reported, as a single admissible target
consistent with H1 and explicitly too thin to support a recency claim.

**Integrity checks run after every target.** All 26 pre-existing result files (v1 `r1_faithful_*` and the six
existing v2 `r1b_fresh_*` target pairs) verified byte-identical against hashes taken before collection, so
§7.6 holds and no checkpoint collision occurred. The derived summary at `r1_faithful_v2_summary.json` also
matches that glob and is byte-identical to its committed version; see §9.

**One finding about the paper's reproducibility, surfaced by these checks and independent of EXP-R1d.**
Re-running `analyze_r1_faithful.py --variant v2` on the six pre-existing targets reproduces every primary
figure exactly — including each instructed cell, each equalized 5-fold cell, every *p* at the
pre-registered 1000 draws, and the pooled 97.0%→43.3% / −53.7 pp at *n* = 300 — but yields three different
**grouped-5-fold** cells than the paper prints:
`tab:r1c_v2` Llama 3.2 3B equalized 46.0% → 36.0%, `tab:r1c_v2` and `tab:r1_variants` Mistral 7B equalized
54.0% → 58.0%, and `tab:r1c_v2`'s pooled equalized 47.7% → 48.3%. This is **not** EXP-R1d's doing: it
reproduces with `ministral-3:8b`'s cells held out, all 24 collected cell files are byte-identical, and two
consecutive runs are byte-identical to each other, so it is not nondeterminism — every estimator seed is
fixed (`StratifiedKFold(random_state=0)`, unshuffled `GroupKFold`, per-cell `zlib.crc32` permutation seed).
The cause is the installed scikit-learn version (1.8.0), which `requirements.txt` does not pin and the
appendix does not state. It is recorded here rather than repaired here, because pinning the version that
yields 46.0/54.0/47.7 and updating the three published cells are different commitments and the choice is the
authors'. (A fourth JSON difference, `llama3.1:8b`'s `instructed_grouped_kfold` serializing as
`0.8600000000000001` rather than `0.86`, is a float-repr artifact of a different summation order: both are
86.0% and the printed table is unchanged.)
