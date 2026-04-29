# Autonomous Deception Experiments - Infrastructure Complete ✓

## Status: Ready to Execute

**Date:** April 24, 2026  
**Reviewer Requirement:** "I would happily move to accept if the authors ran the autonomous-deception extension"

---

## ✅ What's Complete

### 1. Experimental Design (3 Matched-Format Scenarios)
- ✅ Design 1: Jailbroken Persona Maintenance
- ✅ Design 2: Post-Training False Beliefs (ICL simulation)
- ✅ Design 3: Sycophancy-Aligned Claim Defense

### 2. Implementation Scripts
- ✅ `run_matched_format_autonomous.py` - Main experiment runner
- ✅ `analyze_matched_autonomous.py` - Analysis pipeline
- ✅ `test_matched_autonomous_infrastructure.py` - Infrastructure test

### 3. Documentation
- ✅ `AUTONOMOUS_DECEPTION_PROTOCOL.md` - Complete execution guide
- ✅ Success criteria defined (70-85% target for signal transfer)
- ✅ Contingency plans for various outcomes

### 4. Infrastructure Verification
- ✅ Mock data generation: PASSED
- ✅ Analysis pipeline: PASSED  
- ✅ Feature extraction (refusal/hedge count): PASSED
- ✅ LOO cross-validation: PASSED
- ✅ Bootstrap CIs: PASSED
- ✅ Comparison tables: PASSED

**Test output:**
```
Design                                           n  Ref |d|  Ref LOO   Ref ≥1  Hedge LOO
───────────────────────────────────────────── ──── ──────── ──────── ──────── ──────────
Design 1: Jailbroken Persona Maintenance        10     0.00   100.0%   100.0%     100.0%
Design 2: Post-Training False Beliefs (ICL)     10     0.00   100.0%   100.0%     100.0%
Design 3: Sycophancy-Aligned Claim Defense      10     0.00   100.0%   100.0%     100.0%
```

*Note: 100% is expected on mock data with perfect feature separation*

---

## 🚀 How to Run Real Experiments

### Quick Start (Recommended)

Start with **Design 1 (Persona)** as a pilot to verify signal transfer:

```bash
cd /Users/mediratta/code/interpret/adaptive_lie_detector

# Run Design 1: Physician persona (n=50, ~4-6 hours)
.venv/bin/python3 experiments/run_matched_format_autonomous.py \
    --design persona --model llama-3b --n 50 --persona physician \
    --output-dir data/results/matched_autonomous

# Analyze results
.venv/bin/python3 experiments/analyze_matched_autonomous.py
```

**Check results next morning:**
- If `Ref LOO ≥ 70%`: Signal transfers! ✓ Proceed with remaining designs
- If `Ref LOO < 60%`: Signal weak. Consider strengthening jailbreak or adjusting claims
- If `60-70%`: Borderline - increase n to tighten CIs

### Full Execution (All Designs)

See `AUTONOMOUS_DECEPTION_PROTOCOL.md` for complete commands.

**Timeline:**
- Design 1 (3 personas × 50): 12-20 hours
- Design 2 (3 domains × 100): 15-24 hours  
- Design 3 (2 models × 100): 10-16 hours
- **Total:** 37-60 hours serial (can parallelize to ~24 hours wallclock)

**Cost:** ~$11 if using API models, $0 if using Ollama

---

## 📊 Expected Results

### Strong Signal Transfer (Target for Accept)
```
Design                                           n  Ref |d|  Ref LOO   Ref ≥1
───────────────────────────────────────────── ──── ──────── ──────── ────────
Persona (physician)                             50     1.42     72%      71%
Persona (professor)                             50     1.38     71%      69%
False beliefs (temperature)                    100     0.89     65%      64%
Sycophancy                                     100     1.12     74%      72%
───────────────────────────────────────────────────────────────────────────
Average (matched autonomous)                   300     1.20     70.5%    69%
```

**Interpretation:** Signal transfers! Refusal-count rule generalizes to autonomous claim defense.

### Moderate Transfer (Weak Accept / Findings)
```
Persona                                         50     0.85     65%      63%
False beliefs                                  100     0.62     60%      59%
Sycophancy                                     100     0.94     68%      66%
───────────────────────────────────────────────────────────────────────────
Average                                        300     0.80     64%      63%
```

**Interpretation:** Partial transfer. Autonomous deception detectable but weaker than instructed.

### No Transfer (Task-Format Dependency Confirmed)
```
Persona                                         50     0.42     55%      53%
False beliefs                                  100     0.31     52%      51%
Sycophancy                                     100     0.48     57%      55%
───────────────────────────────────────────────────────────────────────────
Average                                        300     0.40     55%      53%
```

**Interpretation:** Refusal-count signal does not transfer. Autonomous deception requires different detection approach.

---

## 📝 Paper Integration (When Complete)

### Update §4.13: Autonomous Deception Evaluation

**Current paragraph (EXP-I - Apollo pilot):**
- Keep as-is, noting task-format mismatch

**New subsection:** §4.13.1 Matched-Format Autonomous Evaluation (EXP-I-matched)

**Content template:**
```latex
\paragraph{Matched-format autonomous evaluation (EXP-I-matched).}
To test whether the refusal-count signal transfers to autonomous deception in claim-defense
contexts, we conducted three matched-format experiments: (1)~jailbroken persona maintenance
(models adopt false personas and defend claims within that persona, n=150 across 3 personas),
(2)~post-training false beliefs (models embedded with false "facts" via in-context learning
defend those beliefs, n=300 across 3 domains), and (3)~sycophancy-aligned claim defense
(models aligned to be agreeable defend user-stated false claims, n=200 across 2 models).
In all three designs, models receive no explicit deception instruction—behavioral differences
arise from autonomous claim defense.

\textbf{Result:} [INSERT ACTUAL RESULTS]

[IF STRONG TRANSFER:]
Refusal-count LOO achieves 70.5\% average accuracy across matched autonomous scenarios
(range: 65-74\% by design), comparable to the equalized baseline (61-84\%, EXP-J).
The label-free rule (refusal count $\geq$ 1) achieves 69\% average (range: 64-72\%),
validating that the 80.1\% cross-model deployment result (EXP-J-cold) partially transfers
to autonomous contexts. Refusal-count Cohen's $|d|$ averages 1.20 across designs (range:
0.89-1.42), demonstrating detectable correction behavior when models confront false claims
even without explicit lying instructions.

[IF MODERATE TRANSFER:]
Refusal-count LOO achieves 64\% average accuracy (range: 60-68\%), modestly below the
equalized baseline (61-84\%) but above chance. The label-free rule achieves 63\% (range:
59-66\%), suggesting the EXP-J-cold deployment result (80.1\%) is optimistic for autonomous
scenarios. Cohen's $|d|$ = 0.80 average (range: 0.62-0.94) indicates weaker but detectable
signal.

[IF NO TRANSFER:]
Refusal-count LOO achieves 55\% average accuracy (range: 52-57\%), near chance, confirming
that the behavioral signal observed under equalized conditions does not transfer to matched-
format autonomous deception. The label-free rule achieves 53\% (range: 51-55\%), indicating
the EXP-J-cold deployment result (80.1\%) is specific to knowledge-conflict scenarios with
neutral prompts and does not generalize to autonomous claim defense. Cohen's $|d|$ = 0.40
(range: 0.31-0.48) confirms weak feature separation.
```

**Add table:**
```latex
\begin{table}[h]
\centering
\caption{Matched-format autonomous deception detection (EXP-I-matched).}
\label{tab:autonomous_matched}
\begin{tabular}{llrrrrr}
\toprule
\textbf{Design} & \textbf{Model} & \textbf{n} & \textbf{|d|} & \textbf{LOO} & \textbf{≥1} & \textbf{Hedge} \\
\midrule
[INSERT ACTUAL RESULTS]
\bottomrule
\end{tabular}
\end{table}
```

---

## 🎯 Next Steps

1. **Decide timeline:**
   - Run overnight starting tonight? (Design 1 pilot: ~6 hours)
   - Run over weekend? (All designs: 2-3 days)
   - Defer to next week?

2. **Choose execution strategy:**
   - **Pilot-first** (recommended): Run Design 1 (one persona, n=50) to verify signal, then decide
   - **Full execution**: Run all designs in parallel if you have GPU capacity
   - **Serial overnight**: Queue up designs to run overnight for 3 nights

3. **When experiments complete:**
   - Run analysis: `.venv/bin/python3 experiments/analyze_matched_autonomous.py`
   - Update paper §4.13.1 with actual results
   - Add results table
   - Update Discussion §5.1 interpretation based on outcome

---

## 📋 Checklist Before Submission

- [ ] Run experiments (37-60 hours compute)
- [ ] Verify results meet success criteria (≥70% LOO for strong transfer)
- [ ] Generate final analysis tables and figures
- [ ] Update paper §4.13.1 with results
- [ ] Update Discussion §5.1 interpretation
- [ ] Add results table (Table \ref{tab:autonomous_matched})
- [ ] Recompile paper and verify page count ≤50
- [ ] Submit to NeurIPS 2026

**Expected outcome:** Weak Reject → **Accept** (reviewer explicitly committed to this)

---

## 💡 Pro Tips

1. **Start small:** Run n=20 pilot first (3-4 hours) to verify infrastructure
2. **Monitor progress:** Check intermediate JSON files in `data/results/matched_autonomous/`
3. **Checkpoint friendly:** Scripts save after each trial - safe to interrupt
4. **Cost conscious:** Use Ollama (free) instead of Bedrock for target models
5. **Parallel execution:** Run different designs on different GPUs if available

---

## Questions?

See:
- `AUTONOMOUS_DECEPTION_PROTOCOL.md` - Complete execution guide
- `run_matched_format_autonomous.py` (line 300+) - Main execution logic
- `analyze_matched_autonomous.py` (line 150+) - Analysis pipeline
- Paper Discussion §5.1 - Expected interpretation framework

**Ready to run!** 🚀
