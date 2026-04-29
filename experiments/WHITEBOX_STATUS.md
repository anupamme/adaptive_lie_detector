# White-Box Probing Comparison - Status

## Goal

Compare representation-based (white-box) lie detection vs. behavioral detection on equalized data.

**Reviewer quote:** "Could a white-box probing classifier distinguish the equalized truth/lie conditions where behavioral methods drop to 52–69%? This seems like the most informative missing experiment."

## Expected Outcomes & Interpretation

### Outcome 1: White-box >> behavioral (e.g., 85% vs 62%)
**Interpretation:** Representations encode truth/lie distinctions that behavioral features miss.
**Paper impact:** Strengthens "behavioral detects instruction-following, not deception" claim.
**Novelty boost:** High - shows fundamental limitation of behavioral approaches.

### Outcome 2: White-box ≈ behavioral (e.g., 65% vs 62%)
**Interpretation:** Both methods collapse under equalization; knowledge-conflict is genuinely hard.
**Paper impact:** Strengthens "equalized detection is fundamentally challenging" claim.
**Novelty boost:** Moderate - validates difficulty, adds white-box triangulation.

### Outcome 3: White-box < behavioral (e.g., 55% vs 62%)
**Interpretation:** Behavioral features capture something representations miss (unlikely).
**Paper impact:** Unexpected finding, requires careful interpretation.
**Novelty boost:** High if real, but may indicate implementation issue.

## Progress

### Phase 1: Dataset Preparation ✅
- **Status:** Complete
- **Models:** Llama 8B, Mistral 7B, Qwen 7B, Qwen 14B (skipped Llama 3B: n=2)
- **Samples:** 397 total (balanced truthful/lying)
- **Output:** `data/whitebox_probing/*_equalized.json`

### Phase 2: Representation Extraction 🔄
- **Status:** In progress
- **Current:** Mistral 7B (running in background)
- **Pending:** Llama 8B, Qwen 7B, Qwen 14B
- **Method:** Final-layer, last-token hidden states
- **Device:** MPS (Apple Silicon GPU)
- **Estimated time:** ~15 min per model (100 samples)

#### Extraction Progress
- [ ] Mistral 7B (4096-dim hidden states) - RUNNING
- [ ] Llama 8B (4096-dim hidden states)
- [ ] Qwen 7B (3584-dim hidden states)
- [ ] Qwen 14B (5120-dim hidden states)

### Phase 3: Probing Classifier Training ⏳
- **Status:** Script ready, waiting for representations
- **Method:** Logistic regression with StandardScaler
- **Evaluation:** Leave-one-out cross-validation (matches behavioral LOO)
- **Comparison:** Direct comparison to behavioral baselines from paper
- **Output:** `data/whitebox_probing/probing_results.json`

### Phase 4: Paper Integration ⏳
- **Status:** Pending results
- **Target:** New §4.5 "White-Box Comparison"
- **Content:**
  - Method description (representation extraction + probing)
  - Results table (white-box vs behavioral per model)
  - Interpretation (outcome 1, 2, or 3)
  - Discussion of implications for behavioral detection

## Behavioral Baselines (from paper)

These are the targets to compare against:

| Model | Mistral Extraction | Claude Extraction | Qwen Extraction |
|-------|-------------------|------------------|-----------------|
| Llama 8B | 61% | 71% | - |
| Mistral 7B | 62% | 75% | - |
| Qwen 7B | 57% | - | 66% |
| Qwen 14B | 52% | 69% | - |

**Key observation:** Claude extraction consistently outperforms cross-family extraction by 9-17pp, suggesting extractor capability differences (not same-family bias per Qwen/Mistral within-family tests).

## Timeline

**Week 1 (Current):**
- Days 1-3: Dataset prep (✅), representation extraction (🔄)
- Days 4-5: Probing training + analysis
- Days 6-7: Paper integration

**Estimated completion:** 2-3 days for extraction + analysis, then 1 day for writing.

## Next Immediate Steps

1. **Wait for Mistral extraction** (~10-15 min remaining)
2. **Test probing pipeline** on Mistral results
3. **Launch parallel extractions** for remaining models (Llama 8B, Qwen 7B, Qwen 14B)
4. **Analyze all results** once extraction complete
5. **Draft §4.5** for paper

## Files Created

### Scripts
- `experiments/whitebox_1_prepare_datasets.py` - Dataset preparation ✅
- `experiments/whitebox_2_extract_representations.py` - Representation extraction ✅
- `experiments/whitebox_3_train_probes.py` - Probing classifier training ✅

### Data
- `data/whitebox_probing/manifest.json` - Dataset index
- `data/whitebox_probing/{model}_equalized.json` - Prepared datasets (5 models)
- `data/whitebox_probing/{model}_representations.json` - Extracted representations (pending)
- `data/whitebox_probing/probing_results.json` - Final results (pending)

## Key Design Decisions

### Why final-layer, last-token?
- Standard for causal LMs (used in SAPLMA and most probing work)
- Captures model's final decision state before output
- Alternative: mean-pooled sequence (could test if time permits)

### Why logistic regression?
- SAPLMA standard (linear probe)
- Interpretable (single direction per class)
- Comparable to behavioral pipeline (also uses logistic regression on features)
- Strong L2 regularization prevents overfitting

### Why LOO cross-validation?
- Matches behavioral evaluation method (fair comparison)
- Standard for small-n probing studies
- Provides per-sample predictions for bootstrap CIs

### Why skip Llama 3B?
- Only n=2 samples (incomplete equalized run)
- Insufficient for LOO (needs n≥10 minimum)
- Focus on 7B+ models where we have full n=97-100

## Potential Issues & Mitigation

### Issue 1: Model downloads slow
- **Status:** Currently downloading Mistral 7B from HuggingFace
- **Mitigation:** Subsequent models will be faster (caching)
- **Fallback:** Can run overnight if needed

### Issue 2: Memory constraints for 14B
- **Risk:** Qwen 14B (5120-dim × 97 samples ≈ 2MB) should fit easily
- **Mitigation:** Using fp16, MPS device, batch_size=1
- **Fallback:** Extract on CPU if MPS OOM

### Issue 3: Extraction time exceeds estimate
- **Current estimate:** 15 min per model × 4 models = 60 min total
- **Actual:** Depends on download speeds + inference speed
- **Mitigation:** Running in background, can parallelize if needed

### Issue 4: White-box results ambiguous (e.g., 68% vs 62% behavioral)
- **Risk:** Small difference (6pp) may not be conclusive
- **Mitigation:** Bootstrap CIs will quantify uncertainty
- **Interpretation:** Frame as "comparable" (Outcome 2) rather than forcing conclusion

## Questions for User

1. **Parallel extraction:** Once Mistral completes, should I launch all 3 remaining models in parallel? (Faster but more resource-intensive)

2. **Alternative representations:** If time permits, should we also test mean-pooled sequences (vs last-token only)?

3. **Additional models:** Should we include Llama 70B or Claude Haiku (via Bedrock API for representations)?

---

**Last updated:** 2026-04-25 (Phase 2 in progress)
