# Camera-ready human ICC study (v2)

This directory holds raw annotator files and aggregated ICC results for the camera-ready validation of correction-density described in `docs/icc_annotation_protocol_v2.md`.

## Schema

Each `annotator_{k}.json` contains:

```json
{
  "metadata": {
    "annotator_id": "prolific:XXXX",
    "study_id": "icc_v2_neurips26",
    "n_trials": 100,
    "date_started": "YYYY-MM-DDTHH:MM:SSZ",
    "date_completed": "YYYY-MM-DDTHH:MM:SSZ",
    "attention_checks_passed": 3
  },
  "ratings": [
    {
      "trial_id": "abc123",
      "ground_truth_blinded": true,
      "consistency": 8,
      "specificity": 7,
      "correction_marker_density": 2,
      "confidence": 9,
      "elaboration": 6
    }
  ]
}
```

`trial_id` corresponds to an entry in the separately-stored `trial_manifest.json` which maps to the underlying equalized transcript (with target model and ground-truth condition unblinded only server-side).

## Files expected after the study runs

- `trial_manifest.json` — sampling manifest (n=100 trials + 3 attention checks; stratified over target and ground-truth).
- `annotator_1.json`, `annotator_2.json`, `annotator_3.json` — raw ratings, one file per annotator.
- `icc_results.json` — aggregated Krippendorff's α, ICC(2,1), ICC(2,k), Spearman ρ per feature and pooled.
- `analysis.ipynb` — Jupyter notebook regenerating the aggregated results from raw annotator files.

## Status (2026-04-30)

Protocol frozen; recruitment scheduled for the NeurIPS 2026 camera-ready window. Exempt determination letter to be attached to the supplementary material.
