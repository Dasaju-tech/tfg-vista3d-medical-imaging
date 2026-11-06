# Block 38 — LOOCV Selective-Trigger Validation

## Motivation

Block 37 found prediction-only features with useful retrospective separation of
the 12 prediction-to-GT-dominated HD95 outliers, led by:

```text
largest_to_second_ratio AUROC = 0.855
largest_component_volume_ml AUROC = 0.853
```

Directly choosing a trigger threshold on the same 100 cases would be
optimistically biased.

Block 38 therefore validates feature and threshold selection out-of-fold.

## Leave-one-out design

For each of the 100 cases:

1. hold that case out;
2. use the other 99 cases to choose the best univariate prediction-geometry
   feature;
3. choose its direction and threshold using training balanced accuracy;
4. apply the resulting trigger to the held-out case.

The held-out target label is never used to generate its own trigger decision.

This produces 100 out-of-fold trigger decisions.

## Trigger metrics

The audit reports:

- sensitivity;
- specificity;
- precision;
- balanced accuracy;
- TP / FP / TN / FN;
- feature-selection frequency across the 100 folds.

## Selective post-processing

The out-of-fold trigger decisions are then combined with the already evaluated
Block 36 filters.

For each held-out case:

```text
trigger false -> preserve original prediction metrics
trigger true  -> use candidate filtered-prediction metrics
```

Both `global_top2` and `side_largest` are evaluated independently.

The analysis reports:

- all-case mean and median HD95;
- remaining HD95 outliers using the frozen original threshold;
- number of target outliers resolved;
- mean target HD95 reduction;
- non-target HD95 and Dice harm.

No prediction file is overwritten.

## Important limitation

LOOCV reduces direct threshold-fitting bias but is not external validation.

The feature family was motivated by Block 37 on the same expanded100 cohort.
Therefore a successful trigger is still a research candidate and should be
validated on additional cases before deployment or generalization claims.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_selective_trigger_loocv/
├── selective_trigger_loocv.json
├── fold_decisions.csv
├── selective_trigger_loocv.md
└── figures/
    └── feature_selection_frequency.png
```
