# Block 37 — Prediction-Geometry Trigger Audit

## Motivation

Block 36 tested two connected-component filters on all 100 frozen predictions.

Both variants were safe on the 78 non-outlier cases but ineffective as global
post-processing:

```text
target outliers resolved: 2/12
target mean HD95 reduction: +0.36 mm
all-case mean HD95: ~13.55 mm
```

Therefore connected-component pruning should not be adopted globally.

A more defensible strategy would require a prediction-only trigger that
identifies cases likely to contain distant or excessive predicted geometry.

## Objective

Block 37 asks whether the 12 Block 35 prediction-to-GT-dominated HD95 outliers
can be separated from the remaining 88 cases using only features derived from
the predicted segmentation.

The retrospective target labels are used only to evaluate separability.

## Prediction-only features

For each frozen prediction, the audit extracts:

- total number of renal connected components;
- left-side component count;
- right-side component count;
- total predicted renal volume;
- largest component volume;
- second-largest component volume;
- volume contained in components ranked third or smaller;
- fraction of predicted renal volume in third-or-smaller components;
- smallest component volume;
- largest-to-second component volume ratio;
- maximum world-space distance between component centroids.

No ground-truth image contributes to these feature values.

## Evaluation

Each feature is evaluated as a univariate trigger candidate using:

- AUROC, allowing either high or low values to indicate risk;
- target/non-target mean and median;
- number of target cases among the 12 most extreme predictions;
- target recall among the 20 most extreme predictions.

This is exploratory trigger discovery, not a registered classifier.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_prediction_geometry_trigger/
├── prediction_geometry_trigger.json
├── prediction_geometry_cases.csv
├── feature_rankings.csv
├── prediction_geometry_trigger.md
└── figures/
    └── feature_auc.png
```

## Interpretation boundary

The 12 target labels are derived from Block 35 and therefore use retrospective
ground-truth evaluation. They must not be used directly at deployment time.

Only the prediction-derived features are deployment-compatible.

A trigger should advance to a subsequent controlled experiment only if it shows
meaningful separation and can be defined without using ground truth.
