# Block 14 — Pilot30 Statistical Audit and Failure Stratification

## Objective

Block 13 established that the frozen 30-case pipeline can run end to end.
Block 14 analyzes the completed cohort before any further scaling or model
changes.

The purpose is not to claim a definitive benchmark. It is to quantify
uncertainty, characterize failure modes, and determine which experimental
question should be addressed next.

## Bootstrap confidence intervals

For the principal supported metrics, Block 14 reports deterministic percentile
bootstrap 95% confidence intervals using:

```text
seed = 20260828
samples = 5000
```

Metrics:

- kidney-and-mass Dice
- kidney-and-mass Surface Dice
- diagnostic HD95
- cyst-positive-only Dice

The bootstrap is used because the pilot cohort is small and metric
distributions can be non-Gaussian.

## Cyst-positive stratification

The pilot contains enough cyst-positive cases to move beyond the smoke-set
single-case observation.

Block 14 separates:

- cyst-positive cases
- cyst-negative cases

and compares kidney performance between the two groups.

Cyst ability remains summarized only on cyst-positive cases. Empty-perfect
cases are never allowed to inflate this estimate.

## Failure ranking

Three deterministic rankings are produced:

1. worst kidney-and-mass Surface Dice,
2. worst HD95,
3. largest bilateral renal volume under-segmentation.

This gives distinct views of boundary error, outlier distance, and systematic
volume bias.

## Correlation diagnostics

The audit reports exploratory Pearson correlations for:

- Surface Dice versus HD95,
- ground-truth versus predicted cyst volume in cyst-positive cases.

These are descriptive diagnostics only. They are not causal claims and no
p-value fishing is performed.

## Figures

The analysis generates:

```text
surface_dice_distribution.png
surface_dice_vs_hd95.png
cyst_volume_reference_vs_prediction.png
```

The third figure is emitted only when cyst-positive cases are available.

## Outputs

```text
outputs/analyses/nv_segment_ct_pilot30/
├── statistical_audit.json
├── ranked_cases.csv
└── figures/
    ├── surface_dice_distribution.png
    ├── surface_dice_vs_hd95.png
    └── cyst_volume_reference_vs_prediction.png
```

## Interpretation boundary

The audit may reveal a reproducible weakness of automatic cyst segmentation.
If so, the next experimental block should investigate that limitation rather
than simply increasing the sample size.

Likewise, if poor renal cases cluster around a specific anatomical or
segmentation condition, that should guide the next controlled experiment.
