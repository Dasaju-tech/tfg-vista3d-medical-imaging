# Block 43 — Tumor Click Depth Audit

## Motivation

Block 42 showed a strong location-dependent interaction pattern:

```text
deep tumor Dice mean      = 0.7557
centroid tumor Dice mean  = 0.7676
shallow 5 mm mean         = 0.4461
```

Centroid clicks retained or slightly improved the deep-oracle result, while
clicks approximately 5 mm from the tumor boundary degraded substantially.

Therefore exact maximum-depth localization is not required, but sufficiently
interior placement appears important.

## Objective

Estimate the practical interior-depth region where one special-positive tumor
click becomes stable.

The audit adds two deterministic GT-simulated depths:

```text
10 mm from tumor boundary
15 mm from tumor boundary
```

and compares them with the existing:

```text
5 mm
centroid
deepest interior point
```

## Runtime cost

Existing deep, centroid, and 5 mm predictions are reused.

Only:

```text
9 cases × 2 new depths = 18 new inferences
```

are required.

## Metrics

For each depth condition:

- tumor Dice mean and median;
- cases with tumor Dice >= 0.50 and >= 0.70;
- mean tumor coverage gain;
- mean and median HEC HD95 reduction;
- number of cases with improved HD95;
- Dice difference relative to the deep oracle;
- number of cases within ±0.05 and ±0.10 Dice of the deep oracle.

## Interpretation

If 10 mm already approaches centroid/deep performance, the interaction is
tolerant to moderate localization error.

If 15 mm is required but 10 mm remains weak, a clinician-assisted workflow
should recommend clicking well inside the visible lesion.

If even 15 mm is substantially inferior, click success depends strongly on
lesion-centre localization and the practical workflow requires additional
guidance or multiple-point interaction.

## Limitation

All click positions remain derived from ground-truth tumor masks.

This block quantifies localization tolerance and should not be described as
autonomous detection or deployment-valid localization.

## Outputs

```text
outputs/experiments/nv_segment_ct_tumor_click_depth_audit/
├── summary.json
├── cases.csv
└── depth_audit.md
```
