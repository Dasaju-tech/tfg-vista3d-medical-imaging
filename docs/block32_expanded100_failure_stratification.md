# Block 32 — Expanded100 Failure Stratification

## Objective

Block 31 showed that several overlap metrics remain comparatively stable when
the cohort increases from 30 to 100 cases, while mean HD95 worsens materially.

Block 32 investigates whether that change is driven by a small geometric
failure tail or by a broad deterioration across the cohort.

No inference is performed and no frozen model or evaluation definition is
changed.

## Inputs

The analysis uses the completed expanded100 cohort:

```text
outputs/cohorts/nv_segment_ct_expanded100/cases.csv
```

and reuses the already downloaded KiTS23 reference segmentations to derive
ground-truth anatomy descriptors.

## Failure views

Five deterministic rankings are generated:

1. worst HD95;
2. worst Surface Dice;
3. worst kidney-and-mass Dice;
4. largest bilateral renal under-segmentation;
5. largest bilateral renal over-segmentation.

The first 15 cases are retained in each ranking.

## HD95 tail definition

A descriptive HD95 outlier threshold is computed as:

```text
Q3 + 1.5 * IQR
```

This is used only to characterize the failure tail. It is not a registered
clinical threshold.

## Subgroups

The cohort is stratified by:

- cyst-positive versus cyst-negative ground truth;
- tumor-positive versus tumor-negative ground truth;
- more than two kidney-and-mass connected components;
- anatomy-QC-flagged versus non-flagged cases.

## Exploratory correlations

Pearson correlations are reported for:

- HD95 vs kidney-and-mass Dice;
- HD95 vs Surface Dice;
- HD95 vs tumor burden;
- HD95 vs cyst burden;
- Surface Dice vs cyst burden;
- cyst Dice vs reference cyst volume among cyst-positive cases.

These correlations are descriptive. No causal interpretation or confirmatory
p-value claim is made.

## Laterality

The single expanded100 case with unresolved ground-truth laterality remains in
all global analyses. It is omitted only from bilateral renal-volume-error
rankings because those quantities are undefined.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_failure_stratification/
├── failure_stratification.json
├── ranked_cases.csv
├── subgroup_table.csv
├── failure_stratification.md
└── figures/
    ├── hd95_distribution.png
    ├── hd95_vs_surface_dice.png
    ├── hd95_vs_cyst_volume.png
    ├── dice_vs_cyst_volume.png
    └── cyst_dice_vs_reference_volume.png
```

## Interpretation boundary

Block 32 is post-hoc failure stratification of the frozen expanded100 cohort.
Its purpose is to choose the next controlled experiment from evidence rather
than intuition.

A subsequent model or prompting change must be implemented as a new experiment
version and compared against the frozen expanded100 baseline.
