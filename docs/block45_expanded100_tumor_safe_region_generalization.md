# Block 45 — Expanded100 Tumor Safe-Region Generalization

## Motivation

Block 44 showed that the nine severe GT→prediction tumor-undercoverage cases
have large physically interior regions:

```text
15 mm safe region:
non-empty = 9/9
mean fraction = 25.9%
median fraction = 24.1%
```

Together with Block 43, this made a well-interior clinician click look
practical in those nine cases.

However, those cases were selected specifically because they were severe
undercoverage outliers and contained large tumors. Their geometry is therefore
not representative of all tumor-positive expanded100 cases.

## Objective

Generalize the safe-region geometry audit to every tumor-positive case in the
frozen expanded100 cohort.

No interactive inference is repeated.

## Tumor target

For each expanded100 case:

1. load KiTS23 ground truth;
2. identify tumor label 2;
3. if tumor is absent, mark the case tumor-negative;
4. otherwise identify the largest 6-connected tumor component;
5. compute the physical Euclidean interior-distance map.

The largest component is used because Blocks 41–44 used a single point targeted
to one connected tumor object.

## Safe-region thresholds

For every tumor-positive case:

```text
5 mm
10 mm
15 mm
```

The audit measures:

- whether a non-empty safe region exists;
- safe-region fraction of the largest tumor component;
- safe-region physical volume;
- maximum interior depth.

## Size stratification

To test the main generalization concern, tumor-positive cases are divided by
largest-component volume quartiles.

For each size quartile, the audit reports 15 mm safe-region feasibility.

This determines whether the 15 mm interaction rule observed in the nine severe
outliers remains geometrically plausible for smaller tumors.

## No GPU

Only frozen KiTS23 ground-truth geometry and the frozen expanded100 case list
are used.

No model inference, prompt, checkpoint, or prediction is modified.

## Interpretation

If 10–15 mm safe regions remain common across the whole tumor-positive cohort,
the clinician-click interpretation becomes substantially stronger.

If 15 mm availability collapses in the smallest tumors while 10 mm remains
common, 10 mm is the more plausible interaction target even though Block 43
showed somewhat lower Dice.

If both 10 and 15 mm regions are often absent in small lesions, a fixed-depth
single-click protocol should not be generalized; multi-click or alternative
interactive guidance would be more appropriate.

## Limitation

All geometry remains derived from ground-truth tumor masks. This block measures
interaction feasibility, not autonomous tumor localization.

## Outputs

```text
outputs/experiments/nv_segment_ct_expanded100_tumor_safe_regions/
├── summary.json
├── cases.csv
└── safe_region_generalization.md
```
