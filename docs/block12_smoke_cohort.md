# Block 12 — Three-Case Real Smoke Cohort

## Objective

Block 12 aggregates the three independently inferred, evaluated, and visually
audited KiTS23 smoke cases into the first cohort-level result table.

The goal is not statistical generalization from three subjects. The goal is to
verify that the full experimental pipeline behaves coherently across
non-identical real cases before scaling to a larger evaluation cohort.

## Primary supported metric

The main cohort summary reports the supported KiTS23
`kidney_and_mass` HEC component:

- Dice
- Surface Dice
- diagnostic HD95

For each metric the output includes:

- count
- mean
- median
- sample standard deviation
- minimum
- maximum

The full KiTS23 score remains unavailable because automatic NV-Segment-CT does
not provide a validated kidney-tumor output for the `mass` and `tumor` HECs.

## Side-aware diagnostics

The cohort also aggregates:

- kidney-core Dice
- left renal-envelope Dice
- right renal-envelope Dice
- side-specific volume errors

These remain engineering diagnostics rather than KiTS23 ranking metrics.

## Cyst reporting rule

Empty-reference cases require special treatment.

If both ground truth and prediction contain no cyst, the per-case binary Dice
is correctly defined as `1.0`. However, averaging such empty-perfect cases
together with cyst-positive cases can strongly overstate cyst segmentation
ability.

Block 12 therefore reports both:

1. `cyst_dice_all_cases`
2. `cyst_dice_cyst_positive_cases`

The second value is the appropriate performance summary when discussing actual
cyst segmentation ability.

Cyst-positive reference and predicted physical volumes are summarized
separately.

## Runtime

When inference manifests are available, wall-clock inference time is aggregated
without silently substituting missing values.

## QC linkage

When Block 11 visual-QC JSON files are available, the per-case CSV carries:

- false-positive voxel count
- false-negative voxel count
- total disagreement voxel count

These values are retained primarily for case-level traceability because raw
voxel counts can depend on acquisition spacing.

## Outputs

```text
outputs/cohorts/nv_segment_ct_smoke/
├── summary.json
└── cases.csv
```

## Interpretation boundary

Three real cases are sufficient to close the smoke-validation phase but not to
make population-level performance claims.

If the cohort is internally coherent, the next experimental phase should scale
to a larger deterministic KiTS23 subset with frozen runtime, frozen prompts,
and the same evaluation protocol.
