# Block 15 — Renal and Cyst Failure Decomposition

## Objective

Block 14 revealed two distinct pilot30 behaviors:

1. high overall renal Dice with a small number of severe HD95 outliers,
2. strong correlation between total cyst volume and reference burden despite
   very low cyst-positive spatial Dice.

Block 15 decomposes these observations before any model or post-processing
change is introduced.

## Renal error components

For every case, connected components of false-positive and false-negative
renal-envelope error are measured in physical space.

Recorded quantities include:

- component count,
- total error volume,
- largest error-component volume,
- farthest error voxel distance from the opposite mask.

This tests whether extreme HD95 values are caused by small distant islands.

## Cyst localization

For cyst-positive cases, KiTS23 label 3 and NV-Segment labels 116/117 are
decomposed into 3D connected components.

Each reference cyst component is compared with the nearest predicted cyst
component centroid in NIfTI world coordinates.

Localization recall is reported at 10, 20, and 30 mm, together with the median
nearest-centroid distance and the fraction of cyst-positive cases with zero
voxel overlap.

These measurements complement Dice rather than replace it.

## Why before interactive prompting?

NV-Segment-CT supports interactive point prompting and automatic+interactive
refinement. The maintained VISTA documentation explicitly recommends
NV-Segment-CT for CT tumor or interactive refinement.

However, supplying GT-derived clicks before characterizing automatic failure
would mix localization and contour-refinement questions.

Block 15 first determines whether automatic predictions are at least spatially
near the true cysts.

## Outputs

```text
outputs/analyses/nv_segment_ct_pilot30/failure_decomposition/
├── failure_decomposition.json
├── failure_cases.csv
└── figures/
    ├── hd95_vs_farthest_fp_distance.png
    └── cyst_dice_vs_centroid_distance.png
```

## Decision rule for the next block

- If HD95 strongly tracks distant false-positive error, test conservative
  prediction-only component filtering.
- If cyst localization recall is high despite poor Dice, test interactive
  contour refinement.
- If cyst localization recall is low, test interactive point prompting as a
  localization intervention before considering fine-tuning.
