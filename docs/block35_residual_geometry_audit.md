# Block 35 — Residual Geometry Audit

## Motivation

Block 34 tested whether the expanded100 HD95 tail was primarily an evaluation
scope artifact caused by tumor being included in the KiTS23 kidney-and-mass
reference while the automatic NV-Segment renal protocol lacks a validated
tumor output.

The sensitivity analysis did not support that explanation:

- mean HD95 worsened after tumor exclusion;
- only 2/22 original outliers fell below the original threshold;
- 25 cases exceeded the original threshold after tumor exclusion.

Therefore the residual HD95 tail must be analyzed as a genuine geometric
prediction/reference mismatch.

## Objective

Block 35 decomposes the 22 frozen HD95 outliers into directional geometric
error without repeating inference.

For each outlier it measures:

- false-positive volume;
- false-negative volume;
- number of FP and FN connected components;
- largest FP component volume;
- largest FN component volume;
- prediction-surface to GT-surface p95 distance;
- GT-surface to prediction-surface p95 distance;
- corresponding maximum directional distances.

## Direction labels

A case is described as:

```text
prediction_to_gt
```

when the prediction-surface p95 exceeds the reverse direction by more than
1 mm. This pattern is consistent with distant or excessive predicted geometry.

A case is described as:

```text
gt_to_prediction
```

when the GT-surface p95 exceeds the prediction-to-GT direction by more than
1 mm. This pattern is consistent with missing reference extent.

Otherwise the case is labelled:

```text
balanced
```

These are descriptive geometry labels, not clinical failure classes.

## Scope

The registered kidney-and-mass reference remains KiTS23 labels `{1,2,3}` and
the prediction remains the frozen union of the maintained NV-Segment renal
labels.

No model, prompt, metric, threshold, or prediction is modified.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_residual_geometry/
├── residual_geometry_audit.json
├── outlier_geometry.csv
├── residual_geometry_audit.md
└── figures/
    ├── hd95_vs_fp_volume.png
    ├── hd95_vs_fn_volume.png
    └── directional_surface_p95.png
```

## Interpretation boundary

The purpose of this audit is to decide whether the next controlled
intervention should target distant false-positive structures, missing
reference extent, or both.

Any post-processing or model change must be introduced as a new experimental
variant and evaluated against the frozen expanded100 baseline.
