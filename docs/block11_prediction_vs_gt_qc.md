# Block 11 — Prediction-versus-Ground-Truth Visual QC

## Objective

Block 11 converts the quantitative Block 10 comparison into deterministic,
high-information visual evidence.

For each orthogonal plane the renderer selects the slice containing the largest
renal-envelope disagreement between KiTS23 and NV-Segment-CT.

## Four-panel figure

Each generated figure contains:

1. CT only,
2. KiTS23 semantic reference,
3. NV-Segment supported renal prediction,
4. renal-envelope agreement/error.

The error map uses four integer states:

```text
0 background
1 true positive
2 false positive
3 false negative
```

## Slice-selection hierarchy

For each plane:

1. maximum prediction/reference disagreement,
2. maximum KiTS23 tumor/cyst area when disagreement is zero,
3. maximum KiTS23 renal area,
4. volume center.

The selected slice and reason are stored in JSON.

This selection rule is deterministic and is preferred over manually choosing
visually attractive slices.

## Interpretation

The Block 10 case_00000 result shows high renal overlap but lower Surface Dice
than volumetric Dice. The Block 11 figures are intended to determine whether
the residual error is concentrated around:

- renal poles,
- tumor-adjacent contour regions,
- partial-volume edges,
- isolated false-positive islands.

No anatomical explanation is assumed before visual inspection.

## Outputs

```text
outputs/qc/nv_segment_ct_comparison/case_XXXXX/
├── axial_prediction_vs_gt.png
├── coronal_prediction_vs_gt.png
├── sagittal_prediction_vs_gt.png
└── comparison_qc.json
```

## Next step

If the visual QC confirms plausible boundary-level residual error, the pipeline
can be run on the remaining real smoke cases and then converted into a batch
inference/evaluation workflow.
