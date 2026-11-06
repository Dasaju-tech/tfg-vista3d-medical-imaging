# Block 41B — Single-Case Tumor Oracle Smoke

## Objective

Run exactly one real interactive VISTA3D inference to test whether the frozen
special-positive tumor point from Block 41A exposes useful tumor segmentation
capability.

The smoke case is:

```text
case_00430
point = (41, 218, 348)
point_label = 3
label_prompt = None
```

The point was frozen in Block 41A from the deepest interior voxel of the largest
KiTS23 tumor component.

## Why case_00430

Block 39 identified `case_00430` as the largest undercoverage failure:

```text
false-negative volume ≈ 2638.10 mL
missing reference fraction ≈ 92%
dominant missing label = tumor
```

Its largest tumor component is approximately 2590.83 mL out of 2590.96 mL
total tumor volume. Therefore a one-point capability test is not confounded by
many similarly sized tumor components.

## Runtime input

The maintained MONAI bundle receives:

```text
image
points = [[41, 218, 348]]
point_labels = [3]
```

No `label_prompt` is supplied.

The experiment remains an oracle upper bound because the point coordinate was
derived from ground truth.

## Evaluation

The interactive foreground is compared against:

1. the exact largest tumor component that generated the point;
2. the full KiTS23 tumor union.

The interactive foreground is then unioned with the frozen automatic renal
prediction only for a diagnostic kidney-and-mass sensitivity comparison.

Reported outputs are:

- interactive largest-tumor Dice;
- interactive full-tumor Dice;
- tumor coverage before/after merge;
- kidney-and-mass Dice before/after merge;
- Surface Dice before/after merge;
- HD95 before/after merge;
- runtime labels, 255/unprocessed presence, and wall time.

## Decision boundary

A clearly useful interactive tumor mask and a material reduction in HEC HD95
would justify Block 41C on all nine frozen tumor-positive undercoverage cases.

If the model produces little tumor overlap even with this deeply interior
GT-derived point, the interactive prompting route should not be expanded
without a new technical hypothesis.

## Outputs

```text
outputs/predictions/nv_segment_ct_interactive_tumor_oracle/
└── case_00430/
    ├── raw/
    ├── prediction.nii.gz
    └── result.json
```
