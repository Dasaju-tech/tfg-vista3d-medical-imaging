# Block 47B — Dual Interior-Point Inference

## Objective

Test whether a second spatially separated interior tumor point improves the
single deepest-point interaction on the same 12 tumor-size-stratified cases
frozen in Blocks 46A and 47A.

## Frozen protocol

No case or point is changed after Block 47A.

For each case:

```text
Point 1 = deepest-interior tumor voxel

Point 2 = farthest tumor voxel from Point 1 among voxels satisfying:
          interior depth >= 50% of maximum tumor depth

point_labels = [3, 3]
label_prompt = None
```

## Reference

The comparator is the single deepest-interior point already evaluated in
Block 46B.

Therefore only the 12 dual-point interactions require new GPU inference.

## Primary hypothesis

The second point should improve tumors whose morphology is not adequately
represented by one local interior seed.

The upper-middle quartile is particularly informative because the Block 46B
single adaptive rule showed its largest loss there:

```text
adaptive-deep tumor Dice delta = -0.1627
```

Block 47A also showed substantial point separations in this quartile:

```text
20–39 mm
```

## Metrics

Dual and deep variants are compared using:

- tumor Dice;
- tumor coverage gain;
- HEC Dice delta;
- HEC Surface Dice delta;
- HEC HD95 reduction.

Paired dual-minus-deep differences are reported:

- overall;
- by tumor-size quartile;
- with deterministic 5000-sample bootstrap 95% confidence intervals for
  mean tumor-Dice and HD95-reduction differences.

The analysis also reports the Pearson association between physical point
separation and paired tumor-Dice gain.

## Decision boundary

The dual-point protocol is promising only if it provides a consistent benefit
rather than helping one size stratum while harming another.

Particular evidence of value would include:

- positive overall paired tumor-Dice delta;
- improved or preserved HD95;
- recovery of the upper-middle quartile;
- no collapse in the smallest quartile.

If dual interaction provides little or inconsistent gain, the project should
stop expanding oracle prompt engineering and preserve Blocks 41–47 as a
systematic interaction-capability study.

## Limitation

Both points are derived from ground-truth tumor masks.

This experiment measures interactive segmentation capability, not autonomous
tumor localization or deployment performance.

## Outputs

```text
outputs/experiments/nv_segment_ct_tumor_dual_interior_point/inference/
├── summary.json
└── cases.csv

outputs/predictions/nv_segment_ct_interactive_tumor_dual_point/
└── case_XXXXX/
    ├── raw/
    ├── prediction.nii.gz
    └── result.json
```
