# Block 47A — Dual Interior-Point Protocol Freeze

## Motivation

Block 46B tested a pre-specified single-point rule at 70% of each tumor's
maximum interior depth.

Overall the adaptive rule was close to the deepest oracle point:

```text
adaptive tumor Dice mean = 0.4982
deep tumor Dice mean     = 0.5219
paired mean delta        = -0.0236
95% bootstrap CI         = [-0.1014, +0.0444]
```

However, performance was not stable across tumor-size strata. The upper-middle
quartile showed a substantial paired Dice loss:

```text
adaptive-deep mean Dice delta = -0.1627
```

A single global relative-depth rule is therefore not sufficiently robust.

## New hypothesis

The remaining failure may be related to spatial coverage rather than simply
point depth.

An irregular or elongated tumor can contain several distant interior regions.
One point, even when deep, may not adequately represent its full geometry.

Block 47 therefore tests two spatially distributed positive points.

## Frozen dual-point rule

The same 12 cases selected in Block 46A are retained. No new cases are chosen.

### Point 1

```text
deepest-interior voxel
```

### Point 2

Among tumor voxels satisfying:

```text
interior depth >= 0.50 × case-specific maximum interior depth
```

choose the voxel with the greatest physical Euclidean distance from Point 1.

Thus the second point is:

- inside the same frozen largest tumor component;
- still substantially interior;
- deliberately spatially separated from the first point;
- scaled automatically to lesion size.

Both points use:

```text
point_labels = [3, 3]
label_prompt = None
```

## Why not tune another single-point percentage?

Block 46B does not justify selecting another percentage after observing the
results. Repeatedly trying 60%, 80%, and similar rules would become
retrospective tuning on a small sample.

The dual-point hypothesis changes the interaction mechanism rather than
retuning the failed single-point parameter.

## Block 47B comparison

The next block can compare:

```text
single deepest point
vs
dual interior points
```

on the same 12 frozen cases.

The single-deep results already exist from Block 46B, so only 12 new dual-point
interactive inferences should be required.

## Primary question

Does adding a second spatially separated interior point improve robustness,
especially in the upper-middle quartile where the single adaptive rule failed?

## Limitation

Both points are generated from ground-truth tumor geometry.

This remains an oracle-simulated interaction study and is not deployment-valid
tumor localization.

## Output

```text
outputs/experiments/nv_segment_ct_tumor_dual_interior_point/
└── protocol.json
```
