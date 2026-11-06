# Block 41A — Tumor Oracle Special-Point Protocol Freeze

## Evidence entering the block

Block 39 found that the nine GT→prediction-dominated HD95 outliers are
predominantly tumor-undercoverage cases: tumor is the dominant missing label in
8/9 cases and the weighted tumor miss fraction is 89.2%.

Block 40 rejected simple prediction-envelope dilation: 2, 5 and 10 mm expansion
resolved 0/9 targets and increasingly harmed non-target cases.

## Runtime contract

The frozen automatic NV-Segment renal vocabulary used by the project is checked
at runtime for:

```text
5   right kidney
14  left kidney
116 left kidney cyst
117 right kidney cyst
```

No kidney-tumor or kidney-mass automatic label is fabricated. If the checked-out
runtime unexpectedly exposes such a class, this block stops and requires a new
protocol review.

The maintained VISTA3D interactive interface documents special point label `3`
as a positive point for an overlapping/special class such as tumor. The planned
capability experiment therefore freezes:

```text
points       = [[x, y, z]]
point_labels = [3]
label_prompt = None
```

## Oracle localization

For every tumor-positive case among the nine GT→prediction cases, the protocol:

1. isolates KiTS23 tumor label 2;
2. selects the largest 6-connected tumor component;
3. computes a physical Euclidean distance transform;
4. chooses the deepest interior voxel;
5. freezes that point in the protocol manifest.

The point is GT-derived and therefore explicitly non-deployment-valid. Its only
purpose is to test whether the interactive model contains useful tumor
segmentation capability when location is supplied.

## Block boundary

Block 41A performs no GPU inference. It freezes the exact prompts and validates
the upstream class contract.

Block 41B should first run a single smoke case (preferably `case_00430`) before
executing the complete prepared tumor-positive subset.

## Output

```text
outputs/experiments/nv_segment_ct_tumor_oracle_point/protocol.json
```
