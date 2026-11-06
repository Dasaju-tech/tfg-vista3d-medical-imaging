# Block 46A — Adaptive Relative-Depth Protocol Freeze

## Motivation

Block 45 showed that fixed absolute click depths do not generalize across tumor
sizes:

```text
15 mm non-empty safe region:
smallest quartile = 0/25
lower-middle      = 3/25
upper-middle      = 23/25
largest quartile  = 25/25
```

The 15 mm rule that worked well in Blocks 43–44 is therefore largely a
large-tumor rule.

A size-adaptive interaction should depend on each lesion's own interior scale,
not a single physical distance.

## Candidate rule

Block 46A freezes one relative-depth candidate before any new inference:

```text
adaptive depth = 0.70 × maximum tumor interior depth
```

The adaptive point is the tumor voxel whose physical Euclidean
distance-to-boundary is closest to that target depth.

The comparison reference is the deepest-interior voxel of the same component.

Both use:

```text
point_label = 3
label_prompt = None
```

## Why 70%

The goal is to place the click clearly inside the lesion while not requiring
the exact maximum-distance point.

Block 43 showed:

- boundary-near 5 mm clicks are fragile;
- increasingly interior 10 and 15 mm clicks progressively recover performance;
- centroid and deepest points are similar.

A 70% relative depth is therefore a pre-specified intermediate rule between
boundary-near and maximum-depth placement.

It is frozen before testing on the new stratified cases.

## Stratified sample

Block 45 expanded100 tumor geometry is sorted by largest tumor-component
volume and divided into four equal-sized quartiles.

Three deterministic volume-ranked representatives are selected per quartile:

```text
3 × 4 = 12 cases
```

No segmentation-performance metric is used for selection.

This intentionally includes small tumors for which 10–15 mm absolute safe
regions may not exist.

## Block 46B design

After this protocol is frozen, Block 46B can compare:

```text
adaptive 70%-depth point
vs
deepest-interior point
```

on the same 12 cases.

That requires:

```text
12 cases × 2 points = 24 interactive inferences
```

unless a selected case already has a cached matching deep result.

## Limitation

Both points remain derived from ground-truth tumor masks.

The experiment tests whether relative depth is a robust interaction principle,
not whether tumor localization can be automated.

## Output

```text
outputs/experiments/nv_segment_ct_tumor_adaptive_relative_depth/
└── protocol.json
```
