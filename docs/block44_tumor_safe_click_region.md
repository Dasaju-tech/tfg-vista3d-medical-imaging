# Block 44 — Tumor Safe-Click Region Audit

## Motivation

Block 43 demonstrated a clear dependence on how far inside the tumor the
special-positive click is placed:

```text
5 mm   -> tumor Dice mean 0.4461
10 mm  -> tumor Dice mean 0.6595
15 mm  -> tumor Dice mean 0.7272
centroid -> 0.7676
deep     -> 0.7557
```

At 15 mm the mean Dice loss relative to the deepest oracle point was only
0.0284, while HD95 improved in all nine cases.

This suggests that a well-interior click may be sufficient. However, before
describing such an interaction as practical, the project must quantify how much
of each tumor actually lies at least 5, 10, or 15 mm from the boundary.

## Objective

For the same frozen largest tumor component used in Blocks 41–43, calculate the
physical Euclidean distance of every tumor voxel to the surrounding background.

For each case, define the safe-click region:

```text
safe_region(d) = tumor voxels with interior distance >= d
```

for:

```text
d = 5, 10, 15 mm
```

## Measurements

Per case and depth:

- safe-region voxel count;
- safe-region physical volume;
- fraction of the frozen tumor component inside the safe region;
- whether the region is non-empty.

Per depth across the nine cases:

- cases with a non-empty region;
- mean, median, minimum, and maximum safe-volume fraction;
- mean and median safe volume;
- cases where at least 5%, 10%, or 25% of tumor volume lies in the safe region.

The audit also imports the already generated Block 43 performance at the same
depths so that geometric availability and segmentation performance can be read
together.

## No new inference

This block uses only:

- frozen KiTS23 ground-truth tumor geometry;
- the Block 41A frozen component identity;
- the Block 43 summary.

No GPU inference is performed.

## Interpretation

A depth is interaction-practical in this cohort only if:

1. the safe region exists in most or all cases;
2. the safe region is not vanishingly small;
3. Block 43 performance at that depth remains useful.

For example, if 15 mm achieves strong segmentation but only a tiny fraction of
tumor volume qualifies, it is a fragile interaction rule.

If 10–15 mm occupies a substantial fraction of these tumors, the clinician-click
workflow becomes much more credible.

## Limitation

The nine cases were selected specifically because they were severe
GT→prediction undercoverage outliers and contain relatively large tumors.

Therefore safe-region fractions must not be generalized to all renal tumors or
treated as universal clinical distance recommendations.

## Outputs

```text
outputs/experiments/nv_segment_ct_tumor_safe_click_region/
├── summary.json
├── cases.csv
└── safe_click_region.md
```
