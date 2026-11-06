# Block 42 — Tumor Click Robustness

## Motivation

Block 41C demonstrated consistent latent interactive tumor capability across
all nine GT→prediction undercoverage cases:

```text
tumor Dice mean   = 0.7557
tumor Dice median = 0.7802
coverage gain     = +52.1 percentage points
HEC HD95 reduction mean = +29.31 mm
HD95 improved     = 9/9
HEC Dice improved = 9/9
```

The remaining question is whether this performance depends on the unusually
precise deepest-interior oracle point.

## Objective

Measure sensitivity to point placement while keeping the tumor component fixed.

Three deterministic point variants are evaluated:

```text
deep
centroid
shallow_5mm
```

### Deep

The frozen Block 41A deepest-interior tumor point.

This result is reused from Block 41C.

### Centroid

The tumor voxel nearest to the voxel-space centroid of the frozen largest tumor
component.

This represents a plausible central user click without requiring a maximum
distance-to-boundary location.

### Shallow 5 mm

A tumor voxel whose physical distance to the tumor boundary is closest to
5 mm.

This intentionally tests a less ideal but still positive click close to the
lesion boundary.

## Important limitation

Centroid and shallow points are still generated from ground truth.

Therefore this block simulates user-click variability but is not itself
deployment-valid.

Its purpose is to determine whether exact oracle localization is necessary.

## Runtime

The deep result is cached.

Centroid and shallow variants require two new interactive inferences per case:

```text
9 cases × 2 new points = 18 new inferences
```

Each inference uses the unchanged VISTA3D special-positive interaction:

```text
point_labels = [3]
label_prompt = None
```

## Metrics

For every point variant:

- tumor Dice;
- tumor coverage gain;
- HEC Dice delta;
- HEC Surface Dice delta;
- HEC HD95 reduction.

The analysis also reports how many centroid/shallow results remain within
±0.05 tumor Dice of the deep oracle result.

## Decision boundary

If centroid clicks preserve most of the Block 41C performance, the project can
credibly move toward a clinician-click interactive workflow.

If shallow clicks are also robust, precise lesion-centre localization is not
required.

If performance collapses outside the deepest point, the interactive capability
exists but depends strongly on localization quality and should not yet be
presented as a practical interaction protocol.

## Outputs

```text
outputs/experiments/nv_segment_ct_tumor_click_robustness/
├── summary.json
└── cases.csv

outputs/predictions/nv_segment_ct_interactive_tumor_click_robustness/
├── centroid/
└── shallow_5mm/
```
