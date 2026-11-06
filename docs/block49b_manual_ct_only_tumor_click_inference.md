# Block 49B — Manual CT-Only Tumor Click Inference

## Motivation

Block 49A froze one manual tumor-location point for each of the 12
tumor-size-stratified cases.

The point-selection viewer displayed only the CT in canonical RAS orientation.
No KiTS23 ground-truth segmentation was displayed or loaded by the viewer.

Block 49B therefore evaluates the first interaction branch in which tumor
localization is not generated from the ground-truth tumor mask.

## Frozen inputs

The 12 manual clicks are immutable inputs from:

```text
outputs/experiments/nv_segment_ct_manual_tumor_clicks/clicks/
```

Each schema-1.1 artifact records:

```text
source = manual_ct_only_ras_display
gt_visible_during_selection = false
display_point_ras_xyz = viewer coordinate
point_xyz = original NIfTI voxel coordinate
```

No click is adjusted after ground truth is inspected.

## Inference

For every case:

```text
points = [manual point]
point_labels = [3]
label_prompt = None
```

The single deepest-interior oracle result from Block 46B is reused as the
capability reference.

Only 12 new manual-click VISTA3D inferences are required.

## Post-hoc localization audit

After the manual point is frozen, ground truth is used to classify the point:

```text
TARGET HIT
    point lies inside the frozen largest tumor component

OTHER TUMOR HIT
    point lies in tumor label 2 but outside the frozen target component

MISS
    point lies outside all tumor voxels
```

For misses, the physical Euclidean distance to the nearest tumor voxel is
reported.

This does not contaminate point selection because the GT is consulted only
after the click artifact already exists.

## Primary questions

1. How often can a simulated user localize the tumor from CT alone?
2. How much tumor-segmentation Dice remains relative to the deep oracle?
3. Does performance loss concentrate in localization misses?
4. Are small tumors harder to localize than large tumors?

## Metrics

### Localization

- target-component hit count;
- any-tumor hit count;
- miss count;
- miss distance to tumor;
- association between distance-to-target and manual tumor Dice.

### Segmentation

Manual and deep:

- tumor Dice;
- tumor coverage gain;
- HEC HD95 reduction.

Paired manual-minus-deep:

- mean difference;
- cases within ±0.05 and ±0.10 Dice;
- deterministic 5000-sample paired bootstrap 95% confidence interval.

Metrics are reported overall and by the four frozen tumor-size quartiles.

## Interpretation

If manual target hits retain a large fraction of deep-oracle segmentation
performance, the dominant practical limitation is tumor localization rather
than segmentation capacity.

If manual points often hit the tumor but segmentation still collapses, the
oracle interaction results were more dependent on precise point placement than
the earlier robustness analyses suggested.

If misses concentrate in the smallest quartile, the experiment supports a
size-dependent localization bottleneck.

## Terminology

The project author selected these clicks.

Therefore results must be described as:

```text
manual CT-only localization
simulated-user interaction
```

and not as clinician or radiologist validation.

## Outputs

```text
outputs/experiments/nv_segment_ct_manual_tumor_clicks/inference/
├── summary.json
├── cases.csv
└── manual_click_report.md
```
