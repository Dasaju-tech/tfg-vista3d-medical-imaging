# Block 46B — Adaptive Relative-Depth Inference

## Objective

Evaluate the frozen Block 46A candidate:

```text
adaptive point depth = 70% × maximum tumor interior depth
```

against the deepest-interior oracle point on the same 12 tumor-size-stratified
expanded100 cases.

## Frozen design

No case selection or point location is changed after Block 46A.

The sample contains:

```text
smallest quartile  3
lower-middle       3
upper-middle       3
largest quartile   3
total             12
```

For every case two one-point VISTA3D interactions are compared:

```text
adaptive 70%-depth point
deepest-interior point
```

Both use:

```text
point_label = 3
label_prompt = None
```

## Why this comparison matters

Block 45 showed that fixed physical-depth rules fail geometrically in small
tumors.

The relevant question is therefore whether a relative-depth point preserves
most of the deep-oracle capability regardless of tumor scale.

## Metrics

For adaptive and deep variants:

- tumor Dice;
- tumor coverage gain;
- HEC Dice delta;
- HEC Surface Dice delta;
- HEC HD95 reduction.

Paired adaptive-minus-deep differences are summarized overall and separately
for each tumor-size quartile.

The primary robustness criteria are:

- mean paired tumor-Dice delta;
- cases within ±0.05 and ±0.10 Dice of deep;
- paired HD95-reduction delta;
- performance in the smallest tumor quartile.

A deterministic 5000-sample paired bootstrap reports 95% confidence intervals
for the overall mean tumor-Dice and HD95-reduction differences.

## Interpretation

A useful adaptive rule should preserve deep-oracle performance not only
overall, but especially in the smallest and lower-middle tumor quartiles.

If 70% relative depth performs similarly to deep across size strata, it is a
better interaction principle than any fixed 10–15 mm rule.

If performance remains strongly size-dependent, a single relative-depth rule
is insufficient and the next interaction design should consider multiple
positive points or morphology-dependent placement.

## Limitation

Both adaptive and deep points are derived from ground-truth tumor masks.

This is an oracle-simulated interaction study, not autonomous localization.

## Outputs

```text
outputs/experiments/nv_segment_ct_tumor_adaptive_relative_depth/inference/
├── summary.json
└── cases.csv
```
