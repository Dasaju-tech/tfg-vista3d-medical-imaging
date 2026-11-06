# Block 19 — Semantic Class + Single Oracle Point for Renal Cysts

## Motivation

Block 18 showed that cumulative point corrections are unstable.

Positive corrective clicks were especially harmful:

```text
positive corrections
n = 11
mean Dice delta = -0.1343
median Dice delta = -0.0967
improved = 3/11
regressed = 6/11
```

Negative corrections were substantially more stable:

```text
negative corrections
n = 23
mean Dice delta = +0.0290
median Dice delta = +0.0063
improved = 19/23
regressed = 4/23
```

Rather than adding more corrections, Block 19 tests whether semantic
conditioning can stabilize the strong but heterogeneous single-click response.

## Maintained bundle capability

NV-Segment-CT supports class prompts and point prompts simultaneously when the
class-prompt batch length is one.

The model therefore receives:

```text
WHAT  = side-specific kidney-cyst class
WHERE = one positive oracle point
```

instead of the Block 16 point-only protocol, which supplies only spatial
localization.

Official cyst classes:

```text
116 = left kidney cyst
117 = right kidney cyst
```

## Oracle status

The experiment remains explicitly non-deployment-valid.

Both pieces of conditioning use reference information:

- the positive point is the frozen GT-derived Block 16 point,
- the left/right semantic class is derived from GT anatomy.

The purpose is an ablation of model conditioning, not an automatic system.

## Laterality derivation

Laterality is determined in NIfTI world coordinates.

1. Build the KiTS23 renal envelope from labels 1, 2, and 3.
2. Identify the two dominant renal connected components.
3. Transform their centroids to world coordinates.
4. Transform the targeted largest-cyst centroid to world coordinates.
5. Use RAS x: smaller x = left, larger x = right.

This avoids assigning laterality from raw voxel indices and is tested under an
x-axis affine flip.

## Input protocol

For a left target:

```text
label_prompt = [116]
points = [[x, y, z]]
point_labels = [1]
```

For a right target:

```text
label_prompt = [117]
points = [[x, y, z]]
point_labels = [1]
```

Only one semantic object is requested, satisfying the maintained bundle
restriction that class+point inference requires label-prompt batch size one.

## Paired comparison

Every case uses the exact same target component and exact same positive point as
Block 16.

Three measurements are reported:

1. automatic side-specific cyst-class Dice,
2. Block 16 point-only Dice,
3. semantic-class + point Dice.

The primary Block 19 comparison is paired:

```text
semantic+point Dice - point-only Dice
```

with a deterministic percentile-bootstrap 95% confidence interval.

## Decision rule

### Semantic conditioning helps

If semantic+point materially improves the paired Dice distribution, the
interactive model benefits from joint WHAT+WHERE conditioning and the next
step should build a practical localization mechanism that can provide the
point without GT.

### No improvement or regression

If semantic conditioning adds little or harms performance, the single positive
point already extracts most available interactive information. The next major
step should then shift toward task-specific adaptation/fine-tuning instead of
more prompt engineering.

## Outputs

```text
outputs/experiments/nv_segment_ct_cyst_semantic_point/
├── summary.json
└── cases.csv

outputs/predictions/nv_segment_ct_semantic_point_cyst/
└── case_XXXXX/
    ├── raw/
    ├── prediction.nii.gz
    └── result.json
```
