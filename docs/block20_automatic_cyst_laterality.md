# Block 20 — Automatic Semantic Class from Renal Segmentation

## Motivation

Block 19 tested:

```text
side-specific cyst class + one positive point
```

using both a GT-derived point and GT-derived cyst laterality.

The paired semantic benefit was heterogeneous:

```text
point-only mean Dice       0.4417
semantic+point mean Dice   0.4692

point-only median Dice     0.4157
semantic+point median Dice 0.5827

paired mean delta          +0.0274
bootstrap 95% CI           [-0.0756, +0.1284]

improved                   11/17
regressed                   6/17
```

The bootstrap interval crosses zero, so semantic conditioning does not show a
robust cohort-wide mean improvement. However, it increases the number of cases
above clinically useful overlap thresholds and raises the marginal median.

Before abandoning this direction, Block 20 removes an unnecessary source of
oracle information: the left/right semantic cyst class.

## Deployable semantic-class rule

The automatic renal segmentation already contains semantic kidney labels:

```text
5  = right kidney
14 = left kidney
```

Given a user click, the side-specific cyst class can therefore be assigned
without reference segmentation:

```text
nearest automatic left kidney  -> class 116
nearest automatic right kidney -> class 117
```

Distance is measured in physical millimetres using NIfTI voxel spacing.

This class-selection rule is deployment-valid **given a user-provided click**.

## Current experiment boundary

The click itself is still generated from GT in Blocks 16–19. Therefore the
overall segmentation experiment remains an oracle simulation of human
interaction.

Block 20 only asks whether the semantic side label can be removed from the
oracle assumptions.

## Audit

For all 17 cyst-positive pilot cases, Block 20 records:

- distance from the frozen click to automatic left kidney,
- distance from the frozen click to automatic right kidney,
- inferred side,
- inferred class 116/117,
- agreement with the GT-derived Block 19 class,
- physical distance margin between the two side candidates.

The GT-derived class is used only as an evaluation reference.

## Interpretation

### Perfect or near-perfect agreement

If automatic renal masks recover the same side-specific cyst class, the Block
19 semantic input can be generated automatically in a future user-guided
workflow.

When the inferred class is identical to the Block 19 class, the model input is
identical and the existing semantic+point predictions require no re-inference.

### Disagreement

If side assignment fails in non-trivial numbers of cases, the mismatches should
be re-run with the automatically inferred class before any practical protocol
claim is made.

## Outputs

```text
outputs/analyses/nv_segment_ct_automatic_cyst_laterality/
├── summary.json
└── cases.csv
```

No GPU inference is performed in this block.
