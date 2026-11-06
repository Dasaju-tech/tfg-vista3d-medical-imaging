# Block 50 — Expanded100 Cyst Failure Stratification

## Motivation

The tumor-interaction branch is now closed.

Block 49C verified that the negative manual CT-only interaction result was not
caused by a coordinate-transform error:

```text
image/GT grids equal = 12/12
max RAS round-trip error = 0 vox
max image->GT world error = 0 mm
manual tumor hits = 0/12
```

The remaining independent weakness in the frozen automatic renal evaluation is
cyst segmentation.

Expanded100 previously reported:

```text
cyst-positive cases = 46
cyst-positive Dice = 0.1958
```

and cyst Dice showed a positive association with reference cyst volume.

## Objective

Block 50 performs a no-GPU retrospective error stratification of the frozen
expanded100 automatic predictions.

No prediction is regenerated or modified.

## Reference and prediction labels

KiTS23 reference:

```text
3 = cyst
```

Maintained NV-Segment-CT renal vocabulary:

```text
116 = left kidney cyst
117 = right kidney cyst
```

The automatic cyst mask is therefore:

```text
prediction in {116, 117}
```

## Per-case measurements

For every cyst-positive case:

- reference cyst volume;
- predicted cyst volume;
- Dice;
- precision;
- recall;
- false-negative cyst volume;
- false-positive cyst volume;
- reference component count;
- predicted component count;
- largest reference component volume;
- largest predicted component volume;
- dominant FN / FP / balanced error type.

## Size stratification

Cyst-positive cases are sorted by reference cyst volume and divided
deterministically into four approximately equal groups:

```text
smallest
lower_middle
upper_middle
largest
```

The analysis reports Dice, precision, recall and zero-Dice frequency for each.

## Primary questions

1. Is poor cyst Dice primarily an under-detection/false-negative problem?
2. Does performance collapse for the smallest cysts?
3. Is the previously observed volume-Dice association driven by recall?
4. Are large false-positive cyst predictions an important secondary problem?

## Decision gate

If the smallest cyst quartiles show low recall and many zero-Dice cases while
larger cysts improve strongly, the next block should study a size/detectability
limit rather than global postprocessing.

If false-positive volume dominates across sizes, a conservative prediction
filtering analysis may be justified.

If both error modes remain substantial, cyst segmentation should be retained as
an unresolved model limitation rather than aggressively optimized on the same
frozen cohort.
