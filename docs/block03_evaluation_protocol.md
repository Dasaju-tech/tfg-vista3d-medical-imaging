# Block 03 — KiTS23 Evaluation Protocol

## Purpose

This block establishes a reproducible segmentation-evaluation layer before
integrating VISTA inference. The implementation is designed to remain usable
for synthetic fixtures, KiTS23 reference masks, and future model predictions.

## KiTS23 semantic labels

The project assumes the official KiTS23 semantic encoding:

- `0`: background
- `1`: kidney
- `2`: tumor
- `3`: cyst

## Hierarchical Evaluation Classes

KiTS23 evaluates three merged regions rather than scoring all semantic labels
independently:

1. `kidney_and_mass = (1, 2, 3)`
2. `mass = (2, 3)`
3. `tumor = (2,)`

The implementation preserves this order.

## Official metrics

The two official KiTS23 metrics are:

- Sørensen–Dice coefficient
- Surface Dice

The Surface Dice tolerances are fixed to the official values:

- kidney and mass: `1.0330772532390826 mm`
- mass: `1.1328796488598762 mm`
- tumor: `1.1498198361434828 mm`

Empty-mask behavior follows the official case-level evaluator:

- both prediction and reference empty: Dice = 1, Surface Dice = 1
- only one mask empty: Dice = 0, Surface Dice = 0

## Diagnostic metric

HD95 is computed as an additional boundary-error diagnostic. It is **not**
an official KiTS23 ranking metric and must not be reported as such.

For the project implementation:

- both masks empty: HD95 = 0 mm
- exactly one mask empty: HD95 = infinity

## Reproducibility rule

All quantitative results used in the TFG should ultimately be reproducible
from saved integer segmentation maps and the source-image voxel spacing.
Model logits or probabilities must be converted to discrete KiTS23 labels
before evaluation.

## Planned parity check

When local KiTS23 data are available, a later block will compare this module
against the official `kits23_compute_metrics` implementation on identical
prediction/reference pairs. This will be treated as an evaluation-parity test
rather than assuming equivalence from implementation alone.
