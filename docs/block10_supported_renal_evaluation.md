# Block 10 — Supported Renal Evaluation Against KiTS23

## Objective

Block 10 performs the first quantitative comparison between a real
NV-Segment-CT prediction and the KiTS23 reference segmentation.

The protocol is intentionally explicit about which parts are official KiTS23
evaluation components and which parts are project-specific diagnostics.

## Official kidney-and-mass HEC component

KiTS23 defines the hierarchical evaluation class:

```text
kidney_and_mass = labels {1, 2, 3}
```

with Surface Dice tolerance:

```text
1.0330772532390826 mm
```

The NV-Segment supported renal output is converted to a binary renal envelope
using:

```text
right kidney
left kidney
left kidney cyst
right kidney cyst
```

The project therefore reports the official Dice and Surface Dice definitions
for the `kidney_and_mass` HEC in isolation.

This is **not** reported as the full KiTS23 challenge score.

## Why the full KiTS23 score is unavailable

The other official HECs require:

```text
mass  = tumor + cyst
tumor = tumor
```

The maintained automatic NV-Segment-CT model does not provide a validated
automatic kidney-tumor class for this protocol.

Therefore:

- kidney-and-mass HEC: available
- mass HEC: unavailable
- tumor HEC: unavailable
- full KiTS23 score: unavailable

The ground-truth tumor volume is still reported to make the missing task
explicit.

## Diagnostic comparisons

Block 10 also reports non-ranking engineering metrics:

### Kidney core

Prediction:

```text
NV left kidney + NV right kidney
```

Reference:

```text
KiTS23 label 1
```

This comparison is diagnostic because lesion voxels may be represented
differently by organ and semantic-label conventions.

### Cyst

Prediction:

```text
NV labels 116 + 117
```

Reference:

```text
KiTS23 label 3
```

### Side-aware renal envelope

KiTS23 left/right renal units are derived using the tested Block 06
world-coordinate component method.

They are compared against the side-specific NV labels.

For every binary comparison the output includes:

- Dice
- reference volume
- prediction volume
- signed volume error
- absolute volume error
- relative volume error when defined
- false-positive physical volume
- false-negative physical volume

## Diagnostic HD95

HD95 is included for the kidney-and-mass HEC as a secondary diagnostic metric.
It is not part of the official KiTS23 ranking metric.

## Output

```text
outputs/evaluations/nv_segment_ct/case_XXXXX/
└── supported_renal_metrics.json
```

## Next block

After validating Block 10 on the smoke case, the next step is prediction-versus-
ground-truth visual QC and then controlled batch inference across the real
subset.
