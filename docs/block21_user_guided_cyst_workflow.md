# Block 21 — GT-Free User-Guided Renal Cyst Workflow

## Motivation

Block 20 showed that side-specific cyst semantics can be recovered from the
automatic renal segmentation for all 17 cyst-positive pilot cases:

```text
agreement with GT-derived semantic class = 17/17
both automatic kidneys present           = 17/17

selected-kidney distance
median = 0.88 mm
max    = 16.48 mm

laterality margin
median = 112.05 mm
minimum = 80.12 mm
```

The semantic side label therefore no longer needs reference segmentation.

Block 21 packages the validated components into a user-facing research
workflow that does not read ground truth.

## Workflow

The user provides one click in original NIfTI voxel coordinates:

```text
[x, y, z]
```

The system already has the automatic renal segmentation from NV-Segment-CT.

It performs:

```text
user click
    |
    +-- physical distance to automatic left kidney (14)
    |
    +-- physical distance to automatic right kidney (5)
                |
                v
        nearest automatic kidney
                |
          +-----+-----+
          |           |
        left        right
         116          117
          |           |
          +-----+-----+
                |
                v
      semantic class + user point
                |
                v
          interactive VISTA
                |
                v
        single cyst prediction
```

## Ground-truth boundary

The workflow itself never opens or evaluates against KiTS23
`segmentation.nii.gz`.

The output manifest records:

```text
ground_truth_used = false
clinical_validation = false
workflow_type = interactive_research_prototype
```

The workflow is therefore GT-free, but it must not be described as clinically
validated.

## Semantic routing

Automatic labels:

```text
5  = right kidney
14 = left kidney
```

Interactive cyst classes:

```text
116 = left kidney cyst
117 = right kidney cyst
```

Physical Euclidean distance uses NIfTI voxel spacing.

An exactly equidistant click is rejected rather than assigned arbitrarily.

## Usage

Example:

```powershell
python .\scripts\run_nv_segment_ct_user_cyst_click.py `
  --case 2 `
  --point 68 372 151
```

The point shown above is the historical Block 16 research point for
`case_00002`. In an actual user-guided workflow the point is selected by the
human operator, not obtained from reference segmentation.

## Outputs

```text
outputs/predictions/nv_segment_ct_user_guided_cyst/
└── case_XXXXX/
    ├── raw/
    ├── prediction.nii.gz
    └── workflow.json
```

The manifest contains:

- click coordinates,
- automatically inferred side and class,
- physical distances to both renal masks,
- laterality margin,
- output labels,
- predicted volume,
- runtime,
- explicit statement that no GT was used.

## Research interpretation

Blocks 16–20 used oracle information to characterize model capability.

Block 21 is different: it is the first packaged interactive workflow whose
runtime logic depends only on:

1. CT image,
2. previously generated automatic renal segmentation,
3. one user-provided click.

This provides a clean bridge from controlled experiments to the eventual
application/reporting layer of the TFG.
