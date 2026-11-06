# Block 22 — GT-Free Structured Findings Contract

## Objective

Block 21 produced the first packaged interactive workflow that requires only:

1. the CT image,
2. automatic renal segmentation,
3. one user-provided click.

Block 22 converts these outputs into a strict machine-readable findings object
before any LLM is introduced.

The design principle is:

```text
segmentation
    -> deterministic measurements
    -> structured facts
    -> deterministic text
    -> later LLM verbalisation
```

The language model must never become the source of measurements.

## Inputs

The reporting layer uses only:

```text
imaging.nii.gz
automatic NV-Segment prediction
user-guided cyst prediction
user-guided workflow.json
```

It does not open the KiTS23 reference segmentation.

Every evidence file is recorded with a SHA-256 digest.

## Automatic measurements

The automatic renal prediction is converted to physical volumes using NIfTI
voxel spacing:

```text
label 14  left kidney core
label 5   right kidney core
label 116 automatic left cyst class
label 117 automatic right cyst class
```

These are explicitly model-derived quantities, not ground-truth anatomy.

## Interactive finding

The user-guided workflow contributes:

- user click coordinates,
- automatically assigned laterality,
- selected semantic cyst class,
- distance to the selected kidney,
- laterality margin,
- interactive predicted mask volume.

## Safety and interpretation boundary

The structured contract explicitly sets:

```text
ground_truth_used = false
clinical_validation = false
```

and carries limitations stating that the measurements are segmentation-derived
research outputs.

The deterministic report does not generate:

- diagnosis,
- malignancy assessment,
- cancer staging,
- treatment recommendation.

## Why deterministic text first?

A deterministic textual report provides the reference verbalisation for later
LLM experiments.

The future LLM layer can then be audited against:

```text
findings.json
```

for:

- numerical consistency,
- laterality consistency,
- unsupported claims,
- omitted limitations,
- hallucinated diagnoses.

This makes the language-generation experiment measurable rather than
subjective.

## Outputs

For each case:

```text
outputs/reports/structured/case_XXXXX/
├── findings.json
└── report.txt
```

A formal JSON Schema is stored at:

```text
schemas/structured_findings_v1.schema.json
```

## Next step

Once Block 22 is validated, the next block can introduce a constrained language
model as a verbalisation layer while treating `findings.json` as the sole
authoritative source of case facts.
