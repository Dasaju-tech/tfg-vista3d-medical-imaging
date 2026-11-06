# Block 09 — First Real NV-Segment-CT Inference

## Objective

This block executes the first real foundation-model inference on an audited
KiTS23 CT volume.

The experiment is deliberately restricted to renal labels that are explicitly
supported by the maintained NV-Segment-CT label vocabulary.

## Supported renal prompt

The inference prompt is:

```text
5   right kidney
14  left kidney
116 left kidney cyst
117 right kidney cyst
```

These indexes are validated at runtime against the checked-out upstream
`configs/label_dict.json`. If the upstream contract changes, inference stops
instead of silently interpreting a label incorrectly.

## Kidney tumor limitation

KiTS23 includes semantic tumor label `2`.

The maintained NV-Segment-CT documentation lists legacy class index `129`
(`kidney mass`) among classes excluded from automatic inference because of
insufficient data. Therefore Block 09 does not fabricate a direct KiTS23 tumor
mapping.

This means that the first experiment evaluates model execution and supported
renal anatomy only. A later experimental block must explicitly choose between:

1. interactive tumor prompting,
2. continual learning / fine-tuning,
3. another justified tumor segmentation head/model.

That choice must be reported as a protocol decision.

## Why not "segment everything"?

The default automatic mode evaluates more than one hundred classes. The TFG
currently needs only renal structures, and the local workstation has one GPU.

Using four explicit label prompts:

- reduces unnecessary class-head work,
- reduces irrelevant output labels,
- makes the prediction contract testable,
- keeps the first experiment focused on the research target.

## Reproducibility

The runner records:

- KiTS23 case ID,
- exact upstream repository commit,
- Conda runtime name,
- input path,
- four requested label indexes,
- whether the model checkpoint existed before execution,
- wall-clock runtime,
- raw upstream output path,
- stable project prediction path,
- SHA-256 of the prediction,
- output shape,
- affine parity,
- unique labels present,
- human-readable labels present.

The first run may include Hugging Face checkpoint download time. This is
recorded through `checkpoint_present_before` so cold-start and warm-start
execution are not mixed silently.

## Geometry

The maintained bundle applies preprocessing and then `Invertd` before
`SaveImaged`. The project nevertheless independently verifies that the final
prediction:

- is 3D,
- has the same shape as the source CT,
- has an affine matching the source CT,
- contains finite, integer-valued labels,
- contains only the requested renal labels or background,
- does not contain unprocessed label `255`.

## Outputs

For `case_00000`:

```text
outputs/predictions/nv_segment_ct/case_00000/
├── raw/
│   └── ... upstream SaveImaged output ...
├── prediction.nii.gz
└── inference.json
```

Raw model output and the stable copied prediction are generated artifacts and
remain outside the Git repository.

## Next step

Once this inference passes, Block 10 will compare supported model anatomy with
KiTS23 ground truth and create prediction-versus-reference QC overlays.

Official KiTS23 HEC scores will not be claimed until a justified renal-tumor
segmentation protocol exists.
