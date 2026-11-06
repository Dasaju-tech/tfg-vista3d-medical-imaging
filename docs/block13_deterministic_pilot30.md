# Block 13 — Deterministic 30-Case KiTS23 Pilot

## Objective

The three-case smoke-validation phase is complete. Block 13 scales the frozen
NV-Segment-CT protocol to a larger real cohort without changing the model,
prompt, evaluation definitions, or runtime during the experiment.

The objective is still engineering and protocol validation. Thirty cases are
not treated as a statistically powered estimate of population performance.

## Frozen case selection

The pilot uses selection revision 2.

Revision 1 made an invalid assumption: the fact that KiTS23 contains 489 cases
does **not** imply that valid case identifiers are the contiguous numeric range
`00000..00488`. The official metadata contains gaps in the numeric case IDs.

This was detected during acquisition, before any 30-case model inference had
been performed. The correction therefore does not use model outcomes.

The valid revision-1 cases already acquired are retained. Four nonexistent
numeric candidates are removed:

```text
00324 00345 00396 00398
```

and replaced with four confirmed official KiTS23 IDs:

```text
00474 00479 00481 00482
```

Frozen revision-2 cohort:

```text
00000 00001 00002 00017 00047 00063 00109 00116 00142 00143
00171 00190 00193 00200 00237 00255 00275 00293 00297 00299
00416 00418 00463 00471 00473 00474 00479 00480 00481 00482
```

The correction is an availability correction only and was frozen before
pilot30 inference.

The exact list and selection rule are written to:

```text
outputs/cohorts/nv_segment_ct_pilot30/selection.json
```

## Why retain the smoke cases?

Keeping cases 0–2:

- preserves direct continuity with the validated smoke phase,
- prevents re-running an unrelated selection merely to obtain a larger N,
- gives immediate regression checks against known results.

The retained and replacement IDs were fixed independently of segmentation
metrics and model predictions.

## Acquisition

The existing atomic KiTS23 acquisition layer is reused.

Existing non-empty files are reused. Missing images and reference
segmentations are downloaded and SHA-256 provenance is recorded.

The complete selected cohort is re-audited before inference.

Outputs:

```text
outputs/manifests/kits23_pilot30_provenance.json
outputs/audits/kits23_pilot30_audit.json
```

## Resumable execution

For every selected case, the runner performs:

```text
audited CT
  -> NV-Segment-CT inference
  -> prediction geometry/label audit
  -> supported KiTS23 renal evaluation
```

Existing prediction and evaluation artifacts are reused unless force options
are explicitly supplied. This allows an interrupted multi-case experiment to
resume without recomputing completed cases.

## Failure behavior

Default behavior is fail-fast.

`--continue-on-error` records a case-level failure, continues with the
remaining cohort, produces summaries for completed cases, and returns a
non-zero exit status at the end.

Failures are never silently removed from the experiment.

## Visual QC scaling rule

Rendering three figures for every case is not required for the first pilot.

After evaluation, the runner ranks cases by kidney-and-mass Surface Dice and
renders full Block 11 visual QC for the five worst cases by default.

This focuses manual review on the highest-risk predictions while retaining a
deterministic selection rule.

## Cyst evaluation

The Block 12 reporting rule is preserved:

- all-case cyst Dice,
- cyst-positive-only Dice,
- cyst-positive GT/predicted volume.

Empty-perfect cases therefore cannot inflate the reported cyst-positive
segmentation result.

## Cohort outputs

```text
outputs/cohorts/nv_segment_ct_pilot30/
├── selection.json
├── cases.csv
├── summary.json
└── run_manifest.json
```

## Protocol freeze

During the 30-case run, do not change:

- NV-Segment upstream commit,
- model checkpoint,
- PyTorch/CUDA runtime,
- MONAI runtime,
- renal label prompt,
- KiTS23 comparison code,
- empty-mask rules,
- Surface Dice tolerance,
- CT QC window.

Any necessary change requires a new experiment version rather than silently
mixing outputs.

## Interpretation boundary

This cohort is large enough to expose pipeline and model behavior that a
three-case smoke subset cannot reveal, including more cyst-positive cases and
more anatomical variability.

It is not presented as a definitive KiTS23 benchmark or population estimate.
The full official KiTS23 challenge score remains unavailable until the tumor
and mass HECs are addressed with a justified prediction protocol.
