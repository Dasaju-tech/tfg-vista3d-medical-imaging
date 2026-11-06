# Block 30B — Expanded100 Acquisition and Frozen-Protocol Execution

## Objective

Block 30A froze a deterministic 100-case KiTS23 cohort before any new model
inference. Block 30B executes that registered cohort without changing the
segmentation model, renal prompt, upstream runtime, evaluation definitions, or
case-level artifact namespace used by pilot30.

The design is nested:

```text
30 retained pilot30 cases
+ 70 newly selected official cases
= 100 expanded100 cases
```

## Reuse rule

The runner intentionally reuses `pilot_case_paths` for case-level inference and
evaluation artifacts. This is not an experiment-name mistake: predictions and
supported renal metrics are properties of the frozen case/model/protocol
combination, not of the cohort summary in which a case later appears.

Therefore a completed pilot30 case is expected to print:

```text
inference: reused
evaluation: reused
```

unless a force flag is supplied.

Cohort-level outputs remain separate under:

```text
outputs/cohorts/nv_segment_ct_expanded100/
```

## Acquisition-first execution

Before consuming GPU time, acquire and audit all 100 frozen cases:

```powershell
python .\scripts\run_nv_segment_ct_expanded100.py --stage acquire
```

Existing pilot files are reused by the acquisition layer. The step writes:

```text
outputs/manifests/kits23_expanded100_provenance.json
outputs/audits/kits23_expanded100_audit.json
```

Inference should not start unless the audit reports `100/100 valid`.

## Full execution

Run:

```powershell
python .\scripts\run_nv_segment_ct_expanded100.py --stage run --continue-on-error
```

The runner is resumable. Existing predictions and evaluations are reused unless
`--force-inference` or `--force-evaluation` is explicitly supplied.

`--continue-on-error` is recommended for the long cohort run. Failures are
recorded and the command returns non-zero at completion; they are never silently
removed from the requested cohort.

## QC scaling

The default manual QC budget is increased from five to ten cases. After metric
calculation, the ten worst kidney-and-mass Surface Dice cases are selected
deterministically for prediction-vs-reference visualization.

## Outputs

```text
outputs/cohorts/nv_segment_ct_expanded100/
├── cases.csv
├── summary.json
├── run_manifest.json
└── expanded_run_metadata.json
```

The additional metadata file records how many inference/evaluation artifacts
were reused and how many were newly computed. This makes the nested design
auditable.

## Protocol freeze

Do not change during Block 30B:

- the frozen Block 30A selection JSON;
- NV-Segment upstream commit;
- model checkpoint;
- PyTorch/CUDA/MONAI runtime;
- renal semantic prompt;
- supported KiTS23 metric definitions;
- empty-mask conventions;
- Surface Dice tolerance;
- QC CT window.

A required model or evaluation change starts a new experiment version rather
than being mixed into expanded100.

## Interpretation boundary

Block 30B estimates the same supported renal protocol at a larger sample size.
It is not a full KiTS23 challenge submission and does not make the unsupported
tumor/mass HECs available. Statistical comparison of pilot30 versus expanded100
is intentionally deferred until all requested cases and failures have been
accounted for.
