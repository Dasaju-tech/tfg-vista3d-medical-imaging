# Block 34 — Tumor-Scope Confounding Audit

## Motivation

Block 33 showed that high ground-truth tumor burden is strongly enriched among
HD95 outliers.

This observation must be interpreted against the frozen evaluation scope.

The official KiTS23 `kidney_and_mass` hierarchical evaluation class uses:

```text
labels {1, 2, 3}
= kidney + tumor + cyst
```

The maintained automatic NV-Segment-CT renal protocol predicts:

```text
left kidney
right kidney
left kidney cyst
right kidney cyst
```

and does not provide a validated automatic tumor class.

Therefore high tumor burden can increase the kidney-and-mass geometric error
even when the predicted kidney envelope itself is anatomically reasonable.

## Objective

Block 34 quantifies this protocol-scope effect before any attempt to modify the
model.

No inference is repeated and no prediction is changed.

## Paired sensitivity comparison

For every frozen expanded100 case, the same NV-Segment prediction is evaluated
against two references:

### Registered/official-scope component

```text
KiTS23 labels {1, 2, 3}
```

This reproduces the existing kidney-and-mass HEC component.

### Tumor-excluded diagnostic

```text
KiTS23 labels {1, 3}
```

This keeps kidney and cyst but excludes tumor from the reference envelope.

The tumor-excluded comparison uses the same Dice, Surface Dice tolerance, and
diagnostic HD95 implementation, but it is explicitly a project-specific
sensitivity diagnostic and is **not** an official KiTS23 HEC.

## Questions

The audit measures:

- change in mean and median HD95;
- change in Dice and Surface Dice;
- how many original HD95 outliers fall below the original fixed threshold;
- correlation between tumor volume and HD95 before and after tumor exclusion;
- correlation between tumor volume and the HD95 reduction;
- scope sensitivity within each Block 33 failure mode.

## Interpretation

A strong reduction in HD95 after tumor exclusion would show that a meaningful
part of the tail reflects task-scope mismatch: the reference contains tumor
voxels for which this automatic model has no validated output.

Residual outliers after tumor exclusion are more defensible candidates for
true renal-boundary or false-positive failure analysis.

The registered kidney-and-mass result is retained unchanged regardless of this
diagnostic outcome.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_tumor_scope/
├── tumor_scope_audit.json
├── case_scope_comparison.csv
├── tumor_scope_audit.md
└── figures/
    ├── tumor_volume_vs_hd95_reduction.png
    └── original_vs_tumor_excluded_hd95.png
```
