# Block 30C — Partial-Laterality Recovery

## Trigger

Expanded100 completed model inference for all 100 registered cases. Evaluation
succeeded for 99 cases and failed only for `case_00563`.

The failure was not a model failure. Ground-truth morphology analysis found one
connected kidney-and-mass component, so a defensible left/right assignment was
unavailable:

```text
kidney_and_mass_components = 1
qc_flags = ["fewer_than_two_kidney_and_mass_components"]
```

## Protocol correction

The primary supported quantities do not require ground-truth left/right
assignment:

- kidney-and-mass Dice;
- kidney-and-mass Surface Dice;
- diagnostic HD95;
- kidney-core Dice;
- global cyst Dice and physical cyst volumes.

Therefore unresolved laterality must not invalidate those quantities.

Side-aware renal-envelope and side-specific cyst diagnostics are nullable.
No anatomical side is guessed from the model prediction or from an arbitrary
coordinate split.

## Cohort semantics

Expanded100 keeps all 100 frozen cases.

Expected recovered cohort:

```text
primary/global evaluation N = 100
laterality-resolved N       = 99
laterality-unresolved N     = 1
```

Left/right cohort summaries are computed only over cases with resolved
ground-truth laterality.

Schema-1.0 evaluation artifacts remain readable and are treated as
laterality-resolved when both historical side objects are present.

## Reproducibility

No prediction is regenerated. Re-running the resumable expanded100 runner
should reuse all 100 inference artifacts and the 99 existing evaluation
artifacts, computing only the missing evaluation for `case_00563`.

The expanded100 run manifest also records the correct experiment identifier
`nv_segment_ct_expanded100`.
