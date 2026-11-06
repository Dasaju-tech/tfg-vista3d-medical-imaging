# Block 31 — Expanded100 Statistical Validation

## Objective

Block 30 established a frozen 100-case KiTS23 evaluation cohort and completed
primary/global evaluation for all 100 cases.

Block 31 asks a different question:

> How stable are the conclusions originally observed in pilot30 when the
> cohort is expanded to 100 frozen cases?

No model inference is performed in this block.

## Nested design

The design is deliberately nested:

```text
pilot30     = 30 frozen cases
additional = 70 frozen cases
expanded100 = pilot30 + additional70
```

This has an important statistical consequence: pilot30 and expanded100 are not
independent samples.

Therefore Block 31 does not report an independent-samples p-value between
pilot30 and expanded100. The change in estimates is treated as a descriptive
stability diagnostic.

## Metrics

The analysis reports:

- kidney-and-mass Dice;
- kidney-and-mass Surface Dice;
- diagnostic HD95;
- kidney-core Dice;
- cyst-positive-only Dice.

For each supported metric it records:

- N;
- mean;
- sample standard deviation;
- median;
- deterministic percentile bootstrap 95% CI of the mean;
- expanded100 minus pilot30 mean;
- additional70 minus pilot30 mean;
- change in bootstrap CI width;
- whether the pilot30 mean lies inside the expanded100 bootstrap interval.

## Bootstrap

The registered analysis uses:

```text
seed = 20260901
samples = 5000
```

The bootstrap is deterministic and performed independently for each displayed
cohort estimate.

## Additional70 sensitivity cohort

The 70 cases added after pilot30 are also summarized separately.

This is not presented as a formal external validation set. Its role is to show
whether the extra cases reproduce or materially shift the original pilot-level
behavior.

## Laterality

Primary/global metrics use all 100 cases.

The single case whose ground-truth left/right assignment is unresolved remains
in the global analysis. Laterality availability is reported explicitly and no
side is guessed.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_validation/
├── statistical_validation.json
├── statistical_validation.md
├── metric_table.csv
└── figures/
    ├── kidney_and_mass_dice_stability.png
    ├── surface_dice_stability.png
    ├── hd95_mm_stability.png
    └── cyst_positive_dice_stability.png
```

## Interpretation boundary

Block 31 validates estimate stability under a larger frozen cohort. It does
not change the frozen model, prompt, runtime, evaluation protocol, or case
selection after observing outcomes.

The full official KiTS23 challenge score remains outside the supported
protocol because the project does not claim a justified full tumor/mass HEC
prediction protocol.
