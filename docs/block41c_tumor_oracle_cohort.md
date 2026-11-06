# Block 41C — Nine-Case Tumor Oracle Cohort

## Motivation

Block 41B showed useful but incomplete latent tumor capability in the worst
undercoverage case:

```text
case_00430

interactive tumor Dice = 0.3777
tumor coverage          = 0.4% -> 23.6%
HEC Dice                = 0.1481 -> 0.4498
HEC HD95                = 97.26 -> 55.00 mm
```

The smoke result is large enough to justify testing whether this behavior is
consistent across all nine frozen GT→prediction tumor-undercoverage cases.

## Frozen protocol

No prompt is changed after Block 41A.

Each case uses:

```text
one GT-derived deepest-interior point
point_label = 3
label_prompt = None
```

The experiment remains an oracle upper bound and is not deployment-valid.

## Reuse

`case_00430` reuses the Block 41B cached result unless `--force` is supplied.

The remaining eight cases require one interactive inference each.

## Aggregate metrics

The cohort reports:

- largest-component tumor Dice mean and median;
- full-tumor Dice mean and median;
- tumor coverage gain mean and median;
- HEC Dice delta mean and median;
- HEC Surface Dice delta mean and median;
- HEC HD95 reduction mean and median;
- cases with improved HD95;
- cases with HD95 reduction greater than 5 mm;
- cases with tumor Dice at least 0.50 and 0.80;
- cases with improved HEC Dice and Surface Dice.

## Decision boundary

If tumor overlap and HD95 improvement are reasonably consistent across cases,
the next research problem becomes deployment-valid tumor localization or
interaction design.

If the oracle benefit is highly inconsistent or weak across the cohort, the
interactive route should not be treated as a robust tumor solution and a
task-specific model/adaptation branch is better justified.

## Outputs

```text
outputs/experiments/nv_segment_ct_tumor_oracle_point/cohort/
├── summary.json
├── cases.csv
└── cohort_report.md
```
