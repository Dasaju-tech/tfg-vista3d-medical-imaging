# Block 36 — Prediction-Only Component-Filter Sensitivity

## Motivation

Block 35 decomposed the 22 expanded100 HD95 outliers into directional error:

```text
prediction -> GT dominated: 12
GT -> prediction dominated: 9
balanced: 1
```

This makes a single universal correction inappropriate.

Block 36 addresses only the prediction-to-GT branch, where distant or excessive
predicted geometry is a plausible contributor.

## Objective

Test whether deterministic connected-component pruning can reduce the
prediction-to-GT HD95 tail without materially harming the rest of the frozen
expanded100 cohort.

No baseline prediction is overwritten and no inference is repeated.

## Candidate A — global top-2

The predicted renal envelope is formed from all maintained renal labels.

Connected components are computed using 6-connectivity and only the two
largest components are retained.

This is a simple anatomy-motivated sensitivity rule: a bilateral renal
prediction usually has at most two dominant renal envelopes.

## Candidate B — side-largest

The maintained NV-Segment labels already distinguish left and right renal
structures.

For each side independently, the union of kidney and kidney-cyst labels is
formed and only the largest connected component is retained.

This uses prediction semantics only. Ground truth does not participate in the
filtering decision.

## Safety evaluation

Both candidates are evaluated on all 100 frozen cases.

The analysis separately reports:

- the 12 Block 35 prediction-to-GT target outliers;
- the other 10 HD95 outliers;
- the 78 non-outlier cases.

For each candidate it measures:

- all-case mean HD95;
- remaining outliers under the original fixed threshold;
- target-case HD95 reduction;
- target outliers resolved;
- non-outlier cases worsened by more than 1 mm HD95;
- non-outlier cases losing more than 0.01 Dice.

## Interpretation boundary

This is a sensitivity analysis, not an adopted post-processing pipeline.

A candidate should only advance to a registered experimental variant if it
meaningfully improves the 12 target cases while showing acceptably small harm
on the 78 non-outliers and the coverage-dominated outliers.

The frozen expanded100 baseline remains unchanged.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_component_filter_sensitivity/
├── component_filter_sensitivity.json
├── case_comparison.csv
├── component_filter_sensitivity.md
└── figures/
    └── target_hd95_sensitivity.png
```
