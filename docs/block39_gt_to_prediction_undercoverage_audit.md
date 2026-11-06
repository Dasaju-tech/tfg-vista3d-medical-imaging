# Block 39 — GT→Prediction Undercoverage Audit

## Motivation

Block 35 separated the 22 frozen HD95 outliers into:

```text
prediction -> GT dominated: 12
GT -> prediction dominated: 9
balanced: 1
```

Blocks 36–38 investigated prediction-side false-positive geometry. The global
filters were ineffective and the retrospective prediction-geometry trigger did
not generalize under leave-one-out validation.

Block 39 therefore changes branch and investigates the nine cases in which the
reference surface is substantially farther from the prediction: genuine
undercoverage.

## Objective

Determine which KiTS23 reference structures account for the missing volume in
the nine GT→prediction-dominated outliers.

No inference is repeated and no model, prompt, metric, or prediction is
changed.

## Missing-volume definition

For the registered kidney-and-mass scope:

```text
reference = KiTS23 labels {1, 2, 3}
prediction = maintained automatic renal-label union
false negative = reference AND NOT prediction
```

The false-negative mask is decomposed by original KiTS23 reference label:

```text
1 = kidney
2 = tumor
3 = cyst
```

## Per-case measurements

For each of the nine cases the audit records:

- total reference volume;
- false-negative volume and fraction;
- kidney, tumor, and cyst reference volumes;
- kidney, tumor, and cyst false-negative volumes;
- label-specific miss fractions;
- number of false-negative connected components;
- largest false-negative component volume;
- dominant missing label;
- dominant-label share of the missing volume;
- the Block 35 GT→prediction surface p95.

## Cohort summaries

The audit reports:

- dominant missing-label counts;
- aggregate fraction of false-negative volume due to kidney/tumor/cyst;
- weighted miss fraction for each label;
- positive-case miss-fraction distributions;
- cases ranked by total missing volume and directional surface distance.

## Interpretation

If undercoverage is dominated by kidney tissue, the next experiment should
focus on renal coverage/prompting.

If tumor dominates, the next experiment should investigate tumor-aware
inference or a protocol-specific multi-target strategy.

If cyst dominates, the result converges with the already identified weak cyst
performance and supports a cyst-focused experimental branch.

The result is descriptive and retrospective; the reference labels are not
deployment-time inputs.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_undercoverage/
├── undercoverage_audit.json
├── undercoverage_cases.csv
├── undercoverage_audit.md
└── figures/
    ├── label_missing_burden.png
    └── case_undercoverage_composition.png
```
