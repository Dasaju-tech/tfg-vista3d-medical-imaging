# Block 48 — Tumor Interaction Study Synthesis & Decision Gate

## Purpose

Blocks 41–47 progressively tested whether VISTA3D's latent tumor interaction
capability could be converted into a robust one- or two-point interaction rule.

The sequence produced both strong positive capability evidence and important
negative generalization results.

Block 48 performs no new inference. It consolidates the complete branch into a
single reproducible decision artifact.

## Included evidence

### Block 41C — oracle capability

A single deeply interior GT-derived tumor point produced:

```text
mean tumor Dice = 0.7557
median tumor Dice = 0.7802
mean tumor coverage gain = +52.1 percentage points
mean HEC HD95 reduction = +29.31 mm
HD95 improved = 9/9
```

This establishes latent interactive tumor capability.

### Blocks 42–43 — location sensitivity

```text
deep      Dice mean = 0.7557
centroid  Dice mean = 0.7676
5 mm      Dice mean = 0.4461
10 mm     Dice mean = 0.6595
15 mm     Dice mean = 0.7272
```

Exact maximum-depth placement is unnecessary, but boundary-near interaction is
substantially less reliable.

### Blocks 44–45 — geometric generalization

The nine severe undercoverage cases had large 15 mm safe regions, but this did
not generalize to expanded100:

```text
15 mm non-empty safe region:
expanded100 tumor-positive = 51/100
smallest tumor quartile     = 0/25
```

Therefore a fixed 15 mm physical-depth rule is rejected.

### Block 46B — relative-depth single point

The pre-specified 70% maximum-depth rule yielded:

```text
paired mean tumor-Dice delta = -0.0236
95% bootstrap CI = [-0.1014, +0.0444]
upper-middle quartile delta = -0.1627
```

The rule is not robust across size strata and is rejected.

### Block 47B — dual interior point

Adding a second spatially separated interior point yielded:

```text
paired mean tumor-Dice delta = +0.0153
95% bootstrap CI = [-0.0325, +0.0594]
upper-middle quartile delta = -0.0285
```

HD95 improved slightly overall, especially in large tumors, but tumor-Dice
benefit was inconsistent and the paired confidence interval crossed zero.

The dual-point rule is therefore not accepted as a robust improvement.

## Final branch decision

Oracle prompt engineering is closed after Block 47.

The strongest defensible conclusion is:

> VISTA3D contains useful latent interactive renal-tumor segmentation
> capability when a positive point is placed well inside the lesion. Exact
> deepest-point placement is not required, but performance is sensitive to
> boundary proximity. Fixed physical-depth, single relative-depth, and
> dual-point oracle rules did not provide robust size-generalizable
> improvements.

The preferred interaction concept remains:

```text
one well-interior positive tumor point
```

but this study does not provide deployment-valid localization because all tumor
points were derived from ground-truth geometry.

## Next validation priority

Further oracle prompt tuning should not be performed on these same cases.

The next meaningful validation should replace GT-derived localization with a
deployment-plausible source, for example:

- clinician clicks;
- independently simulated human clicks from visible lesion centers;
- an independent tumor-localization model.

This separates the demonstrated segmentation capability from the unresolved
localization problem.

## Outputs

```text
outputs/analyses/nv_segment_ct_tumor_interaction_study/
├── interaction_study_summary.json
├── decision_matrix.csv
└── interaction_study.md
```
