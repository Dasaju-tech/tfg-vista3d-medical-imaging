# Block 51 — Cyst Detectability Limit & Size-Response Audit

## Motivation

Block 50 showed a strong size-dependent cyst failure pattern in the frozen
expanded100 cohort:

```text
cyst-positive N = 46
Dice mean / median = 0.1958 / 0.0051
zero-Dice cases = 21/46

dominant errors:
undersegmentation / FN = 34
oversegmentation / FP  = 12

smallest volume quartile:
Dice = 0
recall = 0
zero Dice = 12/12

largest volume quartile:
Dice mean = 0.5667
recall mean = 0.4686
zero Dice = 0/11
```

Reference cyst volume was associated with both Dice (`r=+0.659`) and recall
(`r=+0.674`).

## Objective

Quantify the strength and shape of the size-dependent detectability transition
without modifying predictions or running new inference.

## End points

Two retrospective binary endpoints are defined:

```text
any overlap:
    cyst Dice > 0

useful overlap:
    cyst Dice >= 0.50
```

Reference cyst volume is evaluated as a predictor of each endpoint.

## Analyses

### Volume AUROC

The rank-based AUROC measures how strongly larger reference cyst volume orders
cases toward successful detection.

A deterministic 5000-sample case bootstrap provides a 95% confidence interval.

### Descriptive threshold

Every midpoint between observed cyst volumes is evaluated as:

```text
predicted success = reference volume >= threshold
```

The threshold maximizing balanced accuracy is reported.

This threshold is selected on the same data and is therefore descriptive only.

### Empirical transition zone

The analysis reports:

```text
largest volume among failures
smallest volume among successes
```

If these overlap, there is no hard size boundary even when the overall
association is strong.

### Eight-bin size response

The 46 cyst-positive cases are sorted by reference volume and divided into eight
approximately equal bins.

For each bin:

- median volume;
- any-overlap fraction;
- Dice>=0.50 fraction;
- mean Dice;
- mean recall.

This gives a more informative size-response curve than a single fitted cutoff.

## Interpretation boundary

This is an in-sample retrospective characterization of the frozen expanded100
cohort.

The thresholds must **not** be described as:

- clinical cyst-size cutoffs;
- externally validated detection limits;
- prospective operating points.

A defensible conclusion is instead:

> The maintained automatic cyst output exhibits a strong size-dependent
> detectability transition, dominated by false-negative undersegmentation of
> small cysts.

## Decision gate

If reference volume has high AUROC for any-overlap detection and the lowest
size bins remain near zero while the highest bins improve progressively, cyst
failure should be treated as a model detectability limitation rather than a
global postprocessing problem.

Only if large cysts still show substantial false-positive-driven failure would
further prediction filtering be justified.
