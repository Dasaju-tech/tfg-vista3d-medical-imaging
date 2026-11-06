# Block 52 — Cyst Branch Synthesis & Decision Gate

## Purpose

Blocks 50–51 established that the weak cyst result in expanded100 is not a
uniform segmentation failure.

The branch now has enough evidence to close without further post-hoc tuning.

## Core findings

```text
cyst-positive N = 46
Dice mean / median = 0.1958 / 0.0051
mean recall = 0.1599
zero-Dice = 21/46

FN-dominant = 34/46
FP-dominant = 12/46
```

The smallest reference-volume quartile had:

```text
Dice mean = 0
recall mean = 0
zero-Dice = 12/12
```

while the largest quartile had:

```text
Dice mean = 0.5667
recall mean = 0.4686
zero-Dice = 0/11
```

## Detectability evidence

Reference cyst volume strongly ordered performance:

```text
volume vs Dice Pearson = +0.659
volume vs recall Pearson = +0.674

volume AUROC -> any overlap = 0.950
volume AUROC -> Dice >= 0.50 = 0.914
```

The volume-response bins show a progressive transition from no detection at the
smallest volumes to much stronger overlap in the largest bins.

## No hard cutoff

The outcome is not perfectly separable by volume.

For any overlap:

```text
largest failure = 8.31 mL
smallest success = 2.15 mL
```

Therefore the in-sample threshold around 5.93 mL is a descriptive transition
point only, not a validated detection threshold.

The same limitation applies to the descriptive threshold for Dice >= 0.50.

## Decision

The cyst branch is closed.

The retained interpretation is:

> The maintained automatic cyst output exhibits a strong size-dependent
> detectability limitation dominated by false-negative undersegmentation.
> Very small cysts are frequently missed completely, while larger cysts are
> substantially more recoverable. Reference volume strongly orders
> detectability, but no hard or clinically validated size cutoff is supported.

Global postprocessing is not the preferred remedy because most error is missing
target volume rather than removable false-positive volume.

## Reporting constraint

Blocks 50–52 analyze the same frozen expanded100 predictions retrospectively.

No descriptive threshold from this branch should be presented as a prospective
clinical operating point.
