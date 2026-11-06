# Block 18 — Interactive Cyst Stability and Oracle Early Stopping

## Motivation

Block 17 showed that cumulative oracle clicks are not monotonically beneficial.

Observed fixed-round cohort performance:

```text
1 click: mean 0.4417, median 0.4157
2 clicks: mean 0.3919, median 0.2113
3 clicks: mean 0.3941, median 0.1847
```

At the same time, the number of high-performing cases increased:

```text
Dice >= 0.80
1 click: 4/17
2 clicks: 6/17
3 clicks: 6/17
```

Therefore interaction quality is heterogeneous: additional feedback helps some
targets but can substantially damage others.

## Validity of cumulative point prompting

The maintained NV-Segment-CT interface supports multiple 3D points for one
foreground object, with:

```text
1 = positive / foreground
0 = negative / background
-1 = ignore
```

All points are expressed in original-image coordinates. Re-running the model
with the cumulative click history is therefore a valid point-prompt protocol.

## Questions

Block 18 asks:

1. Which click count produced the best observed Dice for each case?
2. How much performance is lost by forcing all cases to three clicks?
3. Are positive and negative corrections equally stable?
4. Is the second or third correction more likely to regress?
5. What upper bound would an oracle early-stopping rule obtain?

## Oracle early stopping

For each case:

```text
best observed Dice = max(Dice_click1, Dice_click2, Dice_click3)
```

Ties are assigned to the earliest round, representing the least annotation
effort.

This is explicitly **not deployment-valid** because ground truth is used to
select the best round. It measures the value available in the set of generated
interactive predictions and quantifies the cost of inappropriate extra clicks.

## Transition analysis

Every correction transition is recorded as:

```text
case
click 1 -> 2 or click 2 -> 3
positive / negative correction
source error type
Dice before
Dice after
Dice delta
```

Positive and negative correction distributions are summarized separately.

## Regret

The final-round regret is:

```text
best observed Dice - three-click Dice
```

The audit reports the number of cases with regret at least 0.05 and 0.10.

This directly quantifies how often forcing continued interaction destroys a
better earlier prediction.

## Outputs

```text
outputs/analyses/nv_segment_ct_cyst_interaction_stability/
├── stability_summary.json
├── transitions.csv
└── figures/
    ├── correction_delta_by_label.png
    └── fixed_round_vs_oracle_best.png
```

## Next experimental decision

If a particular correction label is associated with regression, the next block
should test a safer prompt strategy rather than simply adding more clicks.

One especially relevant candidate is semantic+spatial conditioning:

```text
kidney-cyst class prompt + one positive point
```

because NV-Segment-CT supports a single class prompt together with point
prompts for one foreground object. This would test whether class semantics can
stabilize the strong but inconsistent point-only response.
