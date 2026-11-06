# Block 17 — Iterative Oracle Click-Response Curve

## Motivation

Block 16 demonstrated that one GT-derived positive click substantially improves
largest-cyst segmentation:

```text
automatic target Dice mean   0.2338
single-click Dice mean       0.4417
automatic target Dice median 0.0064
single-click Dice median     0.4157
improved cases               15/17
automatic zero-overlap rescued 6/7
```

However, a single click remains insufficient for many cases:

```text
Dice >= 0.50: 8/17
Dice >= 0.80: 4/17
```

Block 17 measures whether additional deterministic user-like corrections
continue to improve performance.

## Oracle status

Every correction uses KiTS23 ground truth.

This is an upper-bound interaction experiment and is **not deployment-valid**.

The experiment asks how much segmentation capability can be extracted from the
foundation model when increasingly precise spatial feedback is supplied.

## Click 1

Click 1 is exactly the frozen Block 16 prompt:

- largest GT cyst component,
- deepest physical interior voxel,
- positive label `1`.

The existing Block 16 prediction is reused rather than recomputed.

## Clicks 2 and 3

After each prediction:

```text
FN = target AND NOT predicted foreground
FP = predicted foreground AND NOT target
```

The error type with the larger voxel count is selected.

If FN is larger:

- select the largest FN component,
- choose its deepest physical interior voxel,
- add a positive click (`1`).

If FP is larger:

- select the largest FP component,
- choose its deepest physical interior voxel,
- add a negative click (`0`).

All previous clicks remain in the prompt. The MONAI bundle therefore receives
the cumulative click history.

The maintained NV-Segment-CT interface explicitly supports multiple 3D points
and positive/negative point labels in one inference.

## Why three clicks?

Three cumulative clicks provide a compact interaction-response curve while
keeping compute and oracle supervision bounded:

```text
1 click  -> localization cue
2 clicks -> first explicit error correction
3 clicks -> second explicit error correction
```

If performance is still poor after three oracle corrections, adding many more
clicks is unlikely to be the most informative next experiment; model adaptation
would become a stronger candidate.

## Metrics

For every click count:

- target cyst Dice mean and median,
- minimum and maximum,
- cases with Dice >= 0.50,
- cases with Dice >= 0.80.

The cohort summary also reports:

- final minus single-click Dice,
- improved cases,
- regressed cases,
- automatic zero-overlap cases rescued by the final round.

## Outputs

```text
outputs/experiments/nv_segment_ct_cyst_oracle_click_curve/
├── summary.json
├── trajectories.csv
└── dice_vs_click_count.png

outputs/predictions/nv_segment_ct_interactive_cyst_curve/
└── case_XXXXX/
    ├── click_02/
    ├── click_03/
    └── trajectory.json
```

## Interpretation

Three possible patterns are especially informative.

### Strong monotonic gain

If median Dice rises substantially from one to three clicks, the model has
useful interactive refinement capacity and the automatic localization branch is
the principal bottleneck.

### Early plateau

If click 2 helps but click 3 adds little, a small amount of spatial guidance is
sufficient and further annotation effort has diminishing returns.

### Persistent low Dice

If several oracle corrections still leave poor overlap, the limitation is more
fundamental than localization and should motivate fine-tuning or task-specific
adaptation rather than increasingly elaborate prompting.
