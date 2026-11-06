# Block 16 — Single-Click Oracle Cyst Localization Experiment

## Motivation

The pilot30 failure decomposition shows that automatic cyst segmentation is
primarily a localization problem:

- 17 cyst-positive cases,
- 6/17 cases with zero voxel overlap,
- 116 reference cyst components,
- only 43.1% localized within 30 mm,
- median nearest predicted-component centroid distance of 33.62 mm.

Automatic cyst volume nevertheless correlates strongly with total reference
cyst burden. The next controlled question is therefore:

> Can the interactive branch segment a cyst when its location is supplied?

## Oracle status

This is explicitly an **oracle localization experiment**.

The prompt is generated from the KiTS23 ground truth and therefore cannot be
used as a deployment-valid automatic method. The purpose is to isolate model
capability from automatic localization capability.

No result from this block may be described as automatic performance.

## Target definition

For each cyst-positive pilot30 case:

1. label KiTS23 cyst components with 6-connected 3D connectivity,
2. select the largest cyst component,
3. compute its physical Euclidean distance transform,
4. choose the deepest interior voxel,
5. freeze that voxel as one positive interactive click.

The deepest point is used instead of the arithmetic centroid because the
centroid of a non-convex component can lie outside the object.

Only one component is targeted because the maintained NV-Segment-CT point
interface supports a single foreground object per interactive inference.

## Prompt protocol

The MONAI Bundle input is:

```text
image
points = [[x, y, z]]
point_labels = [1]
```

No class prompt is supplied. This tests point-only zero-shot localization of a
single foreground object.

The maintained NV-Segment-CT documentation defines point coordinates in the
original image space and point label `1` as foreground.

## Comparison

The interactive result is compared against the exact largest GT cyst component
that generated the click.

For a target-matched baseline, the pre-existing automatic cyst union
(labels 116/117) is compared against that same GT component.

Reported quantities include:

- automatic target Dice,
- interactive target Dice,
- Dice improvement,
- interactive target volume error,
- cases improved,
- cases reaching Dice >= 0.50,
- cases reaching Dice >= 0.80,
- automatic zero-overlap cases rescued by one click.

## Interactive output handling

The bundle documents value `255` as an unprocessed/NaN region after
postprocessing. The project excludes `255` from the binary foreground mask and
records whether it occurred.

Any other non-zero label is treated as the single interactive foreground
object.

## Staged execution

First freeze all prompts without running the model:

```text
--stage prepare
```

Then smoke-test a single known cyst-positive case:

```text
--stage run --cases 2
```

Only after inspecting that output should the full 17-case registered experiment
be executed.

## Outputs

```text
outputs/experiments/nv_segment_ct_cyst_oracle_prompt/
├── prompts.json
├── summary.json
└── cases.csv

outputs/predictions/nv_segment_ct_interactive_cyst/
└── case_XXXXX/
    ├── raw/
    ├── prediction.nii.gz
    └── result.json
```

## Interpretation

A large Dice improvement would show that the model contains useful cyst
segmentation capability that the automatic branch fails to localize.

Little or no improvement even after an oracle click would instead suggest a
more fundamental representation/contour problem and would motivate fine-tuning
rather than more elaborate localization logic.
