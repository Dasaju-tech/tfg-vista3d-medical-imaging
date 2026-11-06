# Block 49A — Manual Clinician-Style Click Workflow

## Motivation

Block 48 closed the oracle prompt-engineering branch.

The strongest retained finding is that VISTA3D has useful latent tumor
segmentation capability when a positive point is placed well inside the lesion,
but all interaction points used in Blocks 41–47 were generated from
ground-truth tumor masks.

The next methodological step is to separate segmentation capability from
ground-truth-derived localization.

## Objective

Provide a reproducible CT-only manual point-selection workflow for the same
12-case tumor-size-stratified cohort frozen in Block 46A.

The user selects the point from the CT image itself.

Ground-truth segmentation is deliberately not loaded by the viewer.

## Frozen cohort

The same 12 cases from Block 46A are retained:

```text
smallest quartile   3
lower-middle        3
upper-middle        3
largest quartile    3
```

No new cases are selected based on interactive performance.

## Viewer

For each case the tool shows three orthogonal CT views:

```text
axial
coronal
sagittal
```

Default window:

```text
-200 to +300 HU
```

Controls:

```text
mouse wheel  navigate slices in the active view
left click   place/update the candidate tumor point
Enter        accept the current point
R            reset the point
Q            abort collection
```

A click in any view updates the shared 3D voxel coordinate and cross-hair in all
three views.

## Ground-truth separation

The viewer API accepts only:

```text
CT image path
case ID
quartile
window settings
```

It does not accept or load the KiTS23 segmentation.

Each saved click artifact explicitly records:

```text
gt_visible_during_selection = false
source = manual_ct_only
```

Ground truth may be loaded later for retrospective evaluation, but only after
the click has been frozen.

## Why this is better than another oracle rule

This directly addresses the unresolved localization problem identified in Block
48.

It avoids tuning more percentages, depths, or point combinations using the
known tumor mask.

## Block 49B

After manual clicks are collected, Block 49B can run one VISTA3D interaction per
case:

```text
manual point
point_label = 3
label_prompt = None
```

and then compare the frozen manual click against:

```text
deep oracle
centroid/deep capability reference
```

using GT only for post-hoc evaluation.

## Limitation

The workflow is deployment-plausible in the sense that the click is selected
from the CT rather than the GT mask, but it is not a formal clinician usability
study unless the clicks are provided by qualified clinical readers.

If the project author supplies the clicks, results should be described as
manual CT-only or simulated-user localization rather than clinician validation.

## Outputs

```text
outputs/experiments/nv_segment_ct_manual_tumor_clicks/clicks/
├── case_00141.json
├── ...
└── case_00481.json
```
