# Block 49C — Frozen Manual-Click Geometry & Post-hoc QC

## Motivation

Block 49B produced an extreme result:

```text
manual target hits = 0/12
manual any-tumor hits = 0/12
manual tumor Dice mean = 0.0117
deep-oracle tumor Dice mean = 0.5219
```

Before interpreting this as a true simulated-user localization failure, the
coordinate contract must be independently audited.

## Frozen-data rule

Block 49C does not modify:

- the 12 manual clicks;
- any VISTA3D predictions;
- any oracle reference.

No new inference is run.

Ground truth is used only after the clicks have already been frozen.

## Geometry audit

For every case the analysis checks:

1. imaging and segmentation shapes;
2. imaging and segmentation affines;
3. the saved canonical-RAS display point;
4. RAS-display -> image-voxel round trip;
5. image voxel -> world coordinate;
6. world coordinate -> segmentation voxel;
7. world-space round-trip error;
8. tumor label and physical distance at the mapped segmentation voxel.

This explicitly avoids assuming that the image voxel grid and segmentation
voxel grid are identical.

## Post-hoc figures

For each case a QC image is generated with:

- click-centered axial/coronal/sagittal views;
- tumor-centered axial/coronal/sagittal views;
- the frozen click;
- the nearest GT tumor point;
- GT tumor contour.

These figures are diagnostic artifacts only. They are never used to revise a
click.

## Decision rule

If:

```text
max RAS round-trip error ~= 0 vox
max image->GT world error ~= 0 mm
```

and the post-hoc overlays are anatomically aligned, then the 0/12 manual
localization result is accepted as a genuine result of this simulated-user
experiment.

If a systematic coordinate displacement is visible or world/grid mapping is
non-zero, Block 49B must be corrected before its scientific result is retained.

## Interpretation boundary

Even if geometry is valid, a 0/12 author-selected localization result does not
show that a radiologist would fail. It shows that unguided CT-only interaction
by the project author is not an adequate substitute for expert lesion
localization.

The positive oracle branch and the negative simulated-user branch should then
be interpreted together:

```text
correct lesion localization -> substantial latent tumor capability
unguided non-expert localization -> unreliable / potentially harmful
```
