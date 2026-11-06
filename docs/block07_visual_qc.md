# Block 07 — Deterministic Visual Quality Control

## Objective

This block creates reproducible visual quality-control figures from the real
KiTS23 CT images and reference segmentations.

The purpose is not to create publication figures yet. The immediate goal is to
verify that:

1. the semantic labels align spatially with the CT,
2. the lesion-driven slice-selection logic is sensible,
3. connected-component behavior observed numerically is visible,
4. future VISTA predictions can be inspected using exactly the same renderer.

## Canonical orientation

Both CT and segmentation are converted with NiBabel to the closest canonical
RAS+ orientation before rendering.

This means that QC images are generated under one consistent coordinate
convention even when source NIfTI voxel axes differ.

The canonicalization is applied to image and segmentation together and their
post-transform shape and affine are checked again before visualization.

## Slice selection

For each of the three orthogonal planes:

- axial
- coronal
- sagittal

the renderer first looks for tumor or cyst voxels.

If lesion voxels exist, it chooses the slice with the greatest lesion area.
If no lesion exists, it chooses the slice with the greatest kidney-and-mass
area. If the segmentation is empty, the volume-center slice is used.

The selection source and slice index are stored in JSON.

## Semantic overlays

Every selected slice is rendered as:

- CT background,
- semantic KiTS23 segmentation overlay,
- physical-distance axes in millimetres.

The default engineering display window is:

`[-160, 240] HU`

This value is configurable and is part of the QC result metadata. It is a
visualization setting only and is not used for quantitative measurements or
model preprocessing.

## Component overlays

A second image is generated for the same slice showing connected-component
IDs for the complete kidney-and-mass mask.

This is particularly important for cases where the Block 06 anatomy analysis
finds more than two connected components. It allows small satellite components
to be inspected before changing the component/laterality algorithm.

## Output structure

For each case:

```text
outputs/qc/case_XXXXX/
├── axial_semantic_overlay.png
├── coronal_semantic_overlay.png
├── sagittal_semantic_overlay.png
├── axial_component_overlay.png
├── coronal_component_overlay.png
└── sagittal_component_overlay.png
```

Global metadata:

```text
outputs/qc/kits23_smoke_qc_index.json
```

## Reproducibility

The QC index stores:

- case ID,
- canonical orientation,
- canonical shape,
- canonical voxel spacing,
- CT window,
- connected-component count,
- selected plane/slice indices,
- selection rationale,
- generated file paths.

The JSON index, rather than manual screenshot selection, is therefore the
source of truth for the smoke QC experiment.
