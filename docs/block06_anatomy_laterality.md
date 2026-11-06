# Block 06 — Anatomical Components and Conservative Laterality

## Objective

This block converts the KiTS23 semantic reference segmentation into explicit
renal units and lesion components.

It is intentionally independent of VISTA. Later, the same component-analysis
layer can be applied to model predictions to quantify not only segmentation
overlap but also whether predicted lesions are assigned to the correct renal
side.

## Coordinate convention

NiBabel exposes image world coordinates in RAS+ convention. In RAS+ world
coordinates, the x axis increases from the subject's left toward the subject's
right.

The project assigns laterality by relative x ordering of the two dominant
kidney-and-mass connected components:

- smaller world x centroid -> left
- larger world x centroid -> right

The sign of x is deliberately not used. This avoids assuming that world x=0
coincides with the anatomical mid-sagittal plane.

## Conservative failure behavior

Laterality is not forced when it is not geometrically justified.

- fewer than two kidney-and-mass components: `undetermined`
- two dominant components with indistinguishable world x: `undetermined`
- more than two kidney-and-mass components: the two largest components define
  the paired renal units and a QC flag is raised
- lesions belonging to a non-dominant component: `unassigned`

## Connected-component definition

The analysis uses 6-connected components in 3D, matching the connectivity used
by the existing project morphology module.

The component inventory is an engineering analysis of the consensus semantic
segmentation. It is not claimed to reproduce the individual annotator
instance files included by KiTS23.

## Lesion inventory

Tumor (`label 2`) and cyst (`label 3`) masks are split independently into
connected components.

For every lesion component the pipeline records:

- lesion type
- voxel count
- physical volume
- voxel centroid
- RAS+ world centroid
- parent kidney-and-mass component
- conservative laterality

## QC flags

Possible case-level flags include:

- `fewer_than_two_kidney_and_mass_components`
- `more_than_two_kidney_and_mass_components`
- `ambiguous_renal_laterality`
- `unassigned_tumor_component`
- `unassigned_cyst_component`

These flags must remain available in future report-generation stages so that
uncertain anatomical assignments are not presented as certain facts.

## Output

The real-data smoke script writes:

```text
outputs/anatomy/kits23_smoke_anatomy.json
```
