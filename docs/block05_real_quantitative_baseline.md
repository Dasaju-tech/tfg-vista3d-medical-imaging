# Block 05 — Real KiTS23 Quantitative Baseline

## Objective

This block converts validated KiTS23 reference segmentations into structured
quantitative case descriptors before any VISTA inference is introduced.

The ground-truth pipeline is intentionally model-independent. The same
measurement layer can later be applied to model predictions, allowing direct
comparison between reference anatomy and predicted anatomy.

## Per-case measurements

For every case the pipeline records:

- NIfTI shape
- physical voxel spacing
- image orientation
- global CT intensity statistics
- CT statistics inside the annotated kidney/mass region
- kidney volume
- tumor volume
- cyst volume
- kidney-and-mass volume
- combined mass volume
- connected-component counts
- physical bounding-box dimensions
- voxel-space centroids
- world-coordinate centroids
- tumor burden fraction
- cyst burden fraction
- total mass burden fraction

## Burden definitions

The project currently defines:

`tumor burden = tumor volume / kidney-and-mass volume`

`cyst burden = cyst volume / kidney-and-mass volume`

`mass burden = (tumor + cyst) volume / kidney-and-mass volume`

These are transparent engineering descriptors and must not be presented as
clinical diagnostic scores.

## Laterality

This block stores world-coordinate centroids but deliberately does **not**
assign "left" or "right" kidney labels. Laterality requires a dedicated,
tested anatomical rule that accounts for image orientation and multiple
connected components. That will be implemented separately rather than inferred
from voxel indices.

## Outputs

The smoke analysis creates:

```text
outputs/baselines/kits23_smoke_cases.json
outputs/baselines/kits23_smoke_cohort.json
outputs/baselines/kits23_smoke_table.csv
```

The JSON files are the structured source of truth. The CSV file is intended
for quick inspection, plotting, and inclusion in later notebooks.

## Reproducibility

All measurements are derived directly from:

1. the locally audited KiTS23 CT image,
2. the corresponding KiTS23 reference segmentation,
3. the NIfTI affine,
4. the NIfTI voxel spacing.

No model output or manual transcription is involved in this block.
