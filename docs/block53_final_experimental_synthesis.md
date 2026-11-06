# Block 53 — Final Experimental Synthesis

## Purpose

Block 53 closes the experimental phase by consolidating the project's principal
validated results and rejected hypotheses into one reproducible reporting
artifact.

No new inference is performed.

## Included evidence

The synthesis combines:

- frozen expanded100 automatic baseline;
- HD95 tail and residual-geometry analyses;
- tumor undercoverage analysis;
- Blocks 41–48 interactive tumor branch;
- Blocks 49A–49C simulated-user CT-only localization;
- Blocks 50–52 cyst detectability branch.

## Principal automatic result

```text
expanded100 N = 100
kidney-and-mass Dice = 0.8863 ± 0.1309
surface Dice = 0.7471
HD95 median = 4.53 mm
HD95 mean = 12.10 mm
mean inference wall time = 15.24 s
```

The difference between median and mean HD95 is explained by a concentrated
failure tail rather than uniformly poor boundary geometry.

## HD95 tail

```text
outliers = 22/100
top 5 contribution to mean-H95 burden = 39.2%
top 10 contribution = 54.9%

dominant residual direction:
prediction -> GT = 12
GT -> prediction = 9
balanced = 1
```

Prediction-only postprocessing and selection rules were tested and rejected.

## Tumor interaction

The automatic renal vocabulary does not contain a validated autonomous tumor
output, but GT-derived interactive points revealed latent VISTA3D tumor
capability:

```text
oracle tumor Dice mean = 0.7557
oracle HEC HD95 reduction mean = +29.31 mm
```

Fixed-depth, relative-depth and dual-point oracle refinements did not produce a
robust size-generalizable improvement.

The branch is therefore a capability result, not a deployment-valid
localization result.

## Manual CT-only localization

The 12-case manual experiment separated point selection from ground truth.

Post-hoc geometry QC verified:

```text
image/GT grids equal = 12/12
RAS round-trip error = 0 vox
world-space mapping error = 0 mm
```

Nevertheless:

```text
manual any-tumor hits = 0/12
manual tumor Dice mean = 0.0117
deep-oracle tumor Dice mean = 0.5219
```

This is a negative simulated-user localization result. It is not clinician
validation.

## Cyst branch

```text
cyst-positive N = 46
Dice mean / median = 0.1958 / 0.0051
zero-Dice = 21/46
FN-dominant = 34/46

volume AUROC:
any overlap = 0.950
Dice >= 0.50 = 0.914
```

The dominant limitation is size-dependent false-negative undersegmentation,
especially for small cysts.

No validated clinical size cutoff is claimed.

## Final experimental position

The final experimental narrative is:

1. global automatic renal overlap is strong;
2. boundary-tail failures are concentrated and mechanistically heterogeneous;
3. simplistic automatic postprocessing does not solve the tail;
4. tumor undercoverage reveals substantial latent interactive capability if
   lesion location is supplied;
5. oracle prompt optimization does not generalize robustly;
6. unguided non-expert CT-only localization is not an adequate replacement for
   correct lesion localization;
7. cyst segmentation remains a distinct size-dependent detectability weakness.

This evidence should form the basis of the TFG Results and Discussion chapters.

## Reporting constraints

Do not:

- report an official KiTS23 challenge score;
- describe oracle points as deployment-valid tumor localization;
- describe project-author clicks as clinician/radiologist validation;
- describe cyst-volume transition points as clinical cutoffs;
- claim that prediction-only postprocessing solved the HD95 tail.
