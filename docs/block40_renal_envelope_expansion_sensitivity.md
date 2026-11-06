# Block 40 — Renal-Envelope Expansion Sensitivity

## Motivation

Block 39 showed that the nine GT→prediction-dominated HD95 outliers are
overwhelmingly tumor-driven:

```text
dominant missing label:
kidney  1/9
tumor   8/9
cyst    0/9

weighted tumor miss fraction: 89.2%
```

Before introducing a new tumor-aware inference pathway, Block 40 tests whether
a small prediction-only physical expansion of the current renal envelope can
recover a useful fraction of this missed tumor extent without unacceptable
damage elsewhere.

## No new inference

The frozen expanded100 predictions are reused.

No model, prompt, checkpoint, or registered baseline prediction is modified.

## Candidate sensitivity radii

The predicted renal envelope is dilated in physical space using the NIfTI voxel
spacing:

```text
2 mm
5 mm
10 mm
```

Distance is computed with the Euclidean distance transform and physical voxel
sampling.

The rule uses prediction geometry only.

## Evaluation

Each expanded mask is evaluated against the frozen KiTS23 kidney-and-mass
reference `{1,2,3}` using the existing Dice, Surface Dice, and diagnostic HD95
implementation.

The analysis separately reports the nine Block 35 GT→prediction targets and the
remaining 91 cases.

For each radius it measures:

- all-case mean Dice, Surface Dice, and HD95;
- remaining HD95 outliers under the frozen original threshold;
- mean target HD95 reduction;
- number of target outliers resolved;
- increase in tumor coverage among target cases;
- number of non-target cases harmed by more than 1 mm HD95;
- number of non-target cases losing more than 0.01 Dice;
- volume added by the dilation.

## Interpretation

A useful radius would need to improve tumor coverage and HD95 in the nine
undercoverage cases while leaving the remaining 91 cases largely unchanged.

If all radii show poor target benefit or substantial non-target harm, simple
morphological expansion should be rejected and the project should proceed to a
tumor-aware inference experiment.

This is a sensitivity analysis only; no expanded mask is adopted into the
baseline.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_renal_expansion/
├── renal_expansion_sensitivity.json
├── case_comparison.csv
├── renal_expansion_sensitivity.md
└── figures/
    └── target_expansion_tradeoff.png
```
