# Block 33 — HD95 Tail and Tumor-Burden Audit

## Objective

Block 32 found 22 HD95 outliers in the frozen expanded100 cohort and a much stronger association between HD95 and ground-truth tumor volume than between HD95 and cyst burden.

Block 33 is analysis-only. It separates three descriptive HD95 failure modes before any model modification:

- catastrophic overlap failure: HD95 outlier with Dice < 0.60;
- distant boundary failure: HD95 outlier with Dice >= 0.85;
- mixed HD95 failure: remaining HD95 outliers.

These are analysis labels, not clinical cutoffs.

## Tumor burden

Tumor-positive cases are divided into low, medium and high tertiles of reference tumor volume, with tumor-negative cases retained separately. Each stratum reports HD95, Dice, Surface Dice and HD95-outlier fraction.

The block also reports a descriptive high-tumor-versus-rest HD95-outlier risk ratio.

## Tail concentration

The audit quantifies how much of the cohort's total HD95 is contributed by the worst five, worst ten and all HD95-outlier cases. It also compares all-case HD95 with the non-outlier subset.

## Outputs

```text
outputs/analyses/nv_segment_ct_expanded100_hd95_tail/
├── hd95_tail_audit.json
├── case_taxonomy.csv
└── figures/
    ├── tumor_volume_vs_hd95.png
    ├── dice_vs_hd95.png
    └── hd95_outlier_fraction_by_tumor_burden.png
```

## Interpretation boundary

This is a post-hoc descriptive audit used only to choose the next controlled experiment. It does not establish causality and performs no new inference.
