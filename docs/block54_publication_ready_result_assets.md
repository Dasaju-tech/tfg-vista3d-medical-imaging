# Block 54 — Publication-Ready Figures & LaTeX Tables

## Purpose

Block 53 closed the experimental phase and established a single reproducible
source of truth. Block 54 converts that synthesis into publication-ready
reporting assets without running additional inference or manually copying
numeric results.

## Source

All numbers are read from:

```text
outputs/analyses/nv_segment_ct_final_experimental_synthesis/summary.json
```

## Generated figures

Both vector PDF and 300-dpi PNG versions are produced:

1. automatic baseline overview;
2. HD95 failure-mechanism split;
3. tumor interaction comparison;
4. cyst volume-response curve.

## Generated LaTeX

```text
results_macros.tex
table_final_results.tex
table_claim_matrix.tex
```

The tables assume `booktabs`; the figures can be inserted with `graphicx`.

## Output

```text
outputs/reporting/final_results/
├── figures/
│   ├── fig01_automatic_baseline.{pdf,png}
│   ├── fig02_hd95_failure_mechanisms.{pdf,png}
│   ├── fig03_tumor_interaction_comparison.{pdf,png}
│   └── fig04_cyst_volume_response.{pdf,png}
└── latex/
    ├── results_macros.tex
    ├── table_final_results.tex
    └── table_claim_matrix.tex
```

After visual inspection, the remaining work should move to the manuscript:
Results, Discussion, limitations, and conclusions.
