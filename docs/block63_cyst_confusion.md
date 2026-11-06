# Block 63 — Annotation-relative cyst confusion

Block 62 completed development evaluation for 289/289 cases with exact metric
matches. Block 63 uses those verified sources to explain cyst errors through a
4-by-5 reference/prediction label table. It does not change segmentation or run
the GPU model, full evaluator, surface-distance metrics or laterality analysis.

## Apply and start

Use the main project environment, on top of Block 62:

```powershell
git apply --check .\vista3d-block63-cyst-confusion.patch
git apply .\vista3d-block63-cyst-confusion.patch
python -m pytest -q
python .\scripts\analyze_nv_segment_ct_full489_cyst_confusion.py --run-id 221168ae556a6fdf --validation-id 9499578d340484fa --max-cases 20
```

Expect `partial`, 20/289, zero failures and 269 deferred after this first batch.
Share the console output and the printed `block63_cyst_confusion_report.zip`.
After checking the initial results, complete the remaining cases with:

```powershell
python .\scripts\analyze_nv_segment_ct_full489_cyst_confusion.py --run-id 221168ae556a6fdf --validation-id 9499578d340484fa
```

Verified completed cases are reused without loading voxel arrays again and do
not consume `--max-cases`. Their raw input and artifact hashes are still checked.
Incomplete attempts restart in a new directory. The default stops on the first
error; `--continue-on-error` preserves failure diagnostics and continues. Exit
codes: 0 for partial/complete, 1 for failure, 130 for Ctrl+C. The same execution
lock as previous blocks prevents overlapping full489 jobs.

## What the table means

Rows are KiTS23 reference labels; columns are NV-Segment automatic predictions:

| Axis | Labels |
|---|---|
| Reference | 0 background, 1 kidney, 2 tumor, 3 cyst |
| Prediction | 0 background, 5 right kidney, 14 left kidney, 116 left cyst, 117 right cyst |

The taxonomies differ: this automatic NV-Segment configuration has no validated
tumor prediction class. Reference kidney/tumor labels have no left/right split
in this table. A left/right cyst swap is still overlap in the combined cyst
scope; this block does not measure side-specific error rates.

For each case, the complete matrix conserves the input grid voxel count. Its
twenty cells are exported as voxel counts and physical volumes in mL. Prediction
and reference arrays use the original common geometry without resampling.
The label counter works in slabs to limit temporary allocations; both label
volumes are still loaded into memory. CT intensity voxels are not loaded.

The report separates:

- Predicted cyst over reference background, kidney or tumor: cyst FP relative to
  the annotations, with each class's volume and fraction of total cyst FP.
- Reference cyst predicted as background, right kidney or left kidney: cyst FN,
  with each destination's volume and fraction of total cyst FN.
- The composition of all predicted cyst volume, including true overlap.
- The destination of all reference cyst volume, including true overlap.

A reference cyst predicted as kidney is a class disagreement within the renal
envelope. A reference cyst predicted as background is an envelope omission.
Predicted cyst overlapping reference tumor indicates label disagreement, not a
clinical diagnosis. Predicted cyst on reference background does not by itself
establish a clinical false positive or an omitted annotation. This block selects
no prompts, changes no thresholds and applies no component filter.

## Integrity and metric checks

The command first checks all original development records and the complete
Block 62 certificate, case receipts and full metric payloads. It compares those
payloads with the original saved evaluations. The plan binds source receipts,
the Block 62 certificate, analysis/helper code and package versions.

For a newly analyzed case, raw inputs and original/Block 62 artifacts are checked
before and after processing. Shape, geometry and labels are checked. The matrix
independently reconstructs the complete cyst and kidney-core binary comparison
fields, renal-envelope Dice and reference tumor volume. These are compared to
saved original metrics at 1e-9 absolute/relative tolerance. A mismatch fails the
case and records the differing fields. Surface Dice, HD95 and side assignments
are not recalculated.

The whole frozen development allocation is required. Validation and test case
outputs are never enumerated or opened; there is no partition/unlock option.

## Aggregation and outputs

Outputs are below:

```text
outputs/reporting/full489_block63/<run-id>/<plan-prefix>/
```

- `plan.json`: provenance and label definitions.
- `progress.json`: complete/failed/deferred cases and scope.
- `cases/<case>/attempt_<id>/confusion.json`: counts, volumes and error directions.
- `cases/<case>/complete.json`: completion checksum and artifact digest.
- `case_confusion.csv`: case volumes and the six error directions.
- `confusion_cells.csv`: twenty reference/prediction cells per analyzed case.
- `summary.json`: historical100, additional189 and combined development summaries,
  conditional per-case fraction statistics and top-ten volume rankings.
- `report.md`, `block63_cyst_confusion_report.zip`: English report package.

Pooled physical volumes are computed using each case's spacing before summing.
The volume-weighted error composition differs from the mean of per-case error
fractions. Both are reported, with explicit denominators. Fractions with a zero
denominator are null; per-case summaries exclude them and retain the number of
undefined entries. No overall voxel accuracy is reported because background
dominates the full-grid matrix. Rankings are descriptive, development-only and
use case-ID tie breaks, not deployment-valid decision rules.

Partial exports describe only successfully analyzed cases, even when a group is
named `development289`; inspect `status`, `n_requested`, `n_analyzed` and group
`n_cases`. Do not present a partial report as a full-cohort result. The ZIP has
reports only, without CT/QC images, NIfTI volumes or model weights. After Ctrl+C
or an early source-check failure it may still represent the previous invocation;
rerun normally before sharing it.

After validating this block, commit only its four new files:

```powershell
git add docs/block63_cyst_confusion.md scripts/analyze_nv_segment_ct_full489_cyst_confusion.py src/vista3d_tfg/analysis/full489_cyst_confusion.py tests/test_full489_cyst_confusion.py
git commit -m "feat: audit cyst label confusion across development cohort"
```
