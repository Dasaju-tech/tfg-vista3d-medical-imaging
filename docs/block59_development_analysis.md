# Block 59 - Development289 aggregate analysis

Block 58 completed all 289 development cases without execution failures under
contract `221168ae556a6fdf45870e9e25ac4a3100ca9d424e33bc8c093c50f4bcda036d`.
This establishes execution completeness. Segmentation quality must be assessed
from all saved metrics, rather than the final cases printed by the runner.

This block adds four files. It preserves the Block 58/58a source files, model,
inference fingerprint and existing results. It performs no GPU inference,
downloads, metric recomputation or validation/test output reads.

## Run in the main project environment

Place the patch in the project root, apply it after Block 58a and run:

```powershell
git apply --check .\vista3d-block59-development-analysis.patch
git apply .\vista3d-block59-development-analysis.patch
python -m pytest -q
python .\scripts\analyze_nv_segment_ct_full489_development.py --run-id 221168ae556a6fdf
```

The explicit run ID prevents accidentally pooling results from different models
or runtime configurations. The script supports `--project-root` when needed.
The existing NumPy and Matplotlib dependencies are used in the main environment;
the separate GPU environment and model checkpoint need not be started.

Expected completeness output is:

```text
Block 59 development analysis PASSED
historical100: n=100; ...
additional_development189: n=189; ...
development289: n=289; ...
```

The values printed after those counts are calculated locally from the saved
case results. No cohort metric values are inferred from the execution log.

## Integrity and scope

The committed Block 57 protocol and its original metadata snapshot must still
agree. The run contract checksum, protocol identity, automatic renal prompt and
surface tolerance are checked. Every requested development case must have its
frozen input identity and a valid completion receipt.

The analyzer rehashes all four referenced artifacts: prediction NIfTI, evaluation
JSON, worker JSON and job JSON. It also cross-checks identities, primary metrics,
prediction audit, GPU allocator measurements and bundle timing between those
records. It reads prediction bytes for hashing without loading voxel arrays.
Original CT/GT bytes and model weights are not rehashed in this reporting block;
their recorded identities are cross-checked against the completion receipts.

The final progress JSON is not treated as proof that all artifacts are present
and intact. Missing/corrupt results remain in the requested denominator. Any
failure creates an `integrity_audit.json`, returns exit code 1 and prevents an
aggregate summary or ZIP from being published for that invocation. Existing
inference artifacts and earlier reports remain untouched.

Only development IDs from the frozen protocol are opened. Reserved output
directories are not scanned. The global Block 58 lock prevents simultaneous
execution/acquisition through these runners while the report is being built.

## Statistics

Three groups are reported: historical100, additional development189, and their
union development289. The additional 189 are not all additional389: the other
200 belong to validation80 and test120. Development289 is not an independent
sample from its historical subset.

For kidney-and-mass Dice, Surface Dice, finite HD95, diagnostic kidney-core Dice
and cyst-positive Dice, the report includes the arithmetic mean, sample standard
deviation (ddof=1), median, quartiles, p95, range and available-case counts.
Mean confidence intervals use 5000 case-bootstrap replicates, fixed seed
20260905, and linear percentile interpolation. Each group/metric has a stable
derived random seed. `--bootstrap-reps` accepts at least 1000 and `--seed` a
nonnegative integer; both are recorded with Python, NumPy and analyzer source
identity. A singleton has no estimated sample SD or bootstrap interval.

Positive-infinite HD95 is retained as an explicit outcome, ranked ahead of finite
distances in the worst-case list, and counted separately. Numerical HD95 summaries
and intervals describe the finite subset only. Its ECDF retains all cases in its
denominator, so infinite outcomes prevent the finite curve from reaching one.
NaN, negative infinity, negative distances and invalid Dice values are rejected.

Cyst-positive Dice includes only cases with reference cyst voxels. Reference-
negative cases with predicted cysts and their false-positive volumes are reported
separately. Empty-reference perfect Dice therefore cannot inflate this estimate.
Tumor-positive counts, tumor volume distribution and unresolved laterality counts
provide additional context.

The mean difference between additional and historical development is descriptive.
There is no significance test treating the nested cohorts as independent, no
causal interpretation, and no automatic method selection. Bootstrap intervals
assume independent cases; patient-linkage checks and foundation-model pretraining
overlap remain unresolved. These are exploratory development estimates, not
external generalization, a full KiTS23 challenge score or clinical validation.

## Failure review and execution cost

Top-ten lists identify the lowest kidney-and-mass Dice, lowest Surface Dice,
highest HD95, slowest cases and highest reserved memory. They prioritize review
within development and do not establish clinical failure thresholds.

Original successful-case total, worker-process and bundle times are reported
separately. Their sum is not end-to-end experiment wall time: acquisition, probes,
failed attempts and subsequent cache verification are excluded. GPU allocated and
reserved peaks remain allocator counters; they do not establish physical VRAM
residency or shared system RAM consumption. The report counts cases exceeding
the GPU total recorded by the runtime without claiming why this happened.

## Deliverable outputs

Each invocation uses a new directory below:

```text
outputs/reporting/full489_development/<run-id>/<timestamp-and-id>/
```

The script prints the exact directory and ZIP path. The ZIP contains only:

- `summary.json`: aggregate metrics, uncertainty, ranks and interpretation limits.
- `case_metrics.csv`: one row per development case with input/receipt digests.
- `cohort_summary.csv`: numeric estimates and explicit denominators by cohort.
- `report.md`: English Results-style summary and review tables.
- `development_distributions.png` and `.pdf`: cohort ECDFs and time/volume scatter.
- `integrity_audit.json`: full requested/verified IDs, checksums and verification scope.
- `source_contract.json`: the unmodified source run contract.

Upload `block59_development_report.zip` and the final console summary for the next
analysis step. No CT volumes, segmentation volumes, predictions or model weights
are included. The plots are report figures, not patient-image QC overlays.

After successful verification and report generation:

```powershell
git add docs/block59_development_analysis.md scripts/analyze_nv_segment_ct_full489_development.py src/vista3d_tfg/analysis/full489_development.py tests/test_full489_development.py
git commit -m "feat: audit and summarize full489 development results"
```

The regression tests use synthetic NIfTI inputs and actual existing evaluation
code with a mocked GPU worker. They cover artifact corruption, missing results,
cross-record disagreements, nonfinite outcomes, cyst denominators, bootstrap
reproducibility, split protection, numeric exports, plot/ZIP creation and CLI
success/failure accounting. Real aggregate results must come from the user's
289-case run; the synthetic test values are not experimental evidence.
