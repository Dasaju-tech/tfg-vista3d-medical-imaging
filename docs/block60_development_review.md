# Block 60 — Failure review and CPU evaluation profiling

This block follows the complete Block 59 development report: 289 cases, with
100 historical and 189 additional development cases. Its purpose is to identify
actionable segmentation failures and locate CPU-side cost before choosing a
method change. It uses saved automatic predictions; it does not run the GPU model
or change model settings, labels, weights, existing evaluations or split allocation.

## Apply and start

Run in the main project environment, on top of Block 59:

```powershell
git apply --check .\vista3d-block60-development-review.patch
git apply .\vista3d-block60-development-review.patch
python -m pytest -q
python .\scripts\review_nv_segment_ct_full489_development.py --stage plan --run-id 221168ae556a6fdf
python .\scripts\review_nv_segment_ct_full489_development.py --stage review --run-id 221168ae556a6fdf --max-cases 3
```

The plan stage verifies the 289 saved development records using the Block 59
collector, including prediction byte hashes. It reads no original CT/GT voxel
arrays and calculates no new segmentation metrics. It saves cohort-wide cyst
screening, recorded timing differences and the selected review cases.

The review stage revalidates those sources, then processes at most three new
review attempts in the command above. Repeat the same review command to advance
through pending cases. Completed reviews are verified and reused; they do not
consume the budget. Omitting `--max-cases` processes all pending reviews. Cases
remain serial. A review can take minutes because it includes CPU evaluation and
volume reading, even though it performs no GPU inference.

By default, the first error stops the batch. `--continue-on-error` records errors
and proceeds to other cases. A partial review is expected with a budget. Exit
code 0 permits planned/partial/complete states; inspect the printed status/counts.
Any case failure returns 1. Ctrl+C returns 130; completed reviews remain available.
The same global execution lock as Blocks 58–59 prevents overlapping runs.

## Selection and interpretation

The selection is fixed before rendering or recalculation, with case-ID tie breaks
and deduplication across rules. At most 20 distinct cases are selected:

- Three additional-development cases with lowest kidney-and-mass Dice.
- Three additional-development cases with highest HD95, including infinity first.
- Two largest total reference cyst volumes among positive cases with no predicted cyst.
- Two largest total reference cyst volumes among zero-overlap cases that do predict cyst.
- Three largest predicted cyst false-positive volumes among reference-negative cases.
- Two cyst-positive cases with best Dice as positive controls.
- Three slowest original cases and two cases nearest the median original runtime.

This is an outcome-guided development sample. It is neither representative of the
cohort nor a deployment-valid error detector. References also guide the review
slice selection. Validation80 and test120 case outputs are not enumerated or
opened, and the command exposes no partition/unlock option.

The plan certificate binds the original run/protocol, all development receipt
digests, screening rows, reviewer/collector code and review dependency versions.
Changed sources or review code create a separate review namespace. No inference
fingerprint is weakened or rewritten; the 289 existing GPU results remain intact.

## Cyst screening across all 289 cases

Screening distinguishes reference-negative cases with/without prediction,
positive cases with no prediction, zero overlap despite prediction, and overlapping
cases dominated by FN or FP volume. Similar FN/FP volumes use a 5% relative balance
band, with an absolute numerical floor of 1e-9 mL. These are descriptive error
categories relative to the annotations, not clinical diagnoses.

Cyst-positive cases are grouped by **total annotated cyst volume per case**:
(0,1), [1,5), [5,20), and at least 20 mL. Each bin reports its denominator,
zero-overlap count, Dice and voxel recall summaries. The bins do not describe
individual lesion diameter/size. Precision and recall are undefined (null) when
their predicted/reference denominator is zero.

Recorded timing differences separate total-minus-worker and worker-minus-bundle.
They do not identify which operation inside the parent caused the delay. In the
uploaded Block 59 data, zero-overlap positives divide into 27 with no cyst prediction
and 48 with some prediction but no overlap with annotated cyst voxels. Those facts
motivate separate review categories; they do not establish an anatomical cause.

## Per-case CPU rerun and metric parity

Before review, the original evaluation source hashes and parent package versions
must match the saved contract. Each selected case is measured separately for:

1. Original CT/GT SHA-256 calculation and saved artifact verification.
2. Input geometry and label audit.
3. Prediction geometry, labels and digest audit.
4. The existing complete supported-renal evaluation with `cProfile`.
5. Loading CT/GT/prediction into a common canonical RAS voxel grid.
6. Cyst connected-component diagnostics and figure rendering.
7. Input/artifact verification again before completion is published.

The recomputed evaluation is compared field by field with the saved evaluation,
including all supported renal/cyst and laterality fields. Numeric comparison uses
1e-9 absolute and relative tolerances; equal positive infinities are retained.
NaNs do not pass parity. A mismatch writes `parity_failure.json`, leaves a failed
attempt for diagnosis and does not publish a successful completion receipt.

The profiler CSV contains the 50 largest cumulative function times, self times,
and call counts. Cumulative times include child calls and must not be added
together. These are **instrumented rerun times**, affected by current file caches,
CPU load and profiling overhead. They are not a measured speedup or a reconstruction
of the original wall time. Selected-case timing summaries must not be extrapolated
to the full cohort.

## Review figures and components

Each case has separate `renal_mass_review.png` and `cyst_review.png` figures.
Both show axial, coronal and sagittal cuts with CT, reference, prediction and
overlap/FN/FP panels. Slices maximize mask disagreement; a maximum-reference-area
or center fallback is used when necessary. Reorientation changes axis order/signs,
not the segmentation labels, and spacing sets the physical aspect ratio. These
few selected slices are development QC, not a complete 3D clinical review.

Cyst components use 6-connectivity. For GT and prediction, the JSON reports counts,
largest component volumes, and how many components have any voxel overlap with
the other mask. Connected components are not adjudicated lesions; any-overlap
counts are not validated lesion-detection sensitivity or precision.

## Outputs and resumption

Outputs are written below:

```text
outputs/reporting/full489_block60/<run-id>/<plan-prefix>/
```

- `plan.json`, `review_queue.csv`: selection and reasons.
- `cohort_screen.json`, `case_screening.csv`: all-development screening.
- `review_progress.json`, `stage_timings.csv`: completion, failures and measured stages.
- `cases/<case>/attempt_<id>/`: review JSON, function profile, figures or failure diagnostics.
- `cases/<case>/complete.json`: checksum-protected successful review and artifact digests.
- `report.md`: English index with image links and interpretation limits.
- `block60_review_report.zip`: compact export of the above reports and completed QC figures.

Cached reviews verify their artifact hashes, current input hashes and original
source receipts before reuse. Incomplete attempts use a new directory on retry.
The ZIP is refreshed after a successful command completion, even if the batch is
partial or reports failures; its progress file makes the coverage explicit. A
plan-only ZIP has no reviewed-case figures. A review ZIP **contains CT slice QC
images**, but no full NIfTI volumes, model weights or binary profiler dumps.

Send the final console output and the printed ZIP after the first three cases.
That allows checking parity, slice quality and timing before completing the queue.
No segmentation-method change is proposed as successful by this block; its evidence
will guide the next concrete intervention.

After validation, commit the four new block files:

```powershell
git add docs/block60_development_review.md scripts/review_nv_segment_ct_full489_development.py src/vista3d_tfg/analysis/full489_review.py tests/test_full489_review.py
git commit -m "feat: review development failures and profile CPU evaluation"
```
