# Block 61 — Boolean validation optimization and paired CPU benchmark

Block 60 completed 17 selected development reviews with full metric parity. Its
instrumented evaluator calls totalled 1,411.36 seconds; NumPy array sorting used
876.92 seconds of self time (62.1% of that total). Repeated morphology validation
of Boolean component masks is a concrete optimization target. These figures are
from the selected review cases, not a runtime estimate for all 289 cases.

## Apply and run

Use the main project environment that produced the original CPU evaluations.
Apply on top of Block 60. There are four new files and no existing source edits.

```powershell
git apply --check .\vista3d-block61-cpu-benchmark.patch
git apply .\vista3d-block61-cpu-benchmark.patch
python -m pytest -q
python .\scripts\benchmark_nv_segment_ct_full489_cpu.py --run-id 221168ae556a6fdf --review-id 56fb0bb1df424d0e --max-cases 3
```

This verifies the saved development results and the complete Block 60 review
before starting CPU evaluation. It does not call the GPU worker. Keep the same
environment and source files for the whole benchmark. Avoid overlapping jobs on
the same computer while measuring. Each large case can take several minutes;
the command prints the mode before each evaluator call and its duration after it.

The default is **two paired repetitions**: four evaluator calls per case, two
baseline and two optimized. A case is complete only after every call and pair
passes parity and the input/artifact hashes are rechecked. With `--max-cases 3`,
expect `partial`, 3/17, no failures and 14 deferred. Share the printed
`block61_cpu_benchmark_report.zip` after those first three cases.

After checking the initial report, finish the queue with:

```powershell
python .\scripts\benchmark_nv_segment_ct_full489_cpu.py --run-id 221168ae556a6fdf --review-id 56fb0bb1df424d0e
```

Verified completed cases are reused without reevaluation and do not consume the
case budget. Interrupted or failed case attempts restart in a new directory;
individual timings from unfinished cases are never mixed into a successful run.
The default stops at the first error; `--continue-on-error` exports failures and
continues. Exit codes: 0 for partial/complete, 1 for failure, 130 for interruption.
The archive is refreshed on normal completion, including partial or failed runs;
after Ctrl+C it may still represent the previous invocation until resumption.

## Exact scope of the optimization

An ordinary NumPy ndarray with Boolean dtype and three dimensions already has
binary values. `validate_binary_mask_fast` accepts that case without calling
`np.unique`. All other inputs, including numeric arrays and ndarray subclasses,
use the original validator and retain its input checks and errors.

The benchmark temporarily substitutes only `morphology.validate_binary_mask`
during an optimized evaluator call. It restores the function in a `finally`
block, including on exceptions and Ctrl+C. It never patches NumPy functions.
The substitution is for this serial command; it is not an API for a threaded
service. Nested/concurrent benchmark contexts are rejected.

Original evaluator sources, inference contracts, model weights and all prior
results remain unchanged. This block measures an opt-in candidate optimization;
it does not silently enable it in Blocks 58–60. A later adoption can be based on
the observed parity and performance results.

## Comparison protocol

- Use exactly the completed Block 60 case selection, in its frozen order.
- Alternate baseline→optimized and optimized→baseline within each case; alternate
  the starting mode between cases. `--repeats` must be even and at least two.
- Measure the full evaluator call, including its NIfTI reads, with `perf_counter`.
  No `cProfile` runs. Dictionary conversion, JSON writes and source hashes are
  outside that timer. There is no excluded warmup or forced cache flush.
- Compare every output field against the saved original evaluation, then compare
  the two members of each pair directly. Absolute and relative tolerance are both
  1e-9. Equal positive infinities pass; NaNs do not. Exported infinite values use
  the explicit string `+Infinity` in metric payloads.
- Preserve all measured repetitions, including slow ones. Summarize baseline and
  optimized medians per case. A speedup ratio is baseline median / optimized
  median; values below 1 are slowdowns and remain in the report.

The selected subset is deliberately enriched for failures and long runtimes.
File caches, thermal state and background load affect timings. These paired
observations do not establish an end-to-end GPU speedup, a whole-cohort speedup,
or improved segmentation. Metric parity and performance are separate outcomes.

## Provenance and outputs

The benchmark plan binds the original run/protocol, Block 60 plan and review
receipts, development source receipts, benchmark/helper code, evaluator package
versions, Python/platform/CPU identity and common thread environment variables.
Changing repetitions, code or recorded environment creates a separate output
namespace. Original evaluator source/package drift is rejected.

```text
outputs/reporting/full489_block61/<run-id>/<benchmark-plan-prefix>/
```

- `plan.json`: provenance, selected cases and measurement protocol.
- `progress.json`: complete/failed/deferred cases and descriptive timing summary.
- `case_timings.csv`: paired case median timings and ratios.
- `measurements.csv`: each repetition, order position, mode and duration.
- `cases/<case>/attempt_<id>/benchmark.json`: all full metric payloads and timings.
- `cases/<case>/complete.json`: successful benchmark receipt and artifact hash.
- `report.md` and `block61_cpu_benchmark_report.zip`: reviewable report package.

Failures export diagnostic JSON, including metric differences where available.
The ZIP contains reports, not CT images, NIfTI volumes or weights. No validation
or test case output is opened, and there is no partition/unlock option.

After validating this block on your machine, commit only its four new files:

```powershell
git add docs/block61_cpu_benchmark.md scripts/benchmark_nv_segment_ct_full489_cpu.py src/vista3d_tfg/analysis/full489_cpu.py tests/test_full489_cpu.py
git commit -m "feat: benchmark boolean mask validation optimization with metric parity"
```
