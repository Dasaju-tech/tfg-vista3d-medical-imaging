# Block 62 — Versioned CPU evaluator and full-development parity

Block 61 verified 17/17 selected cases, with 68 timed evaluator calls. All cases
were faster; the median case speedup was 2.26x (range 1.38–4.32x). The ratio of
summed case medians was 3.13x, a 68.06% reduction in that selected benchmark.
Those figures are not extrapolated to the complete development cohort.

Block 62 introduces the explicit evaluator version `supported-renal-cpu-bool-v1`
and compares its complete outputs to saved original results for **all 289 frozen
development cases**. It evaluates each case only once, using its saved prediction.
The original CPU evaluator and GPU model are not rerun by this command.

## Apply and first batch

Apply on top of the completed Block 61, in the same main project environment:

```powershell
git apply --check .\vista3d-block62-cpu-validation.patch
git apply .\vista3d-block62-cpu-validation.patch
python -m pytest -q
python .\scripts\validate_nv_segment_ct_full489_cpu_v1.py --run-id 221168ae556a6fdf --benchmark-id e49f21601d699b29 --max-cases 10
```

Expect `partial`, 10/289, zero failures and 279 deferred after the first batch.
Share the console output and the printed `block62_cpu_validation_report.zip`.
Cases with many components can take minutes even with the Boolean optimization.
The existing execution lock prevents simultaneous full489 jobs.

After reviewing the first batch, complete all remaining cases with:

```powershell
python .\scripts\validate_nv_segment_ct_full489_cpu_v1.py --run-id 221168ae556a6fdf --benchmark-id e49f21601d699b29
```

Completed case records and their current raw input/artifact hashes are verified
before reuse. Cached cases do not consume `--max-cases`. Incomplete attempts
restart in a new attempt directory. By default a failure stops the batch;
`--continue-on-error` records failures and continues through the remaining cases.
Exit codes are 0 for partial/complete, 1 for failure and 130 for Ctrl+C.

## Versioned entry point

The new callable is:

```python
from vista3d_tfg.evaluation.nv_kits23_cpu_v1 import (
    EVALUATOR_VERSION,
    evaluate_supported_renal_prediction_cpu_v1,
)
```

It returns the original `SupportedRenalComparison` result schema. It uses the
exact Block 61 scoped Boolean validator optimization: ordinary three-dimensional
Boolean ndarrays do not require value enumeration. Other input types retain the
original validation. The function restores the original validator even if an
exception or interruption occurs. This entry point is for serial evaluation;
it must not run alongside unrelated evaluator threads in the same process.

All existing source files and inference fingerprints remain unchanged. Old
commands continue using their original evaluator. This explicit entry point can
be selected in subsequent experiments with its own recorded version and hashes.
The version name alone is not a certificate that all development cases passed.

## Evidence and gates

Before starting, the command verifies all saved development records, completed
Block 60 reviews and the completed Block 61 benchmark. It verifies the benchmark
artifact hashes, balanced measurement order and all exported metric payloads
against the saved original results. It requires matching baseline source/package
versions, including loaded Python package sources. No validation/test output is
opened, and the CLI has no partition/unlock option.

The new plan binds the complete development allocation, original receipts, Block
61 evidence, evaluator/optimizer/coordinator/helper code hashes and environment.
Changed code or environment cannot silently reuse results from an old plan.

Each new case checks raw input and original artifact hashes before and after the
evaluation. It compares every field at 1e-9 absolute and relative tolerance,
including shape, labels, renal/cyst metrics and laterality results. Equal positive
infinities are retained and exported as the string `+Infinity`; NaNs fail. Exact
numeric matches are counted separately from matches within the tolerance.

Metric mismatches create `parity_failure.json` and a failed attempt, with no
successful case completion. No aggregate development certificate is issued for
partial or failed coverage. A complete run creates `development_validation.json`
with every case completion checksum and the exact evaluator/protocol identity.
The certificate means numerical parity on development, not clinical validation,
better segmentation or successful evaluation of held-out cases.

## Outputs

```text
outputs/reporting/full489_block62/<run-id>/<plan-prefix>/
```

- `plan.json`: selection, evaluator identity and source evidence.
- `progress.json`: completed/failed/deferred cases, parity counts and CPU times.
- `case_parity.csv`: one row per verified case, exact-match flag and elapsed time.
- `cases/<case>/attempt_<id>/evaluation.json`: full versioned metric payload.
- `cases/<case>/complete.json`: successful case receipt and artifact hash.
- `development_validation.json`: only present after full successful coverage.
- `report.md`, `block62_cpu_validation_report.zip`: shareable report package.

Times measure one optimized call including NIfTI loading and scoped validator
setup, without cProfile. They are descriptive; the historical timings cannot be
used as a paired baseline to claim a speedup across all 289 cases. The ZIP has
reports only, without full CT volumes, QC images or model weights.

The archive is refreshed on normal completion, including partial/failed batches.
After an interruption or an early source-check failure, an older ZIP can still
represent the preceding invocation. Rerun successfully before sharing the ZIP.

After successful validation, commit the five new files:

```powershell
git add docs/block62_cpu_validation.md scripts/validate_nv_segment_ct_full489_cpu_v1.py src/vista3d_tfg/evaluation/nv_kits23_cpu_v1.py src/vista3d_tfg/analysis/full489_cpu_adoption.py tests/test_full489_cpu_adoption.py
git commit -m "feat: validate versioned CPU evaluator parity across development cohort"
```
