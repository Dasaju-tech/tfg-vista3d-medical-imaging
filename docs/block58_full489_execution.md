# Block 58 - Measured and resumable full489 execution

This patch follows the validated Block 57 freeze (425 tests passed on the user's
Windows machine). It adds five files and does not change previous runners or
results. The allocation remains development 289, validation 80 and final internal
test 120; the 100 historical cases belong to development.

## First execution on the desktop GPU

Run commands from the project root in the main project environment. The worker
automatically uses the existing `vista3d-nv` Conda environment. Keep its working
Python 3.11, CUDA, PyTorch, MONAI and checkpoint installation for this measurement.
The patch does not install or upgrade packages or change the external checkout.

If Block 57 is not committed yet, commit the validated freeze first:

```powershell
git add docs/block57_full489_protocol.md scripts/prepare_kits23_full489_protocol.py src/vista3d_tfg/experiments/full489.py tests/test_full489.py configs/experiments/nv_segment_ct_full489_protocol.json
git commit -m "feat: freeze full489 cohort and protected evaluation splits"
```

Apply this patch and run the tests:

```powershell
git apply --check .\vista3d-block58-full489-execution.patch
git apply .\vista3d-block58-full489-execution.patch
python -m pytest -q
python .\scripts\run_nv_segment_ct_full489.py --stage probe
```

The probe checks the committed protocol against the working JSON and the original
metadata snapshot. It checks the checkpoint and renal label contract, records the
actual GPU and VRAM, and executes small CUDA convolutions in FP32 and FP16 inside
the selected runtime. It does not run CT inference or calculate metrics. A passing
probe confirms basic kernel compatibility; the following real CT profile is still
needed to assess memory and timing.

```powershell
python .\scripts\run_nv_segment_ct_full489.py --stage profile
```

The profile selects 20 historical cases deterministically: five with the largest
original CT voxel grids and fifteen ranked by a fixed hash of their IDs. It reads
all 100 historical CT headers to select this group, uses no outcome information,
and saves the selection before inference. Original voxel count is a practical
stress proxy, not a guarantee that these are the five largest cases after resampling.

The console prints the run directory and the `profile_summary.json` path. Save the
full output and that summary to review GPU compatibility, peak memory, case times,
failures and descriptive differences from historical metrics before a large run.
An out-of-memory error fails the case explicitly; no automatic change of patch size,
precision, overlap or prompt is made. A partial profile can be resumed with the
same command. `--max-cases 1` permits a first single-case trial; it is not a complete
profile. All 20 must succeed with the same contract before cohort inference.

The implementation was checked with CPU tests and synthetic NIfTI files. Real
GPU inference and compatibility of the user's installed runtime must be confirmed
by these two commands; synthetic measurements are never presented as GPU results.

## What is measured and preserved

Each case starts a fresh process, calls the same upstream MONAI bundle with the
automatic renal prompt `[5, 14, 116, 117]`, and explicitly binds the existing
`models/model.pt`. The bundle settings, including precision and sliding-window
configuration, are hashed rather than tuned in this block. Ground truth is passed
only to the existing parent-process evaluation function, after prediction.

The report distinguishes:

- `bundle_seconds`: the MONAI bundle call, including model initialization and
  bundle transforms, with CUDA synchronization before and after the call.
- `worker_process_seconds`: the entire Conda/Python subprocess, including imports,
  model/input hashing and process startup.
- `case_total_seconds`: input hashing/audit, worker, prediction audit and evaluation
  until receipt creation. Subsequent cache verification has its own overhead and
  does not replace the original timing.
- `peak_allocated_bytes` and `peak_reserved_bytes`: PyTorch CUDA allocator peaks on
  device 0 inside the inference process. These are not the total usage of every GPU
  process or a system RAM measurement.

Per-case process initialization is retained for a conservative first measurement.
This block provides no claimed speedup, assumed RTX 5070 runtime, fixed VRAM guess,
or extrapolation of 20 selected historical cases to the entire cohort. Later
optimization can use these measurements and must receive a new run contract.

The namespace under `outputs/full489/<contract-prefix>/` binds the protocol,
checkpoint, upstream code/config bytes, worker, relevant evaluation source files,
Python/package versions, GPU/runtime identity, prompt and surface tolerance. A
changed contract creates a separate directory. Free VRAM is diagnostic and is not
part of the stable identity. GPU driver identity is not independently recorded;
retain the machine's software/driver record with experiment notes.

Each completed case has input and ground-truth SHA-256 hashes, output hashes,
timings, allocator measurements, prediction geometry/label audit and a signed-by-hash
completion receipt. This is an integrity checksum, not a digital signature or
tamper-proof access control. Every reused receipt must match the exact contract,
input bytes and the saved prediction, metric, worker and job files.

The old 100 predictions lack the complete input/checkpoint identity required by
this runner. They remain untouched historical results and are not automatically
adopted as its verified cache. The 20 profile cases are recomputed and can then be
reused in development; the other 269 development cases still require a new receipt.
The profile includes descriptive old-versus-current Dice comparisons when old
metrics are available. These are useful for reviewing differences, but cannot
certify numerical parity because the old provenance is incomplete. They do not
select cases or gate success on a preferred metric value.

## Acquisition and later development batches

After reviewing the profile, acquire/audit development inputs and execute the next
20 missing cases:

```powershell
python .\scripts\run_nv_segment_ct_full489.py --stage acquire --partition development
python .\scripts\run_nv_segment_ct_full489.py --stage run --partition development --max-cases 20
```

Rerunning the second command verifies completed receipts and advances to the next
20 missing cases. Omitting `--max-cases` attempts every remaining development case.
The limit counts new attempts, including failed attempts, not successful outputs.
Cache verification is not counted. By default the first error stops the batch;
`--continue-on-error` records the failure and proceeds to other requested cases.
Failures remain in the original denominator and are never replaced by other IDs.

Acquisition uses the existing downloader with its per-file retries and atomic
downloads, audits NIfTI geometry and GT labels, and records input hashes in the
aggregate `outputs/full489/acquisition.json` ledger. Existing nonempty files are
audited before being accepted. An already recorded case is rehashed on resumption.
For acquisition, a case without a ledger entry counts against `--max-cases` even
when its files already existed. The ledger detects compressed CT files with
identical bytes across different case IDs. It is not a full patient-linkage or
decoded-volume duplicate audit, and a duplicate blocks acceptance of that case.

Validation commands are available for the later method-selection phase:

```powershell
python .\scripts\run_nv_segment_ct_full489.py --stage acquire --partition validation
python .\scripts\run_nv_segment_ct_full489.py --stage run --partition validation --max-cases 20
```

Complete input acquisition can be performed independently, with disk space and
bandwidth planned by the user:

```powershell
python .\scripts\run_nv_segment_ct_full489.py --stage acquire --partition all
```

`test` and `all` are accepted only for acquisition. This writes raw inputs and an
aggregate input-integrity ledger, with no predictions, performance metrics or visual
QC. The runner has no test-unlock flag. Inference/evaluation on the final 120 cases
requires a later block after methods and analysis rules have been fixed. Previous
scripts and manual file access are not an access-control boundary; keep the
reserved cases out of those workflows as agreed in Block 57.

## Failure handling and outputs

The existing frozen protocol must be in `HEAD`, and the saved metadata bytes and
expanded100 manifest must still reproduce it. The script fails before running a
model when the freeze or required assets are missing or incompatible. Keep the
Block 57 metadata snapshot when copying the project to another machine.

A single lock under `outputs/full489/execution.lock` prevents two of these commands
running concurrently. On a handled timeout (180 seconds for the probe, 3600 seconds
per inference worker) or Ctrl+C, the launcher terminates the worker process tree.
After a forced machine/process shutdown, check that the old Python/Conda inference
has stopped before manually removing a stale lock. Do not run old and new runners
concurrently on the same GPU. A failed attempt remains available for diagnosis and
does not create a completion receipt. A retry uses a separate attempt directory.

The key output files are:

| File below `outputs/full489/` | Purpose |
| --- | --- |
| `probes/<id>/job.json`, `worker.json` | Raw GPU probe request and result |
| `acquisition.json` | Verified input identities across acquired partitions |
| `acquisition_<partition>_progress.json` | Acquisition completion/failure accounting |
| `<contract>/contract.json` | Frozen runtime/model/evaluation identity |
| `<contract>/latest_probe.json` | Most recent diagnostic probe for that identity |
| `<contract>/profile_plan.json` | Frozen historical profiling selection |
| `<contract>/profile_summary.json` | Profile accounting, timings, memory and comparisons |
| `<contract>/<phase>_progress.json` | Requested, complete, failed and deferred case IDs |
| `<contract>/cases/<case>/complete.json` | Verified completion receipt |
| `<contract>/cases/<case>/attempt_<id>/` | Job, worker report, raw/stable prediction and metrics |

Exit code 0 means no case failed; `partial` with deferred cases is possible when a
budget was supplied. Exit code 1 means a runtime, integrity or case failure. Ctrl+C
returns 130. Inspect status and counts as well as the exit code. Preserve the full
output directory with the data and checkpoint to maintain reproducibility. Disk
usage includes raw and copied predictions plus retained failed attempts.

After tests and GPU profiling succeed, commit only the five new block files:

```powershell
git add docs/block58_full489_execution.md scripts/run_nv_segment_ct_full489.py scripts/nv_segment_ct_full489_worker.py src/vista3d_tfg/experiments/full489_execution.py tests/test_full489_execution.py
git commit -m "feat: profile GPU and resume protected full489 execution"
```

## Interpretation boundary

The evaluation still uses the existing supported renal structures and
kidney-and-mass HEC implementation with its fixed surface tolerance. It is not a
full KiTS23 score, a newly trained model, or clinical validation. The split is
internal to the public KiTS23 training collection. It does not establish absence
of foundation-model pretraining overlap. Patient linkage, broader duplication
checks and the final comparative analysis remain explicit future work.

The tests exercise real input/prediction auditing and metric computation on
synthetic NIfTIs; the GPU worker is mocked. They cover sealed test operations,
committed protocol checks, namespace changes, hash-verified reuse, corrupt and
partial outputs, profile gating, bounded batch resumption, failure denominators,
acquisition-only execution, duplicate bytes and exclusive locking. See the patch
delivery note for the actual test count and host-platform limitations.
