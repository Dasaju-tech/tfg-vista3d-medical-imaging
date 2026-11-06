# Block 58a - Acquisition path serialization and completion accounting

The real downloader returns `DownloadedFile.path` as a pathlib `Path`. Block 58
passed `asdict(record)` directly to JSON, which raises a serialization error for
both WindowsPath and PosixPath. The earlier acquisition tests mocked downloads
with empty tuples and missed this boundary.

The repair converts only the path field to text, preserving the other download
metadata. A case is counted as complete only after the atomic ledger write has
succeeded. A failed write leaves both the on-disk ledger and in-memory ledger
unchanged, so continuing to later cases cannot accidentally commit a failed case.

The CLI now uses `experiments/full489_acquisition.py`. The original helper in
`full489_execution.py` is retained as part of the frozen Block 58 source snapshot;
it must not be used for new acquisition calls. The repaired module is independent
of the inference/evaluation fingerprint. No fingerprint validation is disabled or
relaxed, and no existing contract, receipt or profile is rewritten. This repair
alone therefore does not require repeating the completed 20-case GPU profile.

Apply on top of Block 58, from the project root:

```powershell
git apply --check .\vista3d-block58a-acquisition-fix.patch
git apply .\vista3d-block58a-acquisition-fix.patch
python -m pytest -q
python .\scripts\run_nv_segment_ct_full489.py --stage acquire --partition development
```

There is no need to remove the previous failure report, downloaded data or GPU
outputs. The old failed JSON serialization did not publish a ledger entry. The
next acquisition reuses nonempty local downloads, audits them, writes the corrected
ledger, and refreshes the progress report. Existing valid ledger entries remain
subject to their input hash checks.

Once acquisition reports complete 289/289, the existing bounded development run
can be used. It verifies and reuses the 20 profiled cases under the same contract:

```powershell
python .\scripts\run_nv_segment_ct_full489.py --stage run --partition development --max-cases 20
```

Test inference remains sealed. No GPU settings or external dependencies change.

Regression tests use the real downloader's cached-file branch and real Path-valued
records, Windows path representations, NIfTI audits, failed writes with and without
continuation, old-ledger preservation, duplicate rejection, reserved acquisition,
CLI dispatch and unchanged inference identity.
