# Block 04 — Real-Data Smoke Acquisition and Provenance

## Purpose

The first real-data step deliberately downloads only a small KiTS23 subset.
This prevents a large dataset transfer from becoming a prerequisite for
testing the project infrastructure.

The default smoke subset contains three cases:

- `case_00000`
- `case_00001`
- `case_00002`

The case list is configurable from the command line.

## Data sources

The CT image source follows the URL layout used by the official KiTS23
downloader:

`neheller/KiTS-Challenge-Imaging` on Hugging Face.

Reference segmentations are downloaded from the official `neheller/kits23`
GitHub repository.

The current KiTS23 repository reports dataset version `0.1.4`. The dataset is
distributed under CC BY-NC-SA 4.0.

## Local layout

Real medical data remain outside Git:

```text
data/raw/kits23/
└── case_XXXXX/
    ├── imaging.nii.gz
    └── segmentation.nii.gz
```

## Integrity and provenance

Every acquired file is recorded with:

- source URL,
- local path,
- byte size,
- SHA-256 digest,
- download/reuse status,
- HTTP ETag when available.

The acquisition script then runs the Block 02 NIfTI integrity audit on every
requested case. A successful HTTP request is therefore not sufficient for a
case to be considered valid.

Generated local records:

```text
outputs/manifests/kits23_smoke_provenance.json
outputs/audits/kits23_smoke_audit.json
```

These outputs are reproducibility artifacts and are not raw medical data.

## Warning policy

Known deprecation warnings currently originate from:

1. PyTorch/MONAI code using the deprecated `torch.jit.interface`.
2. DeepMind `surface-distance` using SciPy namespaces deprecated before
   SciPy 2.0.

The test suite suppresses only these exact third-party warning families.
Unrelated warnings remain visible. This keeps the test output useful without
hiding warnings introduced by this project.

## Execution

From the project root:

```powershell
python .\scripts\acquire_kits23_smoke.py
```

A custom smoke subset can be requested:

```powershell
python .\scripts\acquire_kits23_smoke.py --cases 0 1 2 3 4
```

Re-download existing files only when required:

```powershell
python .\scripts\acquire_kits23_smoke.py --cases 0 1 2 --force
```

## Full-dataset policy

This block intentionally does not download all 489 KiTS23 training cases.
After the real-data smoke audit passes, full acquisition should use the
official KiTS23 downloader or an equivalently verified process, followed by
the same project-level audit and provenance checks.
