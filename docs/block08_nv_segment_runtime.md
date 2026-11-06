# Block 08 — Maintained VISTA3D / NV-Segment-CT Runtime

## Motivation

The current MONAI VISTA documentation states that the maintained VISTA3D
model bundles live in the NVIDIA-Medtech `NV-Segment-CTMR` repository.

For this CT-only TFG, the integration target is `NV-Segment-CT`, which is the
maintained MONAI Bundle packaging of the VISTA3D-CT line.

The older CVPR research repository remains useful for paper reproduction, but
it is not the runtime selected for the main project pipeline.

## Environment isolation

The project uses two Conda environments:

### `vista3d-tfg-311`

Owns the TFG code:

- data acquisition
- NIfTI auditing
- quantitative analysis
- KiTS23 evaluation
- visualization
- tests
- reporting

### `vista3d-nv`

Owns the upstream model runtime:

- MONAI Bundle runtime
- upstream PyTorch/MONAI pins
- NV-Segment dependencies
- CUDA inference

The model dependency stack is deliberately not installed over the main TFG
environment. The TFG invokes the model runtime with `conda run`.

This avoids dependency drift in the already-tested analysis environment.

## External checkout

The maintained upstream repository is cloned under:

```text
data/external/NV-Segment-CTMR/
```

This path is already excluded by the project's data ignore policy.

An existing checkout is never automatically pulled. Its current git commit is
recorded instead. Updating the upstream model therefore becomes an explicit
experimental decision rather than an accidental dependency change.

## Runtime provenance

Preparation records:

- upstream repository URL
- exact git commit
- runtime environment name
- requested Python version
- actual Python version
- PyTorch version
- MONAI version
- CUDA availability
- CUDA device count
- GPU name

Output:

```text
outputs/runtime/nv_segment_ct_runtime.json
```

## Maintained inference interface

The integration constructs the upstream MONAI Bundle command:

```text
python -m monai.bundle run
    --config_file configs/inference.json
    --input_dict "{'image': '...'}"
```

executed inside the upstream `NV-Segment-CT` bundle through the isolated
Conda environment.

The upstream documentation reports that model weights are obtained
automatically during the first inference and reused from the local Hugging
Face cache afterwards.

## Block boundary

Block 08 prepares and verifies the model runtime but intentionally does not
claim a segmentation result.

Block 09 will:

1. run the first real KiTS23 CT inference,
2. locate the produced NIfTI mask deterministically,
3. audit its geometry,
4. inventory the upstream label vocabulary,
5. identify the kidney/tumor mapping required for KiTS23 comparison,
6. persist inference provenance and timings.

This separation prevents runtime-installation failures from being confused with
model-performance failures.
