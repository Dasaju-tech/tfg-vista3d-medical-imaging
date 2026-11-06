"""Run inside vista3d-nv; no editable vista3d_tfg installation is required here."""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import time


def digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_snapshot(*, test_kernel: bool = False) -> dict:
    import numpy
    import torch
    import monai

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected model environment.")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        raise RuntimeError("The existing NV-Segment runtime requires Python 3.11.")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if test_kernel:
        with torch.no_grad():
            for dtype in (torch.float32, torch.float16):
                x = torch.ones((1, 1, 8, 8, 8), device="cuda:0", dtype=dtype)
                kernel = torch.ones((1, 1, 3, 3, 3), device="cuda:0", dtype=dtype)
                y = torch.nn.functional.conv3d(x, kernel)
                if not bool(torch.isfinite(y).all()):
                    raise RuntimeError("CUDA convolution produced non-finite values.")
                del x, kernel, y
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    packages = {}
    for package in ("nibabel", "scipy", "einops", "huggingface-hub", "SimpleITK"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    return {
        "identity": {
            "python": platform.python_version(),
            "platform": platform.system(),
            "torch": str(torch.__version__),
            "monai": str(monai.__version__),
            "numpy": str(numpy.__version__),
            "packages": packages,
            "cuda": str(torch.version.cuda),
            "cudnn": torch.backends.cudnn.version(),
            "gpu_name": properties.name,
            "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_total_bytes": properties.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        },
        "free_memory_bytes": free,
        "total_memory_bytes": total,
        "fp32_fp16_conv3d_smoke": test_kernel,
    }


def verify_assets(job: dict) -> None:
    bundle = Path(job["bundle_root"])
    for relative, expected in job["bundle_files"].items():
        if digest_file(bundle / relative) != expected:
            raise RuntimeError(f"Bundle file changed: {relative}")
    checkpoint = bundle / "models/model.pt"
    if digest_file(checkpoint) != job["checkpoint_sha256"]:
        raise RuntimeError("Model checkpoint changed.")


def execute(job: dict) -> dict:
    if job["mode"] == "probe":
        return {"status": "complete", **runtime_snapshot(test_kernel=True)}
    if job["mode"] != "infer":
        raise ValueError("Unknown worker mode.")
    if job["label_prompt"] != [5, 14, 116, 117]:
        raise ValueError("The automatic renal prompt changed.")
    bundle = Path(job["bundle_root"]).resolve()
    # Match python -m monai.bundle from the maintained bundle directory.
    os.chdir(bundle)
    sys.path.insert(0, str(bundle))
    snapshot = runtime_snapshot()
    if snapshot["identity"] != job["runtime_identity"]:
        raise RuntimeError("The actual GPU/runtime differs from the frozen run contract.")
    verify_assets(job)
    image = Path(job["image_path"])
    if digest_file(image) != job["image_sha256"]:
        raise RuntimeError("Input CT changed before inference.")

    import torch
    from monai.bundle import run

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    started = time.perf_counter()
    run(
        config_file=str(bundle / "configs/inference.json"),
        input_dict={"image": image.as_posix(), "label_prompt": job["label_prompt"]},
        output_dir=Path(job["output_dir"]).as_posix(),
        checkpoint_path=(bundle / "models/model.pt").as_posix(),
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "status": "complete",
        "runtime_identity": snapshot["identity"],
        "bundle_wall_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "checkpoint_sha256": job["checkpoint_sha256"],
        "image_sha256": job["image_sha256"],
    }
    # A helper must not silently replace the recorded checkpoint/configuration.
    verify_assets(job)
    if digest_file(image) != job["image_sha256"]:
        raise RuntimeError("Input CT changed during inference.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report.resolve()
    try:
        report = execute(json.loads(args.job.read_text(encoding="utf-8")))
        code = 0
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(report["error"], file=sys.stderr, flush=True)
        code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, report_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
