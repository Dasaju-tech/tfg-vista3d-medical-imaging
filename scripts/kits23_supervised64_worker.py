"""CUDA-only entry point invoked through the existing vista3d-nv environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Select this checkout, even when the GPU environment contains another installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vista3d_tfg.training.contract64 import atomic_json, frozen_json, read_json, check_plan, verify_inputs
from vista3d_tfg.training.supervised64 import train_smoke
from nv_segment_ct_full489_worker import runtime_snapshot


def execute(job: dict) -> dict:
    root, output, plan = Path(job["project_root"]).resolve(), Path(job["output_dir"]).resolve(), job["plan"]
    if root != ROOT:
        raise ValueError("Worker checkout differs from requested project.")
    protocol = read_json(root / "configs/experiments/nv_segment_ct_full489_protocol.json")
    check_plan(root, plan, protocol)
    verify_inputs(root, plan)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime = runtime_snapshot(test_kernel=True)
    frozen_json(output / "runtime.json", runtime)
    pairs = [(cid, root / "data/raw/kits23" / cid / "imaging.nii.gz",
              root / "data/raw/kits23" / cid / "segmentation.nii.gz") for cid in plan["smoke_case_ids"]]
    result = train_smoke(plan, pairs, output)
    check_plan(root, plan, protocol)
    verify_inputs(root, plan)
    if runtime_snapshot()["identity"] != runtime["identity"]:
        raise ValueError("GPU runtime identity changed during training.")
    print(f"Training smoke complete: {result['optimizer_steps']} optimizer steps; checkpoint reload passed.", flush=True)
    print(f"Peak allocated: {result['peak_allocated_bytes']/2**30:.2f} GiB; "
          f"peak reserved: {result['peak_reserved_bytes']/2**30:.2f} GiB; "
          f"wall: {result['wall_seconds']:.2f} s", flush=True)
    return {**result, "runtime_identity": runtime["identity"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = execute(read_json(args.job))
        code = 0
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(report["error"], file=sys.stderr, flush=True)
        code = 1
    atomic_json(args.report, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
