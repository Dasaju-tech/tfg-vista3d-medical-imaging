"""Execute a separately frozen recovery without altering the parent run."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from vista3d_tfg.training import contract65 as c, contract65b as b
from nv_segment_ct_full489_worker import runtime_snapshot


def execute(job):
    root = Path(job["root"]).resolve(); plan = job["plan"]
    if root != ROOT: raise ValueError("Worker belongs to another checkout.")
    protocol = c.read_json(root/"configs/experiments/nv_segment_ct_full489_protocol.json")
    b.check_plan(root, plan, protocol)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime = runtime_snapshot(test_kernel=True)["identity"]
    if runtime != plan["runtime_identity"]: raise ValueError("Frozen CUDA runtime changed.")
    from vista3d_tfg.training.engine65b import train_until
    result = train_until(root, plan, runtime, until_epoch=job["until_epoch"])
    if runtime_snapshot()["identity"] != runtime: raise ValueError("CUDA runtime changed during training.")
    b.check_plan(root, plan, protocol)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try: report = execute(c.read_json(args.job)); code = 0
    except Exception as exc:
        report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}; code = 1
        print(report["error"], file=sys.stderr, flush=True)
    c.atomic_json(args.report, report)
    return code


if __name__ == "__main__": raise SystemExit(main())
