"""Run CPU preprocessing or CUDA training in the existing model environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from vista3d_tfg.training import contract65 as c
from vista3d_tfg.training.cache65 import prepare_cache, runtime_identity
from nv_segment_ct_full489_worker import runtime_snapshot


def execute(job: dict) -> dict:
    root=Path(job["root"]).resolve();plan=job["plan"]
    if root!=ROOT:raise ValueError("Worker belongs to another checkout.")
    protocol=c.read_json(root/"configs/experiments/nv_segment_ct_full489_protocol.json")
    c.check_plan(root,plan,protocol)
    expected=plan["runtime_identity"]
    wanted={"numpy":expected["numpy"],"nibabel":expected["packages"]["nibabel"],"scipy":expected["packages"]["scipy"]}
    if runtime_identity()!=wanted:raise ValueError("Preprocessing package versions differ from frozen GPU environment.")
    if job["stage"]=="cache":
        result=prepare_cache(root,plan,job["partition"],max_cases=job["max_cases"])
    elif job["stage"]=="train":
        from vista3d_tfg.training.engine65 import train
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        runtime=runtime_snapshot(test_kernel=True)
        if runtime["identity"]!=expected:raise ValueError("CUDA runtime differs from frozen Block 64 environment.")
        result=train(root,plan,runtime["identity"],epochs=job["epochs"])
        if runtime_snapshot()["identity"]!=expected:raise ValueError("CUDA runtime changed during training.")
    else:raise ValueError("Unknown worker stage.")
    c.check_plan(root,plan,protocol)
    return {**result,"experiment_sha256":plan["experiment_sha256"],"stage":job["stage"]}


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    try:report=execute(c.read_json(args.job));code=0
    except Exception as exc:
        report={"status":"failed","error":f"{type(exc).__name__}: {exc}"};code=1
        print(report["error"],file=sys.stderr,flush=True)
    c.atomic_json(args.report,report)
    return code


if __name__=="__main__":raise SystemExit(main())
