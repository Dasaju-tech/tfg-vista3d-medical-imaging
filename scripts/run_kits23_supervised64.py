from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, verified_protocol
from vista3d_tfg.experiments.supervised64 import CONFIG, prepare, run_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 64: freeze and run supervised development-only GPU training smoke.")
    parser.add_argument("--stage", required=True, choices=("prepare", "smoke"))
    parser.add_argument("--run-id", default="221168ae556a6fdf")
    parser.add_argument("--confusion-id", default="072cf3bb1fb17561")
    parser.add_argument("--environment", default="vista3d-nv")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        with execution_lock(root / "outputs/full489"):
            if args.stage == "prepare":
                output, plan = prepare(root, protocol, args.run_id, args.confusion_id)
                print(f"Block 64 protocol freeze PASSED: {plan['experiment_sha256']}")
                print(f"Smoke development cases: {', '.join(plan['smoke_case_ids'])}")
                print(f"Protocol: {root / CONFIG}")
                print("GPU training: none during prepare. Next: rerun with --stage smoke.")
                return 0
            state, archive = run_smoke(root, protocol, args.environment)
        print(f"Status: {state['status']}")
        print(f"Optimizer steps: {state.get('optimizer_steps', 'not certified')}/24")
        if "error" in state:
            print(f"FAILED: {state['error']}")
        print(f"Share this ZIP: {archive}")
        print("Training feasibility only. Validation/test data and results remain reserved.")
        return 0 if state["status"] == "complete" else 1
    except KeyboardInterrupt:
        print("Interrupted. Rerun to restart the small smoke in a separate attempt.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 64 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
