from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, verified_protocol
from vista3d_tfg.experiments.supervised65 import prepare, dispatch


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description="Block 65: supervised development training and frozen validation selection.")
    parser.add_argument("--stage",required=True,choices=("prepare","cache","train","report"))
    parser.add_argument("--partition",default="development",choices=("development","validation"))
    parser.add_argument("--epochs",type=int,default=1,help="Additional epochs this call; frozen total stays 50.")
    parser.add_argument("--max-cases",type=int,help="Cache-stage new case budget only.")
    parser.add_argument("--environment",default="vista3d-nv")
    parser.add_argument("--project-root",type=Path)
    args=parser.parse_args(argv)
    if args.epochs<1 or args.max_cases is not None and args.max_cases<1:parser.error("Budgets must be positive")
    if args.max_cases is not None and args.stage!="cache":parser.error("--max-cases is cache-only; training always uses all development cases")
    if args.stage!="cache" and args.partition!="development":parser.error("--partition selects the cache stage only")
    try:
        root=args.project_root.resolve() if args.project_root else discover_project_root()
        protocol=verified_protocol(root)
        with execution_lock(root/"outputs/full489"):
            if args.stage=="prepare":
                output,plan=prepare(root,protocol)
                print(f"Block 65 protocol freeze PASSED: {plan['experiment_sha256']}")
                print(f"Development: {len(plan['partition']['development']['case_ids'])}; validation: {len(plan['partition']['validation']['case_ids'])}")
                print(f"Report directory: {output}")
                return 0
            state,archive=dispatch(root,protocol,stage=args.stage,partition=args.partition,environment=args.environment,
                                   epochs=args.epochs,max_cases=args.max_cases)
        print(f"Status: {state['status']}")
        if "completed_epochs" in state:
            print(f"Completed epochs: {state['completed_epochs']}/{state['planned_epochs']}; optimizer steps: {state['optimizer_steps']}")
            print(f"Selected epoch: {state['selection']['best_epoch']}; validation score: {state['selection']['best_score']}")
        if "completed_case_ids" in state:print(f"Cached: {len(state['completed_case_ids'])}/{len(state['requested_case_ids'])}")
        if "error" in state:print(f"FAILED: {state['error']}")
        print(f"Share this ZIP: {archive}")
        print("Validation is used for model selection. Test120 remains sealed.")
        return 1 if state["status"]=="failed" else 0
    except KeyboardInterrupt:
        print("Interrupted. Rerun: committed epochs are preserved; an unfinished epoch repeats.",file=sys.stderr);return 130
    except Exception as exc:
        print(f"Block 65 FAILED: {type(exc).__name__}: {exc}",file=sys.stderr);return 1


if __name__=="__main__":raise SystemExit(main())
