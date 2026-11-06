"""Block 65b: a traceable AMP recovery from the last complete parent epoch."""
import argparse
from pathlib import Path
import sys

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, verified_protocol
from vista3d_tfg.training import contract65 as c, contract65b as b
from vista3d_tfg.experiments.supervised65b import dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "train", "report"))
    parser.add_argument("--until-epoch", type=int, default=10,
                        help="Absolute target, not additional epochs; safe to repeat after interruption.")
    parser.add_argument("--environment", default="vista3d-nv")
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.until_epoch <= 50: parser.error("Target epoch must be between 1 and 50.")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        with execution_lock(root/"outputs/full489"):
            if args.stage == "prepare":
                plan = b.freeze(root, protocol)
                print(f"Block 65b continuation freeze PASSED: {plan['experiment_sha256']}")
                print(f"Resume from parent epoch: {plan['parent_last_epoch']}; next: {plan['parent_last_epoch']+1}")
                print(f"Report directory: {c.output_root(root, plan)}")
                print("Parent checkpoints/cache preserved. Test120 remains sealed.")
                return 0
            state, archive = dispatch(root, protocol, stage=args.stage,
                                      environment=args.environment, until_epoch=args.until_epoch)
        print(f"Status: {state['status']}")
        if "completed_epochs" in state:
            print(f"Completed epochs: {state['completed_epochs']}/{state['planned_epochs']}; optimizer steps: {state['optimizer_steps']}")
            print(f"Selected epoch: {state['selection']['best_epoch']}; validation score: {state['selection']['best_score']}")
        if "error" in state: print(f"FAILED: {state['error']}")
        print(f"Share this ZIP: {archive}")
        print("Validation is used for model selection. Test120 remains sealed.")
        return 1 if state["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Interrupted; repeat the same --until-epoch command.", file=sys.stderr); return 130
    except Exception as exc:
        print(f"Block 65b FAILED: {type(exc).__name__}: {exc}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
