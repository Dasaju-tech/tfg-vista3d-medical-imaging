from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from vista3d_tfg.analysis.full489_development import collect_development
from vista3d_tfg.analysis.full489_review import package_review, prepare_review, run_review
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import atomic_json, execution_lock, read_json, verified_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 60: development-only failure review and CPU profiling.")
    parser.add_argument("--stage", required=True, choices=("plan", "review"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--max-cases", type=int, help="New review attempts in this invocation; verified cache does not count.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{16}", args.run_id):
        parser.error("--run-id must contain 16 lowercase hexadecimal characters")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.stage == "plan" and (args.max_cases is not None or args.continue_on_error):
        parser.error("--max-cases and --continue-on-error apply to review only")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        run_root = root / "outputs/full489" / args.run_id
        with execution_lock(root / "outputs/full489"):
            contract = read_json(run_root / "contract.json")
            if contract["contract_sha256"][:16] != args.run_id:
                raise ValueError("Run directory and contract disagree.")
            rows, audit = collect_development(run_root, contract, protocol)
            if audit["status"] != "complete":
                path = root / "outputs/reporting/full489_block60" / args.run_id / "source_audit_failed.json"
                atomic_json(path, audit)
                raise ValueError(f"Source audit failed; see {path}")
            output, plan = prepare_review(root, rows, contract)
            print(f"Block 60 plan: {plan['plan_sha256']}")
            print(f"Selected development cases: {len(plan['case_ids'])}")
            print(f"Output: {output}")
            if args.stage == "review":
                state = run_review(root, run_root, output, plan, contract, max_cases=args.max_cases,
                                   continue_on_error=args.continue_on_error)
            else:
                state_path = output / "review_progress.json"
                state = read_json(state_path) if state_path.is_file() else {
                    "status": "planned", "plan_sha256": plan["plan_sha256"], "requested_case_ids": plan["case_ids"],
                    "completed_case_ids": [], "failed_cases": {}, "deferred_case_ids": plan["case_ids"], "gpu_inference": False}
                atomic_json(state_path, state)
            archive = package_review(output, plan, state)
        print(f"Status: {state['status']}")
        print(f"Reviewed: {len(state['completed_case_ids'])}/{len(plan['case_ids'])}")
        print(f"Failed: {len(state['failed_cases'])}; deferred: {len(state['deferred_case_ids'])}")
        print(f"Share this ZIP: {archive}")
        print("GPU inference: none. Validation/test results remain reserved.")
        return 1 if state["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed reviews are preserved; rerun to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 60 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
