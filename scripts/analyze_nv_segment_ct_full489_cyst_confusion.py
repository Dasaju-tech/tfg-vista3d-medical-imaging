from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from vista3d_tfg.analysis.full489_cyst_confusion import completed_validation, prepare_analysis, run_analysis
from vista3d_tfg.analysis.full489_development import collect_development
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, read_json, verified_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 63: annotation-relative cyst confusion across frozen development.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-id", required=True, help="Completed Block 62 plan directory prefix.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--max-cases", type=int, help="New attempts; verified cached cases do not count.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[0-9a-f]{16}", value) for value in (args.run_id, args.validation_id)):
        parser.error("Identifiers must contain 16 lowercase hexadecimal characters")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        run_root = root / "outputs/full489" / args.run_id
        validation_root = root / "outputs/reporting/full489_block62" / args.run_id / args.validation_id
        with execution_lock(root / "outputs/full489"):
            contract = read_json(run_root / "contract.json")
            if contract["contract_sha256"][:16] != args.run_id:
                raise ValueError("Run directory and contract disagree.")
            rows, audit = collect_development(run_root, contract, protocol)
            if audit["status"] != "complete":
                raise ValueError(f"Development source audit failed: {audit['failed_cases']}")
            evidence = completed_validation(root, run_root, validation_root, rows, contract, protocol)
            if evidence["validation_plan_sha256"][:16] != args.validation_id:
                raise ValueError("Block 62 directory and plan disagree.")
            output, plan = prepare_analysis(root, rows, contract, protocol, evidence)
            print(f"Block 63 plan: {plan['plan_sha256']}")
            print(f"Development cases: {len(plan['case_ids'])}; Output: {output}")
            state, archive = run_analysis(root, run_root, output, plan, contract, protocol,
                                          max_cases=args.max_cases, continue_on_error=args.continue_on_error)
        print(f"Status: {state['status']}")
        print(f"Analyzed: {len(state['completed_case_ids'])}/{len(plan['case_ids'])}")
        print(f"Failed: {len(state['failed_cases'])}; deferred: {len(state['deferred_case_ids'])}")
        print(f"Share this ZIP: {archive}")
        print("GPU inference / surface metric recomputation: none. Validation/test results remain reserved.")
        return 1 if state["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed cases are preserved; rerun to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 63 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
