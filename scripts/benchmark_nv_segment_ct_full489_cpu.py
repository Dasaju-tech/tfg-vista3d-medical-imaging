from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from vista3d_tfg.analysis.full489_cpu import check_completed_review, prepare_benchmark, run_benchmark
from vista3d_tfg.analysis.full489_development import collect_development
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, read_json, verified_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 61: paired development-only CPU evaluation benchmark.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--review-id", required=True, help="Completed Block 60 plan directory prefix.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--repeats", type=int, default=2, help="Even number of paired repetitions per case (default: 2).")
    parser.add_argument("--max-cases", type=int, help="New case attempts; completed verified benchmarks do not count.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[0-9a-f]{16}", v) for v in (args.run_id, args.review_id)):
        parser.error("--run-id and --review-id must each contain 16 lowercase hexadecimal characters")
    if args.repeats < 2 or args.repeats % 2:
        parser.error("--repeats must be even and at least 2")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        run_root = root / "outputs/full489" / args.run_id
        review_root = root / "outputs/reporting/full489_block60" / args.run_id / args.review_id
        with execution_lock(root / "outputs/full489"):
            contract = read_json(run_root / "contract.json")
            if contract["contract_sha256"][:16] != args.run_id:
                raise ValueError("Run directory and contract disagree.")
            rows, audit = collect_development(run_root, contract, protocol)
            if audit["status"] != "complete":
                raise ValueError(f"Development source audit failed: {audit['failed_cases']}")
            review = check_completed_review(review_root, rows, contract)
            if review["plan_sha256"][:16] != args.review_id:
                raise ValueError("Block 60 directory and review plan disagree.")
            output, plan = prepare_benchmark(root, contract, review, repeats=args.repeats)
            print(f"Block 61 plan: {plan['plan_sha256']}")
            print(f"Selected cases: {len(plan['case_ids'])}; paired repetitions: {plan['repeats']}")
            print(f"Output: {output}")
            state, archive = run_benchmark(root, run_root, output, plan, contract,
                                          max_cases=args.max_cases, continue_on_error=args.continue_on_error)
        print(f"Status: {state['status']}")
        print(f"Benchmarked: {len(state['completed_case_ids'])}/{len(plan['case_ids'])}")
        print(f"Failed: {len(state['failed_cases'])}; deferred: {len(state['deferred_case_ids'])}")
        print(f"Share this ZIP: {archive}")
        print("GPU inference: none. Validation/test results remain reserved.")
        return 1 if state["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed cases are preserved; the interrupted case restarts on resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 61 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
