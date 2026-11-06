from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

from vista3d_tfg.analysis.full489_cpu_adoption import completed_benchmark, prepare_adoption, run_adoption
from vista3d_tfg.analysis.full489_development import collect_development
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, read_json, verified_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 62: versioned CPU evaluator parity on all frozen development cases.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--benchmark-id", required=True, help="Completed Block 61 plan directory prefix.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--max-cases", type=int, help="New case attempts; verified completions do not count.")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r"[0-9a-f]{16}", v) for v in (args.run_id, args.benchmark_id)):
        parser.error("Identifiers must each contain 16 lowercase hexadecimal characters")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        run_root = root / "outputs/full489" / args.run_id
        benchmark_root = root / "outputs/reporting/full489_block61" / args.run_id / args.benchmark_id
        with execution_lock(root / "outputs/full489"):
            contract = read_json(run_root / "contract.json")
            if contract["contract_sha256"][:16] != args.run_id:
                raise ValueError("Run directory and contract disagree.")
            rows, audit = collect_development(run_root, contract, protocol)
            if audit["status"] != "complete":
                raise ValueError(f"Development source audit failed: {audit['failed_cases']}")
            evidence = completed_benchmark(root, run_root, benchmark_root, rows, contract)
            if evidence["benchmark_plan_sha256"][:16] != args.benchmark_id:
                raise ValueError("Block 61 directory and plan disagree.")
            output, plan = prepare_adoption(root, rows, contract, protocol, evidence)
            print(f"Block 62 plan: {plan['plan_sha256']}")
            print(f"Development cases: {len(plan['case_ids'])}; evaluator: {plan['implementation']['evaluator_version']}")
            print(f"Output: {output}")
            state, archive = run_adoption(root, run_root, output, plan, contract, protocol,
                                         max_cases=args.max_cases, continue_on_error=args.continue_on_error)
        print(f"Status: {state['status']}")
        print(f"Validated: {len(state['completed_case_ids'])}/{len(plan['case_ids'])}")
        print(f"Exact metric matches: {state['exact_metric_matches']}")
        print(f"Failed: {len(state['failed_cases'])}; deferred: {len(state['deferred_case_ids'])}")
        print(f"Share this ZIP: {archive}")
        print("GPU inference: none. Validation/test results remain reserved.")
        return 1 if state["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed cases are preserved; rerun to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 62 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
