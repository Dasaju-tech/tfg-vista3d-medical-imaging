from __future__ import annotations

import argparse
from pathlib import Path
import sys

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_acquisition import acquire_partition
from vista3d_tfg.experiments.full489_execution import (
    execute_batch, execution_lock, prepare_run, read_json,
    selected_cases, verified_protocol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 58: profile and resume frozen full489 runs.")
    parser.add_argument("--stage", required=True, choices=("probe", "profile", "acquire", "run"))
    parser.add_argument("--partition", default="development",
                        choices=("development", "validation", "test", "all"))
    parser.add_argument("--environment", default="vista3d-nv")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--max-cases", type=int,
                        help="Maximum new case attempts; verified cache entries do not consume this budget.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--retries", type=int, default=2, help="Download retries per file.")
    args = parser.parse_args(argv)
    # Check the requested operation before reading data, starting CUDA or downloading.
    if args.stage in ("probe", "profile") and args.partition != "development":
        parser.error("probe/profile use historical development cases only")
    if args.stage == "run" and args.partition not in ("development", "validation"):
        parser.error("Test inference/evaluation is sealed; test/all are acquisition-only")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        with execution_lock(root / "outputs/full489"):
            if args.stage == "acquire":
                state = acquire_partition(
                    root, protocol, args.partition, max_cases=args.max_cases,
                    continue_on_error=args.continue_on_error, retries=args.retries,
                )
                report_path = root / "outputs/full489" / f"acquisition_{args.partition}_progress.json"
            else:
                if args.stage == "run":
                    selected_cases(protocol, args.partition)
                run_root, contract, probe = prepare_run(root, protocol, args.environment)
                gpu = probe["identity"]
                print(f"GPU: {gpu['gpu_name']}; VRAM: {gpu['gpu_total_bytes'] / 2**30:.2f} GiB")
                print(f"CUDA: {gpu['cuda']}; PyTorch: {gpu['torch']}; MONAI: {gpu['monai']}")
                print(f"Contract: {contract['contract_sha256']}")
                print(f"Run directory: {run_root}")
                if args.stage == "probe":
                    print("CUDA FP32/FP16 probe PASSED. CT inference/evaluation: none.")
                    return 0
                state = execute_batch(
                    root, run_root, contract, protocol, args.environment,
                    phase=args.stage, partition=args.partition, max_cases=args.max_cases,
                    continue_on_error=args.continue_on_error,
                )
                label = "profile" if args.stage == "profile" else args.partition
                report_path = run_root / f"{label}_progress.json"
                if args.stage == "profile":
                    report_path = run_root / "profile_summary.json"
                    summary = read_json(report_path)
                    print(f"Median case seconds: {summary['median_case_seconds']}")
                    print(f"Maximum peak reserved GiB: {summary['maximum_peak_reserved_gib']}")
            print(f"Status: {state['status']}")
            print(f"Completed: {len(state['completed_case_ids'])}/{len(state['requested_case_ids'])}")
            print(f"Failed: {len(state['failed_cases'])}; deferred: {len(state['deferred_case_ids'])}")
            print(f"Report: {report_path}")
            print("Final test inference/evaluation remains sealed.")
            return 1 if state["status"] == "failed" else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed receipts are preserved; rerun the same command to resume.",
              file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 58 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
