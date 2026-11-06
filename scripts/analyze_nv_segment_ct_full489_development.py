from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from uuid import uuid4

from vista3d_tfg.analysis.full489_development import collect_development, export_report
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, read_json, verified_protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Block 59: audit and summarize full489 development outputs.")
    parser.add_argument("--run-id", required=True, help="The 16-character contract directory printed by Block 58.")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{16}", args.run_id):
        parser.error("--run-id must contain exactly 16 lowercase hexadecimal characters")
    if args.bootstrap_reps < 1000 or args.seed < 0:
        parser.error("Use at least 1000 bootstrap repetitions and a nonnegative seed")
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        run_root = root / "outputs/full489" / args.run_id
        with execution_lock(root / "outputs/full489"):
            contract = read_json(run_root / "contract.json")
            if contract["contract_sha256"][:16] != args.run_id:
                raise ValueError("Run directory does not match the contract checksum prefix.")
            rows, audit = collect_development(run_root, contract, protocol)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
            output = root / "outputs/reporting/full489_development" / args.run_id / stamp
            summary = export_report(output, rows, audit, contract, reps=args.bootstrap_reps, seed=args.seed)
        print("Block 59 development analysis PASSED")
        for group, cohort in summary["cohorts"].items():
            metrics = cohort["metrics"]
            print(f"{group}: n={cohort['n']}; Dice mean={metrics['kidney_mass_dice']['mean']}; "
                  f"Surface Dice mean={metrics['surface_dice']['mean']}; "
                  f"HD95 finite median={metrics['hd95_mm']['median']}; infinite={metrics['hd95_mm']['n_nonfinite']}")
        print(f"Report directory: {output}")
        print(f"Share this ZIP: {output / 'block59_development_report.zip'}")
        print("GPU inference / metric recomputation / validation-test output reads: none.")
        return 0
    except KeyboardInterrupt:
        print("Analysis interrupted; inference outputs are preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Block 59 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
