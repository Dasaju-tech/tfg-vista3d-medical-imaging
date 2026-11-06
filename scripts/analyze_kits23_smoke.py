from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.analysis.kits23_baseline import (
    analyze_kits23_case,
    write_case_baselines,
    write_case_table_csv,
    write_cohort_summary,
)
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case


DEFAULT_CASES = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute quantitative KiTS23 ground-truth baselines."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=list(DEFAULT_CASES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else discover_project_root()
    )

    dataset_root = project_root / "data" / "raw" / "kits23"

    results = []

    for number in args.cases:
        case_id = f"case_{number:05d}"
        case = discover_case(dataset_root / case_id)

        print(f"Analyzing {case_id} ...")

        result = analyze_kits23_case(
            case_id=case.case_id,
            imaging_path=case.imaging_path,
            segmentation_path=case.segmentation_path,
        )
        results.append(result)

        print(
            "  "
            f"kidney={result.region('kidney').volume_ml:.2f} mL, "
            f"tumor={result.region('tumor').volume_ml:.2f} mL, "
            f"cyst={result.region('cyst').volume_ml:.2f} mL, "
            f"tumor_components={result.region('tumor').component_count}, "
            f"tumor_burden={100.0 * result.tumor_burden_fraction:.2f}%"
        )

    output_root = project_root / "outputs" / "baselines"

    cases_json = output_root / "kits23_smoke_cases.json"
    cohort_json = output_root / "kits23_smoke_cohort.json"
    table_csv = output_root / "kits23_smoke_table.csv"

    write_case_baselines(results, cases_json)
    write_cohort_summary(results, cohort_json)
    write_case_table_csv(results, table_csv)

    print()
    print(f"Case metrics:   {cases_json}")
    print(f"Cohort summary: {cohort_json}")
    print(f"CSV table:      {table_csv}")
    print("KiTS23 quantitative baseline PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
