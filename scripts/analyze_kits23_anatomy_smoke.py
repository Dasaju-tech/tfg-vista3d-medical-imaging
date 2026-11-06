from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.analysis.anatomy import (
    analyze_kits23_anatomy_nifti,
    write_anatomy_results,
)
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case


DEFAULT_CASES = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze renal units, lesion components, and conservative "
            "laterality on the real KiTS23 smoke subset."
        )
    )
    parser.add_argument("--project-root", type=Path, default=None)
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

        result = analyze_kits23_anatomy_nifti(
            case_id=case.case_id,
            segmentation_path=case.segmentation_path,
        )
        results.append(result)

        print(f"Analyzing {case_id} ...")
        print(
            f"  kidney_and_mass_components="
            f"{result.kidney_and_mass_components}, "
            f"laterality_resolved={result.is_laterality_resolved}"
        )

        for side in result.side_summaries:
            print(
                f"  {side.laterality:12s} "
                f"renal={side.renal_volume_ml:8.2f} mL, "
                f"tumors={side.tumor_components} "
                f"({side.tumor_volume_ml:.2f} mL), "
                f"cysts={side.cyst_components} "
                f"({side.cyst_volume_ml:.2f} mL)"
            )

        if result.qc_flags:
            print("  QC:", ", ".join(result.qc_flags))
        else:
            print("  QC: none")

    output_path = (
        project_root
        / "outputs"
        / "anatomy"
        / "kits23_smoke_anatomy.json"
    )
    write_anatomy_results(results, output_path)

    unresolved = sum(
        not result.is_laterality_resolved for result in results
    )

    print()
    print(f"Output: {output_path}")
    print(
        f"Resolved laterality: {len(results) - unresolved}/"
        f"{len(results)} cases"
    )
    print("KiTS23 anatomy smoke analysis PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
