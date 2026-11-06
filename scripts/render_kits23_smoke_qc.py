from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.visualization.qc import (
    DEFAULT_CT_WINDOW_HU,
    build_case_qc,
    write_qc_index,
)


DEFAULT_CASES = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic visual QC figures for the real KiTS23 "
            "smoke subset."
        )
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
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("LOWER_HU", "UPPER_HU"),
        default=list(DEFAULT_CT_WINDOW_HU),
        help="CT display window in HU. Default: -160 240.",
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
    output_root = project_root / "outputs" / "qc"

    results = []

    for number in args.cases:
        case_id = f"case_{number:05d}"
        case = discover_case(dataset_root / case_id)

        print(f"Rendering {case_id} ...")

        result = build_case_qc(
            case_id=case.case_id,
            imaging_path=case.imaging_path,
            segmentation_path=case.segmentation_path,
            output_directory=output_root,
            window_hu=args.window,
        )
        results.append(result)

        print(
            f"  orientation={''.join(result.canonical_orientation)}, "
            f"components={result.kidney_and_mass_components}"
        )

        for selection in result.selected_slices:
            print(
                f"  {selection.plane:8s} "
                f"slice={selection.index:4d} "
                f"source={selection.source}"
            )

    index_path = output_root / "kits23_smoke_qc_index.json"
    write_qc_index(results, index_path)

    print()
    print(f"QC index: {index_path}")
    print("KiTS23 visual QC PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
