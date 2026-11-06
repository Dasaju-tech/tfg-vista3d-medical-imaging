from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.visualization.nv_compare import (
    build_case_comparison_qc,
    write_comparison_qc_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render deterministic NV-Segment prediction-versus-KiTS23 "
            "quality-control figures."
        )
    )
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        metavar=("LOWER_HU", "UPPER_HU"),
        default=(-200.0, 300.0),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    case_id = f"case_{args.case:05d}"

    case = discover_case(
        project_root
        / "data"
        / "raw"
        / "kits23"
        / case_id
    )

    prediction_path = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
        / case_id
        / "prediction.nii.gz"
    )

    if not prediction_path.is_file():
        raise FileNotFoundError(
            f"Prediction not found: {prediction_path}"
        )

    output_directory = (
        project_root
        / "outputs"
        / "qc"
        / "nv_segment_ct_comparison"
    )

    result = build_case_comparison_qc(
        case_id=case.case_id,
        image_path=case.imaging_path,
        ground_truth_path=case.segmentation_path,
        prediction_path=prediction_path,
        output_directory=output_directory,
        window_hu=tuple(args.window),
    )

    index_path = (
        output_directory
        / case.case_id
        / "comparison_qc.json"
    )
    write_comparison_qc_index(result, index_path)

    print("NV-Segment prediction-vs-GT visual QC")
    print(f"Case:        {result.case_id}")
    print(f"Orientation: {''.join(result.orientation)}")
    print(
        f"Errors:      FP={result.total_false_positive_voxels}, "
        f"FN={result.total_false_negative_voxels}, "
        f"total={result.total_disagreement_voxels}"
    )

    for selection, path in zip(
        result.selected_slices,
        result.figure_files,
        strict=True,
    ):
        print(
            f"  {selection.plane:8s} "
            f"slice={selection.index:4d} "
            f"source={selection.source:24s} "
            f"errors={selection.disagreement_voxels:6d}"
        )
        print(f"    {path}")

    print(f"QC index:    {index_path}")
    print("Prediction-vs-GT visual QC PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
