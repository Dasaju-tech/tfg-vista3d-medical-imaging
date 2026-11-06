from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)
from vista3d_tfg.workflows.user_guided_cyst import (
    run_user_guided_cyst_segmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the GT-free user-guided renal-cyst workflow: a user "
            "supplies one voxel-space click, automatic renal masks select "
            "the side-specific cyst class, and NV-Segment-CT performs "
            "semantic+point segmentation."
        )
    )
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument(
        "--point",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        required=True,
        help="User click in original NIfTI voxel coordinates.",
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    nv_paths = runtime_paths(project_root)

    point_xyz = tuple(int(value) for value in args.point)

    result = run_user_guided_cyst_segmentation(
        project_root=project_root,
        case_number=args.case,
        point_xyz=point_xyz,
        nv_paths=nv_paths,
        environment_name=args.environment,
        force=args.force,
    )

    assignment = result.semantic_assignment

    print("NV-Segment-CT user-guided renal cyst segmentation")
    print(f"Case:       {result.case_id}")
    print(f"User point: {result.point_xyz}")
    print()
    print("Automatic semantic routing:")
    print(
        f"  side={assignment.laterality}, "
        f"class={assignment.class_label} "
        f"({assignment.class_name})"
    )
    print(
        f"  distance to left kidney:  "
        f"{assignment.distance_to_left_kidney_mm:.2f} mm"
    )
    print(
        f"  distance to right kidney: "
        f"{assignment.distance_to_right_kidney_mm:.2f} mm"
    )
    print(
        f"  laterality margin:        "
        f"{assignment.laterality_margin_mm:.2f} mm"
    )
    print()
    print(
        f"Prediction volume: "
        f"{result.foreground_volume_ml:.2f} mL "
        f"({result.foreground_voxels} voxels)"
    )
    print(
        f"Labels: {result.labels_present}, "
        f"contains_255={result.contains_255_unprocessed}"
    )
    print(f"Wall time: {result.wall_time_seconds:.2f} s")
    print(f"Prediction: {result.prediction_path}")
    print()
    print("Ground truth used by workflow: NO")
    print("Clinical validation: NO — research prototype")
    print("User-guided cyst workflow PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
