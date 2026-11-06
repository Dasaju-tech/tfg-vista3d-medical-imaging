from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    load_frozen_prompt,
    run_tumor_oracle_smoke_case,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


SMOKE_CASE = "case_00430"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Block 41B single-case tumor oracle "
            "special-positive-point smoke experiment."
        )
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = discover_project_root()

    protocol_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_oracle_point"
        / "protocol.json"
    )

    prompt = load_frozen_prompt(
        protocol_path,
        case_id=SMOKE_CASE,
    )
    nv_paths = runtime_paths(root)

    print("NV-Segment-CT tumor oracle smoke")
    print(f"Case: {prompt.case_id}")
    print(f"Frozen point: {prompt.point_xyz}")
    print("Point label: 3")
    print("Label prompt: none")
    print(
        "Oracle status: GT-derived localization; "
        "not deployment-valid"
    )
    print()

    result = run_tumor_oracle_smoke_case(
        prompt=prompt,
        project_root=root,
        nv_paths=nv_paths,
        environment_name=args.environment,
        force=args.force,
    )

    print("Interactive tumor capability")
    print(
        f"  largest-component Dice: "
        f"{result.interactive_target_dice:.4f}"
    )
    print(
        f"  full-tumor Dice: "
        f"{result.interactive_full_tumor_dice:.4f}"
    )
    print(
        f"  interactive volume: "
        f"{result.interactive_prediction_volume_ml:.2f} mL"
    )
    print(
        "  tumor coverage: "
        f"{100.0 * result.original_tumor_coverage_fraction:.1f}%"
        " -> "
        f"{100.0 * result.merged_tumor_coverage_fraction:.1f}% "
        f"(gain="
        f"{100.0 * result.tumor_coverage_gain:+.1f} pp)"
    )
    print()
    print("Kidney-and-mass HEC sensitivity")
    print(
        f"  Dice: {result.original_hec_dice:.4f} -> "
        f"{result.merged_hec_dice:.4f} "
        f"(delta={result.hec_dice_delta:+.4f})"
    )
    print(
        f"  Surface Dice: "
        f"{result.original_hec_surface_dice:.4f} -> "
        f"{result.merged_hec_surface_dice:.4f} "
        f"(delta={result.hec_surface_dice_delta:+.4f})"
    )
    print(
        f"  HD95: {result.original_hec_hd95_mm:.2f} -> "
        f"{result.merged_hec_hd95_mm:.2f} mm "
        f"(reduction={result.hec_hd95_reduction_mm:+.2f} mm)"
    )
    print()
    print(
        f"Interactive labels: "
        f"{result.interactive_labels_present}"
    )
    print(
        f"Contains 255/unprocessed: "
        f"{result.contains_255_unprocessed}"
    )
    print(
        f"Inference wall time: "
        f"{result.wall_time_seconds:.2f} s"
    )
    print(f"Prediction: {result.prediction_path}")
    print()
    print(
        "Tumor oracle smoke PASSED "
        "(oracle upper-bound; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
