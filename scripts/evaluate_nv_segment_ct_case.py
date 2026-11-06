from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.evaluation.nv_kits23 import (
    evaluate_supported_renal_prediction,
    write_supported_renal_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate supported NV-Segment renal structures against KiTS23."
        )
    )
    parser.add_argument("--case", type=int, default=0)
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
            f"Prediction not found: {prediction_path}. "
            "Run Block 09 inference first."
        )

    result = evaluate_supported_renal_prediction(
        case_id=case.case_id,
        ground_truth_path=case.segmentation_path,
        prediction_path=prediction_path,
    )

    output_path = (
        project_root
        / "outputs"
        / "evaluations"
        / "nv_segment_ct"
        / case_id
        / "supported_renal_metrics.json"
    )
    write_supported_renal_comparison(result, output_path)

    print("NV-Segment-CT vs KiTS23 supported renal evaluation")
    print(f"Case: {result.case_id}")
    print()
    print("Official KiTS23 HEC component:")
    print(
        "  kidney_and_mass "
        f"Dice={result.kidney_and_mass_hec.dice:.4f}, "
        f"SurfaceDice={result.kidney_and_mass_hec.surface_dice:.4f}, "
        f"HD95={result.kidney_and_mass_hec.diagnostic_hd95_mm:.2f} mm"
    )
    print()
    print("Diagnostic comparisons:")
    print(
        "  kidney core     "
        f"Dice={result.kidney_core_diagnostic.dice:.4f}, "
        f"volume error="
        f"{result.kidney_core_diagnostic.signed_volume_error_ml:+.2f} mL"
    )
    print(
        "  cyst            "
        f"Dice={result.cyst.dice:.4f}, "
        f"GT={result.cyst.reference_volume_ml:.2f} mL, "
        f"Pred={result.cyst.prediction_volume_ml:.2f} mL"
    )
    print(
        "  left envelope   "
        f"Dice={result.left_renal_envelope.dice:.4f}, "
        f"volume error="
        f"{result.left_renal_envelope.signed_volume_error_ml:+.2f} mL"
    )
    print(
        "  right envelope  "
        f"Dice={result.right_renal_envelope.dice:.4f}, "
        f"volume error="
        f"{result.right_renal_envelope.signed_volume_error_ml:+.2f} mL"
    )
    print()
    print(
        f"GT tumor volume: {result.gt_tumor_volume_ml:.2f} mL "
        "(reported, not scored by automatic NV-Segment tumor output)"
    )
    print("Full KiTS23 score: unavailable by protocol")
    print(f"Output: {output_path}")
    print("Supported renal evaluation PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
