from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.cohort import (
    load_case_cohort_row,
    summarize_cohort,
    write_cohort_csv,
    write_cohort_json,
)


DEFAULT_CASES = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the real NV-Segment-CT KiTS23 smoke cohort."
        )
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
    project_root = discover_project_root()

    rows = []

    for number in args.cases:
        case_id = f"case_{number:05d}"

        evaluation_path = (
            project_root
            / "outputs"
            / "evaluations"
            / "nv_segment_ct"
            / case_id
            / "supported_renal_metrics.json"
        )
        inference_path = (
            project_root
            / "outputs"
            / "predictions"
            / "nv_segment_ct"
            / case_id
            / "inference.json"
        )
        qc_path = (
            project_root
            / "outputs"
            / "qc"
            / "nv_segment_ct_comparison"
            / case_id
            / "comparison_qc.json"
        )

        row = load_case_cohort_row(
            case_id=case_id,
            evaluation_path=evaluation_path,
            inference_path=inference_path,
            qc_path=qc_path,
        )
        rows.append(row)

    summary = summarize_cohort(rows)

    output_root = (
        project_root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_smoke"
    )
    json_path = output_root / "summary.json"
    csv_path = output_root / "cases.csv"

    write_cohort_json(summary, json_path)
    write_cohort_csv(summary, csv_path)

    print("NV-Segment-CT KiTS23 smoke cohort")
    print(
        f"Cases: {summary.case_count} "
        f"({', '.join(summary.case_ids)})"
    )
    print()
    print("Kidney-and-mass HEC component:")
    print(
        "  Dice        "
        f"mean={summary.kidney_and_mass_dice.mean:.4f}, "
        f"median={summary.kidney_and_mass_dice.median:.4f}, "
        f"std={summary.kidney_and_mass_dice.std:.4f}"
    )
    print(
        "  SurfaceDice "
        f"mean={summary.kidney_and_mass_surface_dice.mean:.4f}, "
        f"median={summary.kidney_and_mass_surface_dice.median:.4f}, "
        f"std={summary.kidney_and_mass_surface_dice.std:.4f}"
    )
    print(
        "  HD95        "
        f"mean={summary.kidney_and_mass_hd95_mm.mean:.2f} mm, "
        f"median={summary.kidney_and_mass_hd95_mm.median:.2f} mm"
    )
    print()
    print("Side-aware diagnostics:")
    print(
        "  Left Dice   "
        f"mean={summary.left_renal_dice.mean:.4f}"
    )
    print(
        "  Right Dice  "
        f"mean={summary.right_renal_dice.mean:.4f}"
    )
    print()
    print("Cyst analysis:")
    print(
        "  all-case Dice "
        f"mean={summary.cyst_dice_all_cases.mean:.4f}"
    )

    if summary.cyst_dice_cyst_positive_cases is not None:
        print(
            "  cyst-positive-only Dice "
            f"mean={summary.cyst_dice_cyst_positive_cases.mean:.4f} "
            f"(n={summary.cyst_positive_case_count})"
        )
        print(
            "  cyst-positive GT volume "
            f"mean="
            f"{summary.cyst_reference_volume_ml_positive_cases.mean:.2f} mL"
        )
        print(
            "  cyst-positive predicted volume "
            f"mean="
            f"{summary.cyst_prediction_volume_ml_positive_cases.mean:.2f} mL"
        )
    else:
        print("  No cyst-positive cases in this cohort.")

    if summary.inference_wall_time_seconds is not None:
        print()
        print(
            "Inference wall time: "
            f"mean={summary.inference_wall_time_seconds.mean:.2f} s, "
            f"median={summary.inference_wall_time_seconds.median:.2f} s"
        )

    print()
    print("Full KiTS23 score: unavailable by protocol")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("Smoke cohort summary PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
