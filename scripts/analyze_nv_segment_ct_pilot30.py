from __future__ import annotations

from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.cohort import load_case_cohort_row
from vista3d_tfg.evaluation.pilot_analysis import (
    analyze_pilot30,
    render_pilot_figures,
    write_analysis_json,
    write_ranked_cases_csv,
)
from vista3d_tfg.experiments.pilot30 import PILOT30_CASES, pilot_case_paths


def main() -> int:
    project_root = discover_project_root()

    rows = []

    for case_number in PILOT30_CASES:
        paths = pilot_case_paths(project_root, case_number)

        rows.append(
            load_case_cohort_row(
                case_id=paths.case_id,
                evaluation_path=paths.evaluation_json,
                inference_path=paths.inference_json,
                qc_path=paths.comparison_qc_json,
            )
        )

    analysis = analyze_pilot30(rows)

    output_root = (
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_pilot30"
    )

    write_analysis_json(
        analysis,
        output_root / "statistical_audit.json",
    )
    write_ranked_cases_csv(
        rows,
        output_root / "ranked_cases.csv",
    )
    figure_paths = render_pilot_figures(
        rows,
        output_root / "figures",
    )

    print("NV-Segment-CT pilot30 statistical audit")
    print(f"Cases: {analysis.case_count}")
    print(
        f"Cyst-positive: {analysis.cyst_positive_count}, "
        f"cyst-negative: {analysis.cyst_negative_count}"
    )
    print()
    print(
        "Kidney-and-mass Dice: "
        f"mean={analysis.kidney_and_mass_dice_mean:.4f}, "
        f"median={analysis.kidney_and_mass_dice_median:.4f}, "
        "bootstrap 95% CI="
        f"[{analysis.kidney_and_mass_dice_bootstrap_95ci.lower:.4f}, "
        f"{analysis.kidney_and_mass_dice_bootstrap_95ci.upper:.4f}]"
    )
    print(
        "Surface Dice: "
        f"mean={analysis.surface_dice_mean:.4f}, "
        f"median={analysis.surface_dice_median:.4f}, "
        "bootstrap 95% CI="
        f"[{analysis.surface_dice_bootstrap_95ci.lower:.4f}, "
        f"{analysis.surface_dice_bootstrap_95ci.upper:.4f}]"
    )
    print(
        "HD95: "
        f"mean={analysis.hd95_mean_mm:.2f} mm, "
        f"median={analysis.hd95_median_mm:.2f} mm, "
        "bootstrap 95% CI="
        f"[{analysis.hd95_bootstrap_95ci_mm.lower:.2f}, "
        f"{analysis.hd95_bootstrap_95ci_mm.upper:.2f}] mm"
    )

    if analysis.cyst_positive_dice_mean is not None:
        print()
        print(
            "Cyst-positive Dice: "
            f"mean={analysis.cyst_positive_dice_mean:.4f}, "
            f"median={analysis.cyst_positive_dice_median:.4f}, "
            "bootstrap 95% CI="
            f"[{analysis.cyst_positive_dice_bootstrap_95ci.lower:.4f}, "
            f"{analysis.cyst_positive_dice_bootstrap_95ci.upper:.4f}]"
        )

    if analysis.kidney_dice_cyst_status_comparison is not None:
        comparison = analysis.kidney_dice_cyst_status_comparison
        print()
        print(
            "Kidney Dice by cyst status: "
            f"{comparison.group_a_name}="
            f"{comparison.group_a_mean:.4f} (n={comparison.group_a_count}), "
            f"{comparison.group_b_name}="
            f"{comparison.group_b_mean:.4f} (n={comparison.group_b_count}), "
            f"difference={comparison.mean_difference_a_minus_b:+.4f}"
        )

    print()
    print(
        "Worst Surface Dice: "
        + ", ".join(analysis.worst_surface_dice_case_ids)
    )
    print(
        "Worst HD95: "
        + ", ".join(analysis.worst_hd95_case_ids)
    )
    print(
        "Largest renal under-segmentation: "
        + ", ".join(
            analysis.largest_undersegmentation_case_ids
        )
    )

    print()
    print(
        "SurfaceDice/HD95 correlation: "
        + (
            f"{analysis.correlation_surface_dice_hd95:.4f}"
            if analysis.correlation_surface_dice_hd95 is not None
            else "undefined"
        )
    )
    print(
        "GT/predicted cyst-volume correlation: "
        + (
            f"{analysis.correlation_cyst_gt_vs_prediction_volume:.4f}"
            if analysis.correlation_cyst_gt_vs_prediction_volume
            is not None
            else "undefined"
        )
    )

    if analysis.interpretation_flags:
        print()
        print(
            "Flags: "
            + ", ".join(analysis.interpretation_flags)
        )

    print()
    print(f"JSON: {output_root / 'statistical_audit.json'}")
    print(f"CSV:  {output_root / 'ranked_cases.csv'}")

    for path in figure_paths:
        print(f"Figure: {path}")

    print("Pilot30 statistical audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
