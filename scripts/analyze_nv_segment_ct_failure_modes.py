from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.cohort import load_case_cohort_row
from vista3d_tfg.evaluation.failure_decomposition import (
    decompose_case_failure,
    render_failure_figures,
    summarize_failure_decomposition,
    write_failure_csv,
    write_failure_json,
)
from vista3d_tfg.experiments.pilot30 import PILOT30_CASES, pilot_case_paths


def main() -> int:
    project_root = discover_project_root()
    cases = []

    print("NV-Segment-CT pilot30 failure decomposition")

    for case_number in PILOT30_CASES:
        paths = pilot_case_paths(project_root, case_number)
        row = load_case_cohort_row(
            case_id=paths.case_id,
            evaluation_path=paths.evaluation_json,
            inference_path=paths.inference_json,
            qc_path=paths.comparison_qc_json,
        )
        cases.append(
            decompose_case_failure(
                row=row,
                ground_truth_path=paths.segmentation_path,
                prediction_path=paths.prediction_path,
            )
        )

    summary = summarize_failure_decomposition(cases)
    output_root = (
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_pilot30"
        / "failure_decomposition"
    )

    json_path = output_root / "failure_decomposition.json"
    csv_path = output_root / "failure_cases.csv"

    write_failure_json(summary, json_path)
    write_failure_csv(summary, csv_path)
    figures = render_failure_figures(summary, output_root / "figures")

    print(f"Cases: {summary.case_count}")
    print()
    print("Renal envelope error components:")
    print(
        f"  FP farther than 20 mm: "
        f"{summary.cases_with_fp_farther_than_20mm}/{summary.case_count}"
    )
    print(
        f"  FP farther than 50 mm: "
        f"{summary.cases_with_fp_farther_than_50mm}/{summary.case_count}"
    )
    print(
        "  HD95 vs farthest FP correlation: "
        + (
            f"{summary.correlation_hd95_vs_farthest_fp_distance:.4f}"
            if summary.correlation_hd95_vs_farthest_fp_distance is not None
            else "undefined"
        )
    )
    print(
        "  HD95 vs farthest FN correlation: "
        + (
            f"{summary.correlation_hd95_vs_farthest_fn_distance:.4f}"
            if summary.correlation_hd95_vs_farthest_fn_distance is not None
            else "undefined"
        )
    )
    print(
        "  Worst far-FP cases: "
        + ", ".join(summary.worst_farthest_fp_case_ids)
    )

    print()
    print("Cyst localization:")
    print(f"  Cyst-positive cases: {summary.cyst_positive_case_count}")
    print(
        f"  Zero voxel-overlap cases: "
        f"{summary.cyst_positive_zero_overlap_cases}/"
        f"{summary.cyst_positive_case_count}"
    )
    print(f"  GT cyst components: {summary.cyst_gt_component_count}")
    print(
        "  Localization recall <=10 mm: "
        + (
            f"{summary.cyst_component_localization_recall_10mm:.4f}"
            if summary.cyst_component_localization_recall_10mm is not None
            else "undefined"
        )
    )
    print(
        "  Localization recall <=20 mm: "
        + (
            f"{summary.cyst_component_localization_recall_20mm:.4f}"
            if summary.cyst_component_localization_recall_20mm is not None
            else "undefined"
        )
    )
    print(
        "  Localization recall <=30 mm: "
        + (
            f"{summary.cyst_component_localization_recall_30mm:.4f}"
            if summary.cyst_component_localization_recall_30mm is not None
            else "undefined"
        )
    )
    print(
        "  Median nearest-centroid distance: "
        + (
            f"{summary.cyst_nearest_centroid_distance_median_mm:.2f} mm"
            if summary.cyst_nearest_centroid_distance_median_mm is not None
            else "undefined"
        )
    )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    for path in figures:
        print(f"Figure: {path}")

    print("Pilot30 failure decomposition PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
