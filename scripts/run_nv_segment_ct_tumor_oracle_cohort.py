from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_oracle_cohort import (
    load_all_frozen_prompts,
    summarize_tumor_oracle_cohort,
    write_cohort_outputs,
    write_markdown_report,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    run_tumor_oracle_smoke_case,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen nine-case tumor oracle "
            "special-positive-point cohort."
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
    output_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_oracle_point"
        / "cohort"
    )

    prompts = load_all_frozen_prompts(
        protocol_path
    )
    nv_paths = runtime_paths(root)

    print("NV-Segment-CT tumor oracle cohort")
    print("Cases: 9")
    print("Protocol: one frozen GT-derived point per case")
    print("point_labels=[3], label_prompt=None")
    print("Oracle status: not deployment-valid")
    print()

    rows = []

    for index, prompt in enumerate(
        prompts,
        start=1,
    ):
        print(
            f"[{index:02d}/09] "
            f"{prompt.case_id} "
            f"point={prompt.point_xyz}"
        )

        result = run_tumor_oracle_smoke_case(
            prompt=prompt,
            project_root=root,
            nv_paths=nv_paths,
            environment_name=args.environment,
            force=args.force,
        )
        rows.append(result)

        print(
            f"  tumor Dice="
            f"{result.interactive_target_dice:.4f}, "
            f"coverage gain="
            f"{100.0 * result.tumor_coverage_gain:+.1f} pp"
        )
        print(
            f"  HEC Dice delta="
            f"{result.hec_dice_delta:+.4f}, "
            f"Surface delta="
            f"{result.hec_surface_dice_delta:+.4f}, "
            f"HD95 reduction="
            f"{result.hec_hd95_reduction_mm:+.2f} mm"
        )
        print(
            f"  runtime={result.wall_time_seconds:.2f} s, "
            f"labels={result.interactive_labels_present}"
        )

    summary = summarize_tumor_oracle_cohort(
        rows
    )

    write_cohort_outputs(
        rows=rows,
        summary=summary,
        output_root=output_root,
    )
    write_markdown_report(
        rows=rows,
        summary=summary,
        path=output_root / "cohort_report.md",
    )

    print()
    print("Tumor oracle cohort summary")
    print(
        "Largest-component tumor Dice: "
        f"mean={summary.interactive_target_dice_mean:.4f}, "
        f"median={summary.interactive_target_dice_median:.4f}"
    )
    print(
        "Full-tumor Dice: "
        f"mean={summary.interactive_full_tumor_dice_mean:.4f}, "
        f"median={summary.interactive_full_tumor_dice_median:.4f}"
    )
    print(
        "Tumor coverage gain: "
        f"mean={100.0 * summary.tumor_coverage_gain_mean:+.1f} pp, "
        f"median={100.0 * summary.tumor_coverage_gain_median:+.1f} pp"
    )
    print(
        "HEC Dice delta: "
        f"mean={summary.hec_dice_delta_mean:+.4f}, "
        f"median={summary.hec_dice_delta_median:+.4f}"
    )
    print(
        "HEC Surface Dice delta: "
        f"mean={summary.hec_surface_dice_delta_mean:+.4f}, "
        f"median={summary.hec_surface_dice_delta_median:+.4f}"
    )
    print(
        "HEC HD95 reduction: "
        f"mean={summary.hec_hd95_reduction_mean_mm:+.2f} mm, "
        f"median={summary.hec_hd95_reduction_median_mm:+.2f} mm"
    )
    print(
        f"HD95 improved: "
        f"{summary.hd95_improved_case_count}/9"
    )
    print(
        f"HD95 reduced >5 mm: "
        f"{summary.hd95_reduced_gt_5mm_count}/9"
    )
    print(
        f"Tumor Dice >=0.50: "
        f"{summary.target_dice_ge_050_count}/9"
    )
    print(
        f"Tumor Dice >=0.80: "
        f"{summary.target_dice_ge_080_count}/9"
    )
    print(
        f"HEC Dice improved: "
        f"{summary.dice_improved_case_count}/9"
    )
    print(
        f"HEC Surface Dice improved: "
        f"{summary.surface_dice_improved_case_count}/9"
    )
    print()
    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(f"MD:   {output_root / 'cohort_report.md'}")
    print(
        "Tumor oracle cohort PASSED "
        "(oracle upper-bound; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
