from __future__ import annotations

import csv
import json
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_click_depth_audit import (
    DepthAuditCase,
    derive_depth_variant,
    summarize_depth_audit,
    write_depth_audit_outputs,
)
from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickRobustnessCase,
    derive_click_variants,
    run_click_variant,
)
from vista3d_tfg.experiments.tumor_oracle_cohort import (
    load_all_frozen_prompts,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def _write_case_csv(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "variant",
        "point_xyz",
        "tumor_dice",
        "tumor_coverage_gain",
        "hec_dice_delta",
        "hec_surface_dice_delta",
        "hec_hd95_reduction_mm",
        "wall_time_seconds",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            assert isinstance(row, dict)
            for variant in (
                "shallow_5mm",
                "depth_10mm",
                "depth_15mm",
                "centroid",
                "deep",
            ):
                result = row[variant]
                assert isinstance(result, dict)
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "variant": variant,
                        "point_xyz": ";".join(
                            str(v)
                            for v in result["point_xyz"]
                        ),
                        "tumor_dice": result[
                            "interactive_target_dice"
                        ],
                        "tumor_coverage_gain": result[
                            "tumor_coverage_gain"
                        ],
                        "hec_dice_delta": result[
                            "hec_dice_delta"
                        ],
                        "hec_surface_dice_delta": result[
                            "hec_surface_dice_delta"
                        ],
                        "hec_hd95_reduction_mm": result[
                            "hec_hd95_reduction_mm"
                        ],
                        "wall_time_seconds": result[
                            "wall_time_seconds"
                        ],
                    }
                )


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    variants = payload["variants"]
    robustness = payload["robustness_vs_deep"]
    assert isinstance(variants, dict)
    assert isinstance(robustness, dict)

    lines = [
        "# Tumor click depth audit",
        "",
        "## Objective",
        "",
        (
            "Quantify how far inside the tumor a single positive click "
            "must lie before the interactive segmentation becomes stable."
        ),
        "",
        "| Variant | Tumor Dice mean | Median | Dice >=0.50 | Dice >=0.70 | Coverage gain | HD95 reduction | HD95 improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in (
        "shallow_5mm",
        "depth_10mm",
        "depth_15mm",
        "centroid",
        "deep",
    ):
        item = variants[name]
        assert isinstance(item, dict)
        lines.append(
            f"| {name} "
            f"| {float(item['tumor_dice_mean']):.4f} "
            f"| {float(item['tumor_dice_median']):.4f} "
            f"| {item['tumor_dice_ge_050_count']}/9 "
            f"| {item['tumor_dice_ge_070_count']}/9 "
            f"| {100.0 * float(item['tumor_coverage_gain_mean']):+.1f} pp "
            f"| {float(item['hec_hd95_reduction_mean_mm']):+.2f} mm "
            f"| {item['hd95_improved_count']}/9 |"
        )

    lines.extend(
        [
            "",
            "## Robustness relative to deep oracle",
            "",
            "| Variant | Mean Dice delta | Within ±0.05 | Within ±0.10 |",
            "|---|---:|---:|---:|",
        ]
    )

    for name in (
        "shallow_5mm",
        "depth_10mm",
        "depth_15mm",
        "centroid",
    ):
        item = robustness[name]
        assert isinstance(item, dict)
        lines.append(
            f"| {name} "
            f"| {float(item['dice_delta_vs_deep_mean']):+.4f} "
            f"| {item['cases_within_0_05_dice_of_deep']}/9 "
            f"| {item['cases_within_0_10_dice_of_deep']}/9 |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "All click locations remain GT-simulated. This audit "
                "quantifies localization tolerance; it does not constitute "
                "an autonomous or deployment-valid tumor detector."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> int:
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
        / "nv_segment_ct_tumor_click_depth_audit"
    )
    dataset_root = (
        root
        / "data"
        / "raw"
        / "kits23"
    )

    prompts = load_all_frozen_prompts(
        protocol_path
    )
    nv_paths = runtime_paths(root)

    rows = []

    print("NV-Segment-CT tumor click depth audit")
    print("Cases: 9")
    print("Existing: deep / centroid / shallow_5mm")
    print("New: depth_10mm / depth_15mm")
    print("New inferences: 18")
    print("GT-simulated clicks; deployment-valid: no")
    print()

    for index, prompt in enumerate(
        prompts,
        start=1,
    ):
        ground_truth_path = (
            dataset_root
            / prompt.case_id
            / "segmentation.nii.gz"
        )

        existing = derive_click_variants(
            prompt=prompt,
            ground_truth_path=ground_truth_path,
        )
        existing_by_name = {
            item.variant: item
            for item in existing
        }

        depth_10 = derive_depth_variant(
            prompt=prompt,
            ground_truth_path=ground_truth_path,
            depth_mm=10.0,
        )
        depth_15 = derive_depth_variant(
            prompt=prompt,
            ground_truth_path=ground_truth_path,
            depth_mm=15.0,
        )

        print(
            f"[{index:02d}/09] {prompt.case_id}"
        )

        results = {}

        for variant in (
            existing_by_name["deep"],
            existing_by_name["centroid"],
            existing_by_name["shallow_5mm"],
            depth_10,
            depth_15,
        ):
            result = run_click_variant(
                base_prompt=prompt,
                variant=variant,
                project_root=root,
                nv_paths=nv_paths,
                environment_name=DEFAULT_RUNTIME_ENV,
                force=False,
            )
            results[variant.variant] = result

            print(
                f"  {variant.variant}: "
                f"depth={variant.interior_radius_mm:.2f} mm, "
                f"Dice={result.interactive_target_dice:.4f}, "
                f"HD95 reduction="
                f"{result.hec_hd95_reduction_mm:+.2f} mm"
            )

        rows.append(
            DepthAuditCase(
                case_id=prompt.case_id,
                shallow_5mm=results["shallow_5mm"],
                depth_10mm=results["depth_10mm"],
                depth_15mm=results["depth_15mm"],
                centroid=results["centroid"],
                deep=results["deep"],
            )
        )

    payload = summarize_depth_audit(rows)
    write_depth_audit_outputs(
        payload=payload,
        output_root=output_root,
    )
    _write_case_csv(
        payload,
        output_root / "cases.csv",
    )
    _write_markdown(
        payload,
        output_root / "depth_audit.md",
    )

    variants = payload["variants"]
    robustness = payload["robustness_vs_deep"]
    assert isinstance(variants, dict)
    assert isinstance(robustness, dict)

    print()
    print("Tumor click depth summary")

    for name in (
        "shallow_5mm",
        "depth_10mm",
        "depth_15mm",
        "centroid",
        "deep",
    ):
        item = variants[name]
        assert isinstance(item, dict)
        print(name)
        print(
            f"  tumor Dice mean/median: "
            f"{item['tumor_dice_mean']:.4f} / "
            f"{item['tumor_dice_median']:.4f}"
        )
        print(
            f"  Dice >=0.50 / >=0.70: "
            f"{item['tumor_dice_ge_050_count']}/9 / "
            f"{item['tumor_dice_ge_070_count']}/9"
        )
        print(
            f"  coverage gain mean: "
            f"{100.0 * item['tumor_coverage_gain_mean']:+.1f} pp"
        )
        print(
            f"  HEC HD95 reduction mean: "
            f"{item['hec_hd95_reduction_mean_mm']:+.2f} mm"
        )
        print(
            f"  HD95 improved: "
            f"{item['hd95_improved_count']}/9"
        )

    print()
    print("Robustness vs deep")
    for name in (
        "shallow_5mm",
        "depth_10mm",
        "depth_15mm",
        "centroid",
    ):
        item = robustness[name]
        assert isinstance(item, dict)
        print(
            f"  {name}: "
            f"Dice delta={item['dice_delta_vs_deep_mean']:+.4f}, "
            f"within±0.05="
            f"{item['cases_within_0_05_dice_of_deep']}/9, "
            f"within±0.10="
            f"{item['cases_within_0_10_dice_of_deep']}/9"
        )

    print()
    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(f"MD:   {output_root / 'depth_audit.md'}")
    print(
        "Tumor click depth audit PASSED "
        "(GT-simulated interaction; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
