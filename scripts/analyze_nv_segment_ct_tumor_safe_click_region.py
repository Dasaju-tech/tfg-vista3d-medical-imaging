from __future__ import annotations

import csv
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_oracle_cohort import (
    load_all_frozen_prompts,
)
from vista3d_tfg.experiments.tumor_safe_click_region import (
    attach_block43_performance,
    evaluate_safe_click_region,
    summarize_safe_click_regions,
    write_safe_click_region_outputs,
)


def _write_case_csv(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "target_volume_ml",
        "max_interior_depth_mm",
        "5mm_fraction",
        "5mm_volume_ml",
        "10mm_fraction",
        "10mm_volume_ml",
        "15mm_fraction",
        "15mm_volume_ml",
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
            safe = row["safe_regions"]
            assert isinstance(safe, list)

            by_name = {
                f"{float(item['depth_mm']):g}mm": item
                for item in safe
            }

            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "target_volume_ml": row[
                        "target_volume_ml"
                    ],
                    "max_interior_depth_mm": row[
                        "max_interior_depth_mm"
                    ],
                    "5mm_fraction": by_name["5mm"][
                        "fraction_of_target"
                    ],
                    "5mm_volume_ml": by_name["5mm"][
                        "volume_ml"
                    ],
                    "10mm_fraction": by_name["10mm"][
                        "fraction_of_target"
                    ],
                    "10mm_volume_ml": by_name["10mm"][
                        "volume_ml"
                    ],
                    "15mm_fraction": by_name["15mm"][
                        "fraction_of_target"
                    ],
                    "15mm_volume_ml": by_name["15mm"][
                        "volume_ml"
                    ],
                }
            )


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    thresholds = payload["thresholds"]
    performance = payload["block43_depth_performance"]
    max_depth = payload["max_interior_depth_mm"]

    assert isinstance(thresholds, dict)
    assert isinstance(performance, dict)
    assert isinstance(max_depth, dict)

    lines = [
        "# Tumor safe-click region audit",
        "",
        "## Objective",
        "",
        (
            "Quantify how much of each frozen tumor component lies at least "
            "5, 10, or 15 mm inside the tumor boundary, and pair that "
            "availability with the click-depth performance measured in "
            "Block 43."
        ),
        "",
        "## Maximum interior depth",
        "",
        (
            f"- Mean: {float(max_depth['mean']):.2f} mm."
        ),
        (
            f"- Median: {float(max_depth['median']):.2f} mm."
        ),
        (
            f"- Range: {float(max_depth['min']):.2f}–"
            f"{float(max_depth['max']):.2f} mm."
        ),
        "",
        "## Safe-region availability and observed performance",
        "",
        "| Depth | Non-empty | Safe fraction mean | Median | >=10% tumor volume | Block43 Dice mean | Dice >=0.50 | HD95 reduction mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in ("5mm", "10mm", "15mm"):
        region = thresholds[name]
        perf = performance[name]
        assert isinstance(region, dict)
        assert isinstance(perf, dict)

        fraction = region["safe_fraction"]
        assert isinstance(fraction, dict)

        lines.append(
            f"| {name} "
            f"| {region['cases_with_nonempty_region']}/9 "
            f"| {100.0 * float(fraction['mean']):.1f}% "
            f"| {100.0 * float(fraction['median']):.1f}% "
            f"| {region['cases_safe_fraction_ge_0_10']}/9 "
            f"| {float(perf['tumor_dice_mean']):.4f} "
            f"| {perf['tumor_dice_ge_050_count']}/9 "
            f"| {float(perf['hec_hd95_reduction_mean_mm']):+.2f} mm |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(payload["interpretation_boundary"]),
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
    block43_summary_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_click_depth_audit"
        / "summary.json"
    )
    dataset_root = (
        root
        / "data"
        / "raw"
        / "kits23"
    )
    output_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_safe_click_region"
    )

    prompts = load_all_frozen_prompts(
        protocol_path
    )

    rows = tuple(
        evaluate_safe_click_region(
            prompt=prompt,
            ground_truth_path=(
                dataset_root
                / prompt.case_id
                / "segmentation.nii.gz"
            ),
        )
        for prompt in prompts
    )

    payload = summarize_safe_click_regions(
        rows
    )
    attach_block43_performance(
        payload=payload,
        block43_summary_path=block43_summary_path,
    )

    write_safe_click_region_outputs(
        payload=payload,
        output_root=output_root,
    )
    _write_case_csv(
        payload,
        output_root / "cases.csv",
    )
    _write_markdown(
        payload,
        output_root / "safe_click_region.md",
    )

    thresholds = payload["thresholds"]
    performance = payload["block43_depth_performance"]
    max_depth = payload["max_interior_depth_mm"]

    assert isinstance(thresholds, dict)
    assert isinstance(performance, dict)
    assert isinstance(max_depth, dict)

    print("NV-Segment-CT tumor safe-click region audit")
    print("Cases: 9")
    print("GPU inference: none")
    print(
        "Maximum interior depth: "
        f"mean={float(max_depth['mean']):.2f} mm, "
        f"median={float(max_depth['median']):.2f} mm, "
        f"range={float(max_depth['min']):.2f}-"
        f"{float(max_depth['max']):.2f} mm"
    )
    print()

    for name in ("5mm", "10mm", "15mm"):
        region = thresholds[name]
        perf = performance[name]
        assert isinstance(region, dict)
        assert isinstance(perf, dict)

        fraction = region["safe_fraction"]
        volume = region["safe_volume_ml"]
        assert isinstance(fraction, dict)
        assert isinstance(volume, dict)

        print(name)
        print(
            f"  non-empty safe region: "
            f"{region['cases_with_nonempty_region']}/9"
        )
        print(
            f"  safe tumor fraction mean/median: "
            f"{100.0 * float(fraction['mean']):.1f}% / "
            f"{100.0 * float(fraction['median']):.1f}%"
        )
        print(
            f"  safe volume mean/median: "
            f"{float(volume['mean']):.2f} / "
            f"{float(volume['median']):.2f} mL"
        )
        print(
            f"  cases with >=10% safe tumor volume: "
            f"{region['cases_safe_fraction_ge_0_10']}/9"
        )
        print(
            f"  Block43 tumor Dice mean: "
            f"{float(perf['tumor_dice_mean']):.4f}"
        )
        print(
            f"  Block43 HD95 reduction mean: "
            f"{float(perf['hec_hd95_reduction_mean_mm']):+.2f} mm"
        )
        print()

    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(f"MD:   {output_root / 'safe_click_region.md'}")
    print(
        "Tumor safe-click region audit PASSED "
        "(GT-derived geometry; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
