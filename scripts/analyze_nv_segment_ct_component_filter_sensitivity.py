from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.component_filter_sensitivity import (
    analyze_component_filter_sensitivity,
    build_sensitivity_cases,
)


def _prediction_to_gt_ids(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    ids = tuple(
        row["case_id"]
        for row in rows
        if row["dominant_direction"] == "prediction_to_gt"
    )

    if len(ids) != 12:
        raise ValueError(
            f"Expected 12 prediction->GT cases, got {len(ids)}."
        )
    return ids


def _write_case_csv(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_id",
        "failure_mode",
        "hd95_outlier",
        "prediction_to_gt_target",
        "original_dice",
        "original_surface_dice",
        "original_hd95_mm",
        "global_top2_dice",
        "global_top2_surface_dice",
        "global_top2_hd95_mm",
        "global_top2_removed_voxels",
        "side_largest_dice",
        "side_largest_surface_dice",
        "side_largest_hd95_mm",
        "side_largest_removed_voxels",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            assert isinstance(row, dict)
            original = row["original"]
            global_top2 = row["global_top2"]
            side_largest = row["side_largest"]
            assert isinstance(original, dict)
            assert isinstance(global_top2, dict)
            assert isinstance(side_largest, dict)

            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "failure_mode": row["failure_mode"],
                    "hd95_outlier": row["hd95_outlier"],
                    "prediction_to_gt_target": (
                        row["prediction_to_gt_target"]
                    ),
                    "original_dice": original["dice"],
                    "original_surface_dice": (
                        original["surface_dice"]
                    ),
                    "original_hd95_mm": original["hd95_mm"],
                    "global_top2_dice": global_top2["dice"],
                    "global_top2_surface_dice": (
                        global_top2["surface_dice"]
                    ),
                    "global_top2_hd95_mm": (
                        global_top2["hd95_mm"]
                    ),
                    "global_top2_removed_voxels": (
                        row["global_top2_removed_voxels"]
                    ),
                    "side_largest_dice": side_largest["dice"],
                    "side_largest_surface_dice": (
                        side_largest["surface_dice"]
                    ),
                    "side_largest_hd95_mm": (
                        side_largest["hd95_mm"]
                    ),
                    "side_largest_removed_voxels": (
                        row["side_largest_removed_voxels"]
                    ),
                }
            )


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    variants = payload["variants"]
    original = payload["original"]

    assert isinstance(variants, dict)
    assert isinstance(original, dict)

    original_hd95 = original["hd95_mm"]
    assert isinstance(original_hd95, dict)

    lines = [
        "# Expanded100 component-filter sensitivity",
        "",
        "## Objective",
        "",
        (
            "Test prediction-only connected-component pruning as a "
            "candidate intervention for the 12 Block 35 "
            "prediction→GT-dominated HD95 outliers."
        ),
        "",
        (
            f"Original all-case mean HD95: "
            f"{float(original_hd95['mean']):.2f} mm."
        ),
        "",
        "## Variants",
        "",
        "- `global_top2`: keep the two largest connected components of the complete predicted renal envelope.",
        "- `side_largest`: independently keep the largest connected component for the predicted left and right renal label groups.",
        "",
        "Neither variant is adopted into the baseline in this block.",
        "",
        "| Variant | Mean HD95 | Remaining outliers | Target mean HD95 reduction | Target resolved | Non-outlier HD95 worsened >1 mm | Non-outlier Dice worsened >0.01 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, values in variants.items():
        assert isinstance(values, dict)
        all_cases = values["all_cases"]
        targets = values["prediction_to_gt_targets"]
        non_outliers = values["non_outliers"]
        assert isinstance(all_cases, dict)
        assert isinstance(targets, dict)
        assert isinstance(non_outliers, dict)

        hd95 = all_cases["hd95_mm"]
        target_reduction = targets["hd95_reduction_mm"]
        assert isinstance(hd95, dict)
        assert isinstance(target_reduction, dict)

        lines.append(
            f"| {name} "
            f"| {float(hd95['mean']):.2f} "
            f"| {all_cases['remaining_outliers_at_original_threshold']} "
            f"| {float(target_reduction['mean']):+.2f} "
            f"| {targets['resolved_below_original_threshold']}/12 "
            f"| {non_outliers['hd95_worsened_gt_1mm']} "
            f"| {non_outliers['dice_worsened_gt_0_01']} |"
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_target_hd95(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    targets = [
        row
        for row in rows
        if row["prediction_to_gt_target"]
    ]

    names = [row["case_id"] for row in targets]
    original = [
        float(row["original"]["hd95_mm"])
        for row in targets
    ]
    global_top2 = [
        float(row["global_top2"]["hd95_mm"])
        for row in targets
    ]
    side_largest = [
        float(row["side_largest"]["hd95_mm"])
        for row in targets
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = list(range(len(names)))
    ax.plot(x, original, marker="o", label="Original")
    ax.plot(x, global_top2, marker="o", label="Global top-2")
    ax.plot(x, side_largest, marker="o", label="Side largest")
    ax.set_xticks(x, names, rotation=60, ha="right")
    ax.set_ylabel("HD95 (mm)")
    ax.set_title("Prediction→GT outliers: component-filter sensitivity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    root = discover_project_root()

    taxonomy_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_hd95_tail"
        / "case_taxonomy.csv"
    )
    geometry_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_residual_geometry"
        / "outlier_geometry.csv"
    )
    dataset_root = root / "data" / "raw" / "kits23"
    prediction_root = (
        root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
    )
    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_component_filter_sensitivity"
    )

    prediction_to_gt_ids = _prediction_to_gt_ids(
        geometry_path
    )

    rows = build_sensitivity_cases(
        taxonomy_path=taxonomy_path,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
    )
    payload = analyze_component_filter_sensitivity(
        rows,
        prediction_to_gt_case_ids=prediction_to_gt_ids,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "component_filter_sensitivity.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "case_comparison.csv"
    _write_case_csv(payload, csv_path)

    md_path = output_root / "component_filter_sensitivity.md"
    _write_markdown(payload, md_path)

    _render_target_hd95(
        payload,
        output_root / "figures" / "target_hd95_sensitivity.png",
    )

    variants = payload["variants"]
    assert isinstance(variants, dict)

    print("NV-Segment-CT component-filter sensitivity")
    print(
        f"Prediction->GT target outliers: "
        f"{payload['prediction_to_gt_target_count']}"
    )
    print()

    for name, values in variants.items():
        assert isinstance(values, dict)
        all_cases = values["all_cases"]
        targets = values["prediction_to_gt_targets"]
        non_outliers = values["non_outliers"]
        assert isinstance(all_cases, dict)
        assert isinstance(targets, dict)
        assert isinstance(non_outliers, dict)

        hd95 = all_cases["hd95_mm"]
        reduction = targets["hd95_reduction_mm"]
        assert isinstance(hd95, dict)
        assert isinstance(reduction, dict)

        print(name)
        print(
            f"  all-case mean HD95: "
            f"{float(hd95['mean']):.2f} mm"
        )
        print(
            f"  remaining outliers: "
            f"{all_cases['remaining_outliers_at_original_threshold']}/100"
        )
        print(
            f"  target mean HD95 reduction: "
            f"{float(reduction['mean']):+.2f} mm"
        )
        print(
            f"  target outliers resolved: "
            f"{targets['resolved_below_original_threshold']}/12"
        )
        print(
            "  non-outlier harm: "
            f"HD95>1mm={non_outliers['hd95_worsened_gt_1mm']}, "
            f"Dice>0.01={non_outliers['dice_worsened_gt_0_01']}"
        )
        print()

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print("Component-filter sensitivity PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
