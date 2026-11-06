from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.renal_envelope_expansion import (
    analyze_expansion_sensitivity,
    build_expansion_cases,
)


def _write_case_csv(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "undercoverage_target",
        "original_dice",
        "original_surface_dice",
        "original_hd95_mm",
        "original_tumor_coverage_fraction",
    ]

    for radius in ("2mm", "5mm", "10mm"):
        fields.extend(
            [
                f"{radius}_dice",
                f"{radius}_surface_dice",
                f"{radius}_hd95_mm",
                f"{radius}_added_volume_ml",
                f"{radius}_tumor_coverage_fraction",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            assert isinstance(row, dict)
            original = row["original"]
            variants = row["variants"]
            assert isinstance(original, dict)
            assert isinstance(variants, list)

            output = {
                "case_id": row["case_id"],
                "undercoverage_target": (
                    row["undercoverage_target"]
                ),
                "original_dice": original["dice"],
                "original_surface_dice": (
                    original["surface_dice"]
                ),
                "original_hd95_mm": original["hd95_mm"],
                "original_tumor_coverage_fraction": (
                    row["original_tumor_coverage_fraction"]
                ),
            }

            for item in variants:
                assert isinstance(item, dict)
                key = f"{float(item['radius_mm']):g}mm"
                metrics = item["metrics"]
                assert isinstance(metrics, dict)

                output[f"{key}_dice"] = metrics["dice"]
                output[f"{key}_surface_dice"] = (
                    metrics["surface_dice"]
                )
                output[f"{key}_hd95_mm"] = metrics["hd95_mm"]
                output[f"{key}_added_volume_ml"] = (
                    item["added_volume_ml"]
                )
                output[f"{key}_tumor_coverage_fraction"] = (
                    item["tumor_coverage_fraction"]
                )

            writer.writerow(output)


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    original = payload["original"]
    variants = payload["variants"]

    assert isinstance(original, dict)
    assert isinstance(variants, dict)

    original_hd95 = original["hd95_mm"]
    assert isinstance(original_hd95, dict)

    lines = [
        "# Expanded100 renal-envelope expansion sensitivity",
        "",
        "## Baseline",
        "",
        (
            f"Original all-case mean HD95: "
            f"{float(original_hd95['mean']):.2f} mm."
        ),
        "",
        "## Expansion variants",
        "",
        "| Radius | Mean HD95 | Remaining outliers | Target mean HD95 reduction | Targets resolved | Target tumor-coverage gain | Non-target HD95 harm | Non-target Dice harm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, values in variants.items():
        assert isinstance(values, dict)
        all_cases = values["all_cases"]
        targets = values["undercoverage_targets"]
        non_targets = values["non_targets"]
        assert isinstance(all_cases, dict)
        assert isinstance(targets, dict)
        assert isinstance(non_targets, dict)

        hd95 = all_cases["hd95_mm"]
        reduction = targets["hd95_reduction_mm"]
        gain = targets["tumor_coverage_gain"]
        assert isinstance(hd95, dict)
        assert isinstance(reduction, dict)

        gain_text = (
            f"{100.0 * float(gain['mean']):+.1f} pp"
            if isinstance(gain, dict)
            else "N/A"
        )

        lines.append(
            f"| {name} "
            f"| {float(hd95['mean']):.2f} "
            f"| {all_cases['remaining_outliers_at_original_threshold']} "
            f"| {float(reduction['mean']):+.2f} "
            f"| {targets['resolved_below_original_threshold']}/9 "
            f"| {gain_text} "
            f"| {non_targets['hd95_worsened_gt_1mm']} "
            f"| {non_targets['dice_worsened_gt_0_01']} |"
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


def _render_target_tradeoff(
    payload: dict[str, object],
    path: Path,
) -> None:
    variants = payload["variants"]
    assert isinstance(variants, dict)

    radii = []
    hd95_reduction = []
    tumor_gain = []

    for values in variants.values():
        assert isinstance(values, dict)
        targets = values["undercoverage_targets"]
        assert isinstance(targets, dict)

        reduction = targets["hd95_reduction_mm"]
        gain = targets["tumor_coverage_gain"]
        assert isinstance(reduction, dict)

        radii.append(float(values["radius_mm"]))
        hd95_reduction.append(float(reduction["mean"]))
        tumor_gain.append(
            100.0 * float(gain["mean"])
            if isinstance(gain, dict)
            else 0.0
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(
        radii,
        hd95_reduction,
        marker="o",
        label="Mean HD95 reduction (mm)",
    )
    ax.plot(
        radii,
        tumor_gain,
        marker="o",
        label="Mean tumor coverage gain (pp)",
    )
    ax.set_xlabel("Expansion radius (mm)")
    ax.set_title("Undercoverage targets: expansion trade-off")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    root = discover_project_root()

    cohort_csv = (
        root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_expanded100"
        / "cases.csv"
    )
    geometry_csv = (
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
        / "nv_segment_ct_expanded100_renal_expansion"
    )

    rows = build_expansion_cases(
        cohort_csv=cohort_csv,
        geometry_csv=geometry_csv,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
    )
    payload = analyze_expansion_sensitivity(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "renal_expansion_sensitivity.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "case_comparison.csv"
    _write_case_csv(payload, csv_path)

    md_path = output_root / "renal_expansion_sensitivity.md"
    _write_markdown(payload, md_path)

    _render_target_tradeoff(
        payload,
        output_root / "figures" / "target_expansion_tradeoff.png",
    )

    variants = payload["variants"]
    assert isinstance(variants, dict)

    print("NV-Segment-CT renal-envelope expansion sensitivity")
    print("Undercoverage targets: 9/100")
    print()

    for name, values in variants.items():
        assert isinstance(values, dict)
        all_cases = values["all_cases"]
        targets = values["undercoverage_targets"]
        non_targets = values["non_targets"]
        assert isinstance(all_cases, dict)
        assert isinstance(targets, dict)
        assert isinstance(non_targets, dict)

        hd95 = all_cases["hd95_mm"]
        reduction = targets["hd95_reduction_mm"]
        gain = targets["tumor_coverage_gain"]
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
            f"  targets resolved: "
            f"{targets['resolved_below_original_threshold']}/9"
        )
        print(
            "  target tumor coverage gain: "
            + (
                f"{100.0 * float(gain['mean']):+.1f} pp"
                if isinstance(gain, dict)
                else "N/A"
            )
        )
        print(
            "  non-target harm: "
            f"HD95>1mm={non_targets['hd95_worsened_gt_1mm']}, "
            f"Dice>0.01={non_targets['dice_worsened_gt_0_01']}"
        )
        print()

    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print("Renal-envelope expansion sensitivity PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
