from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.residual_geometry_audit import (
    analyze_residual_geometry,
    build_residual_geometry_cases,
)


def _write_cases(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scatter(
    *,
    x: list[float],
    y: list[float],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.scatter(x, y, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    directions = payload["direction_counts"]
    summary = payload["all_outliers"]
    modes = payload["failure_modes"]

    assert isinstance(directions, dict)
    assert isinstance(summary, dict)
    assert isinstance(modes, dict)

    lines = [
        "# Expanded100 residual geometry audit",
        "",
        "## Directional decomposition",
        "",
        (
            f"- Prediction-to-GT dominated: "
            f"{directions['prediction_to_gt']}/22."
        ),
        (
            f"- GT-to-prediction dominated: "
            f"{directions['gt_to_prediction']}/22."
        ),
        (
            f"- Balanced directional error: "
            f"{directions['balanced']}/22."
        ),
        "",
        "## Aggregate residual burden",
        "",
        "| Quantity | Mean | Median | Max |",
        "|---|---:|---:|---:|",
    ]

    labels = {
        "fp_volume_ml": "False-positive volume (mL)",
        "fn_volume_ml": "False-negative volume (mL)",
        "prediction_to_gt_surface_p95_mm": (
            "Prediction→GT surface p95 (mm)"
        ),
        "gt_to_prediction_surface_p95_mm": (
            "GT→prediction surface p95 (mm)"
        ),
    }

    for key, label in labels.items():
        values = summary[key]
        assert isinstance(values, dict)
        lines.append(
            f"| {label} "
            f"| {float(values['mean']):.2f} "
            f"| {float(values['median']):.2f} "
            f"| {float(values['max']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Failure-mode decomposition",
            "",
            "| Mode | N | FP volume mean | FN volume mean | Pred→GT p95 mean | GT→Pred p95 mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for mode, values in modes.items():
        assert isinstance(values, dict)
        fp = values["fp_volume_ml"]
        fn = values["fn_volume_ml"]
        p2g = values["prediction_to_gt_surface_p95_mm"]
        g2p = values["gt_to_prediction_surface_p95_mm"]
        assert isinstance(fp, dict)
        assert isinstance(fn, dict)
        assert isinstance(p2g, dict)
        assert isinstance(g2p, dict)

        lines.append(
            f"| {mode} "
            f"| {values['count']} "
            f"| {float(fp['mean']):.2f} "
            f"| {float(fn['mean']):.2f} "
            f"| {float(p2g['mean']):.2f} "
            f"| {float(g2p['mean']):.2f} |"
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


def main() -> int:
    root = discover_project_root()

    taxonomy_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_hd95_tail"
        / "case_taxonomy.csv"
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
        / "nv_segment_ct_expanded100_residual_geometry"
    )

    rows = build_residual_geometry_cases(
        taxonomy_path=taxonomy_path,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
    )
    payload = analyze_residual_geometry(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "residual_geometry_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "outlier_geometry.csv"
    _write_cases(payload, csv_path)

    md_path = output_root / "residual_geometry_audit.md"
    _write_markdown(payload, md_path)

    figures = output_root / "figures"

    _scatter(
        x=[row.fp_volume_ml for row in rows],
        y=[row.original_hd95_mm for row in rows],
        xlabel="False-positive volume (mL)",
        ylabel="Original HD95 (mm)",
        title="HD95 vs false-positive volume",
        path=figures / "hd95_vs_fp_volume.png",
    )
    _scatter(
        x=[row.fn_volume_ml for row in rows],
        y=[row.original_hd95_mm for row in rows],
        xlabel="False-negative volume (mL)",
        ylabel="Original HD95 (mm)",
        title="HD95 vs false-negative volume",
        path=figures / "hd95_vs_fn_volume.png",
    )
    _scatter(
        x=[
            row.prediction_to_gt_surface_p95_mm
            for row in rows
        ],
        y=[
            row.gt_to_prediction_surface_p95_mm
            for row in rows
        ],
        xlabel="Prediction→GT surface p95 (mm)",
        ylabel="GT→prediction surface p95 (mm)",
        title="Directional surface-error decomposition",
        path=figures / "directional_surface_p95.png",
    )

    directions = payload["direction_counts"]
    assert isinstance(directions, dict)

    print("NV-Segment-CT expanded100 residual geometry audit")
    print("HD95 outliers analyzed: 22")
    print(
        "Direction counts: "
        f"prediction->GT={directions['prediction_to_gt']}, "
        f"GT->prediction={directions['gt_to_prediction']}, "
        f"balanced={directions['balanced']}"
    )
    print()

    rankings = payload["rankings"]
    assert isinstance(rankings, dict)

    print("Largest prediction->GT p95:")
    rows_ranked = rankings["largest_prediction_to_gt_p95"]
    assert isinstance(rows_ranked, list)
    for index, row in enumerate(rows_ranked[:5], start=1):
        assert isinstance(row, dict)
        print(
            f"  {index:02d}. {row['case_id']}: "
            f"{float(row['prediction_to_gt_surface_p95_mm']):.2f} mm "
            f"(FP={float(row['fp_volume_ml']):.2f} mL)"
        )

    print()
    print("Largest GT->prediction p95:")
    rows_ranked = rankings["largest_gt_to_prediction_p95"]
    assert isinstance(rows_ranked, list)
    for index, row in enumerate(rows_ranked[:5], start=1):
        assert isinstance(row, dict)
        print(
            f"  {index:02d}. {row['case_id']}: "
            f"{float(row['gt_to_prediction_surface_p95_mm']):.2f} mm "
            f"(FN={float(row['fn_volume_ml']):.2f} mL)"
        )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print("Residual geometry audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
