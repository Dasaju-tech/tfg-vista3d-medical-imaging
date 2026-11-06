from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.undercoverage_audit import (
    analyze_undercoverage,
    build_undercoverage_cases,
)


def _write_case_csv(
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


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    dominant = payload["dominant_missing_label_counts"]
    weighted = payload["weighted_miss_fraction_by_label"]
    shares = payload["aggregate_missing_volume_share"]
    fn_summary = payload["fn_volume_ml"]
    distance = payload["gt_to_prediction_surface_p95_mm"]
    rankings = payload["rankings"]

    assert isinstance(dominant, dict)
    assert isinstance(weighted, dict)
    assert isinstance(shares, dict)
    assert isinstance(fn_summary, dict)
    assert isinstance(distance, dict)
    assert isinstance(rankings, dict)

    def fmt_optional(value: object) -> str:
        if value is None:
            return "N/A"
        return f"{100.0 * float(value):.1f}%"

    lines = [
        "# Expanded100 GT→prediction undercoverage audit",
        "",
        "## Cohort",
        "",
        "Nine frozen HD95 outliers classified in Block 35 as GT→prediction dominated.",
        "",
        "## Global missing burden",
        "",
        (
            f"- Mean false-negative volume: "
            f"{float(fn_summary['mean']):.2f} mL."
        ),
        (
            f"- Median false-negative volume: "
            f"{float(fn_summary['median']):.2f} mL."
        ),
        (
            f"- Mean GT→prediction surface p95: "
            f"{float(distance['mean']):.2f} mm."
        ),
        "",
        "## Missing-label decomposition",
        "",
        "| Label | Dominant cases | Share of all FN volume | Weighted miss fraction |",
        "|---|---:|---:|---:|",
    ]

    for label in ("kidney", "tumor", "cyst"):
        lines.append(
            f"| {label} "
            f"| {dominant[label]}/9 "
            f"| {100.0 * float(shares[label]):.1f}% "
            f"| {fmt_optional(weighted[label])} |"
        )

    worst = rankings["largest_false_negative_volume"]
    assert isinstance(worst, list)

    lines.extend(
        [
            "",
            "## Largest undercoverage cases",
            "",
            "| Rank | Case | FN volume (mL) | FN fraction | Dominant label | Kidney FN | Tumor FN | Cyst FN |",
            "|---:|---|---:|---:|---|---:|---:|---:|",
        ]
    )

    for index, row in enumerate(worst, start=1):
        assert isinstance(row, dict)
        lines.append(
            f"| {index} "
            f"| {row['case_id']} "
            f"| {float(row['fn_volume_ml']):.2f} "
            f"| {100.0 * float(row['fn_fraction']):.1f}% "
            f"| {row['dominant_missing_label']} "
            f"| {float(row['fn_kidney_volume_ml']):.2f} "
            f"| {float(row['fn_tumor_volume_ml']):.2f} "
            f"| {float(row['fn_cyst_volume_ml']):.2f} |"
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


def _render_label_volume(
    payload: dict[str, object],
    path: Path,
) -> None:
    aggregate = payload["aggregate_label_volumes_ml"]
    assert isinstance(aggregate, dict)

    gt = aggregate["ground_truth"]
    fn = aggregate["false_negative"]
    assert isinstance(gt, dict)
    assert isinstance(fn, dict)

    labels = ["kidney", "tumor", "cyst"]
    x = list(range(len(labels)))
    width = 0.38

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(
        [value - width / 2 for value in x],
        [float(gt[label]) for label in labels],
        width=width,
        label="Ground truth",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [float(fn[label]) for label in labels],
        width=width,
        label="False negative",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Volume (mL)")
    ax.set_title("GT→prediction outliers: label-wise missing burden")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _render_case_fn(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    rows = sorted(
        rows,
        key=lambda row: float(row["fn_volume_ml"]),
        reverse=True,
    )

    names = [row["case_id"] for row in rows]
    kidney = [float(row["fn_kidney_volume_ml"]) for row in rows]
    tumor = [float(row["fn_tumor_volume_ml"]) for row in rows]
    cyst = [float(row["fn_cyst_volume_ml"]) for row in rows]

    x = list(range(len(names)))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x, kidney, label="Kidney")
    ax.bar(x, tumor, bottom=kidney, label="Tumor")
    bottom = [
        kidney[i] + tumor[i]
        for i in range(len(names))
    ]
    ax.bar(x, cyst, bottom=bottom, label="Cyst")
    ax.set_xticks(x, names, rotation=55, ha="right")
    ax.set_ylabel("False-negative volume (mL)")
    ax.set_title("Undercoverage composition by case")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    root = discover_project_root()

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
        / "nv_segment_ct_expanded100_undercoverage"
    )

    rows = build_undercoverage_cases(
        geometry_path=geometry_path,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
    )
    payload = analyze_undercoverage(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "undercoverage_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "undercoverage_cases.csv"
    _write_case_csv(payload, csv_path)

    md_path = output_root / "undercoverage_audit.md"
    _write_markdown(payload, md_path)

    figures = output_root / "figures"
    _render_label_volume(
        payload,
        figures / "label_missing_burden.png",
    )
    _render_case_fn(
        payload,
        figures / "case_undercoverage_composition.png",
    )

    dominant = payload["dominant_missing_label_counts"]
    weighted = payload["weighted_miss_fraction_by_label"]

    assert isinstance(dominant, dict)
    assert isinstance(weighted, dict)

    print("NV-Segment-CT expanded100 undercoverage audit")
    print("GT->prediction dominated outliers: 9")
    print()
    print("Dominant missing label:")
    for label in ("kidney", "tumor", "cyst"):
        print(f"  {label}: {dominant[label]}/9")

    print()
    print("Weighted miss fraction by label:")
    for label in ("kidney", "tumor", "cyst"):
        value = weighted[label]
        rendered = (
            f"{100.0 * float(value):.1f}%"
            if value is not None
            else "N/A"
        )
        print(f"  {label}: {rendered}")

    print()
    print("Worst false-negative volume cases:")
    rankings = payload["rankings"]
    assert isinstance(rankings, dict)
    ranked = rankings["largest_false_negative_volume"]
    assert isinstance(ranked, list)

    for index, row in enumerate(ranked[:5], start=1):
        assert isinstance(row, dict)
        print(
            f"  {index:02d}. {row['case_id']}: "
            f"FN={float(row['fn_volume_ml']):.2f} mL, "
            f"missing={100.0 * float(row['fn_fraction']):.1f}%, "
            f"dominant={row['dominant_missing_label']}"
        )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print("Undercoverage audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
