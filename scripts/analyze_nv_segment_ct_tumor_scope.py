from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.tumor_scope_audit import (
    analyze_scope_effect,
    build_scope_cases,
)


def _write_cases(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_id",
        "tumor_volume_ml",
        "failure_mode",
        "original_dice",
        "original_surface_dice",
        "original_hd95_mm",
        "tumor_excluded_dice",
        "tumor_excluded_surface_dice",
        "tumor_excluded_hd95_mm",
        "hd95_reduction_mm",
        "dice_delta",
        "surface_dice_delta",
        "original_hd95_outlier",
        "resolved_by_tumor_exclusion",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            assert isinstance(row, dict)
            writer.writerow(row)


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
    ax.scatter(x, y, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _paired_hd95(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    original = sorted(
        float(row["original_hd95_mm"])
        for row in rows
    )
    excluded = sorted(
        float(row["tumor_excluded_hd95_mm"])
        for row in rows
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(original, label="Original {1,2,3}")
    ax.plot(excluded, label="Tumor-excluded {1,3}")
    ax.set_xlabel("Cases sorted within each metric")
    ax.set_ylabel("HD95 (mm)")
    ax.set_title("HD95 distribution under reference-scope sensitivity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    original = payload["original_hd95_mm"]
    excluded = payload["tumor_excluded_hd95_mm"]
    reduction = payload["hd95_reduction_mm"]
    outliers = payload["outlier_analysis"]
    correlations = payload["tumor_volume_correlations"]
    modes = payload["failure_modes"]

    assert isinstance(original, dict)
    assert isinstance(excluded, dict)
    assert isinstance(reduction, dict)
    assert isinstance(outliers, dict)
    assert isinstance(correlations, dict)
    assert isinstance(modes, dict)

    lines = [
        "# Expanded100 tumor-scope confounding audit",
        "",
        "## Scope distinction",
        "",
        str(payload["official_scope_note"]),
        "",
        str(payload["diagnostic_scope_note"]),
        "",
        "## Global HD95 sensitivity",
        "",
        (
            f"- Original HD95 mean/median: "
            f"{float(original['mean']):.2f} / "
            f"{float(original['median']):.2f} mm."
        ),
        (
            f"- Tumor-excluded HD95 mean/median: "
            f"{float(excluded['mean']):.2f} / "
            f"{float(excluded['median']):.2f} mm."
        ),
        (
            f"- Mean HD95 reduction: "
            f"{float(reduction['mean']):.2f} mm."
        ),
        "",
        "## HD95 outliers",
        "",
        (
            f"- Original outliers: "
            f"{outliers['original_outlier_count']}/100."
        ),
        (
            "- Original outliers falling below the same threshold after "
            f"tumor exclusion: "
            f"{outliers['original_outliers_resolved_by_tumor_exclusion']}."
        ),
        (
            "- Tumor-excluded outliers under the original fixed threshold: "
            f"{outliers['tumor_excluded_outlier_count_fixed_original_threshold']}."
        ),
        "",
        "## Tumor-volume association",
        "",
        "| Relationship | Pearson r |",
        "|---|---:|",
    ]

    for name, value in correlations.items():
        rendered = (
            f"{float(value):+.4f}"
            if value is not None
            else "undefined"
        )
        lines.append(f"| {name} | {rendered} |")

    lines.extend(
        [
            "",
            "## Failure-mode sensitivity",
            "",
            "| Mode | N | Original HD95 mean | Tumor-excluded HD95 mean | Mean reduction |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for name, values in modes.items():
        assert isinstance(values, dict)
        o = values["original_hd95_mm"]
        e = values["tumor_excluded_hd95_mm"]
        r = values["hd95_reduction_mm"]
        assert isinstance(o, dict)
        assert isinstance(e, dict)
        assert isinstance(r, dict)

        lines.append(
            f"| {name} "
            f"| {values['count']} "
            f"| {float(o['mean']):.2f} "
            f"| {float(e['mean']):.2f} "
            f"| {float(r['mean']):+.2f} |"
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
        / "nv_segment_ct_expanded100_tumor_scope"
    )

    rows = build_scope_cases(
        taxonomy_path=taxonomy_path,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
    )
    payload = analyze_scope_effect(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "tumor_scope_audit.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "case_scope_comparison.csv"
    _write_cases(payload, csv_path)

    md_path = output_root / "tumor_scope_audit.md"
    _write_markdown(payload, md_path)

    case_rows = payload["case_rows"]
    assert isinstance(case_rows, list)

    figures = output_root / "figures"
    _scatter(
        x=[
            float(row["tumor_volume_ml"])
            for row in case_rows
        ],
        y=[
            float(row["hd95_reduction_mm"])
            for row in case_rows
        ],
        xlabel="Ground-truth tumor volume (mL)",
        ylabel="HD95 reduction after tumor exclusion (mm)",
        title="Tumor burden vs HD95 scope effect",
        path=figures / "tumor_volume_vs_hd95_reduction.png",
    )
    _paired_hd95(
        payload,
        figures / "original_vs_tumor_excluded_hd95.png",
    )

    original = payload["original_hd95_mm"]
    excluded = payload["tumor_excluded_hd95_mm"]
    outliers = payload["outlier_analysis"]
    correlations = payload["tumor_volume_correlations"]

    assert isinstance(original, dict)
    assert isinstance(excluded, dict)
    assert isinstance(outliers, dict)
    assert isinstance(correlations, dict)

    print("NV-Segment-CT expanded100 tumor-scope confounding audit")
    print(
        "HD95 mean original / tumor-excluded: "
        f"{float(original['mean']):.2f} / "
        f"{float(excluded['mean']):.2f} mm"
    )
    print(
        "HD95 median original / tumor-excluded: "
        f"{float(original['median']):.2f} / "
        f"{float(excluded['median']):.2f} mm"
    )
    print(
        "Original outliers resolved by tumor exclusion: "
        f"{outliers['original_outliers_resolved_by_tumor_exclusion']}/"
        f"{outliers['original_outlier_count']}"
    )
    print(
        "Remaining outliers at original fixed threshold: "
        f"{outliers['tumor_excluded_outlier_count_fixed_original_threshold']}"
    )
    print()
    print("Tumor-volume correlations:")
    for name, value in correlations.items():
        rendered = (
            f"{float(value):+.4f}"
            if value is not None
            else "undefined"
        )
        print(f"  {name}: {rendered}")

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {md_path}")
    print("Tumor-scope confounding audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
