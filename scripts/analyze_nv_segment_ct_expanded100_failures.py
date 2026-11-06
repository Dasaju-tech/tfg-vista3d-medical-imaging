from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.failure_stratification import (
    FailureCase,
    analyze_failure_stratification,
    load_failure_cases,
    write_json,
    write_ranked_cases_csv,
)


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


def _histogram(
    values: list[float],
    *,
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.hist(values, bins=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cases")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_subgroups(
    payload: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subgroups = payload["subgroups"]
    assert isinstance(subgroups, dict)

    fields = [
        "subgroup",
        "count",
        "kidney_and_mass_dice_mean",
        "surface_dice_mean",
        "hd95_median_mm",
        "cyst_dice_mean",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for name, values in subgroups.items():
            assert isinstance(values, dict)
            writer.writerow(
                {
                    "subgroup": name,
                    **values,
                }
            )


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    correlations = payload["correlations"]
    outlier_rule = payload["hd95_outlier_rule"]
    subgroups = payload["subgroups"]
    rankings = payload["rankings"]

    assert isinstance(correlations, dict)
    assert isinstance(outlier_rule, dict)
    assert isinstance(subgroups, dict)
    assert isinstance(rankings, dict)

    lines = [
        "# Expanded100 failure stratification",
        "",
        "## HD95 tail",
        "",
        (
            f"- Outlier rule: Q3 + 1.5*IQR = "
            f"{float(outlier_rule['threshold_mm']):.2f} mm."
        ),
        (
            f"- HD95 outliers: "
            f"{int(outlier_rule['outlier_count'])}/100."
        ),
        "",
        "## Exploratory correlations",
        "",
        "| Relationship | Pearson r |",
        "|---|---:|",
    ]

    for name, value in correlations.items():
        text = (
            f"{float(value):.4f}"
            if value is not None
            else "undefined"
        )
        lines.append(f"| {name} | {text} |")

    lines.extend(
        [
            "",
            "## Subgroups",
            "",
            "| Group | N | Kidney Dice mean | Surface Dice mean | HD95 median (mm) | Cyst Dice mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for name, values in subgroups.items():
        assert isinstance(values, dict)

        def fmt(value: object) -> str:
            if value is None:
                return "N/A"
            return f"{float(value):.4f}"

        lines.append(
            f"| {name} "
            f"| {values['count']} "
            f"| {fmt(values['kidney_and_mass_dice_mean'])} "
            f"| {fmt(values['surface_dice_mean'])} "
            f"| {fmt(values['hd95_median_mm'])} "
            f"| {fmt(values['cyst_dice_mean'])} |"
        )

    worst_hd95 = rankings["worst_hd95"]
    assert isinstance(worst_hd95, list)

    lines.extend(
        [
            "",
            "## Worst HD95 cases",
            "",
            "| Rank | Case | HD95 (mm) | Dice | Surface Dice | Cyst GT (mL) | Tumor GT (mL) | Anatomy flags |",
            "|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )

    for index, row in enumerate(worst_hd95, start=1):
        assert isinstance(row, dict)
        flags = row["anatomy_qc_flags"]
        if isinstance(flags, list):
            flag_text = "; ".join(flags) if flags else "-"
        else:
            flag_text = str(flags)

        lines.append(
            f"| {index} "
            f"| {row['case_id']} "
            f"| {float(row['kidney_and_mass_hd95_mm']):.2f} "
            f"| {float(row['kidney_and_mass_dice']):.4f} "
            f"| {float(row['kidney_and_mass_surface_dice']):.4f} "
            f"| {float(row['cyst_reference_volume_ml']):.2f} "
            f"| {float(row['gt_tumor_volume_ml']):.2f} "
            f"| {flag_text} |"
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

    cohort_csv = (
        root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_expanded100"
        / "cases.csv"
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
        / "analyses"
        / "nv_segment_ct_expanded100_failure_stratification"
    )

    rows = load_failure_cases(
        cohort_csv=cohort_csv,
        dataset_root=dataset_root,
    )
    payload = analyze_failure_stratification(rows)

    write_json(
        payload,
        output_root / "failure_stratification.json",
    )
    write_ranked_cases_csv(
        rows,
        output_root / "ranked_cases.csv",
    )
    _write_subgroups(
        payload,
        output_root / "subgroup_table.csv",
    )
    _write_markdown(
        payload,
        output_root / "failure_stratification.md",
    )

    figures = output_root / "figures"

    _histogram(
        [row.kidney_and_mass_hd95_mm for row in rows],
        xlabel="HD95 (mm)",
        title="Expanded100 HD95 distribution",
        path=figures / "hd95_distribution.png",
    )
    _scatter(
        x=[row.kidney_and_mass_surface_dice for row in rows],
        y=[row.kidney_and_mass_hd95_mm for row in rows],
        xlabel="Surface Dice",
        ylabel="HD95 (mm)",
        title="HD95 vs Surface Dice",
        path=figures / "hd95_vs_surface_dice.png",
    )
    _scatter(
        x=[row.cyst_reference_volume_ml for row in rows],
        y=[row.kidney_and_mass_hd95_mm for row in rows],
        xlabel="GT cyst volume (mL)",
        ylabel="HD95 (mm)",
        title="HD95 vs cyst burden",
        path=figures / "hd95_vs_cyst_volume.png",
    )
    _scatter(
        x=[row.cyst_reference_volume_ml for row in rows],
        y=[row.kidney_and_mass_dice for row in rows],
        xlabel="GT cyst volume (mL)",
        ylabel="Kidney-and-mass Dice",
        title="Dice vs cyst burden",
        path=figures / "dice_vs_cyst_volume.png",
    )

    cyst_positive = [
        row
        for row in rows
        if row.cyst_reference_volume_ml > 0.0
    ]
    if cyst_positive:
        _scatter(
            x=[
                row.cyst_reference_volume_ml
                for row in cyst_positive
            ],
            y=[row.cyst_dice for row in cyst_positive],
            xlabel="GT cyst volume (mL)",
            ylabel="Cyst Dice",
            title="Cyst Dice vs reference cyst volume",
            path=(
                figures
                / "cyst_dice_vs_reference_volume.png"
            ),
        )

    correlations = payload["correlations"]
    outlier_rule = payload["hd95_outlier_rule"]
    rankings = payload["rankings"]

    assert isinstance(correlations, dict)
    assert isinstance(outlier_rule, dict)
    assert isinstance(rankings, dict)

    print("NV-Segment-CT expanded100 failure stratification")
    print(f"Cases: {payload['case_count']}")
    print(
        "HD95 outliers: "
        f"{outlier_rule['outlier_count']}/100 "
        f"(threshold > {float(outlier_rule['threshold_mm']):.2f} mm)"
    )
    print()
    print("Exploratory correlations:")

    for name, value in correlations.items():
        rendered = (
            f"{float(value):+.4f}"
            if value is not None
            else "undefined"
        )
        print(f"  {name}: {rendered}")

    print()
    print("Worst HD95 cases:")
    worst_hd95 = rankings["worst_hd95"]
    assert isinstance(worst_hd95, list)
    for index, row in enumerate(worst_hd95[:10], start=1):
        assert isinstance(row, dict)
        print(
            f"  {index:02d}. {row['case_id']}: "
            f"HD95={float(row['kidney_and_mass_hd95_mm']):.2f} mm, "
            f"Dice={float(row['kidney_and_mass_dice']):.4f}, "
            f"SurfaceDice="
            f"{float(row['kidney_and_mass_surface_dice']):.4f}"
        )

    print()
    print(
        f"JSON: {output_root / 'failure_stratification.json'}"
    )
    print(f"CSV:  {output_root / 'ranked_cases.csv'}")
    print(f"MD:   {output_root / 'failure_stratification.md'}")
    print("Expanded100 failure stratification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
