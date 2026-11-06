from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.expanded_validation import (
    analyze_nested_cohorts,
    read_case_rows,
)


METRIC_LABELS = {
    "kidney_and_mass_dice": "Kidney-and-mass Dice",
    "surface_dice": "Surface Dice",
    "hd95_mm": "HD95 (mm)",
    "kidney_core_dice": "Kidney-core Dice",
    "cyst_positive_dice": "Cyst-positive Dice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen pilot30 with expanded100 and quantify "
            "estimate stability and bootstrap uncertainty."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def write_metric_table(
    payload: dict[str, object],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "cohort",
                "n",
                "mean",
                "std",
                "median",
                "bootstrap_95ci_lower",
                "bootstrap_95ci_upper",
            ]
        )

        metrics = payload["metrics"]
        assert isinstance(metrics, dict)

        for metric_name, metric_payload in metrics.items():
            assert isinstance(metric_payload, dict)

            for cohort in ("pilot30", "additional70", "expanded100"):
                estimate = metric_payload[cohort]
                assert isinstance(estimate, dict)
                ci = estimate["bootstrap_mean_95ci"]
                assert isinstance(ci, dict)

                writer.writerow(
                    [
                        metric_name,
                        cohort,
                        estimate["count"],
                        estimate["mean"],
                        estimate["std"],
                        estimate["median"],
                        ci["lower"],
                        ci["upper"],
                    ]
                )


def render_stability_figure(
    payload: dict[str, object],
    metric_name: str,
    path: Path,
) -> None:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metric = metrics[metric_name]
    assert isinstance(metric, dict)

    cohorts = ("pilot30", "additional70", "expanded100")
    labels = ("Pilot30", "Additional70", "Expanded100")

    means: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []

    for cohort in cohorts:
        estimate = metric[cohort]
        assert isinstance(estimate, dict)
        ci = estimate["bootstrap_mean_95ci"]
        assert isinstance(ci, dict)

        mean = float(estimate["mean"])
        means.append(mean)
        lower_errors.append(mean - float(ci["lower"]))
        upper_errors.append(float(ci["upper"]) - mean)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.errorbar(
        range(len(cohorts)),
        means,
        yerr=[lower_errors, upper_errors],
        fmt="o",
        capsize=5,
    )
    ax.set_xticks(range(len(cohorts)), labels)
    ax.set_ylabel(METRIC_LABELS[metric_name])
    ax.set_title(
        f"{METRIC_LABELS[metric_name]} estimate stability"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    metrics = payload["metrics"]
    design = payload["design"]
    cyst_counts = payload["cyst_positive_counts"]
    laterality = payload["laterality"]

    assert isinstance(metrics, dict)
    assert isinstance(design, dict)
    assert isinstance(cyst_counts, dict)
    assert isinstance(laterality, dict)

    lines = [
        "# Expanded100 statistical validation",
        "",
        "## Design",
        "",
        (
            f"- Pilot30: N={design['pilot30_case_count']}"
        ),
        (
            f"- Expanded100: N={design['expanded100_case_count']}"
        ),
        (
            f"- Additional cases: N={design['additional_case_count']}"
        ),
        "- Cohorts are nested; deltas are descriptive, not independent tests.",
        "",
        "## Metric stability",
        "",
        "| Metric | Pilot30 mean | Expanded100 mean | Δ expanded-pilot | Expanded 95% CI | CI width change |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for metric_name, metric_payload in metrics.items():
        assert isinstance(metric_payload, dict)
        pilot = metric_payload["pilot30"]
        expanded = metric_payload["expanded100"]
        assert isinstance(pilot, dict)
        assert isinstance(expanded, dict)
        ci = expanded["bootstrap_mean_95ci"]
        assert isinstance(ci, dict)

        reduction = metric_payload[
            "expanded_ci_width_reduction_fraction"
        ]
        reduction_text = (
            f"{100.0 * float(reduction):+.1f}%"
            if reduction is not None
            else "N/A"
        )

        lines.append(
            "| "
            + METRIC_LABELS.get(metric_name, metric_name)
            + f" | {float(pilot['mean']):.4f}"
            + f" | {float(expanded['mean']):.4f}"
            + f" | {float(metric_payload['expanded_minus_pilot_mean']):+.4f}"
            + f" | [{float(ci['lower']):.4f}, {float(ci['upper']):.4f}]"
            + f" | {reduction_text} |"
        )

    lines.extend(
        [
            "",
            "## Cohort diagnostics",
            "",
            (
                "- Cyst-positive cases: "
                f"pilot30={cyst_counts['pilot30']}, "
                f"additional70={cyst_counts['additional70']}, "
                f"expanded100={cyst_counts['expanded100']}."
            ),
            (
                "- Ground-truth laterality in expanded100: "
                f"resolved={laterality['expanded100_resolved']}, "
                f"unresolved={laterality['expanded100_unresolved']}."
            ),
            "",
            "## Interpretation boundary",
            "",
            str(design["inferential_boundary"]),
            "",
            (
                "The purpose of this block is estimate-stability validation. "
                "It does not introduce a new model, tune the frozen protocol, "
                "or claim an official full KiTS23 challenge score."
            ),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = discover_project_root()

    pilot_csv = (
        root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_pilot30"
        / "cases.csv"
    )
    expanded_csv = (
        root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_expanded100"
        / "cases.csv"
    )

    if not pilot_csv.is_file():
        raise FileNotFoundError(pilot_csv)
    if not expanded_csv.is_file():
        raise FileNotFoundError(expanded_csv)

    output_root = (
        args.output_root
        if args.output_root is not None
        else (
            root
            / "outputs"
            / "analyses"
            / "nv_segment_ct_expanded100_validation"
        )
    )

    pilot_rows = read_case_rows(pilot_csv)
    expanded_rows = read_case_rows(expanded_csv)
    payload = analyze_nested_cohorts(
        pilot_rows,
        expanded_rows,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "statistical_validation.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    csv_path = output_root / "metric_table.csv"
    write_metric_table(payload, csv_path)

    markdown_path = output_root / "statistical_validation.md"
    write_markdown(payload, markdown_path)

    figures = output_root / "figures"
    for metric_name in (
        "kidney_and_mass_dice",
        "surface_dice",
        "hd95_mm",
        "cyst_positive_dice",
    ):
        metrics = payload["metrics"]
        assert isinstance(metrics, dict)
        if metric_name not in metrics:
            continue

        render_stability_figure(
            payload,
            metric_name,
            figures / f"{metric_name}_stability.png",
        )

    print("NV-Segment-CT expanded100 statistical validation")
    print("Nested cohorts: pilot30 ⊂ expanded100")
    print()

    metrics = payload["metrics"]
    assert isinstance(metrics, dict)

    for metric_name, metric_payload in metrics.items():
        assert isinstance(metric_payload, dict)
        pilot = metric_payload["pilot30"]
        expanded = metric_payload["expanded100"]
        assert isinstance(pilot, dict)
        assert isinstance(expanded, dict)
        ci = expanded["bootstrap_mean_95ci"]
        assert isinstance(ci, dict)

        print(
            f"{METRIC_LABELS.get(metric_name, metric_name)}: "
            f"pilot={float(pilot['mean']):.4f}, "
            f"expanded={float(expanded['mean']):.4f}, "
            f"delta={float(metric_payload['expanded_minus_pilot_mean']):+.4f}, "
            "expanded 95% CI="
            f"[{float(ci['lower']):.4f}, {float(ci['upper']):.4f}]"
        )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"MD:   {markdown_path}")
    print("Expanded100 statistical validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
