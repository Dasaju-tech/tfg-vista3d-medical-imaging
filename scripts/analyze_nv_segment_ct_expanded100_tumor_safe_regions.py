from __future__ import annotations

import csv
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_safe_region_generalization import (
    build_expanded100_safe_region_rows,
    summarize_expanded100_safe_regions,
    write_summary,
)


def _write_case_csv(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = list(rows[0].keys())

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
        writer.writerows(rows)


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    depths = payload["depths"]
    quartiles = payload["volume_quartiles"]
    volume = payload["largest_component_volume_ml"]
    max_depth = payload["max_interior_depth_mm"]

    assert isinstance(depths, dict)
    assert isinstance(quartiles, dict)
    assert isinstance(volume, dict)
    assert isinstance(max_depth, dict)

    lines = [
        "# Expanded100 tumor safe-region generalization",
        "",
        "## Cohort",
        "",
        (
            f"- Expanded100 cases: {payload['case_count']}."
        ),
        (
            f"- Tumor-positive: {payload['tumor_positive_count']}."
        ),
        (
            f"- Tumor-negative: {payload['tumor_negative_count']}."
        ),
        "",
        "## Tumor geometry",
        "",
        (
            f"- Largest-component volume median: "
            f"{float(volume['median']):.2f} mL."
        ),
        (
            f"- Largest-component volume range: "
            f"{float(volume['min']):.2f}–"
            f"{float(volume['max']):.2f} mL."
        ),
        (
            f"- Maximum interior depth median: "
            f"{float(max_depth['median']):.2f} mm."
        ),
        (
            f"- Maximum interior depth range: "
            f"{float(max_depth['min']):.2f}–"
            f"{float(max_depth['max']):.2f} mm."
        ),
        "",
        "## Safe-region availability",
        "",
        "| Depth | Non-empty | Mean safe fraction | Median safe fraction | >=10% tumor volume |",
        "|---:|---:|---:|---:|---:|",
    ]

    for name in ("5mm", "10mm", "15mm"):
        item = depths[name]
        assert isinstance(item, dict)
        fraction = item["safe_fraction"]
        assert isinstance(fraction, dict)

        lines.append(
            f"| {name} "
            f"| {item['cases_with_nonempty_region']}/"
            f"{payload['tumor_positive_count']} "
            f"| {100.0 * float(fraction['mean']):.1f}% "
            f"| {100.0 * float(fraction['median']):.1f}% "
            f"| {item['cases_safe_fraction_ge_0_10']}/"
            f"{payload['tumor_positive_count']} |"
        )

    lines.extend(
        [
            "",
            "## 15 mm feasibility by tumor-size quartile",
            "",
            "| Group | N | Median largest tumor (mL) | Non-empty 15 mm | Median 15 mm safe fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for name in (
        "smallest_quartile",
        "lower_middle",
        "upper_middle",
        "largest_quartile",
    ):
        item = quartiles.get(name)
        if not isinstance(item, dict):
            continue

        size = item["largest_component_volume_ml"]
        depth = item["15mm"]
        assert isinstance(size, dict)
        assert isinstance(depth, dict)
        fraction = depth["safe_fraction"]
        assert isinstance(fraction, dict)

        lines.append(
            f"| {name} "
            f"| {item['count']} "
            f"| {float(size['median']):.2f} "
            f"| {depth['cases_with_nonempty_region']}/{item['count']} "
            f"| {100.0 * float(fraction['median']):.1f}% |"
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
        / "experiments"
        / "nv_segment_ct_expanded100_tumor_safe_regions"
    )

    rows = build_expanded100_safe_region_rows(
        cohort_csv=cohort_csv,
        dataset_root=dataset_root,
    )
    payload = summarize_expanded100_safe_regions(
        rows
    )

    write_summary(
        payload=payload,
        output_root=output_root,
    )
    _write_case_csv(
        payload,
        output_root / "cases.csv",
    )
    _write_markdown(
        payload,
        output_root / "safe_region_generalization.md",
    )

    depths = payload["depths"]
    quartiles = payload["volume_quartiles"]
    assert isinstance(depths, dict)
    assert isinstance(quartiles, dict)

    print("NV-Segment-CT expanded100 tumor safe-region generalization")
    print(f"Cases: {payload['case_count']}")
    print(
        f"Tumor-positive: {payload['tumor_positive_count']}"
    )
    print(
        f"Tumor-negative: {payload['tumor_negative_count']}"
    )
    print("GPU inference: none")
    print()

    for name in ("5mm", "10mm", "15mm"):
        item = depths[name]
        assert isinstance(item, dict)
        fraction = item["safe_fraction"]
        assert isinstance(fraction, dict)

        print(name)
        print(
            f"  non-empty safe region: "
            f"{item['cases_with_nonempty_region']}/"
            f"{payload['tumor_positive_count']}"
        )
        print(
            f"  safe fraction mean/median: "
            f"{100.0 * float(fraction['mean']):.1f}% / "
            f"{100.0 * float(fraction['median']):.1f}%"
        )
        print(
            f"  cases with >=10% safe tumor volume: "
            f"{item['cases_safe_fraction_ge_0_10']}/"
            f"{payload['tumor_positive_count']}"
        )
        print()

    print("15mm feasibility by tumor-size quartile")
    for name in (
        "smallest_quartile",
        "lower_middle",
        "upper_middle",
        "largest_quartile",
    ):
        item = quartiles.get(name)
        if not isinstance(item, dict):
            continue

        size = item["largest_component_volume_ml"]
        depth = item["15mm"]
        assert isinstance(size, dict)
        assert isinstance(depth, dict)
        fraction = depth["safe_fraction"]
        assert isinstance(fraction, dict)

        print(
            f"  {name}: N={item['count']}, "
            f"median_tumor={float(size['median']):.2f} mL, "
            f"nonempty={depth['cases_with_nonempty_region']}/{item['count']}, "
            f"median_safe_fraction="
            f"{100.0 * float(fraction['median']):.1f}%"
        )

    print()
    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(
        f"MD:   {output_root / 'safe_region_generalization.md'}"
    )
    print(
        "Expanded100 tumor safe-region generalization PASSED "
        "(GT-derived geometry; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
