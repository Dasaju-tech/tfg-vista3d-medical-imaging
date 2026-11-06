from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vista3d_tfg.evaluation.expanded_validation import (
    CaseRow,
    analyze_nested_cohorts,
    bootstrap_mean_ci,
    read_case_rows,
)


def make_row(
    index: int,
    *,
    laterality: bool = True,
) -> CaseRow:
    return CaseRow(
        case_id=f"case_{index:05d}",
        kidney_and_mass_dice=0.80 + index / 1000.0,
        kidney_and_mass_surface_dice=0.70 + index / 1000.0,
        kidney_and_mass_hd95_mm=10.0 - index / 100.0,
        kidney_core_dice=0.79 + index / 1000.0,
        cyst_dice=0.20 + index / 2000.0,
        cyst_reference_volume_ml=(
            5.0 if index % 2 == 0 else 0.0
        ),
        inference_wall_time_seconds=12.0 + index / 100.0,
        ground_truth_laterality_resolved=laterality,
    )


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0])
    second = bootstrap_mean_ci([1.0, 2.0, 3.0])

    assert first == second
    assert first.lower <= 2.0 <= first.upper


def test_nested_analysis_counts_and_subset() -> None:
    expanded = tuple(
        make_row(i, laterality=(i != 99))
        for i in range(100)
    )
    pilot = expanded[:30]

    result = analyze_nested_cohorts(pilot, expanded)

    assert result["design"]["pilot30_case_count"] == 30
    assert result["design"]["expanded100_case_count"] == 100
    assert result["design"]["additional_case_count"] == 70
    assert result["laterality"]["expanded100_resolved"] == 99
    assert result["laterality"]["expanded100_unresolved"] == 1
    assert result["cyst_positive_counts"]["expanded100"] == 50


def test_non_nested_cohort_is_rejected() -> None:
    expanded = tuple(make_row(i) for i in range(100))
    pilot = list(expanded[:29])
    pilot.append(
        CaseRow(
            **{
                **make_row(999).__dict__,
                "case_id": "case_missing",
            }
        )
    )

    with pytest.raises(ValueError, match="not a subset"):
        analyze_nested_cohorts(tuple(pilot), expanded)


def test_wrong_expanded_size_is_rejected() -> None:
    expanded = tuple(make_row(i) for i in range(99))
    pilot = expanded[:30]

    with pytest.raises(ValueError, match="N=100"):
        analyze_nested_cohorts(pilot, expanded)


def test_read_case_rows_supports_nullable_laterality(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cases.csv"
    fields = [
        "case_id",
        "kidney_and_mass_dice",
        "kidney_and_mass_surface_dice",
        "kidney_and_mass_hd95_mm",
        "kidney_core_dice",
        "cyst_dice",
        "cyst_reference_volume_ml",
        "inference_wall_time_seconds",
        "ground_truth_laterality_resolved",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case_00563",
                "kidney_and_mass_dice": "0.90",
                "kidney_and_mass_surface_dice": "0.75",
                "kidney_and_mass_hd95_mm": "4.5",
                "kidney_core_dice": "0.89",
                "cyst_dice": "0.20",
                "cyst_reference_volume_ml": "57.0",
                "inference_wall_time_seconds": "15.2",
                "ground_truth_laterality_resolved": "False",
            }
        )

    rows = read_case_rows(path)

    assert len(rows) == 1
    assert rows[0].case_id == "case_00563"
    assert rows[0].ground_truth_laterality_resolved is False
