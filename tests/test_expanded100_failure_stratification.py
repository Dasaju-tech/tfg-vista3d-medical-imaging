from __future__ import annotations

import pytest

from vista3d_tfg.evaluation.failure_stratification import (
    FailureCase,
    analyze_failure_stratification,
    pearson_or_none,
)


def make_row(index: int) -> FailureCase:
    return FailureCase(
        case_id=f"case_{index:05d}",
        kidney_and_mass_dice=0.95 - index / 1000.0,
        kidney_and_mass_surface_dice=(
            0.90 - index / 1000.0
        ),
        kidney_and_mass_hd95_mm=2.0 + index / 10.0,
        kidney_core_dice=0.94 - index / 1000.0,
        cyst_dice=0.2 + index / 10000.0,
        cyst_reference_volume_ml=(
            float(index) if index % 2 == 0 else 0.0
        ),
        cyst_prediction_volume_ml=(
            float(index) / 2 if index % 2 == 0 else 0.0
        ),
        gt_tumor_volume_ml=(
            float(index) / 3 if index % 3 == 0 else 0.0
        ),
        left_renal_volume_error_ml=-float(index),
        right_renal_volume_error_ml=-float(index) / 2,
        ground_truth_laterality_resolved=True,
        kidney_and_mass_components=2,
        tumor_components=1 if index % 3 == 0 else 0,
        cyst_components=1 if index % 2 == 0 else 0,
        anatomy_qc_flags=(),
    )


def test_pearson_or_none() -> None:
    assert pearson_or_none(
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    ) == pytest.approx(-1.0)


def test_constant_correlation_is_undefined() -> None:
    assert pearson_or_none(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ) is None


def test_failure_stratification_requires_100_cases() -> None:
    with pytest.raises(ValueError, match="N=100"):
        analyze_failure_stratification(
            make_row(i) for i in range(99)
        )


def test_failure_stratification_counts_and_rankings() -> None:
    rows = tuple(make_row(i) for i in range(100))

    result = analyze_failure_stratification(rows)

    assert result["case_count"] == 100
    assert result["cyst_positive_count"] == 49
    assert result["tumor_positive_count"] == 33

    worst_hd95 = result["rankings"]["worst_hd95"]
    assert worst_hd95[0]["case_id"] == "case_00099"

    worst_surface = result["rankings"]["worst_surface_dice"]
    assert worst_surface[0]["case_id"] == "case_00099"

    largest_under = result["rankings"][
        "largest_undersegmentation"
    ]
    assert largest_under[0]["case_id"] == "case_00099"


def test_unresolved_laterality_is_counted_without_volume_ranking() -> None:
    rows = [make_row(i) for i in range(100)]

    row = rows[99]
    rows[99] = FailureCase(
        **{
            **row.__dict__,
            "left_renal_volume_error_ml": None,
            "right_renal_volume_error_ml": None,
            "ground_truth_laterality_resolved": False,
            "kidney_and_mass_components": 1,
            "anatomy_qc_flags": (
                "fewer_than_two_kidney_and_mass_components",
            ),
        }
    )

    result = analyze_failure_stratification(rows)

    assert result["laterality_unresolved_count"] == 1

    ranked_ids = {
        item["case_id"]
        for item in result["rankings"][
            "largest_undersegmentation"
        ]
    }
    assert "case_00099" not in ranked_ids
