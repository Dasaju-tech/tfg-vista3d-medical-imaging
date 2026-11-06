from __future__ import annotations

import numpy as np
import pytest

from vista3d_tfg.evaluation.component_filter_sensitivity import (
    SensitivityCase,
    TaxonomyRow,
    VariantMetric,
    analyze_component_filter_sensitivity,
    filter_global_top2,
    filter_side_largest,
)
from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS


def _label(name: str) -> int:
    return int(EXPECTED_RENAL_LABELS[name])


def test_global_top2_removes_third_component() -> None:
    pred = np.zeros((30, 30, 30), dtype=np.int16)

    pred[2:8, 2:8, 2:8] = _label("left kidney")
    pred[15:22, 15:22, 15:22] = _label("right kidney")
    pred[25:27, 2:4, 2:4] = _label("left kidney")

    filtered, removed = filter_global_top2(pred)

    assert removed == 8
    assert np.count_nonzero(filtered[25:27, 2:4, 2:4]) == 0
    assert np.count_nonzero(filtered[2:8, 2:8, 2:8]) > 0
    assert np.count_nonzero(filtered[15:22, 15:22, 15:22]) > 0


def test_side_largest_keeps_one_component_per_side() -> None:
    pred = np.zeros((30, 30, 30), dtype=np.int16)

    pred[2:8, 2:8, 2:8] = _label("left kidney")
    pred[20:22, 2:4, 2:4] = _label("left kidney")
    pred[15:22, 15:22, 15:22] = _label("right kidney")

    filtered, removed = filter_side_largest(pred)

    assert removed == 8
    assert np.count_nonzero(filtered[20:22, 2:4, 2:4]) == 0
    assert np.count_nonzero(filtered[2:8, 2:8, 2:8]) > 0
    assert np.count_nonzero(filtered[15:22, 15:22, 15:22]) > 0


def make_case(index: int) -> SensitivityCase:
    original = VariantMetric(
        dice=0.90,
        surface_dice=0.75,
        hd95_mm=(20.0 if index < 22 else 4.0),
    )
    improved = VariantMetric(
        dice=0.90,
        surface_dice=0.76,
        hd95_mm=(8.0 if index < 12 else original.hd95_mm),
    )

    return SensitivityCase(
        case_id=f"case_{index:05d}",
        failure_mode=(
            "distant_boundary_failure"
            if index < 12
            else "mixed_hd95_failure"
            if index < 22
            else "non_hd95_outlier"
        ),
        hd95_outlier=(index < 22),
        original=original,
        global_top2=improved,
        side_largest=improved,
        global_top2_removed_voxels=10,
        side_largest_removed_voxels=10,
    )


def test_analysis_requires_12_prediction_to_gt_targets() -> None:
    rows = tuple(make_case(i) for i in range(100))

    with pytest.raises(ValueError, match="12"):
        analyze_component_filter_sensitivity(
            rows,
            prediction_to_gt_case_ids=(
                row.case_id for row in rows[:11]
            ),
        )


def test_analysis_reports_target_resolution() -> None:
    rows = tuple(make_case(i) for i in range(100))
    target_ids = tuple(row.case_id for row in rows[:12])

    result = analyze_component_filter_sensitivity(
        rows,
        prediction_to_gt_case_ids=target_ids,
    )

    assert result["case_count"] == 100
    assert result["prediction_to_gt_target_count"] == 12
    assert result["other_hd95_outlier_count"] == 10
    assert result["non_outlier_count"] == 78

    targets = result["variants"]["global_top2"][
        "prediction_to_gt_targets"
    ]
    assert targets["resolved_below_original_threshold"] >= 0
