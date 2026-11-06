from __future__ import annotations

import numpy as np
import pytest

from vista3d_tfg.evaluation.prediction_geometry_trigger import (
    PredictionGeometry,
    analyze_prediction_geometry_trigger,
    rank_auc,
)


def make_row(index: int) -> PredictionGeometry:
    target = index < 12
    return PredictionGeometry(
        case_id=f"case_{index:05d}",
        prediction_to_gt_target=target,
        hd95_outlier=index < 22,
        total_renal_components=(4 if target else 2),
        left_components=(2 if target else 1),
        right_components=(2 if target else 1),
        total_renal_volume_ml=300.0,
        largest_component_volume_ml=150.0,
        second_component_volume_ml=140.0,
        third_plus_volume_ml=(10.0 if target else 0.0),
        third_plus_fraction=(0.033 if target else 0.0),
        smallest_component_volume_ml=(2.0 if target else 140.0),
        largest_to_second_ratio=1.07,
        max_component_centroid_distance_mm=(
            250.0 if target else 120.0
        ),
    )


def test_rank_auc_perfect_separation() -> None:
    auc = rank_auc(
        [0.0, 0.1, 0.9, 1.0],
        [False, False, True, True],
    )
    assert auc == pytest.approx(1.0)


def test_rank_auc_with_ties() -> None:
    auc = rank_auc(
        [1.0, 1.0, 2.0, 2.0],
        [False, True, False, True],
    )
    assert 0.0 <= auc <= 1.0


def test_analysis_requires_100_cases() -> None:
    with pytest.raises(ValueError, match="N=100"):
        analyze_prediction_geometry_trigger(
            make_row(i) for i in range(99)
        )


def test_analysis_finds_component_features() -> None:
    rows = tuple(make_row(i) for i in range(100))

    result = analyze_prediction_geometry_trigger(rows)

    assert result["case_count"] == 100
    assert result["target_count"] == 12
    assert result["best_feature"]["auc"] >= 0.99
    assert (
        result["best_feature"]["top12_target_count"]
        == 12
    )
