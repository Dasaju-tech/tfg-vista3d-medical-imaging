from __future__ import annotations

import pytest

from vista3d_tfg.evaluation.interaction_stability import (
    bootstrap_mean_difference_ci,
    build_transition_observations,
    summarize_interaction_stability,
)
from vista3d_tfg.experiments.cyst_oracle_click_curve import (
    OracleClickRound,
    OracleClickTrajectory,
    OracleCorrectionClick,
)


def trajectory(
    case_id: str,
    *,
    dice: tuple[float, float, float],
    labels: tuple[int, int] = (1, 0),
) -> OracleClickTrajectory:
    clicks = (
        OracleCorrectionClick(
            index=1,
            point_xyz=(1, 1, 1),
            point_label=1,
            source="initial",
            error_component_voxels=10,
            interior_radius_mm=2.0,
        ),
        OracleCorrectionClick(
            index=2,
            point_xyz=(2, 2, 2),
            point_label=labels[0],
            source="correction2",
            error_component_voxels=8,
            interior_radius_mm=1.0,
        ),
        OracleCorrectionClick(
            index=3,
            point_xyz=(3, 3, 3),
            point_label=labels[1],
            source="correction3",
            error_component_voxels=5,
            interior_radius_mm=1.0,
        ),
    )

    rounds = tuple(
        OracleClickRound(
            click_count=index,
            dice=value,
            prediction_volume_ml=10.0,
            signed_volume_error_ml=0.0,
            labels_present=(0, 1),
            contains_255_unprocessed=False,
            wall_time_seconds=1.0,
            prediction_path=f"{case_id}_{index}.nii.gz",
        )
        for index, value in enumerate(dice, start=1)
    )

    return OracleClickTrajectory(
        case_id=case_id,
        target_volume_ml=10.0,
        automatic_target_dice=0.0,
        clicks=clicks,
        rounds=rounds,
    )


def test_transition_observations_capture_labels_and_deltas() -> None:
    observations = build_transition_observations(
        [trajectory("a", dice=(0.4, 0.6, 0.5), labels=(1, 0))]
    )

    assert len(observations) == 2
    assert observations[0].point_label == 1
    assert observations[0].dice_delta == pytest.approx(0.2)
    assert observations[1].point_label == 0
    assert observations[1].dice_delta == pytest.approx(-0.1)


def test_oracle_best_uses_earliest_round_on_tie() -> None:
    summary = summarize_interaction_stability(
        [
            trajectory("a", dice=(0.8, 0.8, 0.7)),
            trajectory("b", dice=(0.2, 0.5, 0.6)),
        ]
    )

    assert summary.best_round_click1_count == 1
    assert summary.best_round_click3_count == 1
    assert summary.best_round_click2_count == 0


def test_final_regret_detects_forced_click_damage() -> None:
    summary = summarize_interaction_stability(
        [
            trajectory("a", dice=(0.9, 0.6, 0.5)),
            trajectory("b", dice=(0.2, 0.4, 0.7)),
        ]
    )

    assert summary.final_regret_ge_010_count == 1
    assert summary.largest_regression_case_ids[0] == "a"
    assert summary.largest_improvement_case_ids[0] == "b"


def test_positive_and_negative_corrections_are_separated() -> None:
    summary = summarize_interaction_stability(
        [
            trajectory("a", dice=(0.2, 0.5, 0.4), labels=(1, 0)),
            trajectory("b", dice=(0.4, 0.3, 0.6), labels=(0, 1)),
        ]
    )

    assert summary.positive_correction is not None
    assert summary.negative_correction is not None
    assert summary.positive_correction.count == 2
    assert summary.negative_correction.count == 2


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_mean_difference_ci([0.0, 0.1, 0.2])
    second = bootstrap_mean_difference_ci([0.0, 0.1, 0.2])

    assert first == second
    assert first.lower <= 0.1 <= first.upper
