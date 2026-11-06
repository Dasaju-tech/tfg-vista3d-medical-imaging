from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vista3d_tfg.experiments.cyst_oracle_click_curve import (
    ClickCountSummary,
    OracleClickRound,
    OracleClickTrajectory,
    OracleCorrectionClick,
    build_multi_point_command,
    select_next_oracle_correction,
    summarize_click_curve,
)


def test_next_click_prefers_larger_false_negative() -> None:
    target = np.zeros((20, 20, 20), dtype=bool)
    target[2:10, 2:10, 2:10] = True

    prediction = np.zeros_like(target)
    prediction[2:5, 2:5, 2:5] = True

    click = select_next_oracle_correction(
        index=2,
        target=target,
        prediction_foreground=prediction,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert click is not None
    assert click.point_label == 1
    assert click.source == "largest_false_negative_component"
    assert target[click.point_xyz]


def test_next_click_can_be_negative_for_large_false_positive() -> None:
    target = np.zeros((20, 20, 20), dtype=bool)
    target[2:4, 2:4, 2:4] = True

    prediction = target.copy()
    prediction[10:18, 10:18, 10:18] = True

    click = select_next_oracle_correction(
        index=2,
        target=target,
        prediction_foreground=prediction,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert click is not None
    assert click.point_label == 0
    assert click.source == "largest_false_positive_component"
    assert not target[click.point_xyz]


def test_perfect_prediction_needs_no_click() -> None:
    target = np.zeros((5, 5, 5), dtype=bool)
    target[1:4, 1:4, 1:4] = True

    assert select_next_oracle_correction(
        index=2,
        target=target,
        prediction_foreground=target.copy(),
        spacing_mm=(1.0, 1.0, 1.0),
    ) is None


def test_multi_point_command_contains_all_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conda = tmp_path / "conda.exe"
    fake_conda.write_bytes(b"fixture")

    monkeypatch.setattr(
        "vista3d_tfg.experiments.cyst_oracle_click_curve.resolve_conda_executable",
        lambda: str(fake_conda),
    )

    clicks = (
        OracleCorrectionClick(
            1, (1, 2, 3), 1, "initial", 10, 2.0
        ),
        OracleCorrectionClick(
            2, (4, 5, 6), 0, "fp", 8, 1.0
        ),
    )

    command = build_multi_point_command(
        environment_name="vista3d-nv",
        image_path=tmp_path / "image.nii.gz",
        output_dir=tmp_path / "output",
        clicks=clicks,
    )

    assert "'points': [[1, 2, 3], [4, 5, 6]]" in command[-3]
    assert "'point_labels': [1, 0]" in command[-3]
    assert all("\n" not in str(value) for value in command)


def make_trajectory(
    case_id: str,
    automatic: float,
    dice_values: tuple[float, ...],
) -> OracleClickTrajectory:
    clicks = tuple(
        OracleCorrectionClick(
            index=index,
            point_xyz=(index, index, index),
            point_label=1,
            source="fixture",
            error_component_voxels=10,
            interior_radius_mm=1.0,
        )
        for index in range(1, len(dice_values) + 1)
    )
    rounds = tuple(
        OracleClickRound(
            click_count=index,
            dice=dice,
            prediction_volume_ml=10.0,
            signed_volume_error_ml=0.0,
            labels_present=(0, 1),
            contains_255_unprocessed=False,
            wall_time_seconds=1.0,
            prediction_path="fixture.nii.gz",
        )
        for index, dice in enumerate(dice_values, start=1)
    )

    return OracleClickTrajectory(
        case_id=case_id,
        target_volume_ml=10.0,
        automatic_target_dice=automatic,
        clicks=clicks,
        rounds=rounds,
    )


def test_click_curve_summary_tracks_improvement() -> None:
    summary = summarize_click_curve(
        [
            make_trajectory("a", 0.0, (0.2, 0.6, 0.8)),
            make_trajectory("b", 0.1, (0.4, 0.5, 0.7)),
        ],
        maximum_clicks=3,
    )

    assert summary.case_count == 2
    assert summary.round_summaries[0].dice_median == pytest.approx(0.3)
    assert summary.round_summaries[2].dice_median == pytest.approx(0.75)
    assert summary.improved_from_click1_count == 2
    assert summary.regressed_from_click1_count == 0
    assert summary.automatic_zero_overlap_rescued_by_final_count == 1


def test_early_perfect_trajectory_is_carried_forward() -> None:
    summary = summarize_click_curve(
        [
            make_trajectory("a", 0.0, (1.0,)),
            make_trajectory("b", 0.0, (0.2, 0.4, 0.6)),
        ],
        maximum_clicks=3,
    )

    assert summary.round_summaries[2].dice_mean == pytest.approx(0.8)
