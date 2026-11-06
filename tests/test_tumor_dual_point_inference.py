from __future__ import annotations

from pathlib import Path

import pytest

from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_dual_point_inference import (
    DualPointComparison,
    build_dual_point_command,
    summarize_dual_point_results,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)


def test_dual_command_uses_two_special_positive_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conda = tmp_path / "conda.exe"
    fake_conda.write_bytes(b"fixture")

    monkeypatch.setattr(
        "vista3d_tfg.experiments.tumor_dual_point_inference.resolve_conda_executable",
        lambda: str(fake_conda),
    )

    command = build_dual_point_command(
        environment_name="vista3d-nv",
        image_path=tmp_path / "image.nii.gz",
        output_dir=tmp_path / "output",
        first_point_xyz=(1, 2, 3),
        second_point_xyz=(4, 5, 6),
    )

    assert "'points': [[1, 2, 3], [4, 5, 6]]" in command[-3]
    assert "'point_labels': [3, 3]" in command[-3]
    assert "label_prompt" not in command[-3]
    assert all(
        "\n" not in str(value)
        for value in command
    )


def _result(
    case_id: str,
    *,
    dice: float,
    hd95_reduction: float,
) -> TumorOracleSmokeResult:
    return TumorOracleSmokeResult(
        case_id=case_id,
        point_xyz=(1, 2, 3),
        point_label=3,
        target_volume_ml=10.0,
        total_tumor_volume_ml=10.0,
        interactive_target_dice=dice,
        interactive_full_tumor_dice=dice,
        original_tumor_coverage_fraction=0.1,
        merged_tumor_coverage_fraction=0.7,
        tumor_coverage_gain=0.6,
        original_hec_dice=0.5,
        merged_hec_dice=0.7,
        hec_dice_delta=0.2,
        original_hec_surface_dice=0.4,
        merged_hec_surface_dice=0.5,
        hec_surface_dice_delta=0.1,
        original_hec_hd95_mm=50.0,
        merged_hec_hd95_mm=(
            50.0 - hd95_reduction
        ),
        hec_hd95_reduction_mm=hd95_reduction,
        interactive_prediction_volume_ml=10.0,
        interactive_labels_present=(0, 1, 255),
        contains_255_unprocessed=True,
        wall_time_seconds=2.0,
        prediction_path="fixture.nii.gz",
    )


def test_summary_is_balanced_and_paired() -> None:
    rows = []
    case_index = 0

    for quartile in QUARTILE_NAMES:
        for _ in range(3):
            case_id = f"case_{case_index:05d}"
            rows.append(
                DualPointComparison(
                    case_id=case_id,
                    quartile=quartile,
                    largest_component_volume_ml=10.0,
                    point_separation_mm=20.0 + case_index,
                    second_point_relative_depth=0.50,
                    dual=_result(
                        case_id,
                        dice=0.82,
                        hd95_reduction=32.0,
                    ),
                    deep=_result(
                        case_id,
                        dice=0.80,
                        hd95_reduction=30.0,
                    ),
                )
            )
            case_index += 1

    payload = summarize_dual_point_results(
        rows
    )

    assert payload["case_count"] == 12

    overall = payload["overall"]
    assert overall["dual"]["tumor_dice"]["mean"] == pytest.approx(
        0.82
    )
    assert overall["deep"]["tumor_dice"]["mean"] == pytest.approx(
        0.80
    )
    assert overall["paired_dual_minus_deep"]["tumor_dice"][
        "mean"
    ] == pytest.approx(0.02)
    assert overall["paired_dual_minus_deep"][
        "cases_within_0_05_tumor_dice"
    ] == 12

    for quartile in QUARTILE_NAMES:
        assert payload["by_quartile"][quartile]["count"] == 3


def test_summary_requires_twelve_cases() -> None:
    with pytest.raises(ValueError, match="12"):
        summarize_dual_point_results([])
