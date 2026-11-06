from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.cyst_oracle_prompt import (
    OracleCystPrompt,
    build_interactive_point_command,
    evaluate_interactive_prediction,
    select_largest_cyst_oracle_prompt,
    summarize_interactive_results,
    InteractiveCystResult,
)


def test_oracle_prompt_is_inside_largest_component() -> None:
    segmentation = np.zeros((20, 20, 20), dtype=np.uint8)
    segmentation[2:5, 2:5, 2:5] = 3
    segmentation[10:18, 10:18, 10:18] = 3

    prompt = select_largest_cyst_oracle_prompt(
        case_id="fixture",
        segmentation=segmentation,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert prompt is not None
    assert segmentation[prompt.point_xyz] == 3
    assert all(10 <= value < 18 for value in prompt.point_xyz)
    assert prompt.interior_radius_mm > 0.0


def test_no_cyst_returns_no_prompt() -> None:
    segmentation = np.zeros((5, 5, 5), dtype=np.uint8)

    assert select_largest_cyst_oracle_prompt(
        case_id="fixture",
        segmentation=segmentation,
        spacing_mm=(1.0, 1.0, 1.0),
    ) is None


def test_command_uses_single_positive_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conda = tmp_path / "conda.exe"
    fake_conda.write_bytes(b"fixture")

    monkeypatch.setattr(
        "vista3d_tfg.experiments.cyst_oracle_prompt.resolve_conda_executable",
        lambda: str(fake_conda),
    )

    command = build_interactive_point_command(
        environment_name="vista3d-nv",
        image_path=tmp_path / "image.nii.gz",
        output_dir=tmp_path / "output",
        point_xyz=(10, 20, 30),
    )

    assert "'points': [[10, 20, 30]]" in command[-3]
    assert "'point_labels': [1]" in command[-3]
    assert "label_prompt" not in command[-3]
    assert all("\n" not in str(value) for value in command)


def save_volume(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        path,
    )


def test_interactive_evaluation_measures_target_component(
    tmp_path: Path,
) -> None:
    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    gt[2:6, 2:6, 2:6] = 3
    gt[12:18, 12:18, 12:18] = 3

    prompt = select_largest_cyst_oracle_prompt(
        case_id="fixture",
        segmentation=gt,
        spacing_mm=(1.0, 1.0, 1.0),
    )
    assert prompt is not None

    automatic = np.zeros_like(gt, dtype=np.uint8)
    interactive = np.zeros_like(gt, dtype=np.uint8)
    interactive[12:18, 12:18, 12:18] = 1
    interactive[0, 0, 0] = 255

    gt_path = tmp_path / "gt.nii.gz"
    auto_path = tmp_path / "auto.nii.gz"
    interactive_path = tmp_path / "interactive.nii.gz"

    save_volume(gt_path, gt)
    save_volume(auto_path, automatic)
    save_volume(interactive_path, interactive)

    result = evaluate_interactive_prediction(
        prompt=prompt,
        ground_truth_path=gt_path,
        automatic_prediction_path=auto_path,
        interactive_prediction_path=interactive_path,
        wall_time_seconds=2.0,
    )

    assert result.automatic_target_dice == 0.0
    assert result.interactive_target_dice == 1.0
    assert result.contains_255_unprocessed is True


def result(
    case_id: str,
    automatic: float,
    interactive: float,
) -> InteractiveCystResult:
    return InteractiveCystResult(
        case_id=case_id,
        point_xyz=(1, 1, 1),
        target_volume_ml=10.0,
        automatic_target_dice=automatic,
        interactive_target_dice=interactive,
        dice_delta=interactive - automatic,
        interactive_prediction_volume_ml=10.0,
        interactive_signed_volume_error_ml=0.0,
        interactive_labels_present=(0, 1),
        contains_255_unprocessed=False,
        wall_time_seconds=1.0,
        prediction_path="fixture.nii.gz",
    )


def test_summary_counts_rescued_zero_overlap() -> None:
    summary = summarize_interactive_results(
        [
            result("a", 0.0, 0.8),
            result("b", 0.2, 0.7),
            result("c", 0.0, 0.0),
        ]
    )

    assert summary.case_count == 3
    assert summary.improved_case_count == 2
    assert summary.automatic_zero_overlap_case_count == 2
    assert summary.zero_overlap_rescued_case_count == 1
    assert summary.interactive_dice_ge_050_count == 2
