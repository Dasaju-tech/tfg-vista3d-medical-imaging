from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_oracle_smoke import (
    build_tumor_special_point_command,
    evaluate_tumor_oracle_smoke,
    load_frozen_prompt,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    SPECIAL_TUMOR_POSITIVE_POINT_LABEL,
    TumorOraclePrompt,
)


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_load_frozen_prompt_requires_oracle_contract(
    tmp_path: Path,
) -> None:
    protocol = {
        "oracle": True,
        "deployment_valid": False,
        "input_protocol": {
            "label_prompt": None,
            "point_labels": [3],
        },
        "prompts": [
            {
                "case_id": "case_00430",
                "component_id": 1,
                "point_xyz": [1, 2, 3],
                "target_voxels": 10,
                "target_volume_ml": 1.0,
                "total_tumor_volume_ml": 1.0,
                "interior_radius_mm": 2.0,
            }
        ],
    }
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )

    prompt = load_frozen_prompt(
        path,
        case_id="case_00430",
    )

    assert prompt.point_xyz == (1, 2, 3)
    assert prompt.component_id == 1


def test_command_uses_special_positive_point_label_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conda = tmp_path / "conda.exe"
    fake_conda.write_bytes(b"fixture")

    monkeypatch.setattr(
        "vista3d_tfg.experiments.tumor_oracle_smoke.resolve_conda_executable",
        lambda: str(fake_conda),
    )

    command = build_tumor_special_point_command(
        environment_name="vista3d-nv",
        image_path=tmp_path / "image.nii.gz",
        output_dir=tmp_path / "output",
        point_xyz=(41, 218, 348),
    )

    assert SPECIAL_TUMOR_POSITIVE_POINT_LABEL == 3
    assert "'points': [[41, 218, 348]]" in command[-3]
    assert "'point_labels': [3]" in command[-3]
    assert "label_prompt" not in command[-3]
    assert all(
        "\n" not in str(value)
        for value in command
    )


def test_perfect_interactive_tumor_improves_hec(
    tmp_path: Path,
) -> None:
    gt = np.zeros((40, 40, 40), dtype=np.uint8)
    automatic = np.zeros_like(gt)
    interactive = np.zeros_like(gt)

    gt[5:15, 5:15, 5:15] = 1
    automatic[5:15, 5:15, 5:15] = 14

    gt[15:25, 8:12, 8:12] = 2
    interactive[15:25, 8:12, 8:12] = 1
    interactive[0, 0, 0] = 255

    gt_path = tmp_path / "gt.nii.gz"
    automatic_path = tmp_path / "auto.nii.gz"
    interactive_path = tmp_path / "interactive.nii.gz"

    _save(gt_path, gt)
    _save(automatic_path, automatic)
    _save(interactive_path, interactive)

    prompt = TumorOraclePrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(18, 9, 9),
        target_voxels=10 * 4 * 4,
        target_volume_ml=0.16,
        total_tumor_volume_ml=0.16,
        interior_radius_mm=2.0,
    )

    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=gt_path,
        automatic_prediction_path=automatic_path,
        interactive_prediction_path=interactive_path,
        wall_time_seconds=1.0,
    )

    assert result.interactive_target_dice == pytest.approx(1.0)
    assert result.interactive_full_tumor_dice == pytest.approx(1.0)
    assert result.merged_tumor_coverage_fraction == pytest.approx(1.0)
    assert result.tumor_coverage_gain > 0.0
    assert result.hec_dice_delta > 0.0
    assert result.hec_hd95_reduction_mm >= 0.0
    assert result.contains_255_unprocessed is True
