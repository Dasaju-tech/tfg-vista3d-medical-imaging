from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vista3d_tfg.evaluation.interaction_stability import BootstrapInterval
from vista3d_tfg.experiments.cyst_oracle_prompt import OracleCystPrompt
from vista3d_tfg.experiments.cyst_semantic_point import (
    LEFT_CYST_LABEL,
    RIGHT_CYST_LABEL,
    SemanticPointResult,
    build_semantic_point_command,
    infer_oracle_cyst_laterality,
    summarize_semantic_point_results,
)


def make_segmentation() -> np.ndarray:
    segmentation = np.zeros((30, 12, 12), dtype=np.uint8)

    # Two renal envelopes in voxel x.
    segmentation[2:10, 2:10, 2:10] = 1
    segmentation[20:28, 2:10, 2:10] = 1

    # Largest cyst belongs to the second voxel-x kidney.
    segmentation[22:26, 4:8, 4:8] = 3

    return segmentation


def make_prompt(
    segmentation: np.ndarray,
) -> OracleCystPrompt:
    target_voxels = int(np.count_nonzero(segmentation == 3))

    return OracleCystPrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(23, 5, 5),
        target_voxels=target_voxels,
        target_volume_ml=float(target_voxels / 1000.0),
        interior_radius_mm=2.0,
    )


def test_laterality_uses_world_coordinates() -> None:
    segmentation = make_segmentation()
    prompt = make_prompt(segmentation)

    # Positive voxel-x is positive RAS-x => second kidney is right.
    target = infer_oracle_cyst_laterality(
        case_id="fixture",
        segmentation=segmentation,
        affine=np.eye(4),
        prompt=prompt,
    )

    assert target.laterality == "right"
    assert target.class_label == RIGHT_CYST_LABEL

    # Flip x in the affine: the very same voxel component is now patient-left.
    affine = np.eye(4)
    affine[0, 0] = -1.0
    affine[0, 3] = 30.0

    target_flipped = infer_oracle_cyst_laterality(
        case_id="fixture",
        segmentation=segmentation,
        affine=affine,
        prompt=prompt,
    )

    assert target_flipped.laterality == "left"
    assert target_flipped.class_label == LEFT_CYST_LABEL


def test_semantic_point_command_combines_class_and_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conda = tmp_path / "conda.exe"
    fake_conda.write_bytes(b"fixture")

    monkeypatch.setattr(
        "vista3d_tfg.experiments.cyst_semantic_point.resolve_conda_executable",
        lambda: str(fake_conda),
    )

    command = build_semantic_point_command(
        environment_name="vista3d-nv",
        image_path=tmp_path / "image.nii.gz",
        output_dir=tmp_path / "output",
        class_label=116,
        point_xyz=(10, 20, 30),
    )

    input_argument = command[-3]

    assert "'label_prompt': [116]" in input_argument
    assert "'points': [[10, 20, 30]]" in input_argument
    assert "'point_labels': [1]" in input_argument
    assert all("\n" not in str(value) for value in command)


def test_invalid_semantic_class_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        build_semantic_point_command(
            environment_name="vista3d-nv",
            image_path=tmp_path / "image.nii.gz",
            output_dir=tmp_path / "output",
            class_label=129,
            point_xyz=(1, 2, 3),
        )


def row(
    case_id: str,
    *,
    point_only: float,
    semantic: float,
    laterality: str,
) -> SemanticPointResult:
    class_label = (
        LEFT_CYST_LABEL
        if laterality == "left"
        else RIGHT_CYST_LABEL
    )

    return SemanticPointResult(
        case_id=case_id,
        laterality=laterality,
        class_label=class_label,
        class_name=f"{laterality} kidney cyst",
        point_xyz=(1, 1, 1),
        target_volume_ml=10.0,
        automatic_side_class_dice=0.1,
        point_only_dice=point_only,
        semantic_plus_point_dice=semantic,
        semantic_plus_point_minus_point_only=semantic - point_only,
        semantic_plus_point_minus_automatic=semantic - 0.1,
        semantic_plus_point_volume_ml=10.0,
        semantic_plus_point_signed_volume_error_ml=0.0,
        labels_present=(0, 1),
        contains_255_unprocessed=False,
        wall_time_seconds=1.0,
        prediction_path="fixture.nii.gz",
    )


def test_summary_is_paired_against_point_only() -> None:
    summary = summarize_semantic_point_results(
        [
            row(
                "a",
                point_only=0.2,
                semantic=0.7,
                laterality="left",
            ),
            row(
                "b",
                point_only=0.8,
                semantic=0.9,
                laterality="right",
            ),
        ]
    )

    assert summary.case_count == 2
    assert summary.point_only_dice_mean == pytest.approx(0.5)
    assert summary.semantic_plus_point_dice_mean == pytest.approx(0.8)
    assert (
        summary.semantic_plus_point_minus_point_only_mean
        == pytest.approx(0.3)
    )
    assert summary.improved_over_point_only_count == 2
    assert summary.left_target_count == 1
    assert summary.right_target_count == 1
