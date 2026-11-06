from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_oracle_protocol import (
    SPECIAL_TUMOR_POSITIVE_POINT_LABEL,
    audit_tumor_runtime_contract,
    largest_tumor_deepest_point,
    prepare_tumor_oracle_prompts,
)


def _save(path: Path, array: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(array, np.eye(4)), str(path))


def test_runtime_contract_requires_current_renal_labels() -> None:
    labels = {
        "right kidney": 5,
        "left kidney": 14,
        "left kidney cyst": 116,
        "right kidney cyst": 117,
    }
    result = audit_tumor_runtime_contract(labels)

    assert SPECIAL_TUMOR_POSITIVE_POINT_LABEL == 3
    assert result["automatic_renal_labels_verified"] is True
    assert result["automatic_kidney_tumor_available"] is False
    assert result["recommended_protocol"] == (
        "oracle_special_positive_point_without_label_prompt"
    )


def test_runtime_contract_detects_kidney_tumor_name() -> None:
    labels = {
        "right kidney": 5,
        "left kidney": 14,
        "left kidney cyst": 116,
        "right kidney cyst": 117,
        "kidney tumor": 129,
    }
    result = audit_tumor_runtime_contract(labels)
    assert result["automatic_kidney_tumor_available"] is True
    assert result["automatic_kidney_tumor_names"] == ["kidney tumor"]


def test_largest_tumor_deepest_point_targets_largest_component() -> None:
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[2:10, 2:10, 2:10] = True
    mask[20:22, 20:22, 20:22] = True

    _, point, voxels, radius = largest_tumor_deepest_point(
        mask,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert voxels == 8 * 8 * 8
    assert mask[point]
    assert radius > 0.0


def test_prepare_prompts_accepts_one_no_tumor_case(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.csv"
    lines = ["case_id,dominant_direction"]
    dataset = tmp_path / "kits23"
    dataset.mkdir()

    for index in range(9):
        case_id = f"case_{index:05d}"
        lines.append(f"{case_id},gt_to_prediction")
        case_root = dataset / case_id
        case_root.mkdir()
        gt = np.zeros((20, 20, 20), dtype=np.uint8)
        if index != 8:
            gt[4:12, 4:12, 4:12] = 2
        _save(case_root / "segmentation.nii.gz", gt)

    geometry.write_text("\n".join(lines) + "\n", encoding="utf-8")

    prompts, no_tumor = prepare_tumor_oracle_prompts(
        geometry_csv=geometry,
        dataset_root=dataset,
    )

    assert len(prompts) == 8
    assert no_tumor == ("case_00008",)
