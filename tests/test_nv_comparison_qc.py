from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.visualization.nv_compare import (
    build_case_comparison_qc,
    disagreement_map,
    select_comparison_slice,
    write_comparison_qc_index,
)


def fixture_volumes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.zeros((12, 10, 14), dtype=np.int16)

    gt = np.zeros_like(image, dtype=np.uint8)
    gt[1:5, 2:8, 2:12] = 1
    gt[7:11, 2:8, 2:12] = 1
    gt[3:4, 4:6, 6:9] = 2

    pred = np.zeros_like(gt, dtype=np.uint8)
    pred[1:5, 2:8, 2:12] = 14
    pred[7:11, 2:8, 2:12] = 5

    pred[7:9, 2:4, 7:10] = 0

    return image, gt, pred


def test_disagreement_map_encodes_false_negative() -> None:
    _, gt, pred = fixture_volumes()

    errors = disagreement_map(gt, pred)

    assert 3 in np.unique(errors)
    assert np.count_nonzero(errors == 2) == 0


def test_slice_selection_prefers_maximum_disagreement() -> None:
    _, gt, pred = fixture_volumes()

    selection = select_comparison_slice(
        gt,
        pred,
        plane="axial",
    )

    assert selection.source == "maximum_disagreement"
    assert selection.disagreement_voxels > 0


def test_identical_envelope_falls_back_to_gt_lesion() -> None:
    _, gt, pred = fixture_volumes()

    pred[:] = 0
    pred[np.isin(gt, (1, 2, 3))] = 5

    selection = select_comparison_slice(
        gt,
        pred,
        plane="coronal",
    )

    assert selection.source == "maximum_ground_truth_lesion"
    assert selection.disagreement_voxels == 0


def save_triplet(
    root: Path,
) -> tuple[Path, Path, Path]:
    image, gt, pred = fixture_volumes()

    image_path = root / "imaging.nii.gz"
    gt_path = root / "segmentation.nii.gz"
    pred_path = root / "prediction.nii.gz"

    affine = np.eye(4)

    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(gt, affine), gt_path)
    nib.save(nib.Nifti1Image(pred, affine), pred_path)

    return image_path, gt_path, pred_path


def test_build_case_qc_creates_three_figures(
    tmp_path: Path,
) -> None:
    image_path, gt_path, pred_path = save_triplet(tmp_path)

    result = build_case_comparison_qc(
        case_id="fixture",
        image_path=image_path,
        ground_truth_path=gt_path,
        prediction_path=pred_path,
        output_directory=tmp_path / "qc",
    )

    assert len(result.figure_files) == 3
    assert all(Path(path).is_file() for path in result.figure_files)
    assert all(
        Path(path).stat().st_size > 0
        for path in result.figure_files
    )
    assert result.total_false_negative_voxels > 0


def test_qc_index_is_structured_json(tmp_path: Path) -> None:
    image_path, gt_path, pred_path = save_triplet(tmp_path)

    result = build_case_comparison_qc(
        case_id="fixture",
        image_path=image_path,
        ground_truth_path=gt_path,
        prediction_path=pred_path,
        output_directory=tmp_path / "qc",
    )

    output = tmp_path / "comparison_qc.json"
    write_comparison_qc_index(result, output)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"] == "KiTS23"
    assert payload["model"] == "NV-Segment-CT"
    assert payload["result"]["case_id"] == "fixture"
    assert len(payload["result"]["selected_slices"]) == 3


def test_unknown_plane_is_rejected() -> None:
    _, gt, pred = fixture_volumes()

    with pytest.raises(ValueError, match="Unknown plane"):
        select_comparison_slice(
            gt,
            pred,
            plane="oblique",
        )
