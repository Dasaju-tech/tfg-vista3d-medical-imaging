from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.evaluation.residual_geometry_audit import (
    analyze_residual_geometry,
    evaluate_residual_geometry,
)


def save_pair(
    tmp_path: Path,
    gt: np.ndarray,
    pred: np.ndarray,
) -> tuple[Path, Path]:
    gt_path = tmp_path / "segmentation.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"

    nib.save(nib.Nifti1Image(gt, np.eye(4)), gt_path)
    nib.save(nib.Nifti1Image(pred, np.eye(4)), pred_path)
    return gt_path, pred_path


def test_distant_false_positive_is_prediction_to_gt_dominated(
    tmp_path: Path,
) -> None:
    gt = np.zeros((40, 40, 40), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[5:15, 5:15, 5:15] = 1
    pred[5:15, 5:15, 5:15] = 14

    # Distant false-positive renal island.
    pred[30:35, 30:35, 30:35] = 14

    gt_path, pred_path = save_pair(tmp_path, gt, pred)

    result = evaluate_residual_geometry(
        case_id="fp_fixture",
        failure_mode="mixed_hd95_failure",
        original_hd95_mm=20.0,
        dice=0.8,
        surface_dice=0.6,
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert result.fp_volume_ml > 0.0
    assert result.fn_volume_ml == 0.0
    assert result.fp_components == 1
    assert result.dominant_direction == "prediction_to_gt"
    assert (
        result.prediction_to_gt_surface_p95_mm
        > result.gt_to_prediction_surface_p95_mm
    )


def test_missing_reference_extent_is_gt_to_prediction_dominated(
    tmp_path: Path,
) -> None:
    gt = np.zeros((40, 40, 40), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[5:15, 5:15, 5:15] = 1
    pred[5:15, 5:15, 5:15] = 14

    # Distant GT-only component.
    gt[30:35, 30:35, 30:35] = 2

    gt_path, pred_path = save_pair(tmp_path, gt, pred)

    result = evaluate_residual_geometry(
        case_id="fn_fixture",
        failure_mode="mixed_hd95_failure",
        original_hd95_mm=20.0,
        dice=0.8,
        surface_dice=0.6,
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert result.fn_volume_ml > 0.0
    assert result.fp_volume_ml == 0.0
    assert result.fn_components == 1
    assert result.dominant_direction == "gt_to_prediction"
    assert (
        result.gt_to_prediction_surface_p95_mm
        > result.prediction_to_gt_surface_p95_mm
    )


def test_audit_requires_frozen_22_outliers() -> None:
    with pytest.raises(ValueError, match="22"):
        analyze_residual_geometry([])
