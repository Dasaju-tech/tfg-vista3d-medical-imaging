from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.evaluation.nv_kits23 import (
    KITS23_KIDNEY_AND_MASS_TOLERANCE_MM,
    compare_binary_masks,
    evaluate_kidney_and_mass_hec,
    evaluate_supported_renal_prediction,
)


def test_binary_comparison_empty_masks_are_perfect() -> None:
    empty = np.zeros((4, 4, 4), dtype=bool)

    result = compare_binary_masks(
        empty,
        empty,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.dice == 1.0
    assert result.reference_volume_ml == 0.0
    assert result.prediction_volume_ml == 0.0


def test_binary_comparison_volume_error_uses_physical_spacing() -> None:
    reference = np.zeros((3, 3, 3), dtype=bool)
    prediction = np.zeros_like(reference)

    reference[0, 0, 0] = True
    prediction[0, 0, 0] = True
    prediction[1, 1, 1] = True

    result = compare_binary_masks(
        reference,
        prediction,
        spacing_mm=(2.0, 2.0, 2.0),
    )

    assert result.reference_volume_ml == pytest.approx(0.008)
    assert result.prediction_volume_ml == pytest.approx(0.016)
    assert result.signed_volume_error_ml == pytest.approx(0.008)


def test_hec_identical_masks_are_perfect() -> None:
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True

    result = evaluate_kidney_and_mass_hec(
        mask,
        mask,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.tolerance_mm == pytest.approx(
        KITS23_KIDNEY_AND_MASS_TOLERANCE_MM
    )
    assert result.dice == 1.0
    assert result.surface_dice == 1.0
    assert result.diagnostic_hd95_mm == 0.0


def test_hec_one_empty_mask_is_zero_and_infinite_hd95() -> None:
    reference = np.zeros((4, 4, 4), dtype=bool)
    reference[1:3, 1:3, 1:3] = True
    prediction = np.zeros_like(reference)

    result = evaluate_kidney_and_mass_hec(
        reference,
        prediction,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.dice == 0.0
    assert result.surface_dice == 0.0
    assert np.isinf(result.diagnostic_hd95_mm)


def save_case(
    tmp_path: Path,
    *,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[Path, Path]:
    ground_truth_path = tmp_path / "segmentation.nii.gz"
    prediction_path = tmp_path / "prediction.nii.gz"

    nib.save(
        nib.Nifti1Image(ground_truth, np.eye(4)),
        ground_truth_path,
    )
    nib.save(
        nib.Nifti1Image(prediction, np.eye(4)),
        prediction_path,
    )

    return ground_truth_path, prediction_path


def bilateral_case() -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros((12, 8, 8), dtype=np.uint8)

    # Lower RAS+ x is left.
    gt[1:5, 1:7, 1:7] = 1
    gt[7:11, 1:7, 1:7] = 1

    gt[2:3, 3:5, 3:5] = 2
    gt[8:9, 3:5, 3:5] = 3

    pred = np.zeros_like(gt, dtype=np.uint8)

    # NV label 14 = left kidney, 5 = right kidney.
    pred[1:5, 1:7, 1:7] = 14
    pred[7:11, 1:7, 1:7] = 5

    # The right cyst is represented explicitly.
    pred[8:9, 3:5, 3:5] = 117

    return gt, pred


def test_supported_case_evaluation_is_side_aware(
    tmp_path: Path,
) -> None:
    gt, pred = bilateral_case()
    gt_path, pred_path = save_case(
        tmp_path,
        ground_truth=gt,
        prediction=pred,
    )

    result = evaluate_supported_renal_prediction(
        case_id="fixture",
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert result.case_id == "fixture"
    assert result.full_kits23_score_available is False
    assert result.gt_tumor_volume_ml > 0.0
    assert result.left_renal_envelope.dice > 0.9
    assert result.right_renal_envelope.dice > 0.9
    assert result.right_cyst.dice == 1.0


def test_prediction_with_unknown_label_is_rejected(
    tmp_path: Path,
) -> None:
    gt, pred = bilateral_case()
    pred[0, 0, 0] = 99

    gt_path, pred_path = save_case(
        tmp_path,
        ground_truth=gt,
        prediction=pred,
    )

    with pytest.raises(ValueError, match="Unexpected NV-Segment labels"):
        evaluate_supported_renal_prediction(
            case_id="fixture",
            ground_truth_path=gt_path,
            prediction_path=pred_path,
        )


def test_affine_mismatch_is_rejected(tmp_path: Path) -> None:
    gt, pred = bilateral_case()

    gt_path = tmp_path / "gt.nii.gz"
    pred_path = tmp_path / "pred.nii.gz"

    nib.save(
        nib.Nifti1Image(gt, np.eye(4)),
        gt_path,
    )

    affine = np.eye(4)
    affine[0, 3] = 20.0

    nib.save(
        nib.Nifti1Image(pred, affine),
        pred_path,
    )

    with pytest.raises(ValueError, match="affines do not match"):
        evaluate_supported_renal_prediction(
            case_id="fixture",
            ground_truth_path=gt_path,
            prediction_path=pred_path,
        )
