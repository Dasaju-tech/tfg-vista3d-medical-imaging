from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.evaluation.undercoverage_audit import (
    UndercoverageCase,
    analyze_undercoverage,
    evaluate_undercoverage_case,
)


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_undercoverage_decomposes_missing_labels(
    tmp_path: Path,
) -> None:
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2:12, 2:12, 2:12] = 1
    pred[2:12, 2:12, 2:12] = 14

    gt[15:25, 15:25, 15:25] = 2
    gt[5:8, 20:23, 5:8] = 3

    gt_path = tmp_path / "segmentation.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"
    _save(gt_path, gt)
    _save(pred_path, pred)

    row = evaluate_undercoverage_case(
        case_id="fixture",
        original_hd95_mm=50.0,
        gt_to_prediction_surface_p95_mm=49.0,
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert row.fn_kidney_volume_ml == pytest.approx(0.0)
    assert row.fn_tumor_volume_ml > 0.0
    assert row.fn_cyst_volume_ml > 0.0
    assert row.dominant_missing_label == "tumor"
    assert row.kidney_miss_fraction == pytest.approx(0.0)
    assert row.tumor_miss_fraction == pytest.approx(1.0)


def _case(index: int, dominant: str) -> UndercoverageCase:
    fn_kidney = 10.0 if dominant == "kidney" else 1.0
    fn_tumor = 10.0 if dominant == "tumor" else 1.0
    fn_cyst = 10.0 if dominant == "cyst" else 1.0
    total = fn_kidney + fn_tumor + fn_cyst

    return UndercoverageCase(
        case_id=f"case_{index:05d}",
        original_hd95_mm=50.0,
        gt_to_prediction_surface_p95_mm=45.0,
        gt_volume_ml=100.0,
        fn_volume_ml=total,
        fn_fraction=total / 100.0,
        gt_kidney_volume_ml=50.0,
        gt_tumor_volume_ml=30.0,
        gt_cyst_volume_ml=20.0,
        fn_kidney_volume_ml=fn_kidney,
        fn_tumor_volume_ml=fn_tumor,
        fn_cyst_volume_ml=fn_cyst,
        kidney_miss_fraction=fn_kidney / 50.0,
        tumor_miss_fraction=fn_tumor / 30.0,
        cyst_miss_fraction=fn_cyst / 20.0,
        fn_components=3,
        largest_fn_component_ml=10.0,
        dominant_missing_label=dominant,
        dominant_missing_label_share=10.0 / total,
    )


def test_analysis_requires_nine_cases() -> None:
    with pytest.raises(ValueError, match="9"):
        analyze_undercoverage(
            _case(i, "kidney")
            for i in range(8)
        )


def test_analysis_counts_dominant_labels() -> None:
    rows = (
        *(_case(i, "kidney") for i in range(4)),
        *(_case(i + 4, "tumor") for i in range(3)),
        *(_case(i + 7, "cyst") for i in range(2)),
    )

    result = analyze_undercoverage(rows)

    assert result["case_count"] == 9
    assert result["dominant_missing_label_counts"] == {
        "kidney": 4,
        "tumor": 3,
        "cyst": 2,
    }
    assert (
        result["weighted_miss_fraction_by_label"]["kidney"]
        is not None
    )
