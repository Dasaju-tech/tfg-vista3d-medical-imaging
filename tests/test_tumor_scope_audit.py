from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.evaluation.tumor_scope_audit import (
    ScopeCase,
    analyze_scope_effect,
    evaluate_scope_pair,
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


def test_tumor_exclusion_recovers_missing_tumor_scope(
    tmp_path: Path,
) -> None:
    gt = np.zeros((40, 20, 20), dtype=np.uint8)
    pred = np.zeros_like(gt)

    # Kidney is predicted correctly.
    gt[3:10, 5:15, 5:15] = 1
    pred[3:10, 5:15, 5:15] = 14

    # A large exophytic tumor is part of the official kidney_and_mass
    # reference but has no automatic tumor output in this protocol.
    gt[10:30, 8:12, 8:12] = 2

    gt_path, pred_path = save_pair(tmp_path, gt, pred)

    original, excluded, tumor_ml = evaluate_scope_pair(
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert tumor_ml > 0.0
    assert excluded.dice == pytest.approx(1.0)
    assert excluded.surface_dice == pytest.approx(1.0)
    assert excluded.diagnostic_hd95_mm == pytest.approx(0.0)
    assert original.dice < excluded.dice
    assert original.diagnostic_hd95_mm > excluded.diagnostic_hd95_mm


def make_case(
    index: int,
    *,
    original_hd95: float,
    excluded_hd95: float,
) -> ScopeCase:
    return ScopeCase(
        case_id=f"case_{index:05d}",
        tumor_volume_ml=float(index),
        failure_mode=(
            "distant_boundary_failure"
            if index >= 90
            else "non_hd95_outlier"
        ),
        original_dice=0.9,
        original_surface_dice=0.75,
        original_hd95_mm=original_hd95,
        tumor_excluded_dice=0.92,
        tumor_excluded_surface_dice=0.80,
        tumor_excluded_hd95_mm=excluded_hd95,
    )


def test_scope_analysis_requires_100_cases() -> None:
    with pytest.raises(ValueError, match="N=100"):
        analyze_scope_effect(
            make_case(
                i,
                original_hd95=4.0,
                excluded_hd95=3.0,
            )
            for i in range(99)
        )


def test_scope_analysis_counts_resolved_outliers() -> None:
    rows = []

    for i in range(100):
        original = 4.0 + i / 100.0
        excluded = original

        if i >= 95:
            original = 50.0 + i
            excluded = 5.0

        rows.append(
            make_case(
                i,
                original_hd95=original,
                excluded_hd95=excluded,
            )
        )

    result = analyze_scope_effect(rows)
    outliers = result["outlier_analysis"]

    assert outliers["original_outlier_count"] >= 5
    assert (
        outliers[
            "original_outliers_resolved_by_tumor_exclusion"
        ]
        >= 5
    )
    assert (
        result["tumor_volume_correlations"][
            "tumor_vs_hd95_reduction"
        ]
        is not None
    )
