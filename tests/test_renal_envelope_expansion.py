from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.evaluation.renal_envelope_expansion import (
    ExpansionCase,
    ExpansionVariant,
    MetricTriplet,
    analyze_expansion_sensitivity,
    evaluate_expansion_case,
    expand_mask_mm,
)


def test_expand_mask_uses_physical_radius() -> None:
    mask = np.zeros((11, 11, 11), dtype=bool)
    mask[5, 5, 5] = True

    expanded = expand_mask_mm(
        mask,
        spacing_mm=(1.0, 1.0, 2.0),
        radius_mm=2.0,
    )

    assert expanded[5, 5, 6]
    assert expanded[5, 5, 4]
    assert expanded[5, 6, 5]
    assert expanded[5, 5, 7] is np.False_ or not expanded[5, 5, 7]


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_expansion_increases_nearby_tumor_coverage(
    tmp_path: Path,
) -> None:
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[5:15, 5:15, 5:15] = 1
    pred[5:15, 5:15, 5:15] = 14

    # Tumor extends 3 voxels from kidney boundary.
    gt[15:18, 8:12, 8:12] = 2

    gt_path = tmp_path / "segmentation.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"
    _save(gt_path, gt)
    _save(pred_path, pred)

    row = evaluate_expansion_case(
        case_id="fixture",
        undercoverage_target=True,
        ground_truth_path=gt_path,
        prediction_path=pred_path,
        radii_mm=(2.0, 5.0),
    )

    assert row.original_tumor_coverage_fraction == pytest.approx(0.0)
    assert (
        row.variants[0].tumor_coverage_fraction
        > row.original_tumor_coverage_fraction
    )
    assert row.variants[1].tumor_coverage_fraction == pytest.approx(1.0)


def _case(index: int) -> ExpansionCase:
    target = index < 9
    original = MetricTriplet(
        dice=0.9,
        surface_dice=0.75,
        hd95_mm=(30.0 if target else 4.0),
    )

    variants = tuple(
        ExpansionVariant(
            radius_mm=radius,
            metrics=MetricTriplet(
                dice=0.9,
                surface_dice=0.76,
                hd95_mm=(
                    10.0 if target else 4.0
                ),
            ),
            added_volume_ml=10.0 * radius,
            tumor_coverage_fraction=(
                min(1.0, radius / 10.0)
                if target
                else None
            ),
        )
        for radius in (2.0, 5.0, 10.0)
    )

    return ExpansionCase(
        case_id=f"case_{index:05d}",
        undercoverage_target=target,
        original=original,
        original_tumor_coverage_fraction=(
            0.0 if target else None
        ),
        variants=variants,
    )


def test_analysis_requires_100_cases() -> None:
    with pytest.raises(ValueError, match="N=100"):
        analyze_expansion_sensitivity(
            _case(i) for i in range(99)
        )


def test_analysis_reports_nine_targets() -> None:
    result = analyze_expansion_sensitivity(
        _case(i) for i in range(100)
    )

    assert result["case_count"] == 100
    assert result["undercoverage_target_count"] == 9
    assert result["non_target_count"] == 91

    ten_mm = result["variants"]["10mm"]
    assert (
        ten_mm["undercoverage_targets"][
            "hd95_reduction_mm"
        ]["mean"]
        > 0.0
    )
