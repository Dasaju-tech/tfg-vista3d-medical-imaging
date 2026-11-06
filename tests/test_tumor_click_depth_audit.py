from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_click_depth_audit import (
    DepthAuditCase,
    derive_depth_variant,
    summarize_depth_audit,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)


def _save(path: Path, array: np.ndarray) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_depth_variant_targets_requested_depth(
    tmp_path: Path,
) -> None:
    gt = np.zeros((50, 50, 50), dtype=np.uint8)
    gt[5:45, 5:45, 5:45] = 2

    gt_path = tmp_path / "segmentation.nii.gz"
    _save(gt_path, gt)

    prompt = TumorOraclePrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(24, 24, 24),
        target_voxels=40 ** 3,
        target_volume_ml=64.0,
        total_tumor_volume_ml=64.0,
        interior_radius_mm=20.0,
    )

    variant = derive_depth_variant(
        prompt=prompt,
        ground_truth_path=gt_path,
        depth_mm=10.0,
    )

    assert variant.variant == "depth_10mm"
    assert variant.interior_radius_mm == pytest.approx(10.0)
    assert gt[variant.point_xyz] == 2


def _result(
    case_id: str,
    dice: float,
) -> TumorOracleSmokeResult:
    return TumorOracleSmokeResult(
        case_id=case_id,
        point_xyz=(1, 2, 3),
        point_label=3,
        target_volume_ml=10.0,
        total_tumor_volume_ml=10.0,
        interactive_target_dice=dice,
        interactive_full_tumor_dice=dice,
        original_tumor_coverage_fraction=0.1,
        merged_tumor_coverage_fraction=0.7,
        tumor_coverage_gain=0.6,
        original_hec_dice=0.5,
        merged_hec_dice=0.7,
        hec_dice_delta=0.2,
        original_hec_surface_dice=0.4,
        merged_hec_surface_dice=0.5,
        hec_surface_dice_delta=0.1,
        original_hec_hd95_mm=50.0,
        merged_hec_hd95_mm=20.0,
        hec_hd95_reduction_mm=30.0,
        interactive_prediction_volume_ml=10.0,
        interactive_labels_present=(0, 1, 255),
        contains_255_unprocessed=True,
        wall_time_seconds=2.0,
        prediction_path="fixture.nii.gz",
    )


def test_depth_summary_orders_expected_variants() -> None:
    rows = tuple(
        DepthAuditCase(
            case_id=f"case_{i:05d}",
            shallow_5mm=_result(f"case_{i:05d}", 0.45),
            depth_10mm=_result(f"case_{i:05d}", 0.65),
            depth_15mm=_result(f"case_{i:05d}", 0.75),
            centroid=_result(f"case_{i:05d}", 0.78),
            deep=_result(f"case_{i:05d}", 0.80),
        )
        for i in range(9)
    )

    payload = summarize_depth_audit(rows)

    assert payload["case_count"] == 9
    assert (
        payload["variants"]["depth_10mm"]["tumor_dice_mean"]
        == pytest.approx(0.65)
    )
    assert (
        payload["variants"]["depth_15mm"]["tumor_dice_mean"]
        == pytest.approx(0.75)
    )
    assert (
        payload["robustness_vs_deep"]["depth_15mm"][
            "cases_within_0_05_dice_of_deep"
        ]
        == 9
    )
