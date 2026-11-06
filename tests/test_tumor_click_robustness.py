from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickRobustnessCase,
    derive_click_variants,
    summarize_click_robustness,
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


def test_click_variants_stay_inside_target(
    tmp_path: Path,
) -> None:
    gt = np.zeros((40, 40, 40), dtype=np.uint8)
    gt[5:30, 5:30, 5:30] = 2

    gt_path = tmp_path / "segmentation.nii.gz"
    _save(gt_path, gt)

    prompt = TumorOraclePrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(17, 17, 17),
        target_voxels=25 ** 3,
        target_volume_ml=15.625,
        total_tumor_volume_ml=15.625,
        interior_radius_mm=13.0,
    )

    variants = derive_click_variants(
        prompt=prompt,
        ground_truth_path=gt_path,
    )

    assert [item.variant for item in variants] == [
        "deep",
        "centroid",
        "shallow_5mm",
    ]
    for item in variants:
        assert gt[item.point_xyz] == 2

    shallow = variants[2]
    assert shallow.interior_radius_mm == pytest.approx(5.0)


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


def test_summary_reports_click_robustness() -> None:
    rows = tuple(
        ClickRobustnessCase(
            case_id=f"case_{i:05d}",
            deep=_result(f"case_{i:05d}", 0.80),
            centroid=_result(f"case_{i:05d}", 0.78),
            shallow_5mm=_result(f"case_{i:05d}", 0.70),
        )
        for i in range(9)
    )

    payload = summarize_click_robustness(rows)

    assert payload["case_count"] == 9
    assert payload["variants"]["deep"]["tumor_dice_mean"] == pytest.approx(0.80)
    assert payload["variants"]["centroid"]["tumor_dice_mean"] == pytest.approx(0.78)
    assert payload["robustness_vs_deep"][
        "centroid_tumor_dice_delta_mean"
    ] == pytest.approx(-0.02)
    assert payload["robustness_vs_deep"][
        "centroid_cases_within_0_05_dice"
    ] == 9
