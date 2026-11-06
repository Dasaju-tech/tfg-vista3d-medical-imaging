from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_safe_click_region import (
    SafeRegionCase,
    SafeRegionDepth,
    attach_block43_performance,
    evaluate_safe_click_region,
    summarize_safe_click_regions,
)


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_safe_region_fraction_decreases_with_depth(
    tmp_path: Path,
) -> None:
    gt = np.zeros((50, 50, 50), dtype=np.uint8)
    gt[5:45, 5:45, 5:45] = 2

    path = tmp_path / "segmentation.nii.gz"
    _save(path, gt)

    prompt = TumorOraclePrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(24, 24, 24),
        target_voxels=40 ** 3,
        target_volume_ml=64.0,
        total_tumor_volume_ml=64.0,
        interior_radius_mm=20.0,
    )

    result = evaluate_safe_click_region(
        prompt=prompt,
        ground_truth_path=path,
    )

    fractions = [
        item.fraction_of_target
        for item in result.safe_regions
    ]

    assert fractions[0] > fractions[1] > fractions[2] > 0.0
    assert result.max_interior_depth_mm >= 20.0


def _case(index: int) -> SafeRegionCase:
    return SafeRegionCase(
        case_id=f"case_{index:05d}",
        target_voxels=1000,
        target_volume_ml=1.0,
        max_interior_depth_mm=20.0,
        safe_regions=(
            SafeRegionDepth(5.0, 800, 0.8, 0.8, True),
            SafeRegionDepth(10.0, 500, 0.5, 0.5, True),
            SafeRegionDepth(15.0, 200, 0.2, 0.2, True),
        ),
    )


def test_summary_reports_safe_region_availability() -> None:
    payload = summarize_safe_click_regions(
        _case(i) for i in range(9)
    )

    assert payload["case_count"] == 9
    assert payload["thresholds"]["15mm"][
        "cases_with_nonempty_region"
    ] == 9
    assert payload["thresholds"]["15mm"][
        "safe_fraction"
    ]["mean"] == pytest.approx(0.2)
    assert payload["thresholds"]["15mm"][
        "cases_safe_fraction_ge_0_10"
    ] == 9


def test_attach_block43_performance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        """
{
  "variants": {
    "shallow_5mm": {
      "tumor_dice_mean": 0.45,
      "tumor_dice_median": 0.40,
      "tumor_dice_ge_050_count": 4,
      "tumor_dice_ge_070_count": 3,
      "tumor_coverage_gain_mean": 0.28,
      "hec_hd95_reduction_mean_mm": 13.4,
      "hd95_improved_count": 8
    },
    "depth_10mm": {
      "tumor_dice_mean": 0.66,
      "tumor_dice_median": 0.64,
      "tumor_dice_ge_050_count": 8,
      "tumor_dice_ge_070_count": 4,
      "tumor_coverage_gain_mean": 0.44,
      "hec_hd95_reduction_mean_mm": 21.0,
      "hd95_improved_count": 9
    },
    "depth_15mm": {
      "tumor_dice_mean": 0.73,
      "tumor_dice_median": 0.73,
      "tumor_dice_ge_050_count": 8,
      "tumor_dice_ge_070_count": 6,
      "tumor_coverage_gain_mean": 0.47,
      "hec_hd95_reduction_mean_mm": 22.9,
      "hd95_improved_count": 9
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    payload: dict[str, object] = {}
    attach_block43_performance(
        payload=payload,
        block43_summary_path=path,
    )

    assert payload["block43_depth_performance"]["15mm"][
        "tumor_dice_mean"
    ] == pytest.approx(0.73)
