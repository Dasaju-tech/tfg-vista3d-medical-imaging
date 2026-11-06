from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    AdaptiveDepthCase,
)
from vista3d_tfg.experiments.tumor_dual_point_protocol import (
    SECOND_POINT_MIN_RELATIVE_DEPTH,
    derive_dual_point_case,
)


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_dual_points_are_distinct_and_interior(
    tmp_path: Path,
) -> None:
    gt = np.zeros(
        (70, 30, 30),
        dtype=np.uint8,
    )
    gt[5:65, 5:25, 5:25] = 2

    path = tmp_path / "segmentation.nii.gz"
    _save(path, gt)

    case = AdaptiveDepthCase(
        case_id="fixture",
        quartile="upper_middle",
        largest_component_volume_ml=24.0,
        component_id=1,
        target_voxels=60 * 20 * 20,
        max_interior_depth_mm=10.0,
        deep_point_xyz=(14, 14, 14),
        adaptive_target_depth_mm=7.0,
        adaptive_actual_depth_mm=7.0,
        adaptive_point_xyz=(11, 11, 11),
    )

    row = derive_dual_point_case(
        case=case,
        segmentation_path=path,
    )

    assert row.first_point_xyz != row.second_point_xyz
    assert gt[row.first_point_xyz] == 2
    assert gt[row.second_point_xyz] == 2
    assert (
        row.second_point_relative_depth
        >= SECOND_POINT_MIN_RELATIVE_DEPTH - 1e-12
    )
    assert row.point_separation_mm > 0.0


def test_dual_second_point_is_spatially_separated(
    tmp_path: Path,
) -> None:
    gt = np.zeros(
        (100, 30, 30),
        dtype=np.uint8,
    )
    gt[5:95, 5:25, 5:25] = 2

    path = tmp_path / "segmentation.nii.gz"
    _save(path, gt)

    case = AdaptiveDepthCase(
        case_id="fixture",
        quartile="largest_quartile",
        largest_component_volume_ml=36.0,
        component_id=1,
        target_voxels=90 * 20 * 20,
        max_interior_depth_mm=10.0,
        deep_point_xyz=(14, 14, 14),
        adaptive_target_depth_mm=7.0,
        adaptive_actual_depth_mm=7.0,
        adaptive_point_xyz=(11, 11, 11),
    )

    row = derive_dual_point_case(
        case=case,
        segmentation_path=path,
    )

    assert row.point_separation_mm >= 40.0


def test_invalid_relative_depth_is_rejected(
    tmp_path: Path,
) -> None:
    dummy = AdaptiveDepthCase(
        case_id="fixture",
        quartile="smallest_quartile",
        largest_component_volume_ml=1.0,
        component_id=1,
        target_voxels=1,
        max_interior_depth_mm=1.0,
        deep_point_xyz=(0, 0, 0),
        adaptive_target_depth_mm=0.7,
        adaptive_actual_depth_mm=0.7,
        adaptive_point_xyz=(0, 0, 0),
    )

    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        derive_dual_point_case(
            case=dummy,
            segmentation_path=tmp_path / "missing.nii.gz",
            second_point_min_relative_depth=1.0,
        )
