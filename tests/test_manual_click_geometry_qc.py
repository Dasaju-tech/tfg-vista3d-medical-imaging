from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.analysis.manual_click_geometry_qc import (
    audit_manual_click_geometry,
    summarize_geometry_qc,
)
from vista3d_tfg.interaction.manual_tumor_clicks import (
    ManualTumorClick,
)


def _save(
    path: Path,
    array: np.ndarray,
    affine: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(
            array,
            affine,
        ),
        str(path),
    )


def test_world_mapping_handles_different_voxel_axes(
    tmp_path: Path,
) -> None:
    image = np.zeros(
        (10, 10, 10),
        dtype=np.float32,
    )
    seg = np.zeros(
        (10, 10, 10),
        dtype=np.uint8,
    )

    image_affine = np.eye(4)
    seg_affine = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Image voxel (2, 3, 4) maps to segmentation voxel (3, 2, 4).
    seg[3, 2, 4] = 2

    image_path = tmp_path / "image.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"

    _save(
        image_path,
        image,
        image_affine,
    )
    _save(
        seg_path,
        seg,
        seg_affine,
    )

    click = ManualTumorClick(
        case_id="fixture",
        quartile="smallest_quartile",
        point_xyz=(2, 3, 4),
        display_point_ras_xyz=(2, 3, 4),
    )

    qc = audit_manual_click_geometry(
        click=click,
        image_path=image_path,
        segmentation_path=seg_path,
    )

    assert qc.grids_equal is False
    assert qc.segmentation_point_xyz == (3, 2, 4)
    assert qc.inside_any_tumor is True
    assert qc.distance_to_any_tumor_mm == pytest.approx(0.0)
    assert qc.world_roundtrip_error_mm == pytest.approx(0.0)


def test_identical_grid_has_zero_roundtrip_error(
    tmp_path: Path,
) -> None:
    image = np.zeros(
        (8, 8, 8),
        dtype=np.float32,
    )
    seg = np.zeros(
        (8, 8, 8),
        dtype=np.uint8,
    )
    seg[4, 4, 4] = 2

    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    seg_path = tmp_path / "seg.nii.gz"

    _save(
        image_path,
        image,
        affine,
    )
    _save(
        seg_path,
        seg,
        affine,
    )

    click = ManualTumorClick(
        case_id="fixture",
        quartile="smallest_quartile",
        point_xyz=(4, 4, 4),
        display_point_ras_xyz=(4, 4, 4),
    )

    qc = audit_manual_click_geometry(
        click=click,
        image_path=image_path,
        segmentation_path=seg_path,
    )

    assert qc.grids_equal is True
    assert qc.display_roundtrip_error_vox == pytest.approx(0.0)
    assert qc.world_roundtrip_error_mm == pytest.approx(0.0)
    assert qc.inside_any_tumor is True


def test_summary_requires_full_frozen_cohort() -> None:
    with pytest.raises(ValueError, match="12"):
        summarize_geometry_qc([])
