from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    ADAPTIVE_DEPTH_FRACTION,
    assign_volume_quartiles,
    derive_adaptive_case,
    select_stratified_cases,
)


def _rows(count: int = 100) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "case_id": f"case_{index:05d}",
            "largest_component_volume_ml": str(
                float(index + 1)
            ),
            "tumor_positive": "True",
        }
        for index in range(count)
    )


def test_quartiles_are_balanced_for_100_cases() -> None:
    groups = assign_volume_quartiles(
        _rows()
    )

    assert [len(group) for group in groups.values()] == [
        25,
        25,
        25,
        25,
    ]


def test_stratified_selection_has_three_per_quartile() -> None:
    selected = select_stratified_cases(
        _rows()
    )

    assert len(selected) == 12

    counts: dict[str, int] = {}
    for quartile, _ in selected:
        counts[quartile] = counts.get(quartile, 0) + 1

    assert set(counts.values()) == {3}


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_adaptive_point_tracks_fraction_of_max_depth(
    tmp_path: Path,
) -> None:
    gt = np.zeros(
        (50, 50, 50),
        dtype=np.uint8,
    )
    gt[5:45, 5:45, 5:45] = 2

    path = (
        tmp_path
        / "segmentation.nii.gz"
    )
    _save(path, gt)

    row = derive_adaptive_case(
        case_id="fixture",
        quartile="smallest_quartile",
        largest_component_volume_ml=64.0,
        segmentation_path=path,
    )

    assert row.max_interior_depth_mm >= 20.0
    assert row.adaptive_target_depth_mm == pytest.approx(
        ADAPTIVE_DEPTH_FRACTION
        * row.max_interior_depth_mm
    )
    assert row.adaptive_actual_depth_mm == pytest.approx(
        row.adaptive_target_depth_mm,
        abs=1.0,
    )
    assert gt[row.adaptive_point_xyz] == 2
    assert gt[row.deep_point_xyz] == 2
