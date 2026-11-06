from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.tumor_safe_region_generalization import (
    TumorSafeRegionCase,
    evaluate_case_safe_region,
    summarize_expanded100_safe_regions,
)


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(array, np.eye(4)),
        str(path),
    )


def test_negative_case_has_no_safe_region(
    tmp_path: Path,
) -> None:
    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    path = tmp_path / "segmentation.nii.gz"
    _save(path, gt)

    row = evaluate_case_safe_region(
        case_id="case_00000",
        segmentation_path=path,
    )

    assert row.tumor_positive is False
    assert row.tumor_volume_ml == pytest.approx(0.0)
    assert row.max_interior_depth_mm is None
    assert row.safe_fraction_15mm is None


def test_positive_case_safe_region_is_monotonic(
    tmp_path: Path,
) -> None:
    gt = np.zeros((50, 50, 50), dtype=np.uint8)
    gt[5:45, 5:45, 5:45] = 2

    path = tmp_path / "segmentation.nii.gz"
    _save(path, gt)

    row = evaluate_case_safe_region(
        case_id="case_00001",
        segmentation_path=path,
    )

    assert row.tumor_positive is True
    assert row.safe_fraction_5mm > row.safe_fraction_10mm
    assert row.safe_fraction_10mm > row.safe_fraction_15mm
    assert row.safe_fraction_15mm > 0.0


def _positive(index: int, volume: float) -> TumorSafeRegionCase:
    return TumorSafeRegionCase(
        case_id=f"case_{index:05d}",
        tumor_positive=True,
        tumor_voxels=100,
        tumor_volume_ml=volume,
        component_count=1,
        largest_component_voxels=100,
        largest_component_volume_ml=volume,
        max_interior_depth_mm=20.0,
        safe_fraction_5mm=0.8,
        safe_fraction_10mm=0.5,
        safe_fraction_15mm=0.2,
        safe_volume_5mm_ml=0.8 * volume,
        safe_volume_10mm_ml=0.5 * volume,
        safe_volume_15mm_ml=0.2 * volume,
    )


def _negative(index: int) -> TumorSafeRegionCase:
    return TumorSafeRegionCase(
        case_id=f"case_{index:05d}",
        tumor_positive=False,
        tumor_voxels=0,
        tumor_volume_ml=0.0,
        component_count=0,
        largest_component_voxels=0,
        largest_component_volume_ml=0.0,
        max_interior_depth_mm=None,
        safe_fraction_5mm=None,
        safe_fraction_10mm=None,
        safe_fraction_15mm=None,
        safe_volume_5mm_ml=None,
        safe_volume_10mm_ml=None,
        safe_volume_15mm_ml=None,
    )


def test_summary_stratifies_tumor_positive_cases() -> None:
    rows = tuple(
        [
            _positive(index, float(index + 1))
            for index in range(40)
        ]
        + [
            _negative(index)
            for index in range(40, 100)
        ]
    )

    payload = summarize_expanded100_safe_regions(rows)

    assert payload["case_count"] == 100
    assert payload["tumor_positive_count"] == 40
    assert payload["tumor_negative_count"] == 60
    assert payload["depths"]["15mm"][
        "cases_with_nonempty_region"
    ] == 40
    assert payload["depths"]["15mm"][
        "safe_fraction"
    ]["mean"] == pytest.approx(0.2)
    assert "smallest_quartile" in payload["volume_quartiles"]
