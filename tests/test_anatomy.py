from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.analysis.anatomy import (
    analyze_kits23_anatomy,
    analyze_kits23_anatomy_nifti,
    write_anatomy_results,
)


def make_bilateral_fixture() -> np.ndarray:
    seg = np.zeros((12, 8, 8), dtype=np.uint8)

    seg[1:5, 1:7, 1:7] = 1
    seg[7:11, 1:7, 1:7] = 1

    seg[2, 2, 2] = 2
    seg[3, 5, 5] = 2

    seg[8:10, 3:5, 3:5] = 3

    return seg


def test_bilateral_renal_units_are_assigned_by_ras_world_x() -> None:
    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=make_bilateral_fixture(),
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert result.is_laterality_resolved
    assert result.kidney_and_mass_components == 2

    sides = {unit.laterality: unit for unit in result.renal_units}

    assert sides["left"].centroid_world_mm[0] < (
        sides["right"].centroid_world_mm[0]
    )


def test_laterality_uses_world_coordinates_not_voxel_order() -> None:
    seg = make_bilateral_fixture()

    affine = np.eye(4)
    affine[0, 0] = -2.0
    affine[0, 3] = 30.0

    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=seg,
        spacing_mm=(2.0, 1.0, 1.0),
        affine=affine,
    )

    sides = {unit.laterality: unit for unit in result.renal_units}

    assert sides["left"].centroid_voxel[0] > (
        sides["right"].centroid_voxel[0]
    )
    assert sides["left"].centroid_world_mm[0] < (
        sides["right"].centroid_world_mm[0]
    )


def test_multiple_tumors_are_assigned_to_same_parent_side() -> None:
    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=make_bilateral_fixture(),
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert len(result.tumor_components) == 2
    assert {item.laterality for item in result.tumor_components} == {"left"}

    left = next(
        item for item in result.side_summaries
        if item.laterality == "left"
    )

    assert left.tumor_components == 2
    assert left.tumor_volume_ml > 0.0


def test_cyst_is_assigned_to_right_side() -> None:
    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=make_bilateral_fixture(),
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert len(result.cyst_components) == 1
    assert result.cyst_components[0].laterality == "right"


def test_single_renal_unit_is_not_given_invented_laterality() -> None:
    seg = np.zeros((8, 8, 8), dtype=np.uint8)
    seg[2:6, 2:6, 2:6] = 1
    seg[3, 3, 3] = 2

    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=seg,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert not result.is_laterality_resolved
    assert result.renal_units[0].laterality == "undetermined"
    assert (
        "fewer_than_two_kidney_and_mass_components"
        in result.qc_flags
    )
    assert result.tumor_components[0].laterality == "undetermined"


def test_third_satellite_component_is_flagged_and_unassigned() -> None:
    seg = make_bilateral_fixture()
    seg[0, 0, 0] = 2

    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=seg,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert result.kidney_and_mass_components == 3
    assert "more_than_two_kidney_and_mass_components" in result.qc_flags
    assert "unassigned_tumor_component" in result.qc_flags
    assert any(
        item.laterality == "unassigned"
        for item in result.tumor_components
    )


def test_absent_cysts_return_empty_component_inventory() -> None:
    seg = make_bilateral_fixture()
    seg[seg == 3] = 1

    result = analyze_kits23_anatomy(
        case_id="fixture",
        segmentation=seg,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert result.cyst_components == ()


@pytest.mark.parametrize(
    "spacing",
    [
        (1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, np.nan, 1.0),
    ],
)
def test_invalid_spacing_is_rejected(spacing) -> None:
    with pytest.raises(ValueError):
        analyze_kits23_anatomy(
            case_id="fixture",
            segmentation=make_bilateral_fixture(),
            spacing_mm=spacing,
            affine=np.eye(4),
        )


def test_invalid_affine_is_rejected() -> None:
    affine = np.zeros((4, 4))

    with pytest.raises(ValueError, match="non-singular"):
        analyze_kits23_anatomy(
            case_id="fixture",
            segmentation=make_bilateral_fixture(),
            spacing_mm=(1.0, 1.0, 1.0),
            affine=affine,
        )


def test_non_integer_segmentation_is_rejected() -> None:
    seg = make_bilateral_fixture().astype(np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        analyze_kits23_anatomy(
            case_id="fixture",
            segmentation=seg,
            spacing_mm=(1.0, 1.0, 1.0),
            affine=np.eye(4),
        )


def test_nifti_entrypoint_and_writer(tmp_path: Path) -> None:
    seg = make_bilateral_fixture()
    path = tmp_path / "segmentation.nii.gz"

    nib.save(
        nib.Nifti1Image(seg, np.eye(4)),
        path,
    )

    result = analyze_kits23_anatomy_nifti(
        case_id="fixture",
        segmentation_path=path,
    )

    output = tmp_path / "out" / "anatomy.json"
    write_anatomy_results([result], output)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"] == "KiTS23"
    assert payload["analysis"] == "anatomical_component_inventory"
    assert payload["cases"][0]["case_id"] == "fixture"
    assert payload["cases"][0]["is_laterality_resolved"] is True
