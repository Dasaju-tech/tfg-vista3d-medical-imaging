from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.visualization.qc import (
    build_case_qc,
    component_label_map,
    load_canonical_case,
    select_representative_slice,
    write_qc_index,
)


def make_case() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((10, 12, 14), dtype=np.int16)
    image += np.arange(14, dtype=np.int16)[None, None, :]

    seg = np.zeros_like(image, dtype=np.uint8)

    seg[1:4, 2:10, 2:12] = 1
    seg[6:9, 2:10, 2:12] = 1

    seg[2:4, 4:6, 7:10] = 2
    seg[7:8, 6:8, 9:12] = 3

    return image, seg


def save_pair(
    root: Path,
    *,
    image: np.ndarray,
    segmentation: np.ndarray,
    affine: np.ndarray,
) -> tuple[Path, Path]:
    imaging_path = root / "imaging.nii.gz"
    segmentation_path = root / "segmentation.nii.gz"

    nib.save(nib.Nifti1Image(image, affine), imaging_path)
    nib.save(
        nib.Nifti1Image(segmentation, affine),
        segmentation_path,
    )

    return imaging_path, segmentation_path


def test_selection_prefers_lesion_over_whole_kidney() -> None:
    _, segmentation = make_case()

    selection = select_representative_slice(
        segmentation,
        plane="axial",
    )

    assert selection.source == "lesion"

    areas = np.count_nonzero(
        np.isin(segmentation, (2, 3)),
        axis=(0, 1),
    )
    assert selection.index == int(np.argmax(areas))


def test_empty_segmentation_selects_volume_center() -> None:
    segmentation = np.zeros((9, 11, 13), dtype=np.uint8)

    selection = select_representative_slice(
        segmentation,
        plane="coronal",
    )

    assert selection.source == "volume_center"
    assert selection.index == 11 // 2


def test_unknown_plane_is_rejected() -> None:
    segmentation = np.zeros((5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unknown plane"):
        select_representative_slice(
            segmentation,
            plane="oblique",
        )


def test_component_map_detects_two_renal_units() -> None:
    _, segmentation = make_case()

    _, count = component_label_map(segmentation)

    assert count == 2


def test_canonicalization_returns_ras_even_for_flipped_input(
    tmp_path: Path,
) -> None:
    image, segmentation = make_case()

    affine = np.eye(4)
    affine[0, 0] = -2.0
    affine[0, 3] = 40.0

    imaging_path, segmentation_path = save_pair(
        tmp_path,
        image=image,
        segmentation=segmentation,
        affine=affine,
    )

    (
        canonical_image,
        canonical_segmentation,
        _,
        spacing,
        orientation,
    ) = load_canonical_case(
        imaging_path,
        segmentation_path,
    )

    assert orientation == ("R", "A", "S")
    assert canonical_image.shape == canonical_segmentation.shape
    assert spacing == pytest.approx((2.0, 1.0, 1.0))


def test_build_case_qc_creates_six_non_empty_pngs(
    tmp_path: Path,
) -> None:
    image, segmentation = make_case()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    imaging_path, segmentation_path = save_pair(
        data_dir,
        image=image,
        segmentation=segmentation,
        affine=np.eye(4),
    )

    result = build_case_qc(
        case_id="case_fixture",
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
        output_directory=tmp_path / "qc",
    )

    files = [
        *(Path(value) for value in result.overlay_files),
        *(Path(value) for value in result.component_files),
    ]

    assert len(files) == 6
    assert all(path.is_file() for path in files)
    assert all(path.stat().st_size > 0 for path in files)

    image_array = mpimg.imread(files[0])
    assert image_array.ndim in (2, 3)


def test_qc_index_is_structured_json(tmp_path: Path) -> None:
    image, segmentation = make_case()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    imaging_path, segmentation_path = save_pair(
        data_dir,
        image=image,
        segmentation=segmentation,
        affine=np.eye(4),
    )

    result = build_case_qc(
        case_id="case_fixture",
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
        output_directory=tmp_path / "qc",
    )

    output = tmp_path / "qc_index.json"
    write_qc_index([result], output)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["dataset"] == "KiTS23"
    assert payload["coordinate_convention"] == "canonical RAS+"
    assert payload["cases"][0]["case_id"] == "case_fixture"
    assert len(payload["cases"][0]["overlay_files"]) == 3


@pytest.mark.parametrize(
    "window",
    [
        (100.0, 100.0),
        (200.0, 100.0),
        (np.nan, 100.0),
    ],
)
def test_invalid_ct_window_is_rejected(
    tmp_path: Path,
    window,
) -> None:
    image, segmentation = make_case()

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    imaging_path, segmentation_path = save_pair(
        data_dir,
        image=image,
        segmentation=segmentation,
        affine=np.eye(4),
    )

    with pytest.raises(ValueError):
        build_case_qc(
            case_id="case_fixture",
            imaging_path=imaging_path,
            segmentation_path=segmentation_path,
            output_directory=tmp_path / "qc",
            window_hu=window,
        )


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    image = np.zeros((5, 5, 5), dtype=np.int16)
    segmentation = np.zeros((6, 5, 5), dtype=np.uint8)

    imaging_path = tmp_path / "imaging.nii.gz"
    segmentation_path = tmp_path / "segmentation.nii.gz"

    nib.save(
        nib.Nifti1Image(image, np.eye(4)),
        imaging_path,
    )
    nib.save(
        nib.Nifti1Image(segmentation, np.eye(4)),
        segmentation_path,
    )

    with pytest.raises(ValueError, match="shapes do not match"):
        load_canonical_case(
            imaging_path,
            segmentation_path,
        )
