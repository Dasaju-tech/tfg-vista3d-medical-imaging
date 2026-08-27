from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.data.audit import audit_case, audit_cases, audit_to_dict, write_audit
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.data.nifti import inspect_nifti, read_segmentation_labels


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _create_case(
    root: Path,
    *,
    case_name: str = "case_00001",
    image_shape: tuple[int, int, int] = (8, 9, 10),
    segmentation_shape: tuple[int, int, int] | None = None,
    image_affine: np.ndarray | None = None,
    segmentation_affine: np.ndarray | None = None,
    labels: tuple[int, ...] = (0, 1, 2, 3),
) -> Path:
    directory = root / case_name
    directory.mkdir()

    segmentation_shape = segmentation_shape or image_shape
    image_affine = image_affine if image_affine is not None else np.diag([1.0, 2.0, 3.0, 1.0])
    segmentation_affine = (
        segmentation_affine if segmentation_affine is not None else image_affine
    )

    image = np.zeros(image_shape, dtype=np.float32)
    segmentation = np.zeros(segmentation_shape, dtype=np.uint8)

    flat = segmentation.reshape(-1)
    for index, label in enumerate(labels):
        if index < flat.size:
            flat[index] = label

    _write_nifti(directory / "imaging.nii.gz", image, image_affine)
    _write_nifti(
        directory / "segmentation.nii.gz",
        segmentation,
        segmentation_affine,
    )

    return directory


def test_inspect_nifti_reads_geometry(tmp_path: Path) -> None:
    directory = _create_case(tmp_path)

    metadata = inspect_nifti(directory / "imaging.nii.gz")

    assert metadata.shape == (8, 9, 10)
    assert metadata.spacing_mm == pytest.approx((1.0, 2.0, 3.0))
    assert metadata.orientation == ("R", "A", "S")
    assert metadata.dtype == "float32"


def test_read_segmentation_labels(tmp_path: Path) -> None:
    directory = _create_case(tmp_path)

    assert read_segmentation_labels(directory / "segmentation.nii.gz") == (0, 1, 2, 3)


def test_valid_kits23_case_passes_audit(tmp_path: Path) -> None:
    directory = _create_case(tmp_path)
    case = discover_case(directory)

    result = audit_case(case)

    assert result.is_valid
    assert result.issues == ()
    assert result.labels == (0, 1, 2, 3)


def test_shape_mismatch_is_reported(tmp_path: Path) -> None:
    directory = _create_case(
        tmp_path,
        segmentation_shape=(8, 9, 9),
    )

    result = audit_case(discover_case(directory))

    assert not result.is_valid
    assert any(issue.startswith("shape_mismatch:") for issue in result.issues)


def test_affine_mismatch_is_reported(tmp_path: Path) -> None:
    image_affine = np.diag([1.0, 2.0, 3.0, 1.0])
    segmentation_affine = np.diag([1.5, 2.0, 3.0, 1.0])

    directory = _create_case(
        tmp_path,
        image_affine=image_affine,
        segmentation_affine=segmentation_affine,
    )

    result = audit_case(discover_case(directory))

    assert not result.is_valid
    assert "affine_mismatch" in result.issues
    assert any(issue.startswith("spacing_mismatch:") for issue in result.issues)


def test_unexpected_label_is_reported(tmp_path: Path) -> None:
    directory = _create_case(tmp_path, labels=(0, 1, 4))

    result = audit_case(discover_case(directory))

    assert not result.is_valid
    assert "unexpected_labels:[4]" in result.issues


def test_non_integer_segmentation_label_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "case_00001"
    directory.mkdir()
    affine = np.eye(4)

    _write_nifti(
        directory / "imaging.nii.gz",
        np.zeros((3, 3, 3), dtype=np.float32),
        affine,
    )

    segmentation = np.zeros((3, 3, 3), dtype=np.float32)
    segmentation[0, 0, 0] = 1.5
    _write_nifti(directory / "segmentation.nii.gz", segmentation, affine)

    with pytest.raises(ValueError, match="non-integer"):
        audit_case(discover_case(directory))


def test_dataset_audit_serializes_to_json(tmp_path: Path) -> None:
    first = discover_case(_create_case(tmp_path, case_name="case_00001"))
    second = discover_case(
        _create_case(tmp_path, case_name="case_00002", labels=(0, 1, 4))
    )

    audit = audit_cases((first, second))
    payload = audit_to_dict(audit)

    assert audit.valid_cases == 1
    assert audit.invalid_cases == 1
    assert not audit.is_valid
    assert payload["number_of_cases"] == 2
    assert payload["label_mapping"]["2"] == "tumor"

    output_path = tmp_path / "outputs" / "audit.json"
    write_audit(audit, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert '"dataset": "KiTS23"' in text
    assert '"invalid_cases": 1' in text
