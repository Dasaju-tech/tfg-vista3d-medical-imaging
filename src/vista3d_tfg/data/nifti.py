from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


KITS23_LABEL_NAMES: dict[int, str] = {
    0: "background",
    1: "kidney",
    2: "tumor",
    3: "cyst",
}


@dataclass(frozen=True, slots=True)
class NiftiMetadata:
    path: Path
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: tuple[str, str, str]
    dtype: str
    affine: tuple[tuple[float, float, float, float], ...]


def _validate_affine(affine: np.ndarray, path: Path) -> None:
    if affine.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine matrix for {path}.")

    if not np.all(np.isfinite(affine)):
        raise ValueError(f"Affine matrix contains non-finite values: {path}")

    determinant = float(np.linalg.det(affine[:3, :3]))
    if np.isclose(determinant, 0.0):
        raise ValueError(f"Affine matrix is singular: {path}")


def inspect_nifti(path: Path) -> NiftiMetadata:
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file does not exist: {path}")

    image = nib.load(str(path))

    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D NIfTI volume, got shape {image.shape}: {path}"
        )

    affine = np.asarray(image.affine, dtype=float)
    _validate_affine(affine, path)

    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    if len(spacing) != 3 or any(
        (not np.isfinite(value)) or value <= 0.0 for value in spacing
    ):
        raise ValueError(f"Invalid voxel spacing {spacing}: {path}")

    orientation_codes = nib.aff2axcodes(affine)
    orientation = tuple(
        code if code is not None else "?" for code in orientation_codes
    )

    return NiftiMetadata(
        path=path,
        shape=tuple(int(value) for value in image.shape),
        spacing_mm=spacing,
        orientation=orientation,
        dtype=str(image.get_data_dtype()),
        affine=tuple(tuple(float(value) for value in row) for row in affine),
    )


def read_segmentation_labels(path: Path) -> tuple[int, ...]:
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Segmentation file does not exist: {path}")

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)

    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D segmentation, got shape {data.shape}: {path}"
        )

    unique_values = np.unique(data)

    if not np.all(np.isfinite(unique_values)):
        raise ValueError(f"Segmentation contains non-finite labels: {path}")

    rounded = np.rint(unique_values)
    if not np.allclose(unique_values, rounded, rtol=0.0, atol=1e-6):
        raise ValueError(f"Segmentation contains non-integer labels: {path}")

    return tuple(int(value) for value in rounded)
