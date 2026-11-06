from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage


DEFAULT_CT_WINDOW_HU = (-160.0, 240.0)
PLANE_TO_AXIS = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}


@dataclass(frozen=True, slots=True)
class SliceSelection:
    plane: str
    axis: int
    index: int
    source: str


@dataclass(frozen=True, slots=True)
class CaseQCResult:
    case_id: str
    canonical_orientation: tuple[str, str, str]
    canonical_shape: tuple[int, int, int]
    canonical_spacing_mm: tuple[float, float, float]
    ct_window_hu: tuple[float, float]
    kidney_and_mass_components: int
    selected_slices: tuple[SliceSelection, ...]
    overlay_files: tuple[str, ...]
    component_files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_window(
    window_hu: Iterable[float],
) -> tuple[float, float]:
    values = tuple(float(value) for value in window_hu)

    if len(values) != 2:
        raise ValueError("window_hu must contain exactly two values.")

    lower, upper = values

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("window_hu values must be finite.")

    if lower >= upper:
        raise ValueError("window_hu lower bound must be smaller than upper.")

    return lower, upper


def load_canonical_case(
    imaging_path: Path,
    segmentation_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    tuple[str, str, str],
]:
    image_nii = nib.load(str(imaging_path))
    segmentation_nii = nib.load(str(segmentation_path))

    if len(image_nii.shape) != 3 or len(segmentation_nii.shape) != 3:
        raise ValueError("QC visualization requires 3D NIfTI volumes.")

    if image_nii.shape != segmentation_nii.shape:
        raise ValueError("Image and segmentation shapes do not match.")

    if not np.allclose(
        image_nii.affine,
        segmentation_nii.affine,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("Image and segmentation affines do not match.")

    image_canonical = nib.as_closest_canonical(image_nii)
    segmentation_canonical = nib.as_closest_canonical(segmentation_nii)

    if image_canonical.shape != segmentation_canonical.shape:
        raise RuntimeError(
            "Canonical image and segmentation shapes unexpectedly differ."
        )

    if not np.allclose(
        image_canonical.affine,
        segmentation_canonical.affine,
        rtol=0.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Canonical image and segmentation affines unexpectedly differ."
        )

    image = np.asanyarray(image_canonical.dataobj)
    segmentation = np.asanyarray(segmentation_canonical.dataobj)

    if not np.issubdtype(segmentation.dtype, np.integer):
        raise TypeError("Segmentation must use an integer dtype.")

    unexpected = set(int(value) for value in np.unique(segmentation)) - {
        0, 1, 2, 3
    }
    if unexpected:
        raise ValueError(
            f"Unexpected KiTS23 labels: {sorted(unexpected)}."
        )

    spacing_mm = tuple(
        float(value)
        for value in image_canonical.header.get_zooms()[:3]
    )
    orientation = tuple(
        str(value)
        for value in nib.aff2axcodes(image_canonical.affine)
    )

    return (
        image,
        segmentation,
        image_canonical.affine,
        spacing_mm,
        orientation,
    )


def select_representative_slice(
    segmentation: np.ndarray,
    *,
    plane: str,
) -> SliceSelection:
    if plane not in PLANE_TO_AXIS:
        raise ValueError(
            f"Unknown plane {plane!r}. "
            f"Expected one of {sorted(PLANE_TO_AXIS)}."
        )

    if segmentation.ndim != 3:
        raise ValueError("segmentation must be 3D.")

    axis = PLANE_TO_AXIS[plane]

    lesion_mask = np.isin(segmentation, (2, 3))
    renal_mask = np.isin(segmentation, (1, 2, 3))

    if np.any(lesion_mask):
        target = lesion_mask
        source = "lesion"
    elif np.any(renal_mask):
        target = renal_mask
        source = "kidney_and_mass"
    else:
        index = segmentation.shape[axis] // 2
        return SliceSelection(
            plane=plane,
            axis=axis,
            index=int(index),
            source="volume_center",
        )

    reduction_axes = tuple(
        candidate
        for candidate in range(3)
        if candidate != axis
    )
    areas = np.count_nonzero(target, axis=reduction_axes)
    index = int(np.argmax(areas))

    return SliceSelection(
        plane=plane,
        axis=axis,
        index=index,
        source=source,
    )


def _slice_2d(
    volume: np.ndarray,
    *,
    axis: int,
    index: int,
) -> np.ndarray:
    section = np.take(volume, indices=index, axis=axis)
    return np.rot90(section)


def _display_spacing(
    spacing_mm: tuple[float, float, float],
    axis: int,
) -> tuple[float, float]:
    remaining = [
        spacing_mm[index]
        for index in range(3)
        if index != axis
    ]

    # np.rot90 swaps the displayed horizontal/vertical directions.
    return float(remaining[1]), float(remaining[0])


def _extent_for_slice(
    shape: tuple[int, int],
    display_spacing_mm: tuple[float, float],
) -> tuple[float, float, float, float]:
    height, width = shape
    x_spacing, y_spacing = display_spacing_mm

    return (
        0.0,
        width * x_spacing,
        0.0,
        height * y_spacing,
    )


def render_semantic_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    *,
    selection: SliceSelection,
    spacing_mm: tuple[float, float, float],
    output_path: Path,
    window_hu: Iterable[float] = DEFAULT_CT_WINDOW_HU,
) -> None:
    lower, upper = _validate_window(window_hu)

    image_slice = _slice_2d(
        image,
        axis=selection.axis,
        index=selection.index,
    )
    segmentation_slice = _slice_2d(
        segmentation,
        axis=selection.axis,
        index=selection.index,
    )

    extent = _extent_for_slice(
        image_slice.shape,
        _display_spacing(spacing_mm, selection.axis),
    )

    figure, axis = plt.subplots(figsize=(7, 7))

    axis.imshow(
        np.clip(image_slice, lower, upper),
        cmap="gray",
        vmin=lower,
        vmax=upper,
        extent=extent,
        origin="lower",
        interpolation="nearest",
    )

    overlay = np.ma.masked_where(
        segmentation_slice == 0,
        segmentation_slice,
    )
    axis.imshow(
        overlay,
        cmap=plt.get_cmap("tab10", 4),
        vmin=0,
        vmax=3,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        alpha=0.45,
    )

    axis.set_title(
        f"{selection.plane.capitalize()} semantic QC "
        f"(slice {selection.index}, source={selection.source})"
    )
    axis.set_xlabel("Physical distance [mm]")
    axis.set_ylabel("Physical distance [mm]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def component_label_map(
    segmentation: np.ndarray,
) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )
    labels, count = ndimage.label(
        np.isin(segmentation, (1, 2, 3)),
        structure=structure,
    )
    return labels.astype(np.int32, copy=False), int(count)


def render_component_overlay(
    image: np.ndarray,
    component_labels: np.ndarray,
    *,
    selection: SliceSelection,
    spacing_mm: tuple[float, float, float],
    output_path: Path,
    window_hu: Iterable[float] = DEFAULT_CT_WINDOW_HU,
) -> None:
    lower, upper = _validate_window(window_hu)

    image_slice = _slice_2d(
        image,
        axis=selection.axis,
        index=selection.index,
    )
    component_slice = _slice_2d(
        component_labels,
        axis=selection.axis,
        index=selection.index,
    )

    extent = _extent_for_slice(
        image_slice.shape,
        _display_spacing(spacing_mm, selection.axis),
    )

    figure, axis = plt.subplots(figsize=(7, 7))

    axis.imshow(
        np.clip(image_slice, lower, upper),
        cmap="gray",
        vmin=lower,
        vmax=upper,
        extent=extent,
        origin="lower",
        interpolation="nearest",
    )

    overlay = np.ma.masked_where(
        component_slice == 0,
        component_slice,
    )

    maximum_component = int(np.max(component_labels))
    color_count = max(2, maximum_component + 1)

    axis.imshow(
        overlay,
        cmap=plt.get_cmap("tab20", color_count),
        vmin=0,
        vmax=max(1, maximum_component),
        extent=extent,
        origin="lower",
        interpolation="nearest",
        alpha=0.50,
    )

    axis.set_title(
        f"{selection.plane.capitalize()} component QC "
        f"(slice {selection.index})"
    )
    axis.set_xlabel("Physical distance [mm]")
    axis.set_ylabel("Physical distance [mm]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_case_qc(
    *,
    case_id: str,
    imaging_path: Path,
    segmentation_path: Path,
    output_directory: Path,
    window_hu: Iterable[float] = DEFAULT_CT_WINDOW_HU,
) -> CaseQCResult:
    window = _validate_window(window_hu)

    (
        image,
        segmentation,
        _,
        spacing_mm,
        orientation,
    ) = load_canonical_case(
        imaging_path,
        segmentation_path,
    )

    selections = tuple(
        select_representative_slice(
            segmentation,
            plane=plane,
        )
        for plane in ("axial", "coronal", "sagittal")
    )

    component_labels, component_count = component_label_map(
        segmentation
    )

    case_output = output_directory / case_id
    overlay_files: list[str] = []
    component_files: list[str] = []

    for selection in selections:
        overlay_path = (
            case_output
            / f"{selection.plane}_semantic_overlay.png"
        )
        component_path = (
            case_output
            / f"{selection.plane}_component_overlay.png"
        )

        render_semantic_overlay(
            image,
            segmentation,
            selection=selection,
            spacing_mm=spacing_mm,
            output_path=overlay_path,
            window_hu=window,
        )
        render_component_overlay(
            image,
            component_labels,
            selection=selection,
            spacing_mm=spacing_mm,
            output_path=component_path,
            window_hu=window,
        )

        overlay_files.append(str(overlay_path))
        component_files.append(str(component_path))

    return CaseQCResult(
        case_id=case_id,
        canonical_orientation=orientation,
        canonical_shape=tuple(int(value) for value in image.shape),
        canonical_spacing_mm=spacing_mm,
        ct_window_hu=window,
        kidney_and_mass_components=component_count,
        selected_slices=selections,
        overlay_files=tuple(overlay_files),
        component_files=tuple(component_files),
    )


def write_qc_index(
    cases: Iterable[CaseQCResult],
    output_path: Path,
) -> None:
    cases = tuple(cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "analysis": "visual_quality_control",
                "coordinate_convention": "canonical RAS+",
                "semantic_labels": {
                    "0": "background",
                    "1": "kidney",
                    "2": "tumor",
                    "3": "cyst",
                },
                "cases": [case.to_dict() for case in cases],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
