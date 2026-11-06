from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.metrics.morphology import summarize_binary_mask


_ALLOWED_LABELS = frozenset({0, 1, 2, 3})
_CONNECTIVITY = 1  # 6-connected voxels in 3D.


@dataclass(frozen=True, slots=True)
class ComponentMeasurement:
    component_id: int
    voxel_count: int
    volume_ml: float
    centroid_voxel: tuple[float, float, float]
    centroid_world_mm: tuple[float, float, float]
    bounding_box_size_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class RenalUnit:
    component_id: int
    laterality: str
    voxel_count: int
    volume_ml: float
    centroid_voxel: tuple[float, float, float]
    centroid_world_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class LesionComponent:
    lesion_id: int
    kind: str
    voxel_count: int
    volume_ml: float
    centroid_voxel: tuple[float, float, float]
    centroid_world_mm: tuple[float, float, float]
    parent_component_id: int | None
    laterality: str


@dataclass(frozen=True, slots=True)
class SideSummary:
    laterality: str
    renal_volume_ml: float
    tumor_components: int
    tumor_volume_ml: float
    cyst_components: int
    cyst_volume_ml: float


@dataclass(frozen=True, slots=True)
class CaseAnatomy:
    case_id: str
    kidney_and_mass_components: int
    renal_units: tuple[RenalUnit, ...]
    tumor_components: tuple[LesionComponent, ...]
    cyst_components: tuple[LesionComponent, ...]
    side_summaries: tuple[SideSummary, ...]
    qc_flags: tuple[str, ...]

    @property
    def is_laterality_resolved(self) -> bool:
        return (
            len(self.renal_units) == 2
            and {item.laterality for item in self.renal_units}
            == {"left", "right"}
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["is_laterality_resolved"] = self.is_laterality_resolved
        return payload


def _validate_segmentation(segmentation: np.ndarray) -> None:
    if not isinstance(segmentation, np.ndarray):
        raise TypeError("segmentation must be a NumPy array.")

    if segmentation.ndim != 3:
        raise ValueError(
            f"segmentation must be 3D, got shape {segmentation.shape}."
        )

    if not np.issubdtype(segmentation.dtype, np.integer):
        raise TypeError("segmentation must use an integer dtype.")

    labels = frozenset(int(value) for value in np.unique(segmentation))
    unexpected = labels - _ALLOWED_LABELS

    if unexpected:
        raise ValueError(
            f"Unexpected KiTS23 labels: {sorted(unexpected)}."
        )


def _validate_spacing(
    spacing_mm: Iterable[float],
) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in spacing_mm)

    if len(spacing) != 3:
        raise ValueError("spacing_mm must contain exactly three values.")

    if not all(np.isfinite(value) and value > 0.0 for value in spacing):
        raise ValueError("spacing_mm values must be finite and positive.")

    return spacing


def _validate_affine(affine: np.ndarray) -> np.ndarray:
    affine = np.asarray(affine, dtype=float)

    if affine.shape != (4, 4):
        raise ValueError("affine must have shape (4, 4).")

    if not np.all(np.isfinite(affine)):
        raise ValueError("affine must contain only finite values.")

    if np.isclose(np.linalg.det(affine[:3, :3]), 0.0):
        raise ValueError("affine must be non-singular.")

    return affine


def _component_measurement(
    labels: np.ndarray,
    component_id: int,
    *,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> ComponentMeasurement:
    mask = labels == component_id
    morphology = summarize_binary_mask(mask, spacing_mm)

    coordinates = np.argwhere(mask)
    centroid_voxel_array = coordinates.mean(axis=0)
    centroid_world_array = nib.affines.apply_affine(
        affine,
        centroid_voxel_array,
    )

    return ComponentMeasurement(
        component_id=component_id,
        voxel_count=morphology.voxel_count,
        volume_ml=morphology.volume_ml,
        centroid_voxel=tuple(
            float(value) for value in centroid_voxel_array
        ),
        centroid_world_mm=tuple(
            float(value) for value in centroid_world_array
        ),
        bounding_box_size_mm=morphology.bounding_box_size_mm,
    )


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=_CONNECTIVITY,
    )
    labels, count = ndimage.label(mask.astype(bool), structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def _dominant_renal_units(
    kidney_and_mass_labels: np.ndarray,
    component_count: int,
    *,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> tuple[tuple[RenalUnit, ...], dict[int, str], list[str]]:
    qc_flags: list[str] = []

    components = [
        _component_measurement(
            kidney_and_mass_labels,
            component_id,
            spacing_mm=spacing_mm,
            affine=affine,
        )
        for component_id in range(1, component_count + 1)
    ]

    components.sort(
        key=lambda item: (-item.voxel_count, item.component_id)
    )

    if component_count < 2:
        qc_flags.append("fewer_than_two_kidney_and_mass_components")

        units = tuple(
            RenalUnit(
                component_id=item.component_id,
                laterality="undetermined",
                voxel_count=item.voxel_count,
                volume_ml=item.volume_ml,
                centroid_voxel=item.centroid_voxel,
                centroid_world_mm=item.centroid_world_mm,
            )
            for item in components
        )
        return units, {
            item.component_id: "undetermined" for item in components
        }, qc_flags

    if component_count > 2:
        qc_flags.append("more_than_two_kidney_and_mass_components")

    dominant = components[:2]

    # NiBabel world coordinates are RAS+: x increases from subject left
    # toward subject right. Relative ordering avoids assuming x=0 is midline.
    by_x = sorted(
        dominant,
        key=lambda item: item.centroid_world_mm[0],
    )

    x_difference = abs(
        by_x[1].centroid_world_mm[0]
        - by_x[0].centroid_world_mm[0]
    )

    if x_difference <= 1e-6:
        qc_flags.append("ambiguous_renal_laterality")
        side_map = {
            item.component_id: "undetermined"
            for item in dominant
        }
    else:
        side_map = {
            by_x[0].component_id: "left",
            by_x[1].component_id: "right",
        }

    units = tuple(
        RenalUnit(
            component_id=item.component_id,
            laterality=side_map[item.component_id],
            voxel_count=item.voxel_count,
            volume_ml=item.volume_ml,
            centroid_voxel=item.centroid_voxel,
            centroid_world_mm=item.centroid_world_mm,
        )
        for item in dominant
    )

    return units, side_map, qc_flags


def _lesion_components(
    segmentation: np.ndarray,
    *,
    semantic_label: int,
    kind: str,
    kidney_and_mass_labels: np.ndarray,
    side_map: dict[int, str],
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> tuple[LesionComponent, ...]:
    lesion_labels, count = _label_components(
        segmentation == semantic_label
    )

    lesions: list[LesionComponent] = []

    for lesion_id in range(1, count + 1):
        measurement = _component_measurement(
            lesion_labels,
            lesion_id,
            spacing_mm=spacing_mm,
            affine=affine,
        )

        lesion_mask = lesion_labels == lesion_id
        parent_ids = np.unique(
            kidney_and_mass_labels[lesion_mask]
        )
        parent_ids = [
            int(value) for value in parent_ids if int(value) != 0
        ]

        parent_component_id = (
            parent_ids[0] if len(parent_ids) == 1 else None
        )
        laterality = (
            side_map.get(parent_component_id, "unassigned")
            if parent_component_id is not None
            else "unassigned"
        )

        lesions.append(
            LesionComponent(
                lesion_id=lesion_id,
                kind=kind,
                voxel_count=measurement.voxel_count,
                volume_ml=measurement.volume_ml,
                centroid_voxel=measurement.centroid_voxel,
                centroid_world_mm=measurement.centroid_world_mm,
                parent_component_id=parent_component_id,
                laterality=laterality,
            )
        )

    return tuple(lesions)


def _side_summaries(
    renal_units: tuple[RenalUnit, ...],
    tumor_components: tuple[LesionComponent, ...],
    cyst_components: tuple[LesionComponent, ...],
) -> tuple[SideSummary, ...]:
    summaries: list[SideSummary] = []

    for side in ("left", "right", "undetermined"):
        renal_volume = sum(
            item.volume_ml
            for item in renal_units
            if item.laterality == side
        )
        side_tumors = [
            item for item in tumor_components if item.laterality == side
        ]
        side_cysts = [
            item for item in cyst_components if item.laterality == side
        ]

        if renal_volume > 0.0 or side_tumors or side_cysts:
            summaries.append(
                SideSummary(
                    laterality=side,
                    renal_volume_ml=float(renal_volume),
                    tumor_components=len(side_tumors),
                    tumor_volume_ml=float(
                        sum(item.volume_ml for item in side_tumors)
                    ),
                    cyst_components=len(side_cysts),
                    cyst_volume_ml=float(
                        sum(item.volume_ml for item in side_cysts)
                    ),
                )
            )

    unassigned_tumors = [
        item for item in tumor_components
        if item.laterality == "unassigned"
    ]
    unassigned_cysts = [
        item for item in cyst_components
        if item.laterality == "unassigned"
    ]

    if unassigned_tumors or unassigned_cysts:
        summaries.append(
            SideSummary(
                laterality="unassigned",
                renal_volume_ml=0.0,
                tumor_components=len(unassigned_tumors),
                tumor_volume_ml=float(
                    sum(item.volume_ml for item in unassigned_tumors)
                ),
                cyst_components=len(unassigned_cysts),
                cyst_volume_ml=float(
                    sum(item.volume_ml for item in unassigned_cysts)
                ),
            )
        )

    return tuple(summaries)


def analyze_kits23_anatomy(
    *,
    case_id: str,
    segmentation: np.ndarray,
    spacing_mm: Iterable[float],
    affine: np.ndarray,
) -> CaseAnatomy:
    _validate_segmentation(segmentation)
    spacing = _validate_spacing(spacing_mm)
    affine = _validate_affine(affine)

    kidney_and_mass_labels, component_count = _label_components(
        np.isin(segmentation, (1, 2, 3))
    )

    renal_units, side_map, qc_flags = _dominant_renal_units(
        kidney_and_mass_labels,
        component_count,
        spacing_mm=spacing,
        affine=affine,
    )

    tumors = _lesion_components(
        segmentation,
        semantic_label=2,
        kind="tumor",
        kidney_and_mass_labels=kidney_and_mass_labels,
        side_map=side_map,
        spacing_mm=spacing,
        affine=affine,
    )
    cysts = _lesion_components(
        segmentation,
        semantic_label=3,
        kind="cyst",
        kidney_and_mass_labels=kidney_and_mass_labels,
        side_map=side_map,
        spacing_mm=spacing,
        affine=affine,
    )

    if any(item.laterality == "unassigned" for item in tumors):
        qc_flags.append("unassigned_tumor_component")

    if any(item.laterality == "unassigned" for item in cysts):
        qc_flags.append("unassigned_cyst_component")

    return CaseAnatomy(
        case_id=case_id,
        kidney_and_mass_components=component_count,
        renal_units=renal_units,
        tumor_components=tumors,
        cyst_components=cysts,
        side_summaries=_side_summaries(
            renal_units,
            tumors,
            cysts,
        ),
        qc_flags=tuple(sorted(set(qc_flags))),
    )


def analyze_kits23_anatomy_nifti(
    *,
    case_id: str,
    segmentation_path: Path,
) -> CaseAnatomy:
    segmentation_nii = nib.load(str(segmentation_path))

    if len(segmentation_nii.shape) != 3:
        raise ValueError("KiTS23 anatomy analysis requires a 3D NIfTI.")

    segmentation = np.asanyarray(segmentation_nii.dataobj)

    return analyze_kits23_anatomy(
        case_id=case_id,
        segmentation=segmentation,
        spacing_mm=segmentation_nii.header.get_zooms()[:3],
        affine=segmentation_nii.affine,
    )


def write_anatomy_results(
    cases: Iterable[CaseAnatomy],
    output_path: Path,
) -> None:
    cases = tuple(cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "analysis": "anatomical_component_inventory",
                "laterality_method": (
                    "paired dominant kidney-and-mass components ordered "
                    "by RAS+ world x coordinate"
                ),
                "cases": [case.to_dict() for case in cases],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
