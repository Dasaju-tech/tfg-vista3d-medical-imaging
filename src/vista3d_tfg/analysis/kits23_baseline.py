from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import nibabel as nib
import numpy as np

from vista3d_tfg.metrics.morphology import summarize_binary_mask


SEMANTIC_REGIONS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("kidney", (1,)),
    ("tumor", (2,)),
    ("cyst", (3,)),
    ("kidney_and_mass", (1, 2, 3)),
    ("mass", (2, 3)),
)


@dataclass(frozen=True, slots=True)
class IntensitySummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    p01: float
    p05: float
    median: float
    p95: float
    p99: float


@dataclass(frozen=True, slots=True)
class RegionSummary:
    name: str
    labels: tuple[int, ...]
    present: bool
    voxel_count: int
    volume_mm3: float
    volume_ml: float
    component_count: int
    bounding_box_size_mm: tuple[float, float, float]
    bounding_box_diagonal_mm: float
    centroid_voxel: tuple[float, float, float] | None
    centroid_world_mm: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class CaseBaseline:
    case_id: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: tuple[str, str, str]
    ct_global: IntensitySummary
    ct_annotated_region: IntensitySummary
    regions: tuple[RegionSummary, ...]
    tumor_burden_fraction: float
    cyst_burden_fraction: float
    mass_burden_fraction: float

    def region(self, name: str) -> RegionSummary:
        for item in self.regions:
            if item.name == name:
                return item
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _describe(values: np.ndarray) -> IntensitySummary:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError("Intensity statistics require at least one finite value.")

    percentiles = np.percentile(
        finite,
        [1.0, 5.0, 50.0, 95.0, 99.0],
    )

    return IntensitySummary(
        count=int(finite.size),
        minimum=float(np.min(finite)),
        maximum=float(np.max(finite)),
        mean=float(np.mean(finite)),
        standard_deviation=float(np.std(finite)),
        p01=float(percentiles[0]),
        p05=float(percentiles[1]),
        median=float(percentiles[2]),
        p95=float(percentiles[3]),
        p99=float(percentiles[4]),
    )


def _centroid(
    mask: np.ndarray,
    affine: np.ndarray,
) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        return None, None

    centroid_voxel_array = coordinates.mean(axis=0)
    centroid_world_array = nib.affines.apply_affine(
        affine,
        centroid_voxel_array,
    )

    centroid_voxel = tuple(
        float(value) for value in centroid_voxel_array
    )
    centroid_world = tuple(
        float(value) for value in centroid_world_array
    )

    return centroid_voxel, centroid_world


def _region_summary(
    segmentation: np.ndarray,
    *,
    name: str,
    labels: tuple[int, ...],
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> RegionSummary:
    mask = np.isin(segmentation, labels)
    morphology = summarize_binary_mask(mask, spacing_mm)
    centroid_voxel, centroid_world = _centroid(mask, affine)

    return RegionSummary(
        name=name,
        labels=labels,
        present=morphology.voxel_count > 0,
        voxel_count=morphology.voxel_count,
        volume_mm3=morphology.volume_mm3,
        volume_ml=morphology.volume_ml,
        component_count=morphology.component_count,
        bounding_box_size_mm=morphology.bounding_box_size_mm,
        bounding_box_diagonal_mm=morphology.bounding_box_diagonal_mm,
        centroid_voxel=centroid_voxel,
        centroid_world_mm=centroid_world,
    )


def _safe_fraction(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def analyze_kits23_case(
    *,
    case_id: str,
    imaging_path: Path,
    segmentation_path: Path,
) -> CaseBaseline:
    """
    Compute a quantitative ground-truth baseline for one KiTS23 case.

    This function deliberately performs its own geometry checks even when the
    Block 02 dataset audit has already passed, so standalone use remains safe.
    """

    image_nii = nib.load(str(imaging_path))
    segmentation_nii = nib.load(str(segmentation_path))

    if len(image_nii.shape) != 3 or len(segmentation_nii.shape) != 3:
        raise ValueError("KiTS23 baseline analysis requires 3D NIfTI volumes.")

    if image_nii.shape != segmentation_nii.shape:
        raise ValueError(
            "Image and segmentation shapes do not match: "
            f"{image_nii.shape} != {segmentation_nii.shape}."
        )

    if not np.allclose(
        image_nii.affine,
        segmentation_nii.affine,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("Image and segmentation affines do not match.")

    image = np.asanyarray(image_nii.dataobj)
    segmentation = np.asanyarray(segmentation_nii.dataobj)

    if not np.issubdtype(segmentation.dtype, np.integer):
        raise TypeError(
            "KiTS23 reference segmentation must use an integer dtype."
        )

    labels = {int(value) for value in np.unique(segmentation)}
    unexpected = labels - {0, 1, 2, 3}
    if unexpected:
        raise ValueError(
            f"Unexpected KiTS23 labels: {sorted(unexpected)}."
        )

    spacing_mm = tuple(
        float(value)
        for value in image_nii.header.get_zooms()[:3]
    )
    orientation = tuple(
        str(value)
        for value in nib.aff2axcodes(image_nii.affine)
    )

    annotated_mask = segmentation > 0
    if np.any(annotated_mask):
        annotated_values = image[annotated_mask]
    else:
        annotated_values = image

    regions = tuple(
        _region_summary(
            segmentation,
            name=name,
            labels=region_labels,
            spacing_mm=spacing_mm,
            affine=image_nii.affine,
        )
        for name, region_labels in SEMANTIC_REGIONS
    )

    by_name = {item.name: item for item in regions}
    kidney_and_mass = by_name["kidney_and_mass"].volume_ml

    return CaseBaseline(
        case_id=case_id,
        shape=tuple(int(value) for value in image_nii.shape),
        spacing_mm=spacing_mm,
        orientation=orientation,
        ct_global=_describe(image),
        ct_annotated_region=_describe(annotated_values),
        regions=regions,
        tumor_burden_fraction=_safe_fraction(
            by_name["tumor"].volume_ml,
            kidney_and_mass,
        ),
        cyst_burden_fraction=_safe_fraction(
            by_name["cyst"].volume_ml,
            kidney_and_mass,
        ),
        mass_burden_fraction=_safe_fraction(
            by_name["mass"].volume_ml,
            kidney_and_mass,
        ),
    )


def cohort_summary(
    cases: Iterable[CaseBaseline],
) -> dict[str, object]:
    cases = tuple(cases)

    if not cases:
        raise ValueError("At least one case is required.")

    region_names = [name for name, _ in SEMANTIC_REGIONS]

    regions: dict[str, dict[str, object]] = {}
    for region_name in region_names:
        region_items = [case.region(region_name) for case in cases]
        volumes = [item.volume_ml for item in region_items]

        regions[region_name] = {
            "present_cases": sum(item.present for item in region_items),
            "prevalence": float(
                sum(item.present for item in region_items) / len(region_items)
            ),
            "mean_volume_ml": float(mean(volumes)),
            "median_volume_ml": float(median(volumes)),
            "minimum_volume_ml": float(min(volumes)),
            "maximum_volume_ml": float(max(volumes)),
            "total_components": int(
                sum(item.component_count for item in region_items)
            ),
        }

    return {
        "schema_version": "1.0",
        "dataset": "KiTS23",
        "analysis": "ground_truth_quantitative_baseline",
        "number_of_cases": len(cases),
        "case_ids": [case.case_id for case in cases],
        "regions": regions,
        "tumor_burden_fraction": {
            "mean": float(mean(case.tumor_burden_fraction for case in cases)),
            "median": float(median(case.tumor_burden_fraction for case in cases)),
        },
        "cyst_burden_fraction": {
            "mean": float(mean(case.cyst_burden_fraction for case in cases)),
            "median": float(median(case.cyst_burden_fraction for case in cases)),
        },
        "mass_burden_fraction": {
            "mean": float(mean(case.mass_burden_fraction for case in cases)),
            "median": float(median(case.mass_burden_fraction for case in cases)),
        },
    }


def write_case_baselines(
    cases: Iterable[CaseBaseline],
    output_path: Path,
) -> None:
    cases = tuple(cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "analysis": "ground_truth_quantitative_baseline",
                "cases": [case.to_dict() for case in cases],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_cohort_summary(
    cases: Iterable[CaseBaseline],
    output_path: Path,
) -> None:
    payload = cohort_summary(cases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_case_table_csv(
    cases: Iterable[CaseBaseline],
    output_path: Path,
) -> None:
    cases = tuple(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "kidney_volume_ml",
        "tumor_volume_ml",
        "cyst_volume_ml",
        "mass_volume_ml",
        "kidney_and_mass_volume_ml",
        "tumor_components",
        "cyst_components",
        "tumor_burden_fraction",
        "cyst_burden_fraction",
        "mass_burden_fraction",
        "ct_p01",
        "ct_median",
        "ct_p99",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for case in cases:
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "spacing_x_mm": case.spacing_mm[0],
                    "spacing_y_mm": case.spacing_mm[1],
                    "spacing_z_mm": case.spacing_mm[2],
                    "kidney_volume_ml": case.region("kidney").volume_ml,
                    "tumor_volume_ml": case.region("tumor").volume_ml,
                    "cyst_volume_ml": case.region("cyst").volume_ml,
                    "mass_volume_ml": case.region("mass").volume_ml,
                    "kidney_and_mass_volume_ml": (
                        case.region("kidney_and_mass").volume_ml
                    ),
                    "tumor_components": case.region("tumor").component_count,
                    "cyst_components": case.region("cyst").component_count,
                    "tumor_burden_fraction": case.tumor_burden_fraction,
                    "cyst_burden_fraction": case.cyst_burden_fraction,
                    "mass_burden_fraction": case.mass_burden_fraction,
                    "ct_p01": case.ct_global.p01,
                    "ct_median": case.ct_global.median,
                    "ct_p99": case.ct_global.p99,
                }
            )
