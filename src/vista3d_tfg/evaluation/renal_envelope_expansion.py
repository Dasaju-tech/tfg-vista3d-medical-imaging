from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.evaluation.nv_kits23 import (
    evaluate_kidney_and_mass_hec,
)
from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS


EXPANSION_RADII_MM = (2.0, 5.0, 10.0)


@dataclass(frozen=True)
class MetricTriplet:
    dice: float
    surface_dice: float
    hd95_mm: float


@dataclass(frozen=True)
class ExpansionVariant:
    radius_mm: float
    metrics: MetricTriplet
    added_volume_ml: float
    tumor_coverage_fraction: float | None


@dataclass(frozen=True)
class ExpansionCase:
    case_id: str
    undercoverage_target: bool
    original: MetricTriplet
    original_tumor_coverage_fraction: float | None
    variants: tuple[ExpansionVariant, ...]


def _load_integer_nifti(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(path))
    if len(nii.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")

    data = np.asanyarray(nii.dataobj)
    if not np.allclose(data, np.rint(data), atol=1e-5, rtol=0.0):
        raise TypeError(f"Expected integer-valued labels: {path}")

    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return np.rint(data).astype(np.int16, copy=False), spacing


def _prediction_values() -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in EXPECTED_RENAL_LABELS.values()
    )


def expand_mask_mm(
    mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    radius_mm: float,
) -> np.ndarray:
    if radius_mm < 0.0:
        raise ValueError("radius_mm must be non-negative.")
    if radius_mm == 0.0:
        return mask.astype(bool, copy=True)
    if not np.any(mask):
        return mask.astype(bool, copy=True)

    distance_to_mask = ndimage.distance_transform_edt(
        ~mask,
        sampling=spacing_mm,
    )
    return mask | (distance_to_mask <= radius_mm)


def _tumor_coverage_fraction(
    tumor_mask: np.ndarray,
    predicted_envelope: np.ndarray,
) -> float | None:
    tumor_voxels = int(np.count_nonzero(tumor_mask))
    if tumor_voxels == 0:
        return None

    covered = int(
        np.count_nonzero(tumor_mask & predicted_envelope)
    )
    return covered / tumor_voxels


def _evaluate(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> MetricTriplet:
    result = evaluate_kidney_and_mass_hec(
        reference,
        prediction,
        spacing_mm=spacing_mm,
    )
    return MetricTriplet(
        dice=result.dice,
        surface_dice=result.surface_dice,
        hd95_mm=result.diagnostic_hd95_mm,
    )


def evaluate_expansion_case(
    *,
    case_id: str,
    undercoverage_target: bool,
    ground_truth_path: Path,
    prediction_path: Path,
    radii_mm: tuple[float, ...] = EXPANSION_RADII_MM,
) -> ExpansionCase:
    gt_nii = nib.load(str(ground_truth_path))
    pred_nii = nib.load(str(prediction_path))

    if gt_nii.shape != pred_nii.shape:
        raise ValueError(f"{case_id}: shape mismatch.")
    if not np.allclose(
        gt_nii.affine,
        pred_nii.affine,
        atol=1e-4,
        rtol=0.0,
    ):
        raise ValueError(f"{case_id}: affine mismatch.")

    ground_truth, spacing = _load_integer_nifti(
        ground_truth_path
    )
    prediction, pred_spacing = _load_integer_nifti(
        prediction_path
    )

    if not np.allclose(
        spacing,
        pred_spacing,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{case_id}: spacing mismatch.")

    reference = np.isin(ground_truth, (1, 2, 3))
    tumor = ground_truth == 2
    predicted_envelope = np.isin(
        prediction,
        _prediction_values(),
    )

    original = _evaluate(
        reference,
        predicted_envelope,
        spacing_mm=spacing,
    )
    original_tumor_coverage = _tumor_coverage_fraction(
        tumor,
        predicted_envelope,
    )

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    variants = []

    for radius_mm in radii_mm:
        expanded = expand_mask_mm(
            predicted_envelope,
            spacing_mm=spacing,
            radius_mm=float(radius_mm),
        )
        added_voxels = int(
            np.count_nonzero(expanded & ~predicted_envelope)
        )

        variants.append(
            ExpansionVariant(
                radius_mm=float(radius_mm),
                metrics=_evaluate(
                    reference,
                    expanded,
                    spacing_mm=spacing,
                ),
                added_volume_ml=(
                    added_voxels * voxel_volume_ml
                ),
                tumor_coverage_fraction=(
                    _tumor_coverage_fraction(
                        tumor,
                        expanded,
                    )
                ),
            )
        )

    return ExpansionCase(
        case_id=case_id,
        undercoverage_target=undercoverage_target,
        original=original,
        original_tumor_coverage_fraction=(
            original_tumor_coverage
        ),
        variants=tuple(variants),
    )


def read_undercoverage_target_ids(
    path: Path,
) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    ids = {
        row["case_id"]
        for row in rows
        if row["dominant_direction"] == "gt_to_prediction"
    }

    if len(ids) != 9:
        raise ValueError(
            f"Expected 9 GT->prediction targets, got {len(ids)}."
        )
    return ids


def read_expanded100_case_ids(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    ids = tuple(row["case_id"] for row in rows)

    if len(ids) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(ids)}."
        )
    if len(set(ids)) != 100:
        raise ValueError("Duplicate expanded100 case IDs.")

    return ids


def build_expansion_cases(
    *,
    cohort_csv: Path,
    geometry_csv: Path,
    dataset_root: Path,
    prediction_root: Path,
) -> tuple[ExpansionCase, ...]:
    case_ids = read_expanded100_case_ids(cohort_csv)
    target_ids = read_undercoverage_target_ids(geometry_csv)

    return tuple(
        evaluate_expansion_case(
            case_id=case_id,
            undercoverage_target=(case_id in target_ids),
            ground_truth_path=(
                dataset_root
                / case_id
                / "segmentation.nii.gz"
            ),
            prediction_path=(
                prediction_root
                / case_id
                / "prediction.nii.gz"
            ),
        )
        for case_id in case_ids
    )


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize empty values.")

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": (
            float(array.std(ddof=1))
            if array.size > 1
            else 0.0
        ),
    }


def _positive_tumor_coverage(
    values: Iterable[float | None],
) -> dict[str, float | int] | None:
    finite = [
        float(value)
        for value in values
        if value is not None
    ]
    if not finite:
        return None
    return _summary(finite)


def analyze_expansion_sensitivity(
    rows: Iterable[ExpansionCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(rows)}."
        )
    if sum(row.undercoverage_target for row in rows) != 9:
        raise ValueError(
            "Expected exactly 9 undercoverage targets."
        )

    original_hd95 = np.asarray(
        [row.original.hd95_mm for row in rows],
        dtype=float,
    )
    q1, q3 = np.percentile(original_hd95, [25.0, 75.0])
    original_threshold = float(q3 + 1.5 * (q3 - q1))

    targets = tuple(
        row for row in rows
        if row.undercoverage_target
    )
    non_targets = tuple(
        row for row in rows
        if not row.undercoverage_target
    )

    radii = tuple(
        variant.radius_mm
        for variant in rows[0].variants
    )

    variants_payload = {}

    for radius_index, radius_mm in enumerate(radii):
        def variant(row: ExpansionCase) -> ExpansionVariant:
            selected = row.variants[radius_index]
            if selected.radius_mm != radius_mm:
                raise ValueError(
                    "Variant radius ordering differs across cases."
                )
            return selected

        all_variants = tuple(variant(row) for row in rows)
        target_variants = tuple(
            variant(row) for row in targets
        )
        non_target_variants = tuple(
            variant(row) for row in non_targets
        )

        target_hd95_reduction = [
            row.original.hd95_mm
            - variant(row).metrics.hd95_mm
            for row in targets
        ]
        target_tumor_gain = [
            variant(row).tumor_coverage_fraction
            - row.original_tumor_coverage_fraction
            for row in targets
            if (
                variant(row).tumor_coverage_fraction
                is not None
                and row.original_tumor_coverage_fraction
                is not None
            )
        ]

        non_target_hd95_reduction = [
            row.original.hd95_mm
            - variant(row).metrics.hd95_mm
            for row in non_targets
        ]
        non_target_dice_delta = [
            variant(row).metrics.dice
            - row.original.dice
            for row in non_targets
        ]

        variants_payload[f"{radius_mm:g}mm"] = {
            "radius_mm": radius_mm,
            "all_cases": {
                "dice": _summary(
                    item.metrics.dice
                    for item in all_variants
                ),
                "surface_dice": _summary(
                    item.metrics.surface_dice
                    for item in all_variants
                ),
                "hd95_mm": _summary(
                    item.metrics.hd95_mm
                    for item in all_variants
                ),
                "added_volume_ml": _summary(
                    item.added_volume_ml
                    for item in all_variants
                ),
                "remaining_outliers_at_original_threshold": sum(
                    item.metrics.hd95_mm
                    > original_threshold
                    for item in all_variants
                ),
            },
            "undercoverage_targets": {
                "count": len(targets),
                "hd95_reduction_mm": _summary(
                    target_hd95_reduction
                ),
                "tumor_coverage_gain": (
                    _summary(target_tumor_gain)
                    if target_tumor_gain
                    else None
                ),
                "tumor_coverage": _positive_tumor_coverage(
                    item.tumor_coverage_fraction
                    for item in target_variants
                ),
                "resolved_below_original_threshold": sum(
                    row.original.hd95_mm > original_threshold
                    and variant(row).metrics.hd95_mm
                    <= original_threshold
                    for row in targets
                ),
            },
            "non_targets": {
                "count": len(non_targets),
                "hd95_worsened_gt_1mm": sum(
                    reduction < -1.0
                    for reduction in non_target_hd95_reduction
                ),
                "dice_worsened_gt_0_01": sum(
                    delta < -0.01
                    for delta in non_target_dice_delta
                ),
            },
        }

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_renal_envelope_expansion_sensitivity"
        ),
        "case_count": len(rows),
        "undercoverage_target_count": len(targets),
        "non_target_count": len(non_targets),
        "original_hd95_outlier_threshold_mm": original_threshold,
        "original": {
            "dice": _summary(
                row.original.dice for row in rows
            ),
            "surface_dice": _summary(
                row.original.surface_dice for row in rows
            ),
            "hd95_mm": _summary(
                row.original.hd95_mm for row in rows
            ),
            "target_tumor_coverage": _positive_tumor_coverage(
                row.original_tumor_coverage_fraction
                for row in targets
            ),
        },
        "variants": variants_payload,
        "case_rows": [
            {
                "case_id": row.case_id,
                "undercoverage_target": row.undercoverage_target,
                "original": asdict(row.original),
                "original_tumor_coverage_fraction": (
                    row.original_tumor_coverage_fraction
                ),
                "variants": [
                    {
                        "radius_mm": item.radius_mm,
                        "metrics": asdict(item.metrics),
                        "added_volume_ml": item.added_volume_ml,
                        "tumor_coverage_fraction": (
                            item.tumor_coverage_fraction
                        ),
                    }
                    for item in row.variants
                ],
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "This is a morphology sensitivity analysis, not an adopted "
            "post-processing method. Expansion uses prediction geometry "
            "only, while ground truth is used solely for retrospective "
            "evaluation. A useful radius would need to improve the 9 "
            "undercoverage targets without unacceptable damage to the "
            "remaining 91 cases."
        ),
    }
