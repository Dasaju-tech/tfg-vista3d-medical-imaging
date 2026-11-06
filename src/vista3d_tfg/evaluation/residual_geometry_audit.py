from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS


@dataclass(frozen=True)
class ResidualGeometryCase:
    case_id: str
    failure_mode: str
    original_hd95_mm: float
    dice: float
    surface_dice: float
    fp_voxels: int
    fn_voxels: int
    fp_volume_ml: float
    fn_volume_ml: float
    fp_components: int
    fn_components: int
    largest_fp_component_ml: float
    largest_fn_component_ml: float
    prediction_to_gt_surface_p95_mm: float
    gt_to_prediction_surface_p95_mm: float
    prediction_to_gt_surface_max_mm: float
    gt_to_prediction_surface_max_mm: float
    dominant_direction: str


def _load_labelmap(
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


def _surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )
    return mask & ~eroded


def _surface_distances(
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> np.ndarray:
    source_surface = _surface(source_mask)
    target_surface = _surface(target_mask)

    if not np.any(source_surface):
        return np.asarray([], dtype=float)

    if not np.any(target_surface):
        return np.asarray([float("inf")], dtype=float)

    distance_map = ndimage.distance_transform_edt(
        ~target_surface,
        sampling=spacing_mm,
    )
    return np.asarray(
        distance_map[source_surface],
        dtype=float,
    )


def _component_stats(
    mask: np.ndarray,
    *,
    voxel_volume_ml: float,
) -> tuple[int, float]:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)

    if count == 0:
        return 0, 0.0

    counts = np.bincount(labels.ravel())[1:]
    largest = int(counts.max()) if counts.size else 0
    return int(count), float(largest * voxel_volume_ml)


def _percentile_or_zero(
    values: np.ndarray,
    percentile: float,
) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def _max_or_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _dominant_direction(
    prediction_to_gt_p95: float,
    gt_to_prediction_p95: float,
) -> str:
    tolerance_mm = 1.0

    if prediction_to_gt_p95 > gt_to_prediction_p95 + tolerance_mm:
        return "prediction_to_gt"
    if gt_to_prediction_p95 > prediction_to_gt_p95 + tolerance_mm:
        return "gt_to_prediction"
    return "balanced"


def evaluate_residual_geometry(
    *,
    case_id: str,
    failure_mode: str,
    original_hd95_mm: float,
    dice: float,
    surface_dice: float,
    ground_truth_path: Path,
    prediction_path: Path,
) -> ResidualGeometryCase:
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

    gt, spacing = _load_labelmap(ground_truth_path)
    pred, pred_spacing = _load_labelmap(prediction_path)

    if not np.allclose(spacing, pred_spacing, atol=1e-6, rtol=0.0):
        raise ValueError(f"{case_id}: spacing mismatch.")

    gt_mask = np.isin(gt, (1, 2, 3))
    pred_mask = np.isin(
        pred,
        tuple(EXPECTED_RENAL_LABELS.values()),
    )

    fp = pred_mask & ~gt_mask
    fn = gt_mask & ~pred_mask

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)

    fp_components, largest_fp_ml = _component_stats(
        fp,
        voxel_volume_ml=voxel_volume_ml,
    )
    fn_components, largest_fn_ml = _component_stats(
        fn,
        voxel_volume_ml=voxel_volume_ml,
    )

    pred_to_gt = _surface_distances(
        pred_mask,
        gt_mask,
        spacing_mm=spacing,
    )
    gt_to_pred = _surface_distances(
        gt_mask,
        pred_mask,
        spacing_mm=spacing,
    )

    pred_to_gt_p95 = _percentile_or_zero(pred_to_gt, 95.0)
    gt_to_pred_p95 = _percentile_or_zero(gt_to_pred, 95.0)

    return ResidualGeometryCase(
        case_id=case_id,
        failure_mode=failure_mode,
        original_hd95_mm=original_hd95_mm,
        dice=dice,
        surface_dice=surface_dice,
        fp_voxels=int(np.count_nonzero(fp)),
        fn_voxels=int(np.count_nonzero(fn)),
        fp_volume_ml=float(np.count_nonzero(fp) * voxel_volume_ml),
        fn_volume_ml=float(np.count_nonzero(fn) * voxel_volume_ml),
        fp_components=fp_components,
        fn_components=fn_components,
        largest_fp_component_ml=largest_fp_ml,
        largest_fn_component_ml=largest_fn_ml,
        prediction_to_gt_surface_p95_mm=pred_to_gt_p95,
        gt_to_prediction_surface_p95_mm=gt_to_pred_p95,
        prediction_to_gt_surface_max_mm=_max_or_zero(pred_to_gt),
        gt_to_prediction_surface_max_mm=_max_or_zero(gt_to_pred),
        dominant_direction=_dominant_direction(
            pred_to_gt_p95,
            gt_to_pred_p95,
        ),
    )


def read_outlier_taxonomy(
    path: Path,
) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    outliers = tuple(
        row
        for row in rows
        if row["hd95_outlier"].strip().lower() == "true"
    )

    if len(outliers) != 22:
        raise ValueError(
            f"Expected 22 frozen HD95 outliers, got {len(outliers)}."
        )

    return outliers


def build_residual_geometry_cases(
    *,
    taxonomy_path: Path,
    dataset_root: Path,
    prediction_root: Path,
) -> tuple[ResidualGeometryCase, ...]:
    rows = read_outlier_taxonomy(taxonomy_path)

    output = []

    for row in rows:
        case_id = row["case_id"]
        output.append(
            evaluate_residual_geometry(
                case_id=case_id,
                failure_mode=row["failure_mode"],
                original_hd95_mm=float(row["hd95_mm"]),
                dice=float(row["dice"]),
                surface_dice=float(row["surface_dice"]),
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
        )

    return tuple(output)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize empty values.")

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def analyze_residual_geometry(
    rows: Iterable[ResidualGeometryCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 22:
        raise ValueError(
            f"Expected 22 frozen HD95 outliers, got {len(rows)}."
        )

    directions = {
        "prediction_to_gt": tuple(
            row for row in rows
            if row.dominant_direction == "prediction_to_gt"
        ),
        "gt_to_prediction": tuple(
            row for row in rows
            if row.dominant_direction == "gt_to_prediction"
        ),
        "balanced": tuple(
            row for row in rows
            if row.dominant_direction == "balanced"
        ),
    }

    failure_modes = {}
    for mode in sorted({row.failure_mode for row in rows}):
        group = tuple(row for row in rows if row.failure_mode == mode)
        failure_modes[mode] = {
            "count": len(group),
            "fp_volume_ml": _summary(
                row.fp_volume_ml for row in group
            ),
            "fn_volume_ml": _summary(
                row.fn_volume_ml for row in group
            ),
            "prediction_to_gt_surface_p95_mm": _summary(
                row.prediction_to_gt_surface_p95_mm for row in group
            ),
            "gt_to_prediction_surface_p95_mm": _summary(
                row.gt_to_prediction_surface_p95_mm for row in group
            ),
        }

    largest_fp = sorted(
        rows,
        key=lambda row: (
            -row.largest_fp_component_ml,
            row.case_id,
        ),
    )[:10]
    largest_fn = sorted(
        rows,
        key=lambda row: (
            -row.largest_fn_component_ml,
            row.case_id,
        ),
    )[:10]
    largest_pred_distance = sorted(
        rows,
        key=lambda row: (
            -row.prediction_to_gt_surface_p95_mm,
            row.case_id,
        ),
    )[:10]
    largest_gt_distance = sorted(
        rows,
        key=lambda row: (
            -row.gt_to_prediction_surface_p95_mm,
            row.case_id,
        ),
    )[:10]

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_residual_geometry_audit"
        ),
        "outlier_count": len(rows),
        "direction_counts": {
            name: len(group)
            for name, group in directions.items()
        },
        "all_outliers": {
            "fp_volume_ml": _summary(
                row.fp_volume_ml for row in rows
            ),
            "fn_volume_ml": _summary(
                row.fn_volume_ml for row in rows
            ),
            "prediction_to_gt_surface_p95_mm": _summary(
                row.prediction_to_gt_surface_p95_mm for row in rows
            ),
            "gt_to_prediction_surface_p95_mm": _summary(
                row.gt_to_prediction_surface_p95_mm for row in rows
            ),
        },
        "failure_modes": failure_modes,
        "rankings": {
            "largest_fp_component": [
                asdict(row) for row in largest_fp
            ],
            "largest_fn_component": [
                asdict(row) for row in largest_fn
            ],
            "largest_prediction_to_gt_p95": [
                asdict(row) for row in largest_pred_distance
            ],
            "largest_gt_to_prediction_p95": [
                asdict(row) for row in largest_gt_distance
            ],
        },
        "case_rows": [asdict(row) for row in rows],
        "interpretation_boundary": (
            "Directional surface distances and FP/FN components are "
            "descriptive diagnostics of the frozen predictions. They do "
            "not modify the registered metric and are used only to choose "
            "the next controlled intervention."
        ),
    }
