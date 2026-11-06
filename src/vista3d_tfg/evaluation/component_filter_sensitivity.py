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


@dataclass(frozen=True)
class TaxonomyRow:
    case_id: str
    failure_mode: str
    hd95_outlier: bool


@dataclass(frozen=True)
class VariantMetric:
    dice: float
    surface_dice: float
    hd95_mm: float


@dataclass(frozen=True)
class SensitivityCase:
    case_id: str
    failure_mode: str
    hd95_outlier: bool
    original: VariantMetric
    global_top2: VariantMetric
    side_largest: VariantMetric
    global_top2_removed_voxels: int
    side_largest_removed_voxels: int


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


def _renal_values() -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in EXPECTED_RENAL_LABELS.values()
    )


def _side_values(side: str) -> tuple[int, ...]:
    prefix = f"{side} kidney"
    values = tuple(
        int(value)
        for name, value in EXPECTED_RENAL_LABELS.items()
        if name.startswith(prefix)
    )

    if not values:
        raise RuntimeError(
            f"No EXPECTED_RENAL_LABELS entries found for {side!r}."
        )
    return values


def _largest_component_mask(
    mask: np.ndarray,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)

    if count <= 1:
        return mask.astype(bool, copy=True)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    selected = int(np.argmax(counts))
    return labels == selected


def _largest_n_components_mask(
    mask: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive.")
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    structure = ndimage.generate_binary_structure(3, 1)
    labels, component_count = ndimage.label(
        mask,
        structure=structure,
    )

    if component_count <= count:
        return mask.astype(bool, copy=True)

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = np.argsort(sizes)[-count:]
    return np.isin(labels, selected)


def filter_global_top2(
    prediction: np.ndarray,
) -> tuple[np.ndarray, int]:
    renal_values = _renal_values()
    renal_mask = np.isin(prediction, renal_values)

    keep = _largest_n_components_mask(
        renal_mask,
        count=2,
    )

    output = prediction.copy()
    remove = renal_mask & ~keep
    output[remove] = 0

    return output, int(np.count_nonzero(remove))


def filter_side_largest(
    prediction: np.ndarray,
) -> tuple[np.ndarray, int]:
    output = prediction.copy()
    removed_total = 0

    for side in ("left", "right"):
        values = _side_values(side)
        side_mask = np.isin(prediction, values)
        keep = _largest_component_mask(side_mask)
        remove = side_mask & ~keep
        output[remove] = 0
        removed_total += int(np.count_nonzero(remove))

    return output, removed_total


def _evaluate_mask(
    *,
    reference: np.ndarray,
    prediction_labels: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> VariantMetric:
    prediction_mask = np.isin(
        prediction_labels,
        _renal_values(),
    )
    result = evaluate_kidney_and_mass_hec(
        reference,
        prediction_mask,
        spacing_mm=spacing_mm,
    )
    return VariantMetric(
        dice=result.dice,
        surface_dice=result.surface_dice,
        hd95_mm=result.diagnostic_hd95_mm,
    )


def read_taxonomy(path: Path) -> tuple[TaxonomyRow, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    if len(rows) != 100:
        raise ValueError(
            f"Expected 100 case-taxonomy rows, got {len(rows)}."
        )

    output = []
    for row in rows:
        outlier = row["hd95_outlier"].strip().lower()
        if outlier not in {"true", "false"}:
            raise ValueError(
                f"Invalid hd95_outlier value: {row['hd95_outlier']!r}"
            )

        output.append(
            TaxonomyRow(
                case_id=row["case_id"],
                failure_mode=row["failure_mode"],
                hd95_outlier=(outlier == "true"),
            )
        )

    return tuple(output)


def evaluate_case_variants(
    *,
    taxonomy: TaxonomyRow,
    ground_truth_path: Path,
    prediction_path: Path,
) -> SensitivityCase:
    gt_nii = nib.load(str(ground_truth_path))
    pred_nii = nib.load(str(prediction_path))

    if gt_nii.shape != pred_nii.shape:
        raise ValueError(
            f"{taxonomy.case_id}: GT/prediction shape mismatch."
        )
    if not np.allclose(
        gt_nii.affine,
        pred_nii.affine,
        atol=1e-4,
        rtol=0.0,
    ):
        raise ValueError(
            f"{taxonomy.case_id}: GT/prediction affine mismatch."
        )

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
        raise ValueError(
            f"{taxonomy.case_id}: GT/prediction spacing mismatch."
        )

    reference = np.isin(ground_truth, (1, 2, 3))

    global_filtered, global_removed = filter_global_top2(
        prediction
    )
    side_filtered, side_removed = filter_side_largest(
        prediction
    )

    return SensitivityCase(
        case_id=taxonomy.case_id,
        failure_mode=taxonomy.failure_mode,
        hd95_outlier=taxonomy.hd95_outlier,
        original=_evaluate_mask(
            reference=reference,
            prediction_labels=prediction,
            spacing_mm=spacing,
        ),
        global_top2=_evaluate_mask(
            reference=reference,
            prediction_labels=global_filtered,
            spacing_mm=spacing,
        ),
        side_largest=_evaluate_mask(
            reference=reference,
            prediction_labels=side_filtered,
            spacing_mm=spacing,
        ),
        global_top2_removed_voxels=global_removed,
        side_largest_removed_voxels=side_removed,
    )


def build_sensitivity_cases(
    *,
    taxonomy_path: Path,
    dataset_root: Path,
    prediction_root: Path,
) -> tuple[SensitivityCase, ...]:
    taxonomy = read_taxonomy(taxonomy_path)

    return tuple(
        evaluate_case_variants(
            taxonomy=row,
            ground_truth_path=(
                dataset_root
                / row.case_id
                / "segmentation.nii.gz"
            ),
            prediction_path=(
                prediction_root
                / row.case_id
                / "prediction.nii.gz"
            ),
        )
        for row in taxonomy
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


def _variant_payload(
    rows: tuple[SensitivityCase, ...],
    *,
    variant: str,
    original_threshold_mm: float,
) -> dict[str, object]:
    if variant not in {"global_top2", "side_largest"}:
        raise ValueError(f"Unknown variant: {variant}")

    variant_metrics = [
        getattr(row, variant)
        for row in rows
    ]

    original_hd95 = np.asarray(
        [row.original.hd95_mm for row in rows],
        dtype=float,
    )
    variant_hd95 = np.asarray(
        [metric.hd95_mm for metric in variant_metrics],
        dtype=float,
    )

    delta_hd95 = original_hd95 - variant_hd95
    delta_dice = np.asarray(
        [
            metric.dice - row.original.dice
            for row, metric in zip(rows, variant_metrics)
        ],
        dtype=float,
    )
    delta_surface = np.asarray(
        [
            metric.surface_dice - row.original.surface_dice
            for row, metric in zip(rows, variant_metrics)
        ],
        dtype=float,
    )

    fp_dominated = tuple(
        row
        for row in rows
        if row.failure_mode in {
            "distant_boundary_failure",
            "mixed_hd95_failure",
            "catastrophic_overlap_failure",
        }
        and row.hd95_outlier
    )

    # The Block 35 audit is the authoritative directional diagnosis.
    # Block 36 cannot infer direction from the taxonomy alone, so targeted
    # analysis is supplied separately by the runner after joining Block 35.

    return {
        "all_cases": {
            "hd95_mm": _summary(
                metric.hd95_mm for metric in variant_metrics
            ),
            "dice": _summary(
                metric.dice for metric in variant_metrics
            ),
            "surface_dice": _summary(
                metric.surface_dice for metric in variant_metrics
            ),
            "hd95_reduction_mm": _summary(delta_hd95),
            "dice_delta": _summary(delta_dice),
            "surface_dice_delta": _summary(delta_surface),
            "remaining_outliers_at_original_threshold": int(
                np.count_nonzero(
                    variant_hd95 > original_threshold_mm
                )
            ),
            "cases_hd95_improved_gt_1mm": int(
                np.count_nonzero(delta_hd95 > 1.0)
            ),
            "cases_hd95_worsened_gt_1mm": int(
                np.count_nonzero(delta_hd95 < -1.0)
            ),
            "cases_dice_worsened_gt_0_01": int(
                np.count_nonzero(delta_dice < -0.01)
            ),
        },
        "taxonomy_hd95_outlier_count": len(fp_dominated),
    }


def analyze_component_filter_sensitivity(
    rows: Iterable[SensitivityCase],
    *,
    prediction_to_gt_case_ids: Iterable[str],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(rows)}."
        )

    pred_to_gt_ids = set(prediction_to_gt_case_ids)
    if len(pred_to_gt_ids) != 12:
        raise ValueError(
            "Expected 12 prediction->GT dominated outliers from Block 35."
        )

    case_ids = {row.case_id for row in rows}
    if not pred_to_gt_ids.issubset(case_ids):
        raise ValueError(
            "Block 35 prediction->GT IDs are not a subset of expanded100."
        )

    original_hd95 = np.asarray(
        [row.original.hd95_mm for row in rows],
        dtype=float,
    )
    q1, q3 = np.percentile(original_hd95, [25.0, 75.0])
    threshold = float(q3 + 1.5 * (q3 - q1))

    target_rows = tuple(
        row for row in rows
        if row.case_id in pred_to_gt_ids
    )
    other_outliers = tuple(
        row for row in rows
        if row.hd95_outlier
        and row.case_id not in pred_to_gt_ids
    )
    non_outliers = tuple(
        row for row in rows
        if not row.hd95_outlier
    )

    variants = {}

    for variant in ("global_top2", "side_largest"):
        base = _variant_payload(
            rows,
            variant=variant,
            original_threshold_mm=threshold,
        )

        def subgroup(group: tuple[SensitivityCase, ...]) -> dict[str, object]:
            metrics = [getattr(row, variant) for row in group]
            reductions = [
                row.original.hd95_mm - metric.hd95_mm
                for row, metric in zip(group, metrics)
            ]
            dice_delta = [
                metric.dice - row.original.dice
                for row, metric in zip(group, metrics)
            ]
            return {
                "count": len(group),
                "hd95_reduction_mm": _summary(reductions),
                "dice_delta": _summary(dice_delta),
                "resolved_below_original_threshold": sum(
                    row.original.hd95_mm > threshold
                    and metric.hd95_mm <= threshold
                    for row, metric in zip(group, metrics)
                ),
                "hd95_worsened_gt_1mm": sum(
                    reduction < -1.0
                    for reduction in reductions
                ),
                "dice_worsened_gt_0_01": sum(
                    delta < -0.01
                    for delta in dice_delta
                ),
            }

        base["prediction_to_gt_targets"] = subgroup(
            target_rows
        )
        base["other_hd95_outliers"] = subgroup(
            other_outliers
        )
        base["non_outliers"] = subgroup(
            non_outliers
        )
        variants[variant] = base

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_component_filter_sensitivity"
        ),
        "case_count": len(rows),
        "original_hd95_outlier_threshold_mm": threshold,
        "prediction_to_gt_target_count": len(target_rows),
        "other_hd95_outlier_count": len(other_outliers),
        "non_outlier_count": len(non_outliers),
        "original": {
            "hd95_mm": _summary(
                row.original.hd95_mm for row in rows
            ),
            "dice": _summary(
                row.original.dice for row in rows
            ),
            "surface_dice": _summary(
                row.original.surface_dice for row in rows
            ),
        },
        "variants": variants,
        "case_rows": [
            {
                "case_id": row.case_id,
                "failure_mode": row.failure_mode,
                "hd95_outlier": row.hd95_outlier,
                "prediction_to_gt_target": (
                    row.case_id in pred_to_gt_ids
                ),
                "original": asdict(row.original),
                "global_top2": asdict(row.global_top2),
                "side_largest": asdict(row.side_largest),
                "global_top2_removed_voxels": (
                    row.global_top2_removed_voxels
                ),
                "side_largest_removed_voxels": (
                    row.side_largest_removed_voxels
                ),
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "This is a post-processing sensitivity analysis. Neither "
            "candidate is adopted into the frozen baseline. The candidates "
            "use prediction geometry only; ground truth is used solely for "
            "retrospective evaluation."
        ),
    }
