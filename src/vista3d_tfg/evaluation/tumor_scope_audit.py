from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from vista3d_tfg.evaluation.nv_kits23 import (
    KidneyAndMassHEC,
    evaluate_kidney_and_mass_hec,
)
from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS


@dataclass(frozen=True)
class ScopeCase:
    case_id: str
    tumor_volume_ml: float
    failure_mode: str
    original_dice: float
    original_surface_dice: float
    original_hd95_mm: float
    tumor_excluded_dice: float
    tumor_excluded_surface_dice: float
    tumor_excluded_hd95_mm: float

    @property
    def hd95_reduction_mm(self) -> float:
        return self.original_hd95_mm - self.tumor_excluded_hd95_mm

    @property
    def dice_delta(self) -> float:
        return self.tumor_excluded_dice - self.original_dice

    @property
    def surface_dice_delta(self) -> float:
        return (
            self.tumor_excluded_surface_dice
            - self.original_surface_dice
        )


def _load_integer_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(path))
    if len(nii.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")

    array = np.asanyarray(nii.dataobj)
    if not np.allclose(array, np.rint(array), atol=1e-5, rtol=0.0):
        raise TypeError(f"Expected integer-valued labels: {path}")

    spacing = tuple(
        float(value)
        for value in nii.header.get_zooms()[:3]
    )
    return np.rint(array).astype(np.int16, copy=False), spacing


def evaluate_scope_pair(
    *,
    ground_truth_path: Path,
    prediction_path: Path,
) -> tuple[KidneyAndMassHEC, KidneyAndMassHEC, float]:
    gt_nii = nib.load(str(ground_truth_path))
    pred_nii = nib.load(str(prediction_path))

    if gt_nii.shape != pred_nii.shape:
        raise ValueError("Ground truth and prediction shapes differ.")
    if not np.allclose(
        gt_nii.affine,
        pred_nii.affine,
        atol=1e-4,
        rtol=0.0,
    ):
        raise ValueError("Ground truth and prediction affines differ.")

    ground_truth, spacing = _load_integer_nifti(ground_truth_path)
    prediction, pred_spacing = _load_integer_nifti(prediction_path)

    if not np.allclose(spacing, pred_spacing, atol=1e-6, rtol=0.0):
        raise ValueError("Ground truth and prediction spacing differs.")

    original_reference = np.isin(ground_truth, (1, 2, 3))
    tumor_excluded_reference = np.isin(ground_truth, (1, 3))
    predicted_renal_envelope = np.isin(
        prediction,
        tuple(EXPECTED_RENAL_LABELS.values()),
    )

    original = evaluate_kidney_and_mass_hec(
        original_reference,
        predicted_renal_envelope,
        spacing_mm=spacing,
    )
    tumor_excluded = evaluate_kidney_and_mass_hec(
        tumor_excluded_reference,
        predicted_renal_envelope,
        spacing_mm=spacing,
    )

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    tumor_volume_ml = (
        float(np.count_nonzero(ground_truth == 2))
        * voxel_volume_ml
    )

    return original, tumor_excluded, tumor_volume_ml


def read_case_taxonomy(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    if len(rows) != 100:
        raise ValueError(
            f"Expected 100 case-taxonomy rows, got {len(rows)}."
        )

    ids = [row["case_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in case taxonomy.")

    return rows


def build_scope_cases(
    *,
    taxonomy_path: Path,
    dataset_root: Path,
    prediction_root: Path,
) -> tuple[ScopeCase, ...]:
    taxonomy = read_case_taxonomy(taxonomy_path)

    output: list[ScopeCase] = []

    for row in taxonomy:
        case_id = row["case_id"]

        ground_truth_path = (
            dataset_root
            / case_id
            / "segmentation.nii.gz"
        )
        prediction_path = (
            prediction_root
            / case_id
            / "prediction.nii.gz"
        )

        if not ground_truth_path.is_file():
            raise FileNotFoundError(ground_truth_path)
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)

        original, tumor_excluded, tumor_volume = evaluate_scope_pair(
            ground_truth_path=ground_truth_path,
            prediction_path=prediction_path,
        )

        output.append(
            ScopeCase(
                case_id=case_id,
                tumor_volume_ml=tumor_volume,
                failure_mode=row["failure_mode"],
                original_dice=original.dice,
                original_surface_dice=original.surface_dice,
                original_hd95_mm=original.diagnostic_hd95_mm,
                tumor_excluded_dice=tumor_excluded.dice,
                tumor_excluded_surface_dice=(
                    tumor_excluded.surface_dice
                ),
                tumor_excluded_hd95_mm=(
                    tumor_excluded.diagnostic_hd95_mm
                ),
            )
        )

    return tuple(output)


def pearson_or_none(
    x: Iterable[float],
    y: Iterable[float],
) -> float | None:
    x_arr = np.asarray(tuple(x), dtype=float)
    y_arr = np.asarray(tuple(y), dtype=float)

    if x_arr.size != y_arr.size:
        raise ValueError("Correlation inputs must have equal length.")
    if x_arr.size < 2:
        return None
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None

    return float(np.corrcoef(x_arr, y_arr)[0, 1])


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


def analyze_scope_effect(
    rows: Iterable[ScopeCase],
) -> dict[str, object]:
    rows = tuple(rows)
    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(rows)}."
        )

    original_hd95 = np.asarray(
        [row.original_hd95_mm for row in rows],
        dtype=float,
    )
    q1, q3 = np.percentile(original_hd95, [25.0, 75.0])
    original_threshold = float(q3 + 1.5 * (q3 - q1))

    excluded_hd95 = np.asarray(
        [row.tumor_excluded_hd95_mm for row in rows],
        dtype=float,
    )
    eq1, eq3 = np.percentile(excluded_hd95, [25.0, 75.0])
    excluded_threshold = float(eq3 + 1.5 * (eq3 - eq1))

    original_outliers = tuple(
        row for row in rows
        if row.original_hd95_mm > original_threshold
    )
    excluded_fixed_threshold_outliers = tuple(
        row for row in rows
        if row.tumor_excluded_hd95_mm > original_threshold
    )
    excluded_reestimated_outliers = tuple(
        row for row in rows
        if row.tumor_excluded_hd95_mm > excluded_threshold
    )

    resolved_by_scope = tuple(
        row
        for row in original_outliers
        if row.tumor_excluded_hd95_mm <= original_threshold
    )

    failure_modes = sorted(
        {row.failure_mode for row in rows}
    )
    mode_summaries = {}

    for mode in failure_modes:
        group = tuple(
            row for row in rows
            if row.failure_mode == mode
        )
        if not group:
            continue

        mode_summaries[mode] = {
            "count": len(group),
            "original_hd95_mm": _summary(
                row.original_hd95_mm for row in group
            ),
            "tumor_excluded_hd95_mm": _summary(
                row.tumor_excluded_hd95_mm for row in group
            ),
            "hd95_reduction_mm": _summary(
                row.hd95_reduction_mm for row in group
            ),
            "dice_delta": _summary(
                row.dice_delta for row in group
            ),
            "surface_dice_delta": _summary(
                row.surface_dice_delta for row in group
            ),
        }

    tumor_positive = tuple(
        row for row in rows
        if row.tumor_volume_ml > 0.0
    )
    tumor_negative = tuple(
        row for row in rows
        if row.tumor_volume_ml == 0.0
    )

    largest_reductions = sorted(
        rows,
        key=lambda row: (
            -row.hd95_reduction_mm,
            row.case_id,
        ),
    )[:20]

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_tumor_scope_confounding_audit"
        ),
        "case_count": len(rows),
        "official_scope_note": (
            "The original kidney_and_mass HEC reference includes KiTS23 "
            "labels {1,2,3}. The automatic NV-Segment renal protocol has "
            "no validated tumor output class."
        ),
        "diagnostic_scope_note": (
            "The tumor-excluded reference {1,3} is a project-specific "
            "sensitivity diagnostic and is not an official KiTS23 HEC."
        ),
        "original_hd95_mm": _summary(
            row.original_hd95_mm for row in rows
        ),
        "tumor_excluded_hd95_mm": _summary(
            row.tumor_excluded_hd95_mm for row in rows
        ),
        "hd95_reduction_mm": _summary(
            row.hd95_reduction_mm for row in rows
        ),
        "dice_delta": _summary(
            row.dice_delta for row in rows
        ),
        "surface_dice_delta": _summary(
            row.surface_dice_delta for row in rows
        ),
        "outlier_analysis": {
            "original_threshold_mm": original_threshold,
            "original_outlier_count": len(original_outliers),
            "tumor_excluded_outlier_count_fixed_original_threshold": (
                len(excluded_fixed_threshold_outliers)
            ),
            "tumor_excluded_reestimated_threshold_mm": (
                excluded_threshold
            ),
            "tumor_excluded_outlier_count_reestimated_threshold": (
                len(excluded_reestimated_outliers)
            ),
            "original_outliers_resolved_by_tumor_exclusion": (
                len(resolved_by_scope)
            ),
            "resolved_case_ids": [
                row.case_id for row in resolved_by_scope
            ],
        },
        "tumor_volume_correlations": {
            "tumor_vs_original_hd95": pearson_or_none(
                (row.tumor_volume_ml for row in rows),
                (row.original_hd95_mm for row in rows),
            ),
            "tumor_vs_tumor_excluded_hd95": pearson_or_none(
                (row.tumor_volume_ml for row in rows),
                (row.tumor_excluded_hd95_mm for row in rows),
            ),
            "tumor_vs_hd95_reduction": pearson_or_none(
                (row.tumor_volume_ml for row in rows),
                (row.hd95_reduction_mm for row in rows),
            ),
        },


        "tumor_status": {
            "positive": {
                "count": len(tumor_positive),
                "hd95_reduction_mm": (
                    _summary(
                        row.hd95_reduction_mm
                        for row in tumor_positive
                    )
                    if tumor_positive
                    else None
                ),
            },
            "negative": {
                "count": len(tumor_negative),
                "hd95_reduction_mm": (
                    _summary(
                        row.hd95_reduction_mm
                        for row in tumor_negative
                    )
                    if tumor_negative
                    else None
                ),
            },
        },


        "failure_modes": mode_summaries,
        "largest_hd95_reductions": [
            {
                **asdict(row),
                "hd95_reduction_mm": row.hd95_reduction_mm,
                "dice_delta": row.dice_delta,
                "surface_dice_delta": row.surface_dice_delta,
            }
            for row in largest_reductions
        ],
        "case_rows": [
            {
                **asdict(row),
                "hd95_reduction_mm": row.hd95_reduction_mm,
                "dice_delta": row.dice_delta,
                "surface_dice_delta": row.surface_dice_delta,
                "original_hd95_outlier": (
                    row.original_hd95_mm > original_threshold
                ),
                "resolved_by_tumor_exclusion": (
                    row.original_hd95_mm > original_threshold
                    and row.tumor_excluded_hd95_mm
                    <= original_threshold
                ),
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "This is a scope-sensitivity analysis. The tumor-excluded "
            "metric is not an official KiTS23 score and must not replace "
            "the registered kidney_and_mass HEC result. It is used only "
            "to quantify how much missing automatic tumor coverage "
            "contributes to the observed geometric error tail."
        ),
    }
