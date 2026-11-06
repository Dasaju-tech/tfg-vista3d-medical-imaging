from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import median
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.evaluation.cohort import CaseCohortRow

NV_RENAL_LABELS = (5, 14, 116, 117)
NV_CYST_LABELS = (116, 117)


@dataclass(frozen=True, slots=True)
class ErrorComponentSummary:
    component_count: int
    total_volume_ml: float
    largest_component_volume_ml: float
    farthest_distance_mm: float


@dataclass(frozen=True, slots=True)
class CystLocalizationSummary:
    gt_component_count: int
    prediction_component_count: int
    gt_volume_ml: float
    prediction_volume_ml: float
    voxel_dice: float
    within_10mm: int
    within_20mm: int
    within_30mm: int
    nearest_centroid_median_mm: float | None
    zero_voxel_overlap: bool


@dataclass(frozen=True, slots=True)
class CaseFailureDecomposition:
    case_id: str
    kidney_and_mass_dice: float
    kidney_and_mass_surface_dice: float
    kidney_and_mass_hd95_mm: float
    false_positive: ErrorComponentSummary
    false_negative: ErrorComponentSummary
    cyst: CystLocalizationSummary | None


@dataclass(frozen=True, slots=True)
class FailureDecompositionSummary:
    case_count: int
    cyst_positive_case_count: int
    cases_with_fp_farther_than_20mm: int
    cases_with_fp_farther_than_50mm: int
    correlation_hd95_vs_farthest_fp_distance: float | None
    correlation_hd95_vs_farthest_fn_distance: float | None
    cyst_positive_zero_overlap_cases: int
    cyst_positive_zero_overlap_fraction: float | None
    cyst_gt_component_count: int
    cyst_component_localization_recall_10mm: float | None
    cyst_component_localization_recall_20mm: float | None
    cyst_component_localization_recall_30mm: float | None
    cyst_nearest_centroid_distance_median_mm: float | None
    worst_farthest_fp_case_ids: tuple[str, ...]
    worst_farthest_fn_case_ids: tuple[str, ...]
    cases: tuple[CaseFailureDecomposition, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _pearson(
    x: Iterable[float],
    y: Iterable[float],
) -> float | None:
    x_values = np.asarray(tuple(float(v) for v in x), dtype=float)
    y_values = np.asarray(tuple(float(v) for v in y), dtype=float)

    if x_values.size != y_values.size:
        raise ValueError("Correlation inputs must have equal length.")
    if x_values.size < 2:
        return None
    if np.isclose(np.std(x_values), 0.0) or np.isclose(np.std(y_values), 0.0):
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _load_masks(
    *,
    ground_truth_path: Path,
    prediction_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float], np.ndarray]:
    gt_nii = nib.load(str(ground_truth_path))
    pred_nii = nib.load(str(prediction_path))

    if gt_nii.shape != pred_nii.shape:
        raise ValueError("Ground truth and prediction shapes differ.")
    if not np.allclose(gt_nii.affine, pred_nii.affine, rtol=0.0, atol=1e-4):
        raise ValueError("Ground truth and prediction affines differ.")

    gt = np.asanyarray(gt_nii.dataobj)
    pred = np.asanyarray(pred_nii.dataobj)

    if not np.issubdtype(gt.dtype, np.integer):
        raise TypeError("Ground truth must use integer labels.")
    if not np.all(np.isfinite(pred)):
        raise ValueError("Prediction contains non-finite values.")
    if not np.allclose(pred, np.rint(pred), rtol=0.0, atol=1e-5):
        raise TypeError("Prediction must be integer-valued.")

    pred = np.rint(pred).astype(np.int16, copy=False)
    spacing = tuple(float(v) for v in gt_nii.header.get_zooms()[:3])

    return gt, pred, spacing, gt_nii.affine


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, count = ndimage.label(mask.astype(bool, copy=False), structure=structure)
    return labels.astype(np.int32, copy=False), int(count)


def summarize_error_components(
    error_mask: np.ndarray,
    *,
    target_mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> ErrorComponentSummary:
    labels, count = _label_components(error_mask)
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)

    if count == 0:
        return ErrorComponentSummary(0, 0.0, 0.0, 0.0)

    sizes = np.bincount(labels.ravel())[1:]
    largest_ml = float(int(sizes.max()) * voxel_ml)

    if np.any(target_mask):
        distance = ndimage.distance_transform_edt(
            ~target_mask.astype(bool, copy=False),
            sampling=spacing_mm,
        )
        farthest = float(np.max(distance[error_mask]))
    else:
        farthest = float("inf")

    return ErrorComponentSummary(
        component_count=count,
        total_volume_ml=float(np.count_nonzero(error_mask) * voxel_ml),
        largest_component_volume_ml=largest_ml,
        farthest_distance_mm=farthest,
    )


def _component_centroids_world(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, ...]:
    labels, count = _label_components(mask)
    centroids = []
    for component_id in range(1, count + 1):
        coords = np.argwhere(labels == component_id)
        voxel_centroid = coords.mean(axis=0)
        centroids.append(
            np.asarray(nib.affines.apply_affine(affine, voxel_centroid), dtype=float)
        )
    return tuple(centroids)


def _binary_dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    r = int(np.count_nonzero(reference))
    p = int(np.count_nonzero(prediction))
    if r + p == 0:
        return 1.0
    intersection = int(np.count_nonzero(reference & prediction))
    return float(2.0 * intersection / (r + p))


def summarize_cyst_localization(
    *,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> CystLocalizationSummary | None:
    gt_cyst = ground_truth == 3
    if not np.any(gt_cyst):
        return None

    pred_cyst = np.isin(prediction, NV_CYST_LABELS)
    gt_centroids = _component_centroids_world(gt_cyst, affine)
    pred_centroids = _component_centroids_world(pred_cyst, affine)

    nearest = []
    for gt_centroid in gt_centroids:
        if not pred_centroids:
            nearest.append(float("inf"))
        else:
            nearest.append(
                min(
                    float(np.linalg.norm(gt_centroid - pred_centroid))
                    for pred_centroid in pred_centroids
                )
            )

    finite = [value for value in nearest if np.isfinite(value)]
    voxel_ml = float(np.prod(spacing_mm) / 1000.0)

    return CystLocalizationSummary(
        gt_component_count=len(gt_centroids),
        prediction_component_count=len(pred_centroids),
        gt_volume_ml=float(np.count_nonzero(gt_cyst) * voxel_ml),
        prediction_volume_ml=float(np.count_nonzero(pred_cyst) * voxel_ml),
        voxel_dice=_binary_dice(gt_cyst, pred_cyst),
        within_10mm=sum(value <= 10.0 for value in nearest),
        within_20mm=sum(value <= 20.0 for value in nearest),
        within_30mm=sum(value <= 30.0 for value in nearest),
        nearest_centroid_median_mm=(
            float(median(finite)) if finite else None
        ),
        zero_voxel_overlap=not bool(np.any(gt_cyst & pred_cyst)),
    )


def decompose_case_failure(
    *,
    row: CaseCohortRow,
    ground_truth_path: Path,
    prediction_path: Path,
) -> CaseFailureDecomposition:
    gt, pred, spacing, affine = _load_masks(
        ground_truth_path=ground_truth_path,
        prediction_path=prediction_path,
    )

    gt_renal = np.isin(gt, (1, 2, 3))
    pred_renal = np.isin(pred, NV_RENAL_LABELS)

    fp = pred_renal & ~gt_renal
    fn = gt_renal & ~pred_renal

    return CaseFailureDecomposition(
        case_id=row.case_id,
        kidney_and_mass_dice=row.kidney_and_mass_dice,
        kidney_and_mass_surface_dice=row.kidney_and_mass_surface_dice,
        kidney_and_mass_hd95_mm=row.kidney_and_mass_hd95_mm,
        false_positive=summarize_error_components(
            fp,
            target_mask=gt_renal,
            spacing_mm=spacing,
        ),
        false_negative=summarize_error_components(
            fn,
            target_mask=pred_renal,
            spacing_mm=spacing,
        ),
        cyst=summarize_cyst_localization(
            ground_truth=gt,
            prediction=pred,
            spacing_mm=spacing,
            affine=affine,
        ),
    )


def summarize_failure_decomposition(
    cases: Iterable[CaseFailureDecomposition],
) -> FailureDecompositionSummary:
    cases = tuple(cases)
    if not cases:
        raise ValueError("At least one case is required.")

    cyst_cases = tuple(case for case in cases if case.cyst is not None)
    total_gt_components = sum(case.cyst.gt_component_count for case in cyst_cases)
    within_10 = sum(case.cyst.within_10mm for case in cyst_cases)
    within_20 = sum(case.cyst.within_20mm for case in cyst_cases)
    within_30 = sum(case.cyst.within_30mm for case in cyst_cases)

    centroid_medians = [
        case.cyst.nearest_centroid_median_mm
        for case in cyst_cases
        if case.cyst.nearest_centroid_median_mm is not None
    ]
    zero_overlap = sum(case.cyst.zero_voxel_overlap for case in cyst_cases)

    def recall(value: int) -> float | None:
        return None if total_gt_components == 0 else float(value / total_gt_components)

    worst_fp = tuple(
        case.case_id
        for case in sorted(
            cases,
            key=lambda item: (-item.false_positive.farthest_distance_mm, item.case_id),
        )[:5]
    )
    worst_fn = tuple(
        case.case_id
        for case in sorted(
            cases,
            key=lambda item: (-item.false_negative.farthest_distance_mm, item.case_id),
        )[:5]
    )

    return FailureDecompositionSummary(
        case_count=len(cases),
        cyst_positive_case_count=len(cyst_cases),
        cases_with_fp_farther_than_20mm=sum(
            case.false_positive.farthest_distance_mm > 20.0 for case in cases
        ),
        cases_with_fp_farther_than_50mm=sum(
            case.false_positive.farthest_distance_mm > 50.0 for case in cases
        ),
        correlation_hd95_vs_farthest_fp_distance=_pearson(
            (case.kidney_and_mass_hd95_mm for case in cases),
            (case.false_positive.farthest_distance_mm for case in cases),
        ),
        correlation_hd95_vs_farthest_fn_distance=_pearson(
            (case.kidney_and_mass_hd95_mm for case in cases),
            (case.false_negative.farthest_distance_mm for case in cases),
        ),
        cyst_positive_zero_overlap_cases=zero_overlap,
        cyst_positive_zero_overlap_fraction=(
            float(zero_overlap / len(cyst_cases)) if cyst_cases else None
        ),
        cyst_gt_component_count=total_gt_components,
        cyst_component_localization_recall_10mm=recall(within_10),
        cyst_component_localization_recall_20mm=recall(within_20),
        cyst_component_localization_recall_30mm=recall(within_30),
        cyst_nearest_centroid_distance_median_mm=(
            float(median(centroid_medians)) if centroid_medians else None
        ),
        worst_farthest_fp_case_ids=worst_fp,
        worst_farthest_fn_case_ids=worst_fn,
        cases=cases,
    )


def write_failure_json(
    summary: FailureDecompositionSummary,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "model": "NV-Segment-CT",
                "analysis": "pilot30_failure_decomposition",
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_failure_csv(
    summary: FailureDecompositionSummary,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "kidney_and_mass_dice",
        "kidney_and_mass_surface_dice",
        "kidney_and_mass_hd95_mm",
        "fp_component_count",
        "fp_total_volume_ml",
        "fp_largest_component_volume_ml",
        "fp_farthest_distance_mm",
        "fn_component_count",
        "fn_total_volume_ml",
        "fn_largest_component_volume_ml",
        "fn_farthest_distance_mm",
        "cyst_gt_components",
        "cyst_pred_components",
        "cyst_dice",
        "cyst_zero_overlap",
        "cyst_nearest_centroid_median_mm",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for case in summary.cases:
            cyst = case.cyst
            writer.writerow(
                {
                    "case_id": case.case_id,
                    "kidney_and_mass_dice": case.kidney_and_mass_dice,
                    "kidney_and_mass_surface_dice": case.kidney_and_mass_surface_dice,
                    "kidney_and_mass_hd95_mm": case.kidney_and_mass_hd95_mm,
                    "fp_component_count": case.false_positive.component_count,
                    "fp_total_volume_ml": case.false_positive.total_volume_ml,
                    "fp_largest_component_volume_ml": (
                        case.false_positive.largest_component_volume_ml
                    ),
                    "fp_farthest_distance_mm": (
                        case.false_positive.farthest_distance_mm
                    ),
                    "fn_component_count": case.false_negative.component_count,
                    "fn_total_volume_ml": case.false_negative.total_volume_ml,
                    "fn_largest_component_volume_ml": (
                        case.false_negative.largest_component_volume_ml
                    ),
                    "fn_farthest_distance_mm": (
                        case.false_negative.farthest_distance_mm
                    ),
                    "cyst_gt_components": cyst.gt_component_count if cyst else None,
                    "cyst_pred_components": (
                        cyst.prediction_component_count if cyst else None
                    ),
                    "cyst_dice": cyst.voxel_dice if cyst else None,
                    "cyst_zero_overlap": cyst.zero_voxel_overlap if cyst else None,
                    "cyst_nearest_centroid_median_mm": (
                        cyst.nearest_centroid_median_mm if cyst else None
                    ),
                }
            )


def render_failure_figures(
    summary: FailureDecompositionSummary,
    output_directory: Path,
) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = []

    hd95 = np.asarray(
        [case.kidney_and_mass_hd95_mm for case in summary.cases],
        dtype=float,
    )
    fp_distance = np.asarray(
        [case.false_positive.farthest_distance_mm for case in summary.cases],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(fp_distance, hd95)
    axis.set_xlabel("Farthest false-positive renal voxel [mm]")
    axis.set_ylabel("Kidney-and-mass HD95 [mm]")
    axis.set_title("HD95 versus distant false-positive error")
    figure.tight_layout()
    path = output_directory / "hd95_vs_farthest_fp_distance.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    cyst_cases = [
        case for case in summary.cases
        if (
            case.cyst is not None
            and case.cyst.nearest_centroid_median_mm is not None
        )
    ]

    if cyst_cases:
        distances = np.asarray(
            [case.cyst.nearest_centroid_median_mm for case in cyst_cases],
            dtype=float,
        )
        dice = np.asarray(
            [case.cyst.voxel_dice for case in cyst_cases],
            dtype=float,
        )

        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(distances, dice)
        axis.set_xlabel("Median nearest GT-to-predicted cyst centroid distance [mm]")
        axis.set_ylabel("Cyst Dice")
        axis.set_title("Cyst overlap versus localization distance")
        figure.tight_layout()
        path = output_directory / "cyst_dice_vs_centroid_distance.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)

    return tuple(outputs)
