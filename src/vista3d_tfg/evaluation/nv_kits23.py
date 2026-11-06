from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from surface_distance import metrics as surface_metrics

from vista3d_tfg.analysis.anatomy import analyze_kits23_anatomy
from vista3d_tfg.integration.nv_inference import (
    EXPECTED_RENAL_LABELS,
)


KITS23_KIDNEY_AND_MASS_TOLERANCE_MM = 1.0330772532390826


@dataclass(frozen=True, slots=True)
class BinaryComparison:
    reference_voxels: int
    prediction_voxels: int
    intersection_voxels: int
    dice: float
    reference_volume_ml: float
    prediction_volume_ml: float
    signed_volume_error_ml: float
    absolute_volume_error_ml: float
    relative_volume_error_percent: float | None
    false_positive_volume_ml: float
    false_negative_volume_ml: float


@dataclass(frozen=True, slots=True)
class KidneyAndMassHEC:
    tolerance_mm: float
    dice: float
    surface_dice: float
    diagnostic_hd95_mm: float


@dataclass(frozen=True, slots=True)
class SupportedRenalComparison:
    case_id: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    prediction_labels_present: tuple[int, ...]
    gt_tumor_volume_ml: float
    full_kits23_score_available: bool
    full_kits23_score_reason: str
    kidney_and_mass_hec: KidneyAndMassHEC
    kidney_core_diagnostic: BinaryComparison
    cyst: BinaryComparison
    ground_truth_laterality_resolved: bool
    ground_truth_laterality_reason: str | None
    left_renal_envelope: BinaryComparison | None
    right_renal_envelope: BinaryComparison | None
    left_cyst: BinaryComparison | None
    right_cyst: BinaryComparison | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _voxel_volume_ml(spacing_mm: tuple[float, float, float]) -> float:
    return float(np.prod(spacing_mm) / 1000.0)


def compare_binary_masks(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> BinaryComparison:
    if reference.shape != prediction.shape:
        raise ValueError("Binary masks must have identical shapes.")

    reference = reference.astype(bool, copy=False)
    prediction = prediction.astype(bool, copy=False)

    reference_voxels = int(np.count_nonzero(reference))
    prediction_voxels = int(np.count_nonzero(prediction))
    intersection_voxels = int(
        np.count_nonzero(reference & prediction)
    )

    denominator = reference_voxels + prediction_voxels

    if denominator == 0:
        dice = 1.0
    else:
        dice = 2.0 * intersection_voxels / denominator

    voxel_ml = _voxel_volume_ml(spacing_mm)

    reference_volume = reference_voxels * voxel_ml
    prediction_volume = prediction_voxels * voxel_ml
    signed_error = prediction_volume - reference_volume

    if reference_volume > 0.0:
        relative_error = 100.0 * signed_error / reference_volume
    else:
        relative_error = None

    false_positive = int(
        np.count_nonzero(prediction & ~reference)
    )
    false_negative = int(
        np.count_nonzero(reference & ~prediction)
    )

    return BinaryComparison(
        reference_voxels=reference_voxels,
        prediction_voxels=prediction_voxels,
        intersection_voxels=intersection_voxels,
        dice=float(dice),
        reference_volume_ml=float(reference_volume),
        prediction_volume_ml=float(prediction_volume),
        signed_volume_error_ml=float(signed_error),
        absolute_volume_error_ml=float(abs(signed_error)),
        relative_volume_error_percent=(
            float(relative_error)
            if relative_error is not None
            else None
        ),
        false_positive_volume_ml=float(false_positive * voxel_ml),
        false_negative_volume_ml=float(false_negative * voxel_ml),
    )


def evaluate_kidney_and_mass_hec(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> KidneyAndMassHEC:
    basic = compare_binary_masks(
        reference,
        prediction,
        spacing_mm=spacing_mm,
    )

    reference_any = bool(np.any(reference))
    prediction_any = bool(np.any(prediction))

    if not reference_any and not prediction_any:
        surface_dice = 1.0
        hd95 = 0.0
    elif reference_any != prediction_any:
        surface_dice = 0.0
        hd95 = float("inf")
    else:
        distances = surface_metrics.compute_surface_distances(
            reference.astype(bool, copy=False),
            prediction.astype(bool, copy=False),
            spacing_mm=spacing_mm,
        )
        surface_dice = (
            surface_metrics.compute_surface_dice_at_tolerance(
                distances,
                KITS23_KIDNEY_AND_MASS_TOLERANCE_MM,
            )
        )
        hd95 = surface_metrics.compute_robust_hausdorff(
            distances,
            95.0,
        )

    return KidneyAndMassHEC(
        tolerance_mm=KITS23_KIDNEY_AND_MASS_TOLERANCE_MM,
        dice=basic.dice,
        surface_dice=float(surface_dice),
        diagnostic_hd95_mm=float(hd95),
    )


def _load_pair(
    ground_truth_path: Path,
    prediction_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    np.ndarray,
]:
    ground_truth_nii = nib.load(str(ground_truth_path))
    prediction_nii = nib.load(str(prediction_path))

    if len(ground_truth_nii.shape) != 3 or len(prediction_nii.shape) != 3:
        raise ValueError("Ground truth and prediction must be 3D.")

    if ground_truth_nii.shape != prediction_nii.shape:
        raise ValueError(
            "Ground truth and prediction shapes do not match."
        )

    if not np.allclose(
        ground_truth_nii.affine,
        prediction_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Ground truth and prediction affines do not match."
        )

    ground_truth = np.asanyarray(ground_truth_nii.dataobj)
    prediction = np.asanyarray(prediction_nii.dataobj)

    if not np.issubdtype(ground_truth.dtype, np.integer):
        raise TypeError("KiTS23 ground truth must use integer labels.")

    if not np.allclose(
        prediction,
        np.rint(prediction),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError("NV-Segment prediction must be integer-valued.")

    prediction = np.rint(prediction).astype(np.int16, copy=False)

    spacing = tuple(
        float(value)
        for value in ground_truth_nii.header.get_zooms()[:3]
    )

    return ground_truth, prediction, spacing, ground_truth_nii.affine


def _ground_truth_side_masks(
    ground_truth: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    # Global kidney-and-mass, kidney-core, and cyst metrics do not require
    # left/right assignment. Unresolved laterality disables only side-aware
    # diagnostics rather than invalidating the entire case.
    anatomy = analyze_kits23_anatomy(
        case_id="evaluation",
        segmentation=ground_truth,
        spacing_mm=spacing_mm,
        affine=affine,
    )

    if not anatomy.is_laterality_resolved:
        reason = (
            ";".join(anatomy.qc_flags)
            if anatomy.qc_flags
            else "ground_truth_laterality_unresolved"
        )
        return None, reason

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )
    component_labels, _ = ndimage.label(
        np.isin(ground_truth, (1, 2, 3)),
        structure=structure,
    )

    side_masks: dict[str, np.ndarray] = {}

    for unit in anatomy.renal_units:
        if unit.laterality not in {"left", "right"}:
            continue

        side_masks[unit.laterality] = (
            component_labels == unit.component_id
        )

    if set(side_masks) != {"left", "right"}:
        return None, "dominant_ground_truth_renal_masks_unavailable"

    return side_masks, None


def evaluate_supported_renal_prediction(
    *,
    case_id: str,
    ground_truth_path: Path,
    prediction_path: Path,
) -> SupportedRenalComparison:
    (
        ground_truth,
        prediction,
        spacing_mm,
        affine,
    ) = _load_pair(
        ground_truth_path,
        prediction_path,
    )

    labels_present = tuple(
        sorted(int(value) for value in np.unique(prediction))
    )

    allowed_prediction_labels = {
        0,
        *EXPECTED_RENAL_LABELS.values(),
    }
    unexpected = set(labels_present) - allowed_prediction_labels

    if unexpected:
        raise ValueError(
            f"Unexpected NV-Segment labels: {sorted(unexpected)}."
        )

    gt_renal_envelope = np.isin(
        ground_truth,
        (1, 2, 3),
    )
    predicted_renal_envelope = np.isin(
        prediction,
        tuple(EXPECTED_RENAL_LABELS.values()),
    )

    gt_kidney_core = ground_truth == 1
    predicted_kidneys = np.isin(
        prediction,
        (
            EXPECTED_RENAL_LABELS["left kidney"],
            EXPECTED_RENAL_LABELS["right kidney"],
        ),
    )

    gt_cyst = ground_truth == 3
    predicted_cyst = np.isin(
        prediction,
        (
            EXPECTED_RENAL_LABELS["left kidney cyst"],
            EXPECTED_RENAL_LABELS["right kidney cyst"],
        ),
    )

    gt_side_masks, laterality_reason = _ground_truth_side_masks(
        ground_truth,
        spacing_mm=spacing_mm,
        affine=affine,
    )

    left_gt = (
        gt_side_masks["left"]
        if gt_side_masks is not None
        else None
    )
    right_gt = (
        gt_side_masks["right"]
        if gt_side_masks is not None
        else None
    )

    left_prediction = (
        (prediction == EXPECTED_RENAL_LABELS["left kidney"])
        | (
            prediction
            == EXPECTED_RENAL_LABELS["left kidney cyst"]
        )
    )
    right_prediction = (
        (prediction == EXPECTED_RENAL_LABELS["right kidney"])
        | (
            prediction
            == EXPECTED_RENAL_LABELS["right kidney cyst"]
        )
    )

    left_gt_cyst = (
        gt_cyst & left_gt
        if left_gt is not None
        else None
    )
    right_gt_cyst = (
        gt_cyst & right_gt
        if right_gt is not None
        else None
    )

    left_pred_cyst = (
        prediction == EXPECTED_RENAL_LABELS["left kidney cyst"]
    )
    right_pred_cyst = (
        prediction == EXPECTED_RENAL_LABELS["right kidney cyst"]
    )

    voxel_ml = _voxel_volume_ml(spacing_mm)
    tumor_volume_ml = (
        float(np.count_nonzero(ground_truth == 2)) * voxel_ml
    )

    return SupportedRenalComparison(
        case_id=case_id,
        shape=tuple(int(value) for value in ground_truth.shape),
        spacing_mm=spacing_mm,
        prediction_labels_present=labels_present,
        gt_tumor_volume_ml=float(tumor_volume_ml),
        full_kits23_score_available=False,
        full_kits23_score_reason=(
            "NV-Segment-CT automatic inference does not provide a "
            "validated kidney-tumor class for the KiTS23 mass and tumor HECs."
        ),
        kidney_and_mass_hec=evaluate_kidney_and_mass_hec(
            gt_renal_envelope,
            predicted_renal_envelope,
            spacing_mm=spacing_mm,
        ),
        kidney_core_diagnostic=compare_binary_masks(
            gt_kidney_core,
            predicted_kidneys,
            spacing_mm=spacing_mm,
        ),
        cyst=compare_binary_masks(
            gt_cyst,
            predicted_cyst,
            spacing_mm=spacing_mm,
        ),
        ground_truth_laterality_resolved=(gt_side_masks is not None),
        ground_truth_laterality_reason=laterality_reason,
        left_renal_envelope=(
            compare_binary_masks(
                left_gt,
                left_prediction,
                spacing_mm=spacing_mm,
            )
            if left_gt is not None
            else None
        ),
        right_renal_envelope=(
            compare_binary_masks(
                right_gt,
                right_prediction,
                spacing_mm=spacing_mm,
            )
            if right_gt is not None
            else None
        ),
        left_cyst=(
            compare_binary_masks(
                left_gt_cyst,
                left_pred_cyst,
                spacing_mm=spacing_mm,
            )
            if left_gt_cyst is not None
            else None
        ),
        right_cyst=(
            compare_binary_masks(
                right_gt_cyst,
                right_pred_cyst,
                spacing_mm=spacing_mm,
            )
            if right_gt_cyst is not None
            else None
        ),
    )


def write_supported_renal_comparison(
    result: SupportedRenalComparison,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "dataset": "KiTS23",
                "model": "NV-Segment-CT",
                "protocol": "supported_renal_structures",
                "official_scope": (
                    "kidney_and_mass HEC only; no full KiTS23 score"
                ),
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
