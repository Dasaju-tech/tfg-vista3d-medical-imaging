from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Iterable

import numpy as np

try:
    from surface_distance import (
        compute_robust_hausdorff,
        compute_surface_dice_at_tolerance,
        compute_surface_distances,
    )
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "KiTS23 surface metrics require the 'surface-distance' package. "
        "Install it with: pip install -r requirements-kits23-metrics.txt"
    ) from exc


KITS23_ALLOWED_LABELS = frozenset({0, 1, 2, 3})


@dataclass(frozen=True, slots=True)
class HECDefinition:
    """Definition of one KiTS23 Hierarchical Evaluation Class (HEC)."""

    name: str
    labels: tuple[int, ...]
    surface_tolerance_mm: float


KITS23_HECS: tuple[HECDefinition, ...] = (
    HECDefinition(
        name="kidney_and_mass",
        labels=(1, 2, 3),
        surface_tolerance_mm=1.0330772532390826,
    ),
    HECDefinition(
        name="mass",
        labels=(2, 3),
        surface_tolerance_mm=1.1328796488598762,
    ),
    HECDefinition(
        name="tumor",
        labels=(2,),
        surface_tolerance_mm=1.1498198361434828,
    ),
)


@dataclass(frozen=True, slots=True)
class HECEvaluation:
    """Metrics for one KiTS23 HEC."""

    name: str
    labels: tuple[int, ...]
    surface_tolerance_mm: float
    dice: float
    surface_dice: float
    hd95_mm: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KiTS23CaseEvaluation:
    """Complete KiTS23-compatible evaluation for one 3D case."""

    hecs: tuple[HECEvaluation, ...]

    @property
    def mean_dice(self) -> float:
        return float(np.mean([item.dice for item in self.hecs]))

    @property
    def mean_surface_dice(self) -> float:
        return float(np.mean([item.surface_dice for item in self.hecs]))

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol": "KiTS23",
            "official_metrics": ["dice", "surface_dice"],
            "diagnostic_metrics": ["hd95_mm"],
            "mean_dice": self.mean_dice,
            "mean_surface_dice": self.mean_surface_dice,
            "hecs": [item.to_dict() for item in self.hecs],
        }


def _validate_spacing(spacing_mm: Iterable[float]) -> tuple[float, float, float]:
    spacing = tuple(float(value) for value in spacing_mm)

    if len(spacing) != 3:
        raise ValueError("spacing_mm must contain exactly three values.")

    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("All spacing_mm values must be finite and positive.")

    return spacing


def validate_kits23_segmentation(
    segmentation: np.ndarray,
    *,
    name: str = "segmentation",
) -> None:
    """Validate a discrete KiTS23 semantic segmentation map."""

    if not isinstance(segmentation, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array.")

    if segmentation.ndim != 3:
        raise ValueError(
            f"{name} must be 3D, got shape {segmentation.shape}."
        )

    if not np.issubdtype(segmentation.dtype, np.integer):
        raise TypeError(
            f"{name} must use an integer dtype, got {segmentation.dtype}."
        )

    labels = frozenset(int(value) for value in np.unique(segmentation))
    unexpected = labels - KITS23_ALLOWED_LABELS

    if unexpected:
        raise ValueError(
            f"{name} contains unexpected KiTS23 labels: "
            f"{sorted(unexpected)}."
        )


def construct_hec_mask(
    segmentation: np.ndarray,
    labels: tuple[int, ...],
) -> np.ndarray:
    """Merge semantic labels into a boolean HEC mask."""

    validate_kits23_segmentation(segmentation)
    return np.isin(segmentation, labels)


def dice_score(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Compute case-level Dice with KiTS23 empty-mask semantics.

    Both empty -> 1.0.
    Exactly one empty -> 0.0.
    """

    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)

    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference masks must have the same shape.")

    prediction_count = int(np.count_nonzero(prediction))
    reference_count = int(np.count_nonzero(reference))

    if prediction_count == 0 and reference_count == 0:
        return 1.0

    if prediction_count == 0 or reference_count == 0:
        return 0.0

    intersection = int(np.count_nonzero(prediction & reference))

    return float(
        2.0 * intersection / (prediction_count + reference_count)
    )


def evaluate_hec_masks(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    spacing_mm: Iterable[float],
    definition: HECDefinition,
) -> HECEvaluation:
    """
    Evaluate one HEC using KiTS23 Dice and Surface Dice.

    HD95 is included as a diagnostic metric and is not part of the official
    KiTS23 ranking.
    """

    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)

    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference masks must have the same shape.")

    if prediction.ndim != 3:
        raise ValueError("Prediction and reference masks must be 3D.")

    spacing = _validate_spacing(spacing_mm)

    prediction_empty = not np.any(prediction)
    reference_empty = not np.any(reference)

    if prediction_empty and reference_empty:
        return HECEvaluation(
            name=definition.name,
            labels=definition.labels,
            surface_tolerance_mm=definition.surface_tolerance_mm,
            dice=1.0,
            surface_dice=1.0,
            hd95_mm=0.0,
        )

    if prediction_empty or reference_empty:
        return HECEvaluation(
            name=definition.name,
            labels=definition.labels,
            surface_tolerance_mm=definition.surface_tolerance_mm,
            dice=0.0,
            surface_dice=0.0,
            hd95_mm=inf,
        )

    distances = compute_surface_distances(
        reference,
        prediction,
        spacing_mm=spacing,
    )

    return HECEvaluation(
        name=definition.name,
        labels=definition.labels,
        surface_tolerance_mm=definition.surface_tolerance_mm,
        dice=dice_score(prediction, reference),
        surface_dice=float(
            compute_surface_dice_at_tolerance(
                distances,
                tolerance_mm=definition.surface_tolerance_mm,
            )
        ),
        hd95_mm=float(
            compute_robust_hausdorff(
                distances,
                percent=95.0,
            )
        ),
    )


def evaluate_kits23_case(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    spacing_mm: Iterable[float],
) -> KiTS23CaseEvaluation:
    """Evaluate all three official KiTS23 HECs for one case."""

    validate_kits23_segmentation(prediction, name="prediction")
    validate_kits23_segmentation(reference, name="reference")

    if prediction.shape != reference.shape:
        raise ValueError(
            "Prediction and reference segmentations must have the same shape."
        )

    spacing = _validate_spacing(spacing_mm)

    evaluations: list[HECEvaluation] = []

    for definition in KITS23_HECS:
        prediction_mask = np.isin(prediction, definition.labels)
        reference_mask = np.isin(reference, definition.labels)

        evaluations.append(
            evaluate_hec_masks(
                prediction_mask,
                reference_mask,
                spacing_mm=spacing,
                definition=definition,
            )
        )

    return KiTS23CaseEvaluation(hecs=tuple(evaluations))
