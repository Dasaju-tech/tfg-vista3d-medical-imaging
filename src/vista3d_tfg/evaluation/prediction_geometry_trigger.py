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
class PredictionGeometry:
    case_id: str
    prediction_to_gt_target: bool
    hd95_outlier: bool
    total_renal_components: int
    left_components: int
    right_components: int
    total_renal_volume_ml: float
    largest_component_volume_ml: float
    second_component_volume_ml: float
    third_plus_volume_ml: float
    third_plus_fraction: float
    smallest_component_volume_ml: float
    largest_to_second_ratio: float | None
    max_component_centroid_distance_mm: float


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
        raise RuntimeError(f"No labels found for side={side!r}.")
    return values


def _load_prediction(
    path: Path,
) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    nii = nib.load(str(path))
    if len(nii.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")

    data = np.asanyarray(nii.dataobj)
    if not np.allclose(data, np.rint(data), atol=1e-5, rtol=0.0):
        raise TypeError(f"Expected integer-valued labels: {path}")

    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return (
        np.rint(data).astype(np.int16, copy=False),
        spacing,
        np.asarray(nii.affine, dtype=float),
    )


def _component_features(
    mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> tuple[
    int,
    list[float],
    list[np.ndarray],
]:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)

    if count == 0:
        return 0, [], []

    voxel_volume_ml = float(np.prod(spacing_mm) / 1000.0)
    volumes = []
    centroids_world = []

    for component_id in range(1, count + 1):
        component = labels == component_id
        voxel_count = int(np.count_nonzero(component))
        volumes.append(voxel_count * voxel_volume_ml)

        centroid_voxel = np.asarray(
            ndimage.center_of_mass(component),
            dtype=float,
        )
        homogeneous = np.append(centroid_voxel, 1.0)
        world = affine @ homogeneous
        centroids_world.append(world[:3])

    order = np.argsort(volumes)[::-1]
    volumes = [float(volumes[i]) for i in order]
    centroids_world = [
        np.asarray(centroids_world[i], dtype=float)
        for i in order
    ]

    return int(count), volumes, centroids_world


def _max_centroid_distance(
    centroids: list[np.ndarray],
) -> float:
    if len(centroids) < 2:
        return 0.0

    maximum = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = float(
                np.linalg.norm(centroids[i] - centroids[j])
            )
            maximum = max(maximum, distance)

    return maximum


def extract_prediction_geometry(
    *,
    case_id: str,
    prediction_path: Path,
    prediction_to_gt_target: bool,
    hd95_outlier: bool,
) -> PredictionGeometry:
    prediction, spacing, affine = _load_prediction(
        prediction_path
    )

    renal_mask = np.isin(prediction, _renal_values())
    total_count, volumes, centroids = _component_features(
        renal_mask,
        spacing_mm=spacing,
        affine=affine,
    )

    left_count, _, _ = _component_features(
        np.isin(prediction, _side_values("left")),
        spacing_mm=spacing,
        affine=affine,
    )
    right_count, _, _ = _component_features(
        np.isin(prediction, _side_values("right")),
        spacing_mm=spacing,
        affine=affine,
    )

    total_volume = float(sum(volumes))
    largest = volumes[0] if volumes else 0.0
    second = volumes[1] if len(volumes) >= 2 else 0.0
    third_plus = float(sum(volumes[2:])) if len(volumes) >= 3 else 0.0
    smallest = volumes[-1] if volumes else 0.0

    return PredictionGeometry(
        case_id=case_id,
        prediction_to_gt_target=prediction_to_gt_target,
        hd95_outlier=hd95_outlier,
        total_renal_components=total_count,
        left_components=left_count,
        right_components=right_count,
        total_renal_volume_ml=total_volume,
        largest_component_volume_ml=largest,
        second_component_volume_ml=second,
        third_plus_volume_ml=third_plus,
        third_plus_fraction=(
            third_plus / total_volume
            if total_volume > 0.0
            else 0.0
        ),
        smallest_component_volume_ml=smallest,
        largest_to_second_ratio=(
            largest / second
            if second > 0.0
            else None
        ),
        max_component_centroid_distance_mm=(
            _max_centroid_distance(centroids)
        ),
    )


def read_labels(
    *,
    taxonomy_path: Path,
    geometry_path: Path,
) -> tuple[tuple[str, bool, bool], ...]:
    with taxonomy_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        taxonomy = {
            row["case_id"]: row
            for row in csv.DictReader(handle)
        }

    with geometry_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        geometry_rows = tuple(csv.DictReader(handle))

    target_ids = {
        row["case_id"]
        for row in geometry_rows
        if row["dominant_direction"] == "prediction_to_gt"
    }

    if len(target_ids) != 12:
        raise ValueError(
            f"Expected 12 prediction->GT targets, got {len(target_ids)}."
        )

    if len(taxonomy) != 100:
        raise ValueError(
            f"Expected 100 taxonomy rows, got {len(taxonomy)}."
        )

    output = []
    for case_id, row in taxonomy.items():
        outlier = row["hd95_outlier"].strip().lower()
        if outlier not in {"true", "false"}:
            raise ValueError(
                f"Invalid hd95_outlier value for {case_id}."
            )

        output.append(
            (
                case_id,
                case_id in target_ids,
                outlier == "true",
            )
        )

    return tuple(sorted(output))


def build_prediction_geometry_rows(
    *,
    taxonomy_path: Path,
    geometry_path: Path,
    prediction_root: Path,
) -> tuple[PredictionGeometry, ...]:
    labels = read_labels(
        taxonomy_path=taxonomy_path,
        geometry_path=geometry_path,
    )

    return tuple(
        extract_prediction_geometry(
            case_id=case_id,
            prediction_path=(
                prediction_root
                / case_id
                / "prediction.nii.gz"
            ),
            prediction_to_gt_target=is_target,
            hd95_outlier=is_outlier,
        )
        for case_id, is_target, is_outlier in labels
    )


def rank_auc(
    scores: Iterable[float],
    labels: Iterable[bool],
) -> float | None:
    scores = np.asarray(tuple(scores), dtype=float)
    labels = np.asarray(tuple(labels), dtype=bool)

    if scores.size != labels.size:
        raise ValueError("scores and labels must have equal length.")

    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)

    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks = np.empty_like(scores, dtype=float)

    start = 0
    while start < scores.size:
        end = start + 1
        while (
            end < scores.size
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels].sum())
    u = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    )

    return u / (positives * negatives)


def _feature_value(
    row: PredictionGeometry,
    feature: str,
) -> float:
    value = getattr(row, feature)
    if value is None:
        return 0.0
    return float(value)


def _feature_summary(
    rows: tuple[PredictionGeometry, ...],
    *,
    feature: str,
) -> dict[str, object]:
    labels = [row.prediction_to_gt_target for row in rows]
    values = [_feature_value(row, feature) for row in rows]

    target_values = np.asarray(
        [
            value
            for value, label in zip(values, labels)
            if label
        ],
        dtype=float,
    )
    non_target_values = np.asarray(
        [
            value
            for value, label in zip(values, labels)
            if not label
        ],
        dtype=float,
    )

    auc_high = rank_auc(values, labels)
    auc_low = rank_auc([-value for value in values], labels)

    direction = (
        "high"
        if (auc_high or 0.0) >= (auc_low or 0.0)
        else "low"
    )
    auc = max(auc_high or 0.0, auc_low or 0.0)

    sorted_rows = sorted(
        rows,
        key=lambda row: _feature_value(row, feature),
        reverse=(direction == "high"),
    )

    top12 = sorted_rows[:12]
    top20 = sorted_rows[:20]

    return {
        "feature": feature,
        "direction": direction,
        "auc": auc,
        "target_mean": float(target_values.mean()),
        "target_median": float(np.median(target_values)),
        "non_target_mean": float(non_target_values.mean()),
        "non_target_median": float(np.median(non_target_values)),
        "top12_target_count": sum(
            row.prediction_to_gt_target for row in top12
        ),
        "top12_precision": (
            sum(row.prediction_to_gt_target for row in top12)
            / 12.0
        ),
        "top20_target_count": sum(
            row.prediction_to_gt_target for row in top20
        ),
        "top20_recall": (
            sum(row.prediction_to_gt_target for row in top20)
            / 12.0
        ),
    }


def analyze_prediction_geometry_trigger(
    rows: Iterable[PredictionGeometry],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(rows)}."
        )

    if sum(row.prediction_to_gt_target for row in rows) != 12:
        raise ValueError(
            "Expected exactly 12 prediction->GT target cases."
        )

    features = (
        "total_renal_components",
        "left_components",
        "right_components",
        "total_renal_volume_ml",
        "largest_component_volume_ml",
        "second_component_volume_ml",
        "third_plus_volume_ml",
        "third_plus_fraction",
        "smallest_component_volume_ml",
        "largest_to_second_ratio",
        "max_component_centroid_distance_mm",
    )

    summaries = tuple(
        _feature_summary(
            rows,
            feature=feature,
        )
        for feature in features
    )

    ranked = sorted(
        summaries,
        key=lambda item: (
            -float(item["auc"]),
            str(item["feature"]),
        ),
    )

    best = ranked[0]

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_prediction_geometry_trigger_audit"
        ),
        "case_count": len(rows),
        "target_count": 12,
        "feature_rankings": list(ranked),
        "best_feature": best,
        "case_rows": [asdict(row) for row in rows],
        "interpretation_boundary": (
            "Target labels come from the retrospective Block 35 "
            "directional audit. Prediction-geometry features use prediction "
            "data only. This block evaluates separability and does not "
            "register a trigger or post-processing rule."
        ),
    }
