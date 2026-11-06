from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS


GT_LABELS = {
    "kidney": 1,
    "tumor": 2,
    "cyst": 3,
}


@dataclass(frozen=True)
class UndercoverageCase:
    case_id: str
    original_hd95_mm: float
    gt_to_prediction_surface_p95_mm: float
    gt_volume_ml: float
    fn_volume_ml: float
    fn_fraction: float
    gt_kidney_volume_ml: float
    gt_tumor_volume_ml: float
    gt_cyst_volume_ml: float
    fn_kidney_volume_ml: float
    fn_tumor_volume_ml: float
    fn_cyst_volume_ml: float
    kidney_miss_fraction: float | None
    tumor_miss_fraction: float | None
    cyst_miss_fraction: float | None
    fn_components: int
    largest_fn_component_ml: float
    dominant_missing_label: str
    dominant_missing_label_share: float


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


def _renal_prediction_values() -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in EXPECTED_RENAL_LABELS.values()
    )


def _component_summary(
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


def _fraction_or_none(
    numerator: float,
    denominator: float,
) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def evaluate_undercoverage_case(
    *,
    case_id: str,
    original_hd95_mm: float,
    gt_to_prediction_surface_p95_mm: float,
    ground_truth_path: Path,
    prediction_path: Path,
) -> UndercoverageCase:
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

    ground_truth, spacing = _load_labelmap(ground_truth_path)
    prediction, pred_spacing = _load_labelmap(prediction_path)

    if not np.allclose(
        spacing,
        pred_spacing,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{case_id}: spacing mismatch.")

    reference = np.isin(ground_truth, tuple(GT_LABELS.values()))
    predicted = np.isin(
        prediction,
        _renal_prediction_values(),
    )
    missing = reference & ~predicted

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)

    gt_volumes: dict[str, float] = {}
    fn_volumes: dict[str, float] = {}

    for name, label in GT_LABELS.items():
        gt_label = ground_truth == label
        fn_label = missing & gt_label

        gt_volumes[name] = (
            float(np.count_nonzero(gt_label))
            * voxel_volume_ml
        )
        fn_volumes[name] = (
            float(np.count_nonzero(fn_label))
            * voxel_volume_ml
        )

    gt_volume_ml = float(np.count_nonzero(reference) * voxel_volume_ml)
    fn_volume_ml = float(np.count_nonzero(missing) * voxel_volume_ml)

    fn_components, largest_fn_component_ml = _component_summary(
        missing,
        voxel_volume_ml=voxel_volume_ml,
    )

    dominant_missing_label = max(
        GT_LABELS,
        key=lambda name: (
            fn_volumes[name],
            name,
        ),
    )
    dominant_missing_volume = fn_volumes[dominant_missing_label]

    return UndercoverageCase(
        case_id=case_id,
        original_hd95_mm=original_hd95_mm,
        gt_to_prediction_surface_p95_mm=(
            gt_to_prediction_surface_p95_mm
        ),
        gt_volume_ml=gt_volume_ml,
        fn_volume_ml=fn_volume_ml,
        fn_fraction=_fraction_or_none(
            fn_volume_ml,
            gt_volume_ml,
        ) or 0.0,
        gt_kidney_volume_ml=gt_volumes["kidney"],
        gt_tumor_volume_ml=gt_volumes["tumor"],
        gt_cyst_volume_ml=gt_volumes["cyst"],
        fn_kidney_volume_ml=fn_volumes["kidney"],
        fn_tumor_volume_ml=fn_volumes["tumor"],
        fn_cyst_volume_ml=fn_volumes["cyst"],
        kidney_miss_fraction=_fraction_or_none(
            fn_volumes["kidney"],
            gt_volumes["kidney"],
        ),
        tumor_miss_fraction=_fraction_or_none(
            fn_volumes["tumor"],
            gt_volumes["tumor"],
        ),
        cyst_miss_fraction=_fraction_or_none(
            fn_volumes["cyst"],
            gt_volumes["cyst"],
        ),
        fn_components=fn_components,
        largest_fn_component_ml=largest_fn_component_ml,
        dominant_missing_label=dominant_missing_label,
        dominant_missing_label_share=(
            dominant_missing_volume / fn_volume_ml
            if fn_volume_ml > 0.0
            else 0.0
        ),
    )


def read_gt_to_prediction_cases(
    path: Path,
) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    selected = tuple(
        row
        for row in rows
        if row["dominant_direction"] == "gt_to_prediction"
    )

    if len(selected) != 9:
        raise ValueError(
            f"Expected 9 GT->prediction outliers, got {len(selected)}."
        )

    return selected


def build_undercoverage_cases(
    *,
    geometry_path: Path,
    dataset_root: Path,
    prediction_root: Path,
) -> tuple[UndercoverageCase, ...]:
    rows = read_gt_to_prediction_cases(geometry_path)

    output = []

    for row in rows:
        case_id = row["case_id"]
        output.append(
            evaluate_undercoverage_case(
                case_id=case_id,
                original_hd95_mm=float(
                    row["original_hd95_mm"]
                ),
                gt_to_prediction_surface_p95_mm=float(
                    row["gt_to_prediction_surface_p95_mm"]
                ),
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


def _summary(
    values: Iterable[float],
) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)

    if array.size == 0:
        raise ValueError("Cannot summarize empty values.")

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _positive_fraction_summary(
    rows: tuple[UndercoverageCase, ...],
    *,
    attribute: str,
) -> dict[str, float | int] | None:
    values = [
        getattr(row, attribute)
        for row in rows
        if getattr(row, attribute) is not None
    ]
    if not values:
        return None
    return _summary(float(value) for value in values)


def _weighted_miss_fraction(
    rows: tuple[UndercoverageCase, ...],
    *,
    gt_attribute: str,
    fn_attribute: str,
) -> float | None:
    gt_total = sum(
        float(getattr(row, gt_attribute))
        for row in rows
    )
    fn_total = sum(
        float(getattr(row, fn_attribute))
        for row in rows
    )

    if gt_total <= 0.0:
        return None
    return fn_total / gt_total


def analyze_undercoverage(
    rows: Iterable[UndercoverageCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 9:
        raise ValueError(
            f"Expected 9 GT->prediction cases, got {len(rows)}."
        )

    dominant_counts = {
        label: sum(
            row.dominant_missing_label == label
            for row in rows
        )
        for label in GT_LABELS
    }

    ranked_fn = sorted(
        rows,
        key=lambda row: (
            -row.fn_volume_ml,
            row.case_id,
        ),
    )
    ranked_distance = sorted(
        rows,
        key=lambda row: (
            -row.gt_to_prediction_surface_p95_mm,
            row.case_id,
        ),
    )

    total_gt_by_label = {
        "kidney": sum(row.gt_kidney_volume_ml for row in rows),
        "tumor": sum(row.gt_tumor_volume_ml for row in rows),
        "cyst": sum(row.gt_cyst_volume_ml for row in rows),
    }
    total_fn_by_label = {
        "kidney": sum(row.fn_kidney_volume_ml for row in rows),
        "tumor": sum(row.fn_tumor_volume_ml for row in rows),
        "cyst": sum(row.fn_cyst_volume_ml for row in rows),
    }

    total_fn = sum(total_fn_by_label.values())

    label_missing_share = {
        label: (
            total_fn_by_label[label] / total_fn
            if total_fn > 0.0
            else 0.0
        )
        for label in GT_LABELS
    }

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_gt_to_prediction_undercoverage_audit"
        ),
        "case_count": len(rows),
        "fn_volume_ml": _summary(
            row.fn_volume_ml for row in rows
        ),
        "fn_fraction": _summary(
            row.fn_fraction for row in rows
        ),
        "fn_components": _summary(
            float(row.fn_components) for row in rows
        ),
        "largest_fn_component_ml": _summary(
            row.largest_fn_component_ml for row in rows
        ),
        "gt_to_prediction_surface_p95_mm": _summary(
            row.gt_to_prediction_surface_p95_mm
            for row in rows
        ),
        "dominant_missing_label_counts": dominant_counts,
        "aggregate_label_volumes_ml": {
            "ground_truth": total_gt_by_label,
            "false_negative": total_fn_by_label,
        },
        "aggregate_missing_volume_share": label_missing_share,
        "weighted_miss_fraction_by_label": {
            "kidney": _weighted_miss_fraction(
                rows,
                gt_attribute="gt_kidney_volume_ml",
                fn_attribute="fn_kidney_volume_ml",
            ),
            "tumor": _weighted_miss_fraction(
                rows,
                gt_attribute="gt_tumor_volume_ml",
                fn_attribute="fn_tumor_volume_ml",
            ),
            "cyst": _weighted_miss_fraction(
                rows,
                gt_attribute="gt_cyst_volume_ml",
                fn_attribute="fn_cyst_volume_ml",
            ),
        },
        "case_positive_miss_fraction": {
            "kidney": _positive_fraction_summary(
                rows,
                attribute="kidney_miss_fraction",
            ),
            "tumor": _positive_fraction_summary(
                rows,
                attribute="tumor_miss_fraction",
            ),
            "cyst": _positive_fraction_summary(
                rows,
                attribute="cyst_miss_fraction",
            ),
        },
        "rankings": {
            "largest_false_negative_volume": [
                asdict(row)
                for row in ranked_fn
            ],
            "largest_gt_to_prediction_p95": [
                asdict(row)
                for row in ranked_distance
            ],
        },
        "case_rows": [asdict(row) for row in rows],
        "interpretation_boundary": (
            "This is a retrospective audit of the 9 frozen "
            "GT-to-prediction-dominated HD95 outliers. Missing-volume "
            "decomposition uses KiTS23 reference labels and must not be "
            "interpreted as a deployment-time trigger. Its purpose is to "
            "identify which structures drive undercoverage before any "
            "prompting or inference change."
        ),
    }
