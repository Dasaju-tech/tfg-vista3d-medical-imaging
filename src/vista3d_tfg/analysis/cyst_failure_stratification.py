from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


CYST_GT_LABEL = 3
AUTO_CYST_LABELS = (116, 117)


@dataclass(frozen=True, slots=True)
class CystCaseFailure:
    case_id: str
    reference_volume_ml: float
    prediction_volume_ml: float
    intersection_volume_ml: float
    dice: float
    precision: float
    recall: float
    false_negative_volume_ml: float
    false_positive_volume_ml: float
    reference_component_count: int
    prediction_component_count: int
    largest_reference_component_ml: float
    largest_prediction_component_ml: float
    dominant_error: str


def _load_integer_label(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(path))
    raw = np.asanyarray(nii.dataobj)

    if raw.ndim != 3:
        raise ValueError(
            f"Expected 3D label image, got {raw.shape}: {path}"
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"Non-finite label values: {path}")
    if not np.allclose(raw, np.rint(raw), rtol=0.0, atol=1e-5):
        raise TypeError(f"Label image is not integer-valued: {path}")

    array = np.rint(raw).astype(np.int16, copy=False)
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return array, spacing


def _largest_component_volume_ml(
    mask: np.ndarray,
    *,
    voxel_volume_ml: float,
) -> tuple[int, float]:
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, count = ndimage.label(mask, structure=structure)

    if count == 0:
        return 0, 0.0

    sizes = np.bincount(labeled.ravel())[1:]
    largest = int(sizes.max())
    return int(count), float(largest * voxel_volume_ml)


def analyze_cyst_case(
    *,
    case_id: str,
    ground_truth_path: Path,
    automatic_prediction_path: Path,
) -> CystCaseFailure:
    gt, spacing = _load_integer_label(ground_truth_path)
    pred, pred_spacing = _load_integer_label(automatic_prediction_path)

    if gt.shape != pred.shape:
        raise ValueError(
            f"{case_id}: GT/prediction shape mismatch {gt.shape} vs {pred.shape}."
        )
    if not np.allclose(spacing, pred_spacing, rtol=0.0, atol=1e-5):
        raise ValueError(
            f"{case_id}: GT/prediction spacing mismatch."
        )

    gt_mask = gt == CYST_GT_LABEL
    pred_mask = np.isin(pred, AUTO_CYST_LABELS)

    if not np.any(gt_mask):
        raise ValueError(
            f"{case_id}: case is not cyst-positive."
        )

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)

    tp = int(np.count_nonzero(gt_mask & pred_mask))
    gt_count = int(np.count_nonzero(gt_mask))
    pred_count = int(np.count_nonzero(pred_mask))
    fn = gt_count - tp
    fp = pred_count - tp

    denominator = gt_count + pred_count
    dice = (
        1.0
        if denominator == 0
        else 2.0 * tp / denominator
    )
    precision = (
        0.0
        if pred_count == 0
        else tp / pred_count
    )
    recall = tp / gt_count

    ref_components, largest_ref = _largest_component_volume_ml(
        gt_mask,
        voxel_volume_ml=voxel_volume_ml,
    )
    pred_components, largest_pred = _largest_component_volume_ml(
        pred_mask,
        voxel_volume_ml=voxel_volume_ml,
    )

    fn_ml = float(fn * voxel_volume_ml)
    fp_ml = float(fp * voxel_volume_ml)

    tolerance = max(1e-9, 0.05 * max(fn_ml, fp_ml, 1e-9))
    if abs(fn_ml - fp_ml) <= tolerance:
        dominant = "balanced"
    elif fn_ml > fp_ml:
        dominant = "undersegmentation_fn"
    else:
        dominant = "oversegmentation_fp"

    return CystCaseFailure(
        case_id=case_id,
        reference_volume_ml=float(gt_count * voxel_volume_ml),
        prediction_volume_ml=float(pred_count * voxel_volume_ml),
        intersection_volume_ml=float(tp * voxel_volume_ml),
        dice=float(dice),
        precision=float(precision),
        recall=float(recall),
        false_negative_volume_ml=fn_ml,
        false_positive_volume_ml=fp_ml,
        reference_component_count=ref_components,
        prediction_component_count=pred_components,
        largest_reference_component_ml=largest_ref,
        largest_prediction_component_ml=largest_pred,
        dominant_error=dominant,
    )


def _summary(values: Iterable[float]) -> dict[str, float | int]:
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


def _quartile_groups(
    rows: tuple[CystCaseFailure, ...],
) -> dict[str, tuple[CystCaseFailure, ...]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.reference_volume_ml,
            row.case_id,
        ),
    )
    indices = np.array_split(
        np.arange(len(ordered)),
        4,
    )
    names = (
        "smallest",
        "lower_middle",
        "upper_middle",
        "largest",
    )
    return {
        name: tuple(
            ordered[int(index)]
            for index in group
        )
        for name, group in zip(names, indices)
    }


def _group_payload(
    rows: tuple[CystCaseFailure, ...],
) -> dict[str, object]:
    return {
        "count": len(rows),
        "reference_volume_ml": _summary(
            row.reference_volume_ml
            for row in rows
        ),
        "dice": _summary(
            row.dice
            for row in rows
        ),
        "precision": _summary(
            row.precision
            for row in rows
        ),
        "recall": _summary(
            row.recall
            for row in rows
        ),
        "false_negative_volume_ml": _summary(
            row.false_negative_volume_ml
            for row in rows
        ),
        "false_positive_volume_ml": _summary(
            row.false_positive_volume_ml
            for row in rows
        ),
        "zero_dice_count": sum(
            row.dice == 0.0
            for row in rows
        ),
        "dice_ge_050_count": sum(
            row.dice >= 0.50
            for row in rows
        ),
        "dominant_error_counts": {
            label: sum(
                row.dominant_error == label
                for row in rows
            )
            for label in (
                "undersegmentation_fn",
                "oversegmentation_fp",
                "balanced",
            )
        },
    }


def summarize_cyst_failures(
    rows: Iterable[CystCaseFailure],
) -> dict[str, object]:
    rows = tuple(rows)
    if not rows:
        raise ValueError("No cyst-positive cases supplied.")

    volumes = np.asarray(
        [
            row.reference_volume_ml
            for row in rows
        ],
        dtype=float,
    )
    dice = np.asarray(
        [
            row.dice
            for row in rows
        ],
        dtype=float,
    )
    recall = np.asarray(
        [
            row.recall
            for row in rows
        ],
        dtype=float,
    )

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) == 0.0 or np.std(b) == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    quartiles = _quartile_groups(rows)

    return {
        "schema_version": "1.0",
        "analysis": "expanded100_cyst_failure_stratification",
        "case_count": len(rows),
        "overall": _group_payload(rows),
        "by_reference_volume_quartile": {
            name: _group_payload(group)
            for name, group in quartiles.items()
        },
        "associations": {
            "reference_volume_vs_dice_pearson": corr(
                volumes,
                dice,
            ),
            "reference_volume_vs_recall_pearson": corr(
                volumes,
                recall,
            ),
        },
        "rows": [
            asdict(row)
            for row in rows
        ],
        "interpretation_boundary": (
            "This is a retrospective error analysis of the frozen expanded100 "
            "automatic predictions. It does not alter predictions and does "
            "not constitute a new model evaluation cohort."
        ),
    }


def write_cyst_failure_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = payload["rows"]
    assert isinstance(rows, list)
    if not rows:
        raise ValueError("No cyst rows.")

    with (output_root / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_cyst_failure_report(
    *,
    payload: dict[str, object],
    path: Path,
) -> None:
    overall = payload["overall"]
    quartiles = payload["by_reference_volume_quartile"]
    associations = payload["associations"]

    assert isinstance(overall, dict)
    assert isinstance(quartiles, dict)
    assert isinstance(associations, dict)

    lines = [
        "# Expanded100 cyst failure stratification",
        "",
        f"Cyst-positive cases: {payload['case_count']}",
        "",
        "## Overall",
        "",
        (
            f"- Dice mean/median: "
            f"{overall['dice']['mean']:.4f} / "
            f"{overall['dice']['median']:.4f}."
        ),
        (
            f"- Precision mean: "
            f"{overall['precision']['mean']:.4f}."
        ),
        (
            f"- Recall mean: "
            f"{overall['recall']['mean']:.4f}."
        ),
        (
            f"- Zero-Dice cases: "
            f"{overall['zero_dice_count']}/{payload['case_count']}."
        ),
        "",
        "## By reference cyst-volume quartile",
        "",
        "| Quartile | N | Ref volume median mL | Dice mean | Recall mean | Zero Dice |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in (
        "smallest",
        "lower_middle",
        "upper_middle",
        "largest",
    ):
        group = quartiles[name]
        lines.append(
            f"| {name} "
            f"| {group['count']} "
            f"| {group['reference_volume_ml']['median']:.2f} "
            f"| {group['dice']['mean']:.4f} "
            f"| {group['recall']['mean']:.4f} "
            f"| {group['zero_dice_count']} |"
        )

    lines.extend(
        [
            "",
            "## Associations",
            "",
            (
                f"- Reference cyst volume vs Dice Pearson: "
                f"{associations['reference_volume_vs_dice_pearson']:+.3f}."
            ),
            (
                f"- Reference cyst volume vs recall Pearson: "
                f"{associations['reference_volume_vs_recall_pearson']:+.3f}."
            ),
            "",
            "## Interpretation boundary",
            "",
            str(payload["interpretation_boundary"]),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
