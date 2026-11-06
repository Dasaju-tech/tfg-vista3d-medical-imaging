from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.experiments.cyst_oracle_prompt import OracleCystPrompt
from vista3d_tfg.experiments.cyst_semantic_point import (
    LEFT_CYST_LABEL,
    RIGHT_CYST_LABEL,
)


LEFT_KIDNEY_LABEL = 14
RIGHT_KIDNEY_LABEL = 5


@dataclass(frozen=True, slots=True)
class AutomaticLateralityDecision:
    case_id: str
    point_xyz: tuple[int, int, int]
    inferred_laterality: str
    inferred_cyst_class: int
    oracle_cyst_class: int
    agrees_with_oracle: bool
    distance_to_left_kidney_mm: float
    distance_to_right_kidney_mm: float
    distance_margin_mm: float
    selected_distance_mm: float
    left_kidney_present: bool
    right_kidney_present: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AutomaticLateralitySummary:
    case_count: int
    agreement_count: int
    disagreement_count: int
    agreement_fraction: float
    left_inferred_count: int
    right_inferred_count: int
    both_kidneys_present_count: int
    selected_distance_mean_mm: float
    selected_distance_median_mm: float
    selected_distance_max_mm: float
    margin_mean_mm: float
    margin_median_mm: float
    margin_minimum_mm: float
    disagreement_case_ids: tuple[str, ...]
    decisions: tuple[AutomaticLateralityDecision, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _distance_from_point_to_mask(
    mask: np.ndarray,
    *,
    point_xyz: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> float:
    if not np.any(mask):
        return float("inf")

    distance = ndimage.distance_transform_edt(
        ~mask.astype(bool, copy=False),
        sampling=spacing_mm,
    )

    return float(distance[point_xyz])


def infer_cyst_class_from_automatic_kidneys(
    *,
    case_id: str,
    automatic_prediction: np.ndarray,
    spacing_mm: tuple[float, float, float],
    point_xyz: tuple[int, int, int],
    oracle_cyst_class: int,
) -> AutomaticLateralityDecision:
    if automatic_prediction.ndim != 3:
        raise ValueError("Automatic prediction must be 3D.")

    if any(
        coordinate < 0 or coordinate >= automatic_prediction.shape[axis]
        for axis, coordinate in enumerate(point_xyz)
    ):
        raise ValueError("Prompt point lies outside prediction volume.")

    rounded = np.rint(automatic_prediction).astype(
        np.int16,
        copy=False,
    )

    left_kidney = rounded == LEFT_KIDNEY_LABEL
    right_kidney = rounded == RIGHT_KIDNEY_LABEL

    left_present = bool(np.any(left_kidney))
    right_present = bool(np.any(right_kidney))

    if not left_present and not right_present:
        raise RuntimeError(
            f"{case_id}: automatic prediction contains neither kidney "
            f"label {LEFT_KIDNEY_LABEL} nor {RIGHT_KIDNEY_LABEL}."
        )

    left_distance = _distance_from_point_to_mask(
        left_kidney,
        point_xyz=point_xyz,
        spacing_mm=spacing_mm,
    )
    right_distance = _distance_from_point_to_mask(
        right_kidney,
        point_xyz=point_xyz,
        spacing_mm=spacing_mm,
    )

    if np.isclose(left_distance, right_distance, atol=1e-8, rtol=0.0):
        raise RuntimeError(
            f"{case_id}: automatic kidney laterality is ambiguous at "
            f"point {point_xyz}; equal physical distances."
        )

    if left_distance < right_distance:
        inferred_laterality = "left"
        inferred_class = LEFT_CYST_LABEL
        selected_distance = left_distance
    else:
        inferred_laterality = "right"
        inferred_class = RIGHT_CYST_LABEL
        selected_distance = right_distance

    finite_distances = [
        value
        for value in (left_distance, right_distance)
        if np.isfinite(value)
    ]

    if len(finite_distances) == 2:
        margin = abs(left_distance - right_distance)
    else:
        margin = float("inf")

    return AutomaticLateralityDecision(
        case_id=case_id,
        point_xyz=point_xyz,
        inferred_laterality=inferred_laterality,
        inferred_cyst_class=inferred_class,
        oracle_cyst_class=int(oracle_cyst_class),
        agrees_with_oracle=inferred_class == int(oracle_cyst_class),
        distance_to_left_kidney_mm=left_distance,
        distance_to_right_kidney_mm=right_distance,
        distance_margin_mm=float(margin),
        selected_distance_mm=float(selected_distance),
        left_kidney_present=left_present,
        right_kidney_present=right_present,
    )


def audit_case_laterality(
    *,
    prompt: OracleCystPrompt,
    automatic_prediction_path: Path,
    oracle_semantic_result_path: Path,
) -> AutomaticLateralityDecision:
    prediction_nii = nib.load(str(automatic_prediction_path))
    prediction = np.asanyarray(prediction_nii.dataobj)

    spacing = tuple(
        float(value)
        for value in prediction_nii.header.get_zooms()[:3]
    )

    payload = json.loads(
        oracle_semantic_result_path.read_text(encoding="utf-8")
    )
    semantic_target = payload.get("semantic_target")

    if not isinstance(semantic_target, dict):
        raise ValueError(
            f"Semantic result lacks semantic_target: "
            f"{oracle_semantic_result_path}"
        )

    oracle_class = int(semantic_target["class_label"])

    return infer_cyst_class_from_automatic_kidneys(
        case_id=prompt.case_id,
        automatic_prediction=prediction,
        spacing_mm=spacing,
        point_xyz=prompt.point_xyz,
        oracle_cyst_class=oracle_class,
    )


def summarize_automatic_laterality(
    decisions: Iterable[AutomaticLateralityDecision],
) -> AutomaticLateralitySummary:
    decisions = tuple(decisions)

    if not decisions:
        raise ValueError("At least one laterality decision is required.")

    selected_distances = [
        decision.selected_distance_mm
        for decision in decisions
        if np.isfinite(decision.selected_distance_mm)
    ]
    margins = [
        decision.distance_margin_mm
        for decision in decisions
        if np.isfinite(decision.distance_margin_mm)
    ]

    if not selected_distances:
        raise RuntimeError("No finite selected kidney distances were found.")
    if not margins:
        raise RuntimeError("No finite bilateral distance margins were found.")

    agreement_count = sum(
        decision.agrees_with_oracle
        for decision in decisions
    )

    disagreement_case_ids = tuple(
        decision.case_id
        for decision in decisions
        if not decision.agrees_with_oracle
    )

    return AutomaticLateralitySummary(
        case_count=len(decisions),
        agreement_count=agreement_count,
        disagreement_count=len(decisions) - agreement_count,
        agreement_fraction=float(
            agreement_count / len(decisions)
        ),
        left_inferred_count=sum(
            decision.inferred_laterality == "left"
            for decision in decisions
        ),
        right_inferred_count=sum(
            decision.inferred_laterality == "right"
            for decision in decisions
        ),
        both_kidneys_present_count=sum(
            decision.left_kidney_present
            and decision.right_kidney_present
            for decision in decisions
        ),
        selected_distance_mean_mm=float(mean(selected_distances)),
        selected_distance_median_mm=float(median(selected_distances)),
        selected_distance_max_mm=float(max(selected_distances)),
        margin_mean_mm=float(mean(margins)),
        margin_median_mm=float(median(margins)),
        margin_minimum_mm=float(min(margins)),
        disagreement_case_ids=disagreement_case_ids,
        decisions=decisions,
    )


def write_laterality_outputs(
    summary: AutomaticLateralitySummary,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "analysis": "automatic_cyst_semantic_class_from_renal_masks",
                "deployment_valid_semantic_class": True,
                "click_source_in_current_experiment": "GT-derived oracle click",
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fields = list(asdict(summary.decisions[0]).keys())

    with (output_root / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for decision in summary.decisions:
            writer.writerow(asdict(decision))
