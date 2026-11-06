from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable


@dataclass(frozen=True, slots=True)
class MetricSummary:
    count: int
    mean: float
    median: float
    std: float | None
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class CaseCohortRow:
    case_id: str
    kidney_and_mass_dice: float
    kidney_and_mass_surface_dice: float
    kidney_and_mass_hd95_mm: float
    kidney_core_dice: float
    cyst_dice: float
    cyst_reference_volume_ml: float
    cyst_prediction_volume_ml: float
    left_renal_dice: float | None
    right_renal_dice: float | None
    left_renal_volume_error_ml: float | None
    right_renal_volume_error_ml: float | None
    gt_tumor_volume_ml: float
    inference_wall_time_seconds: float | None
    qc_false_positive_voxels: int | None
    qc_false_negative_voxels: int | None
    qc_total_disagreement_voxels: int | None
    ground_truth_laterality_resolved: bool = True


@dataclass(frozen=True, slots=True)
class CohortSummary:
    schema_version: str
    dataset: str
    model: str
    protocol: str
    case_count: int
    case_ids: tuple[str, ...]
    full_kits23_score_available: bool
    full_kits23_score_reason: str
    kidney_and_mass_dice: MetricSummary
    kidney_and_mass_surface_dice: MetricSummary
    kidney_and_mass_hd95_mm: MetricSummary
    kidney_core_dice: MetricSummary
    laterality_resolved_case_count: int
    laterality_unresolved_case_count: int
    left_renal_dice: MetricSummary | None
    right_renal_dice: MetricSummary | None
    cyst_dice_all_cases: MetricSummary
    cyst_positive_case_count: int
    cyst_dice_cyst_positive_cases: MetricSummary | None
    cyst_reference_volume_ml_positive_cases: MetricSummary | None
    cyst_prediction_volume_ml_positive_cases: MetricSummary | None
    gt_tumor_volume_ml: MetricSummary
    inference_wall_time_seconds: MetricSummary | None
    rows: tuple[CaseCohortRow, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def summarize_values(values: Iterable[float]) -> MetricSummary:
    values = tuple(float(value) for value in values)

    if not values:
        raise ValueError("Cannot summarize an empty value sequence.")

    return MetricSummary(
        count=len(values),
        mean=float(mean(values)),
        median=float(median(values)),
        std=float(stdev(values)) if len(values) > 1 else None,
        minimum=float(min(values)),
        maximum=float(max(values)),
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")

    return payload


def load_case_cohort_row(
    *,
    case_id: str,
    evaluation_path: Path,
    inference_path: Path | None = None,
    qc_path: Path | None = None,
) -> CaseCohortRow:
    evaluation = _read_json(evaluation_path)

    result = evaluation.get("result")
    if not isinstance(result, dict):
        raise ValueError(
            f"Evaluation JSON does not contain result object: "
            f"{evaluation_path}"
        )

    kidney_hec = result["kidney_and_mass_hec"]
    kidney_core = result["kidney_core_diagnostic"]
    cyst = result["cyst"]
    left = result.get("left_renal_envelope")
    right = result.get("right_renal_envelope")

    if left is not None and not isinstance(left, dict):
        raise ValueError("left_renal_envelope must be an object or null.")
    if right is not None and not isinstance(right, dict):
        raise ValueError("right_renal_envelope must be an object or null.")

    laterality_value = result.get("ground_truth_laterality_resolved")
    if laterality_value is None:
        laterality_resolved = left is not None and right is not None
    elif isinstance(laterality_value, bool):
        laterality_resolved = laterality_value
    else:
        raise ValueError(
            "ground_truth_laterality_resolved must be boolean when present."
        )

    inference_wall_time: float | None = None

    if inference_path is not None and inference_path.is_file():
        inference = _read_json(inference_path)
        inference_result = inference.get("result")

        if not isinstance(inference_result, dict):
            raise ValueError(
                f"Inference JSON does not contain result object: "
                f"{inference_path}"
            )

        inference_wall_time = float(
            inference_result["wall_time_seconds"]
        )

    qc_fp: int | None = None
    qc_fn: int | None = None
    qc_total: int | None = None

    if qc_path is not None and qc_path.is_file():
        qc = _read_json(qc_path)
        qc_result = qc.get("result")

        if not isinstance(qc_result, dict):
            raise ValueError(
                f"QC JSON does not contain result object: {qc_path}"
            )

        qc_fp = int(qc_result["total_false_positive_voxels"])
        qc_fn = int(qc_result["total_false_negative_voxels"])
        qc_total = int(qc_result["total_disagreement_voxels"])

    return CaseCohortRow(
        case_id=case_id,
        kidney_and_mass_dice=float(kidney_hec["dice"]),
        kidney_and_mass_surface_dice=float(
            kidney_hec["surface_dice"]
        ),
        kidney_and_mass_hd95_mm=float(
            kidney_hec["diagnostic_hd95_mm"]
        ),
        kidney_core_dice=float(kidney_core["dice"]),
        cyst_dice=float(cyst["dice"]),
        cyst_reference_volume_ml=float(
            cyst["reference_volume_ml"]
        ),
        cyst_prediction_volume_ml=float(
            cyst["prediction_volume_ml"]
        ),
        left_renal_dice=(
            float(left["dice"])
            if left is not None
            else None
        ),
        right_renal_dice=(
            float(right["dice"])
            if right is not None
            else None
        ),
        left_renal_volume_error_ml=(
            float(left["signed_volume_error_ml"])
            if left is not None
            else None
        ),
        right_renal_volume_error_ml=(
            float(right["signed_volume_error_ml"])
            if right is not None
            else None
        ),
        gt_tumor_volume_ml=float(result["gt_tumor_volume_ml"]),
        inference_wall_time_seconds=inference_wall_time,
        qc_false_positive_voxels=qc_fp,
        qc_false_negative_voxels=qc_fn,
        qc_total_disagreement_voxels=qc_total,
        ground_truth_laterality_resolved=laterality_resolved,
    )


def summarize_cohort(
    rows: Iterable[CaseCohortRow],
) -> CohortSummary:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one cohort row is required.")

    case_ids = tuple(row.case_id for row in rows)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case IDs are not allowed.")

    cyst_positive = tuple(
        row
        for row in rows
        if row.cyst_reference_volume_ml > 0.0
    )

    runtime_values = tuple(
        row.inference_wall_time_seconds
        for row in rows
        if row.inference_wall_time_seconds is not None
    )

    laterality_resolved_count = sum(
        1
        for row in rows
        if row.ground_truth_laterality_resolved
    )
    left_dice_values = tuple(
        row.left_renal_dice
        for row in rows
        if row.left_renal_dice is not None
    )
    right_dice_values = tuple(
        row.right_renal_dice
        for row in rows
        if row.right_renal_dice is not None
    )

    return CohortSummary(
        schema_version="1.1",
        dataset="KiTS23",
        model="NV-Segment-CT",
        protocol="supported_renal_structures_smoke_cohort",
        case_count=len(rows),
        case_ids=case_ids,
        full_kits23_score_available=False,
        full_kits23_score_reason=(
            "Automatic NV-Segment-CT inference does not provide a "
            "validated kidney-tumor class for the KiTS23 mass and tumor HECs."
        ),
        kidney_and_mass_dice=summarize_values(
            row.kidney_and_mass_dice for row in rows
        ),
        kidney_and_mass_surface_dice=summarize_values(
            row.kidney_and_mass_surface_dice for row in rows
        ),
        kidney_and_mass_hd95_mm=summarize_values(
            row.kidney_and_mass_hd95_mm for row in rows
        ),
        kidney_core_dice=summarize_values(
            row.kidney_core_dice for row in rows
        ),
        laterality_resolved_case_count=laterality_resolved_count,
        laterality_unresolved_case_count=(
            len(rows) - laterality_resolved_count
        ),
        left_renal_dice=(
            summarize_values(left_dice_values)
            if left_dice_values
            else None
        ),
        right_renal_dice=(
            summarize_values(right_dice_values)
            if right_dice_values
            else None
        ),
        cyst_dice_all_cases=summarize_values(
            row.cyst_dice for row in rows
        ),
        cyst_positive_case_count=len(cyst_positive),
        cyst_dice_cyst_positive_cases=(
            summarize_values(row.cyst_dice for row in cyst_positive)
            if cyst_positive
            else None
        ),
        cyst_reference_volume_ml_positive_cases=(
            summarize_values(
                row.cyst_reference_volume_ml
                for row in cyst_positive
            )
            if cyst_positive
            else None
        ),
        cyst_prediction_volume_ml_positive_cases=(
            summarize_values(
                row.cyst_prediction_volume_ml
                for row in cyst_positive
            )
            if cyst_positive
            else None
        ),
        gt_tumor_volume_ml=summarize_values(
            row.gt_tumor_volume_ml for row in rows
        ),
        inference_wall_time_seconds=(
            summarize_values(runtime_values)
            if runtime_values
            else None
        ),
        rows=rows,
    )


def write_cohort_json(
    summary: CohortSummary,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            summary.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_cohort_csv(
    summary: CohortSummary,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(summary.rows[0]).keys())

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary.rows:
            writer.writerow(asdict(row))
