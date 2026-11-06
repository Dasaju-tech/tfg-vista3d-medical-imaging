from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class ReportingCohortRow:
    case_id: str
    point_xyz: tuple[int, int, int]
    laterality: str
    class_label: int

    workflow_volume_ml: float
    selected_kidney_distance_mm: float
    laterality_margin_mm: float

    llm_parsed: bool
    llm_accepted: bool
    llm_issue_count: int
    llm_wall_time_seconds: float
    llm_response_sha256: str | None
    llm_candidate_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReportingCohortSummary:
    case_count: int

    left_case_count: int
    right_case_count: int

    workflow_volume_mean_ml: float
    workflow_volume_median_ml: float

    selected_kidney_distance_mean_mm: float
    selected_kidney_distance_median_mm: float
    selected_kidney_distance_max_mm: float

    laterality_margin_mean_mm: float
    laterality_margin_median_mm: float
    laterality_margin_min_mm: float

    llm_parse_success_count: int
    llm_parse_success_fraction: float

    llm_accepted_count: int
    llm_acceptance_fraction: float

    llm_total_issue_count: int
    llm_rejected_case_ids: tuple[str, ...]

    llm_wall_time_mean_seconds: float
    llm_wall_time_median_seconds: float
    llm_wall_time_min_seconds: float
    llm_wall_time_max_seconds: float

    rows: tuple[ReportingCohortRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_reporting_cohort(
    rows: Iterable[ReportingCohortRow],
) -> ReportingCohortSummary:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one cohort row is required.")

    case_ids = tuple(row.case_id for row in rows)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Cohort contains duplicate case IDs.")

    volumes = tuple(row.workflow_volume_ml for row in rows)
    selected_distances = tuple(
        row.selected_kidney_distance_mm
        for row in rows
    )
    margins = tuple(
        row.laterality_margin_mm
        for row in rows
    )
    wall_times = tuple(
        row.llm_wall_time_seconds
        for row in rows
    )

    parsed = sum(row.llm_parsed for row in rows)
    accepted = sum(row.llm_accepted for row in rows)

    rejected_case_ids = tuple(
        row.case_id
        for row in rows
        if not row.llm_accepted
    )

    return ReportingCohortSummary(
        case_count=len(rows),

        left_case_count=sum(
            row.laterality == "left"
            for row in rows
        ),
        right_case_count=sum(
            row.laterality == "right"
            for row in rows
        ),

        workflow_volume_mean_ml=float(mean(volumes)),
        workflow_volume_median_ml=float(median(volumes)),

        selected_kidney_distance_mean_mm=float(
            mean(selected_distances)
        ),
        selected_kidney_distance_median_mm=float(
            median(selected_distances)
        ),
        selected_kidney_distance_max_mm=float(
            max(selected_distances)
        ),

        laterality_margin_mean_mm=float(mean(margins)),
        laterality_margin_median_mm=float(median(margins)),
        laterality_margin_min_mm=float(min(margins)),

        llm_parse_success_count=parsed,
        llm_parse_success_fraction=float(parsed / len(rows)),

        llm_accepted_count=accepted,
        llm_acceptance_fraction=float(accepted / len(rows)),

        llm_total_issue_count=sum(
            row.llm_issue_count
            for row in rows
        ),
        llm_rejected_case_ids=rejected_case_ids,

        llm_wall_time_mean_seconds=float(mean(wall_times)),
        llm_wall_time_median_seconds=float(median(wall_times)),
        llm_wall_time_min_seconds=float(min(wall_times)),
        llm_wall_time_max_seconds=float(max(wall_times)),

        rows=rows,
    )


def write_reporting_cohort_outputs(
    summary: ReportingCohortSummary,
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.json"
    csv_path = output_root / "cases.csv"

    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "local_llm_reporting_cohort",
                "deployment_valid": False,
                "interaction_provenance": (
                    "Frozen Block 16 GT-derived oracle points replayed as "
                    "experimental user-click surrogates. Runtime segmentation "
                    "and reporting do not open ground truth."
                ),
                "llm_factual_source": (
                    "Per-case structured findings generated from model outputs."
                ),
                "clinical_validation": False,
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fieldnames = list(asdict(summary.rows[0]).keys())

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in summary.rows:
            writer.writerow(asdict(row))

    return summary_path, csv_path
