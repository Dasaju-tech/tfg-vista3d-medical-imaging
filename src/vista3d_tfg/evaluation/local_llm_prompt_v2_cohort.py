from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class PromptV2CaseResult:
    case_id: str
    v1_final_accepted: bool | None

    v2_parsed: bool
    v2_accepted: bool
    v2_issue_count: int
    v2_issue_codes: tuple[str, ...]

    wall_time_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None

    prompt_sha256: str
    response_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptV2CohortSummary:
    case_count: int

    v2_parse_success_count: int
    v2_parse_success_fraction: float

    v2_accepted_count: int
    v2_acceptance_fraction: float

    v2_rejected_case_ids: tuple[str, ...]
    v2_issue_code_counts: dict[str, int]

    v1_comparable_case_count: int
    v1_accepted_count: int | None
    v1_acceptance_fraction: float | None

    improved_case_ids: tuple[str, ...]
    regressed_case_ids: tuple[str, ...]
    unchanged_pass_case_ids: tuple[str, ...]
    unchanged_fail_case_ids: tuple[str, ...]

    acceptance_delta_count: int | None
    acceptance_delta_fraction: float | None

    wall_time_mean_seconds: float
    wall_time_median_seconds: float
    wall_time_min_seconds: float
    wall_time_max_seconds: float

    rows: tuple[PromptV2CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_prompt_v2_cohort(
    rows: Iterable[PromptV2CaseResult],
) -> PromptV2CohortSummary:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one case result is required.")

    case_ids = tuple(row.case_id for row in rows)

    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Duplicate case IDs are not allowed.")

    parsed_count = sum(row.v2_parsed for row in rows)
    accepted_count = sum(row.v2_accepted for row in rows)

    rejected = tuple(
        row.case_id
        for row in rows
        if not row.v2_accepted
    )

    issue_counts: dict[str, int] = {}

    for row in rows:
        for code in row.v2_issue_codes:
            issue_counts[code] = issue_counts.get(code, 0) + 1

    comparable = tuple(
        row
        for row in rows
        if row.v1_final_accepted is not None
    )

    if comparable:
        v1_accepted_count = sum(
            bool(row.v1_final_accepted)
            for row in comparable
        )
        v1_fraction = v1_accepted_count / len(comparable)

        improved = tuple(
            row.case_id
            for row in comparable
            if row.v1_final_accepted is False
            and row.v2_accepted
        )
        regressed = tuple(
            row.case_id
            for row in comparable
            if row.v1_final_accepted is True
            and not row.v2_accepted
        )
        unchanged_pass = tuple(
            row.case_id
            for row in comparable
            if row.v1_final_accepted is True
            and row.v2_accepted
        )
        unchanged_fail = tuple(
            row.case_id
            for row in comparable
            if row.v1_final_accepted is False
            and not row.v2_accepted
        )

        delta_count = accepted_count - v1_accepted_count
        delta_fraction = (
            accepted_count / len(rows)
            - v1_fraction
        )
    else:
        v1_accepted_count = None
        v1_fraction = None
        improved = ()
        regressed = ()
        unchanged_pass = ()
        unchanged_fail = ()
        delta_count = None
        delta_fraction = None

    wall_times = tuple(row.wall_time_seconds for row in rows)

    return PromptV2CohortSummary(
        case_count=len(rows),

        v2_parse_success_count=parsed_count,
        v2_parse_success_fraction=parsed_count / len(rows),

        v2_accepted_count=accepted_count,
        v2_acceptance_fraction=accepted_count / len(rows),

        v2_rejected_case_ids=rejected,
        v2_issue_code_counts=issue_counts,

        v1_comparable_case_count=len(comparable),
        v1_accepted_count=v1_accepted_count,
        v1_acceptance_fraction=v1_fraction,

        improved_case_ids=improved,
        regressed_case_ids=regressed,
        unchanged_pass_case_ids=unchanged_pass,
        unchanged_fail_case_ids=unchanged_fail,

        acceptance_delta_count=delta_count,
        acceptance_delta_fraction=delta_fraction,

        wall_time_mean_seconds=float(mean(wall_times)),
        wall_time_median_seconds=float(median(wall_times)),
        wall_time_min_seconds=float(min(wall_times)),
        wall_time_max_seconds=float(max(wall_times)),

        rows=rows,
    )


def write_prompt_v2_cohort_outputs(
    summary: PromptV2CohortSummary,
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.json"
    csv_path = output_root / "cases.csv"

    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "local_llm_reporting_cohort_prompt_v2",
                "prompt_protocol": "minimal-facts-v2",
                "generation_is_prospective": True,
                "vista_repeated": False,
                "segmentation_repeated": False,
                "auditor_changed_from_block26c": False,
                "model_changed": False,
                "temperature_changed": False,
                "seed_changed": False,
                "deployment_valid": False,
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
            rendered = asdict(row)
            rendered["v2_issue_codes"] = ";".join(
                row.v2_issue_codes
            )
            writer.writerow(rendered)

    return summary_path, csv_path
