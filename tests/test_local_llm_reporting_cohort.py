from __future__ import annotations

import pytest

from vista3d_tfg.evaluation.local_llm_reporting_cohort import (
    ReportingCohortRow,
    summarize_reporting_cohort,
)


def row(
    case_id: str,
    *,
    laterality: str = "right",
    accepted: bool = True,
    parsed: bool = True,
    issues: int = 0,
    wall: float = 2.0,
) -> ReportingCohortRow:
    class_label = 116 if laterality == "left" else 117

    return ReportingCohortRow(
        case_id=case_id,
        point_xyz=(1, 2, 3),
        laterality=laterality,
        class_label=class_label,
        workflow_volume_ml=10.0,
        selected_kidney_distance_mm=5.0,
        laterality_margin_mm=100.0,
        llm_parsed=parsed,
        llm_accepted=accepted,
        llm_issue_count=issues,
        llm_wall_time_seconds=wall,
        llm_response_sha256="a" * 64,
        llm_candidate_sha256="b" * 64,
    )


def test_all_accepted_cohort_has_full_pass_rate() -> None:
    summary = summarize_reporting_cohort(
        [
            row("case_00001", laterality="left"),
            row("case_00002", laterality="right"),
        ]
    )

    assert summary.case_count == 2
    assert summary.left_case_count == 1
    assert summary.right_case_count == 1
    assert summary.llm_parse_success_fraction == 1.0
    assert summary.llm_acceptance_fraction == 1.0
    assert summary.llm_rejected_case_ids == ()


def test_rejection_is_preserved_in_summary() -> None:
    summary = summarize_reporting_cohort(
        [
            row("case_00001"),
            row(
                "case_00002",
                accepted=False,
                issues=2,
            ),
        ]
    )

    assert summary.llm_accepted_count == 1
    assert summary.llm_acceptance_fraction == pytest.approx(0.5)
    assert summary.llm_total_issue_count == 2
    assert summary.llm_rejected_case_ids == ("case_00002",)


def test_parse_failure_is_distinct_from_audit_rejection() -> None:
    summary = summarize_reporting_cohort(
        [
            row("case_00001"),
            row(
                "case_00002",
                accepted=False,
                parsed=False,
                issues=0,
            ),
        ]
    )

    assert summary.llm_parse_success_count == 1
    assert summary.llm_accepted_count == 1


def test_latency_summary_is_computed() -> None:
    summary = summarize_reporting_cohort(
        [
            row("case_00001", wall=1.5),
            row("case_00002", wall=2.5),
            row("case_00003", wall=2.0),
        ]
    )

    assert summary.llm_wall_time_mean_seconds == pytest.approx(2.0)
    assert summary.llm_wall_time_median_seconds == pytest.approx(2.0)
    assert summary.llm_wall_time_min_seconds == pytest.approx(1.5)
    assert summary.llm_wall_time_max_seconds == pytest.approx(2.5)


def test_duplicate_case_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        summarize_reporting_cohort(
            [
                row("case_00001"),
                row("case_00001"),
            ]
        )


def test_empty_cohort_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_reporting_cohort([])
