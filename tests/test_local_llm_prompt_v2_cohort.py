from __future__ import annotations

import pytest

from vista3d_tfg.evaluation.local_llm_prompt_v2_cohort import (
    PromptV2CaseResult,
    summarize_prompt_v2_cohort,
)


def row(
    case_id: str,
    *,
    v1: bool | None,
    v2: bool,
    parsed: bool = True,
    issues: tuple[str, ...] = (),
    wall: float = 2.0,
) -> PromptV2CaseResult:
    return PromptV2CaseResult(
        case_id=case_id,
        v1_final_accepted=v1,
        v2_parsed=parsed,
        v2_accepted=v2,
        v2_issue_count=len(issues),
        v2_issue_codes=issues,
        wall_time_seconds=wall,
        prompt_tokens=900,
        output_tokens=200,
        prompt_sha256="a" * 64,
        response_sha256="b" * 64,
    )


def test_v2_summary_detects_improvement_without_regression() -> None:
    summary = summarize_prompt_v2_cohort(
        [
            row("case_00001", v1=True, v2=True),
            row("case_00002", v1=False, v2=True),
        ]
    )

    assert summary.v1_accepted_count == 1
    assert summary.v2_accepted_count == 2
    assert summary.acceptance_delta_count == 1
    assert summary.acceptance_delta_fraction == pytest.approx(0.5)
    assert summary.improved_case_ids == ("case_00002",)
    assert summary.regressed_case_ids == ()


def test_v2_summary_preserves_rejection_codes() -> None:
    summary = summarize_prompt_v2_cohort(
        [
            row(
                "case_00237",
                v1=False,
                v2=False,
                issues=("distance_relation_mismatch",),
            ),
        ]
    )

    assert summary.v2_acceptance_fraction == 0.0
    assert summary.v2_rejected_case_ids == ("case_00237",)
    assert summary.v2_issue_code_counts == {
        "distance_relation_mismatch": 1
    }


def test_v2_summary_detects_regression() -> None:
    summary = summarize_prompt_v2_cohort(
        [
            row("case_00001", v1=True, v2=False),
            row("case_00002", v1=True, v2=True),
        ]
    )

    assert summary.regressed_case_ids == ("case_00001",)
    assert summary.acceptance_delta_count == -1


def test_v2_summary_handles_missing_v1_baseline() -> None:
    summary = summarize_prompt_v2_cohort(
        [
            row("case_00001", v1=None, v2=True),
            row("case_00002", v1=None, v2=True),
        ]
    )

    assert summary.v1_comparable_case_count == 0
    assert summary.v1_accepted_count is None
    assert summary.acceptance_delta_count is None


def test_duplicate_cases_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_prompt_v2_cohort(
            [
                row("case_00001", v1=True, v2=True),
                row("case_00001", v1=True, v2=True),
            ]
        )


def test_empty_cohort_is_rejected() -> None:
    with pytest.raises(ValueError, match="At least one"):
        summarize_prompt_v2_cohort([])
