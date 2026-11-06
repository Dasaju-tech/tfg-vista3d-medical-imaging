from __future__ import annotations

from dataclasses import replace

import pytest

from vista3d_tfg.evaluation.local_llm_reproducibility import (
    build_reproducibility_run,
    canonical_json_sha256,
    summarize_local_llm_reproducibility,
)
from vista3d_tfg.reports.ollama_local import LocalLLMRun


def local_run(
    *,
    accepted: bool = True,
    response_hash: str = "a" * 64,
    narrative: str = "fixture",
    wall_time: float = 2.0,
) -> LocalLLMRun:
    audit = {
        "passed": accepted,
        "issue_count": 0 if accepted else 1,
        "issues": [],
    }

    return LocalLLMRun(
        case_id="case_00002",
        provider="ollama-local",
        endpoint="http://127.0.0.1:11434",
        model="qwen3:4b-instruct",
        ollama_version="0.33.1",
        generation_config={
            "temperature": 0.0,
            "seed": 20260828,
        },
        prompt_sha256="p" * 64,
        response_sha256=response_hash,
        wall_time_seconds=wall_time,
        parsed=True,
        parse_error=None,
        candidate={
            "case_id": "case_00002",
            "narrative": narrative,
            "limitations": ["fixture"],
        },
        audit=audit,
        accepted=accepted,
    )


def row(
    index: int,
    *,
    accepted: bool = True,
    response_hash: str = "a" * 64,
    narrative: str = "fixture",
    wall_time: float = 2.0,
):
    return build_reproducibility_run(
        run_index=index,
        local_run=local_run(
            accepted=accepted,
            response_hash=response_hash,
            narrative=narrative,
            wall_time=wall_time,
        ),
        prompt_eval_count=1551,
        eval_count=244,
        done_reason="stop",
    )


def test_canonical_candidate_hash_ignores_key_order() -> None:
    first = {
        "a": 1,
        "b": 2,
    }
    second = {
        "b": 2,
        "a": 1,
    }

    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_identical_runs_are_exactly_reproducible() -> None:
    rows = [
        row(1, wall_time=3.0),
        row(2, wall_time=2.0),
        row(3, wall_time=2.2),
    ]

    summary = summarize_local_llm_reproducibility(
        case_id="case_00002",
        provider="ollama-local",
        model="qwen3:4b-instruct",
        requested_runs=3,
        rows=rows,
    )

    assert summary.acceptance_fraction == 1.0
    assert summary.unique_raw_response_count == 1
    assert summary.unique_candidate_count == 1
    assert summary.all_raw_responses_identical is True
    assert summary.all_candidates_identical is True
    assert summary.raw_match_to_first_fraction == 1.0
    assert summary.candidate_match_to_first_fraction == 1.0
    assert summary.warm_run_count == 2
    assert summary.warm_wall_time_mean_seconds == pytest.approx(2.1)


def test_textual_variation_can_still_be_factually_accepted() -> None:
    rows = [
        row(
            1,
            response_hash="a" * 64,
            narrative="version one",
        ),
        row(
            2,
            response_hash="b" * 64,
            narrative="version two",
        ),
    ]

    summary = summarize_local_llm_reproducibility(
        case_id="case_00002",
        provider="ollama-local",
        model="qwen3:4b-instruct",
        requested_runs=2,
        rows=rows,
    )

    assert summary.acceptance_fraction == 1.0
    assert summary.unique_raw_response_count == 2
    assert summary.unique_candidate_count == 2
    assert summary.all_raw_responses_identical is False
    assert summary.all_candidates_identical is False


def test_rejected_generation_reduces_factual_reproducibility() -> None:
    rows = [
        row(1, accepted=True),
        row(
            2,
            accepted=False,
            response_hash="b" * 64,
            narrative="bad",
        ),
    ]

    summary = summarize_local_llm_reproducibility(
        case_id="case_00002",
        provider="ollama-local",
        model="qwen3:4b-instruct",
        requested_runs=2,
        rows=rows,
    )

    assert summary.accepted_count == 1
    assert summary.acceptance_fraction == pytest.approx(0.5)
    assert summary.parse_success_fraction == 1.0


def test_requested_and_completed_runs_must_match() -> None:
    with pytest.raises(ValueError, match="must equal"):
        summarize_local_llm_reproducibility(
            case_id="case_00002",
            provider="ollama-local",
            model="qwen3:4b-instruct",
            requested_runs=3,
            rows=[row(1), row(2)],
        )


def test_run_indices_must_be_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        summarize_local_llm_reproducibility(
            case_id="case_00002",
            provider="ollama-local",
            model="qwen3:4b-instruct",
            requested_runs=2,
            rows=[row(1), row(3)],
        )


def test_candidate_hash_absent_when_parse_failed() -> None:
    failed = replace(
        local_run(),
        parsed=False,
        parse_error="invalid json",
        candidate=None,
        audit=None,
        accepted=False,
    )

    result = build_reproducibility_run(
        run_index=1,
        local_run=failed,
        prompt_eval_count=None,
        eval_count=None,
        done_reason=None,
    )

    assert result.canonical_candidate_sha256 is None
    assert result.audit_passed is False
