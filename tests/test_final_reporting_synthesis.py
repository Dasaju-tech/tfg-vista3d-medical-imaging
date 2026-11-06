from __future__ import annotations

from vista3d_tfg.evaluation.final_reporting_synthesis import (
    SynthesisStage,
    build_final_reporting_synthesis,
)


def stage(
    name: str,
    *,
    cases: int,
    accepted: int,
    prompt_tokens: float | None,
) -> SynthesisStage:
    return SynthesisStage(
        stage=name,
        description="fixture",
        case_count=cases,
        parse_success_count=cases,
        parse_success_fraction=1.0,
        accepted_count=accepted,
        acceptance_fraction=accepted / cases,
        mean_wall_time_seconds=2.0,
        median_wall_time_seconds=1.8,
        prompt_tokens_mean=prompt_tokens,
        prompt_tokens_min=(
            int(prompt_tokens)
            if prompt_tokens is not None
            else None
        ),
        prompt_tokens_max=(
            int(prompt_tokens)
            if prompt_tokens is not None
            else None
        ),
        output_tokens_mean=230.0,
        output_tokens_min=220,
        output_tokens_max=240,
    )


def test_final_synthesis_computes_acceptance_gain() -> None:
    result = build_final_reporting_synthesis(
        block25_stage=stage(
            "Block 25",
            cases=5,
            accepted=5,
            prompt_tokens=1551.0,
        ),
        block25_result={
            "all_raw_responses_identical": True,
            "all_candidates_identical": True,
        },
        block26_stage=stage(
            "Block 26",
            cases=17,
            accepted=16,
            prompt_tokens=None,
        ),
        block26_final={
            "persistent_rejected_case_ids": [
                "case_00237"
            ],
        },
        block27_stage=stage(
            "Block 27",
            cases=17,
            accepted=17,
            prompt_tokens=808.0,
        ),
        block27_result={
            "improved_case_ids": ["case_00237"],
            "regressed_case_ids": [],
        },
    )

    assert result.v1_to_v2_acceptance_delta_percentage_points > 5.8
    assert result.v1_to_v2_acceptance_delta_percentage_points < 6.0


def test_final_synthesis_computes_prompt_reduction() -> None:
    result = build_final_reporting_synthesis(
        block25_stage=stage(
            "Block 25",
            cases=5,
            accepted=5,
            prompt_tokens=1551.0,
        ),
        block25_result={
            "all_raw_responses_identical": True,
            "all_candidates_identical": True,
        },
        block26_stage=stage(
            "Block 26",
            cases=17,
            accepted=16,
            prompt_tokens=None,
        ),
        block26_final={
            "persistent_rejected_case_ids": [
                "case_00237"
            ],
        },
        block27_stage=stage(
            "Block 27",
            cases=17,
            accepted=17,
            prompt_tokens=808.0,
        ),
        block27_result={
            "improved_case_ids": ["case_00237"],
            "regressed_case_ids": [],
        },
    )

    assert result.prompt_token_reduction_percent > 47.0
    assert result.prompt_token_reduction_percent < 49.0


def test_final_synthesis_preserves_failure_and_improvement_case() -> None:
    result = build_final_reporting_synthesis(
        block25_stage=stage(
            "Block 25",
            cases=5,
            accepted=5,
            prompt_tokens=1551.0,
        ),
        block25_result={
            "all_raw_responses_identical": True,
            "all_candidates_identical": True,
        },
        block26_stage=stage(
            "Block 26",
            cases=17,
            accepted=16,
            prompt_tokens=None,
        ),
        block26_final={
            "persistent_rejected_case_ids": [
                "case_00237"
            ],
        },
        block27_stage=stage(
            "Block 27",
            cases=17,
            accepted=17,
            prompt_tokens=808.0,
        ),
        block27_result={
            "improved_case_ids": ["case_00237"],
            "regressed_case_ids": [],
        },
    )

    assert result.block26_persistent_failure_case_ids == (
        "case_00237",
    )
    assert result.block27_improved_case_ids == (
        "case_00237",
    )
    assert result.block27_regressed_case_ids == ()


def test_exact_reproducibility_flags_are_preserved() -> None:
    result = build_final_reporting_synthesis(
        block25_stage=stage(
            "Block 25",
            cases=5,
            accepted=5,
            prompt_tokens=1551.0,
        ),
        block25_result={
            "all_raw_responses_identical": True,
            "all_candidates_identical": True,
        },
        block26_stage=stage(
            "Block 26",
            cases=17,
            accepted=16,
            prompt_tokens=None,
        ),
        block26_final={
            "persistent_rejected_case_ids": [],
        },
        block27_stage=stage(
            "Block 27",
            cases=17,
            accepted=17,
            prompt_tokens=808.0,
        ),
        block27_result={
            "improved_case_ids": [],
            "regressed_case_ids": [],
        },
    )

    assert result.block25_exact_raw_reproducibility is True
    assert result.block25_exact_candidate_reproducibility is True
