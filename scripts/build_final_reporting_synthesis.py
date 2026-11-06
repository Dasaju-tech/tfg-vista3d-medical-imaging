from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.final_reporting_synthesis import (
    build_final_reporting_synthesis,
    load_block25_stage,
    load_block26_stage,
    load_block27_stage,
    write_final_reporting_synthesis,
)


def main() -> int:
    project_root = discover_project_root()
    model_slug = "qwen3_4b-instruct"

    block25_path = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reproducibility"
        / "case_00002"
        / model_slug
        / "summary.json"
    )

    block26_root = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort"
        / model_slug
    )
    block26_original_path = block26_root / "summary.json"
    block26_final_path = block26_root / "reaudit_summary.json"

    block27_path = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort_v2"
        / model_slug
        / "summary.json"
    )

    block25_stage, block25_result = load_block25_stage(
        block25_path
    )
    (
        block26_stage,
        _block26_original,
        block26_final,
    ) = load_block26_stage(
        block26_original_path,
        block26_final_path,
    )
    block27_stage, block27_result = load_block27_stage(
        block27_path
    )

    synthesis = build_final_reporting_synthesis(
        block25_stage=block25_stage,
        block25_result=block25_result,
        block26_stage=block26_stage,
        block26_final=block26_final,
        block27_stage=block27_stage,
        block27_result=block27_result,
    )

    output_root = (
        project_root
        / "outputs"
        / "final_reporting_synthesis"
    )

    outputs = write_final_reporting_synthesis(
        synthesis,
        output_root,
    )

    print("Final local-LLM reporting synthesis")
    print()
    print(
        f"Block 25 exact raw reproducibility: "
        f"{synthesis.block25_exact_raw_reproducibility}"
    )
    print(
        f"Block 25 exact candidate reproducibility: "
        f"{synthesis.block25_exact_candidate_reproducibility}"
    )
    print()
    print(
        f"Prompt v1 acceptance: "
        f"{synthesis.block26_v1.accepted_count}/"
        f"{synthesis.block26_v1.case_count} "
        f"({synthesis.block26_v1.acceptance_fraction:.1%})"
    )
    print(
        f"Prompt v2 acceptance: "
        f"{synthesis.block27_v2.accepted_count}/"
        f"{synthesis.block27_v2.case_count} "
        f"({synthesis.block27_v2.acceptance_fraction:.1%})"
    )
    print(
        f"Acceptance delta: "
        f"{synthesis.v1_to_v2_acceptance_delta_percentage_points:+.1f} pp"
    )
    print()
    print(
        f"Mean prompt tokens: "
        f"{synthesis.block25.prompt_tokens_mean:.1f} -> "
        f"{synthesis.block27_v2.prompt_tokens_mean:.1f}"
    )
    print(
        f"Prompt-token reduction: "
        f"{synthesis.prompt_token_reduction_percent:.1f}%"
    )
    print()
    print(
        f"Prompt-v1 persistent failures: "
        + (
            ", ".join(
                synthesis.block26_persistent_failure_case_ids
            )
            if synthesis.block26_persistent_failure_case_ids
            else "none"
        )
    )
    print(
        f"Prompt-v2 improved cases: "
        + (
            ", ".join(
                synthesis.block27_improved_case_ids
            )
            if synthesis.block27_improved_case_ids
            else "none"
        )
    )
    print(
        f"Prompt-v2 regressions: "
        + (
            ", ".join(
                synthesis.block27_regressed_case_ids
            )
            if synthesis.block27_regressed_case_ids
            else "none"
        )
    )
    print()
    print("Outputs")

    for label, path in outputs.items():
        print(f"  {label}: {path}")

    print()
    print("Final reporting synthesis PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
