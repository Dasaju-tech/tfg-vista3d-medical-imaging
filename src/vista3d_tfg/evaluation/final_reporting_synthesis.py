from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import matplotlib.pyplot as plt


@dataclass(frozen=True, slots=True)
class SynthesisStage:
    stage: str
    description: str
    case_count: int
    parse_success_count: int | None
    parse_success_fraction: float | None
    accepted_count: int
    acceptance_fraction: float
    mean_wall_time_seconds: float | None
    median_wall_time_seconds: float | None
    prompt_tokens_mean: float | None
    prompt_tokens_min: int | None
    prompt_tokens_max: int | None
    output_tokens_mean: float | None
    output_tokens_min: int | None
    output_tokens_max: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FinalReportingSynthesis:
    block25: SynthesisStage
    block26_v1: SynthesisStage
    block27_v2: SynthesisStage

    prompt_token_reduction_fraction: float
    prompt_token_reduction_percent: float

    v1_to_v2_acceptance_delta_fraction: float
    v1_to_v2_acceptance_delta_percentage_points: float

    block25_exact_raw_reproducibility: bool
    block25_exact_candidate_reproducibility: bool

    block26_persistent_failure_case_ids: tuple[str, ...]
    block27_improved_case_ids: tuple[str, ...]
    block27_regressed_case_ids: tuple[str, ...]

    interpretation_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _result(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("result")

    if isinstance(value, dict):
        return value

    return payload


def _mean_optional_int(values: Iterable[int | None]) -> float | None:
    finite = [int(value) for value in values if value is not None]
    return float(mean(finite)) if finite else None


def _min_optional_int(values: Iterable[int | None]) -> int | None:
    finite = [int(value) for value in values if value is not None]
    return min(finite) if finite else None


def _max_optional_int(values: Iterable[int | None]) -> int | None:
    finite = [int(value) for value in values if value is not None]
    return max(finite) if finite else None


def load_block25_stage(path: Path) -> tuple[SynthesisStage, dict[str, Any]]:
    payload = _load_json(path)
    result = _result(payload)

    rows = tuple(result.get("run_rows", ()))

    prompt_tokens = tuple(
        row.get("prompt_eval_count")
        for row in rows
    )
    output_tokens = tuple(
        row.get("eval_count")
        for row in rows
    )

    stage = SynthesisStage(
        stage="Block 25",
        description="Single-case local LLM reproducibility",
        case_count=int(result["completed_runs"]),
        parse_success_count=int(result["parse_success_count"]),
        parse_success_fraction=float(result["parse_success_fraction"]),
        accepted_count=int(result["accepted_count"]),
        acceptance_fraction=float(result["acceptance_fraction"]),
        mean_wall_time_seconds=float(
            result["wall_time_mean_seconds"]
        ),
        median_wall_time_seconds=float(
            result["wall_time_median_seconds"]
        ),
        prompt_tokens_mean=_mean_optional_int(prompt_tokens),
        prompt_tokens_min=_min_optional_int(prompt_tokens),
        prompt_tokens_max=_max_optional_int(prompt_tokens),
        output_tokens_mean=_mean_optional_int(output_tokens),
        output_tokens_min=_min_optional_int(output_tokens),
        output_tokens_max=_max_optional_int(output_tokens),
    )

    return stage, result


def load_block26_stage(
    original_summary_path: Path,
    final_reaudit_path: Path,
) -> tuple[SynthesisStage, dict[str, Any], dict[str, Any]]:
    original_payload = _load_json(original_summary_path)
    original = _result(original_payload)
    final = _load_json(final_reaudit_path)

    rows = tuple(original.get("rows", ()))

    stage = SynthesisStage(
        stage="Block 26",
        description="Prompt v1, 17-case reporting cohort",
        case_count=int(final["case_count"]),
        parse_success_count=int(
            original["llm_parse_success_count"]
        ),
        parse_success_fraction=float(
            original["llm_parse_success_fraction"]
        ),
        accepted_count=int(final["accepted_count"]),
        acceptance_fraction=float(final["acceptance_fraction"]),
        mean_wall_time_seconds=float(
            original["llm_wall_time_mean_seconds"]
        ),
        median_wall_time_seconds=float(
            original["llm_wall_time_median_seconds"]
        ),
        prompt_tokens_mean=None,
        prompt_tokens_min=None,
        prompt_tokens_max=None,
        output_tokens_mean=None,
        output_tokens_min=None,
        output_tokens_max=None,
    )

    return stage, original, final


def load_block27_stage(path: Path) -> tuple[SynthesisStage, dict[str, Any]]:
    payload = _load_json(path)
    result = _result(payload)
    rows = tuple(result["rows"])

    prompt_tokens = tuple(
        row.get("prompt_tokens")
        for row in rows
    )
    output_tokens = tuple(
        row.get("output_tokens")
        for row in rows
    )

    stage = SynthesisStage(
        stage="Block 27",
        description="Minimal-facts prompt v2, 17-case prospective cohort",
        case_count=int(result["case_count"]),
        parse_success_count=int(result["v2_parse_success_count"]),
        parse_success_fraction=float(
            result["v2_parse_success_fraction"]
        ),
        accepted_count=int(result["v2_accepted_count"]),
        acceptance_fraction=float(result["v2_acceptance_fraction"]),
        mean_wall_time_seconds=float(
            result["wall_time_mean_seconds"]
        ),
        median_wall_time_seconds=float(
            result["wall_time_median_seconds"]
        ),
        prompt_tokens_mean=_mean_optional_int(prompt_tokens),
        prompt_tokens_min=_min_optional_int(prompt_tokens),
        prompt_tokens_max=_max_optional_int(prompt_tokens),
        output_tokens_mean=_mean_optional_int(output_tokens),
        output_tokens_min=_min_optional_int(output_tokens),
        output_tokens_max=_max_optional_int(output_tokens),
    )

    return stage, result


def build_final_reporting_synthesis(
    *,
    block25_stage: SynthesisStage,
    block25_result: dict[str, Any],
    block26_stage: SynthesisStage,
    block26_final: dict[str, Any],
    block27_stage: SynthesisStage,
    block27_result: dict[str, Any],
) -> FinalReportingSynthesis:
    if block25_stage.prompt_tokens_mean is None:
        raise ValueError(
            "Block 25 prompt-token mean is required."
        )

    if block27_stage.prompt_tokens_mean is None:
        raise ValueError(
            "Block 27 prompt-token mean is required."
        )

    token_reduction_fraction = (
        1.0
        - (
            block27_stage.prompt_tokens_mean
            / block25_stage.prompt_tokens_mean
        )
    )

    acceptance_delta = (
        block27_stage.acceptance_fraction
        - block26_stage.acceptance_fraction
    )

    return FinalReportingSynthesis(
        block25=block25_stage,
        block26_v1=block26_stage,
        block27_v2=block27_stage,

        prompt_token_reduction_fraction=float(
            token_reduction_fraction
        ),
        prompt_token_reduction_percent=float(
            token_reduction_fraction * 100.0
        ),

        v1_to_v2_acceptance_delta_fraction=float(
            acceptance_delta
        ),
        v1_to_v2_acceptance_delta_percentage_points=float(
            acceptance_delta * 100.0
        ),

        block25_exact_raw_reproducibility=bool(
            block25_result["all_raw_responses_identical"]
        ),
        block25_exact_candidate_reproducibility=bool(
            block25_result["all_candidates_identical"]
        ),

        block26_persistent_failure_case_ids=tuple(
            block26_final.get(
                "persistent_rejected_case_ids",
                (),
            )
        ),
        block27_improved_case_ids=tuple(
            block27_result.get(
                "improved_case_ids",
                (),
            )
        ),
        block27_regressed_case_ids=tuple(
            block27_result.get(
                "regressed_case_ids",
                (),
            )
        ),

        interpretation_boundary=(
            "Research prototype only. The 17-case cohort replays frozen "
            "GT-derived oracle click coordinates as experimental user-click "
            "surrogates; therefore cohort-level results are not "
            "deployment-valid or clinically validated."
        ),
    )


def _write_stage_csv(
    synthesis: FinalReportingSynthesis,
    path: Path,
) -> None:
    stages = (
        synthesis.block25,
        synthesis.block26_v1,
        synthesis.block27_v2,
    )

    fieldnames = list(asdict(stages[0]).keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for stage in stages:
            writer.writerow(asdict(stage))


def _write_markdown(
    synthesis: FinalReportingSynthesis,
    path: Path,
) -> None:
    b25 = synthesis.block25
    b26 = synthesis.block26_v1
    b27 = synthesis.block27_v2

    def pct(value: float | None) -> str:
        return (
            "n/a"
            if value is None
            else f"{100.0 * value:.1f}%"
        )

    def num(value: float | None, digits: int = 2) -> str:
        return (
            "n/a"
            if value is None
            else f"{value:.{digits}f}"
        )

    lines = [
        "# Final local-LLM reporting synthesis",
        "",
        "## Quantitative summary",
        "",
        "| Stage | Scope | JSON parse | Accepted | Median latency | Mean prompt tokens |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| Block 25 | repeated single case ({b25.case_count} runs) | "
            f"{pct(b25.parse_success_fraction)} | "
            f"{pct(b25.acceptance_fraction)} | "
            f"{num(b25.median_wall_time_seconds)} s | "
            f"{num(b25.prompt_tokens_mean, 1)} |"
        ),
        (
            f"| Block 26 | prompt v1 ({b26.case_count} cases) | "
            f"{pct(b26.parse_success_fraction)} | "
            f"{pct(b26.acceptance_fraction)} | "
            f"{num(b26.median_wall_time_seconds)} s | "
            "n/a |"
        ),
        (
            f"| Block 27 | minimal-facts prompt v2 ({b27.case_count} cases) | "
            f"{pct(b27.parse_success_fraction)} | "
            f"{pct(b27.acceptance_fraction)} | "
            f"{num(b27.median_wall_time_seconds)} s | "
            f"{num(b27.prompt_tokens_mean, 1)} |"
        ),
        "",
        "## Key findings",
        "",
        (
            f"- Block 25 achieved exact raw-response reproducibility: "
            f"`{synthesis.block25_exact_raw_reproducibility}`."
        ),
        (
            f"- Block 25 achieved exact canonical-candidate reproducibility: "
            f"`{synthesis.block25_exact_candidate_reproducibility}`."
        ),
        (
            f"- Prompt v1 final acceptance was "
            f"{b26.accepted_count}/{b26.case_count} "
            f"({100.0 * b26.acceptance_fraction:.1f}%)."
        ),
        (
            f"- Prompt v2 acceptance was "
            f"{b27.accepted_count}/{b27.case_count} "
            f"({100.0 * b27.acceptance_fraction:.1f}%)."
        ),
        (
            f"- Prompt v2 improved acceptance by "
            f"{synthesis.v1_to_v2_acceptance_delta_percentage_points:+.1f} "
            "percentage points."
        ),
        (
            f"- Relative to the Block 25 prompt, the minimal-facts prompt "
            f"reduced mean input tokens by "
            f"{synthesis.prompt_token_reduction_percent:.1f}%."
        ),
        (
            "- Persistent prompt-v1 rejected cases after auditor calibration: "
            + (
                ", ".join(synthesis.block26_persistent_failure_case_ids)
                if synthesis.block26_persistent_failure_case_ids
                else "none"
            )
            + "."
        ),
        (
            "- Prompt-v2 improved cases: "
            + (
                ", ".join(synthesis.block27_improved_case_ids)
                if synthesis.block27_improved_case_ids
                else "none"
            )
            + "."
        ),
        (
            "- Prompt-v2 regressed cases: "
            + (
                ", ".join(synthesis.block27_regressed_case_ids)
                if synthesis.block27_regressed_case_ids
                else "none"
            )
            + "."
        ),
        "",
        "## Interpretation boundary",
        "",
        synthesis.interpretation_boundary,
        "",
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _plot_acceptance(
    synthesis: FinalReportingSynthesis,
    path: Path,
) -> None:
    labels = ["Prompt v1", "Prompt v2"]
    values = [
        100.0 * synthesis.block26_v1.acceptance_fraction,
        100.0 * synthesis.block27_v2.acceptance_fraction,
    ]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Deterministic acceptance (%)")
    ax.set_ylim(0.0, 105.0)
    ax.set_title("Local LLM factual acceptance: prompt v1 vs v2")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_latency(
    synthesis: FinalReportingSynthesis,
    path: Path,
) -> None:
    labels = ["Block 25", "Prompt v1", "Prompt v2"]
    values = [
        synthesis.block25.median_wall_time_seconds,
        synthesis.block26_v1.median_wall_time_seconds,
        synthesis.block27_v2.median_wall_time_seconds,
    ]

    if any(value is None for value in values):
        raise ValueError(
            "Median latency is required for all synthesis stages."
        )

    numeric = [float(value) for value in values if value is not None]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(labels, numeric)
    ax.set_ylabel("Median LLM wall time (s)")
    ax.set_title("Local LLM generation latency")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, numeric):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_prompt_tokens(
    synthesis: FinalReportingSynthesis,
    path: Path,
) -> None:
    old = synthesis.block25.prompt_tokens_mean
    new = synthesis.block27_v2.prompt_tokens_mean

    if old is None or new is None:
        raise ValueError(
            "Prompt token means are required."
        )

    labels = ["Original constrained prompt", "Minimal-facts prompt v2"]
    values = [old, new]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(labels, values)
    ax.set_ylabel("Mean prompt tokens")
    ax.set_title("Prompt context reduction")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + max(values) * 0.015,
            f"{value:.0f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_final_reporting_synthesis(
    synthesis: FinalReportingSynthesis,
    output_root: Path,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "final_reporting_synthesis.json"
    csv_path = output_root / "final_reporting_stages.csv"
    md_path = output_root / "final_reporting_synthesis.md"
    acceptance_path = output_root / "acceptance_v1_vs_v2.png"
    latency_path = output_root / "llm_latency.png"
    tokens_path = output_root / "prompt_token_reduction.png"

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "final_local_llm_reporting_synthesis",
                "result": synthesis.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    _write_stage_csv(synthesis, csv_path)
    _write_markdown(synthesis, md_path)
    _plot_acceptance(synthesis, acceptance_path)
    _plot_latency(synthesis, latency_path)
    _plot_prompt_tokens(synthesis, tokens_path)

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "acceptance_figure": acceptance_path,
        "latency_figure": latency_path,
        "prompt_tokens_figure": tokens_path,
    }
