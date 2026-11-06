from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from vista3d_tfg.reports.ollama_local import LocalLLMRun


@dataclass(frozen=True, slots=True)
class ReproducibilityRun:
    run_index: int
    accepted: bool
    parsed: bool
    audit_passed: bool
    issue_count: int
    wall_time_seconds: float
    raw_response_sha256: str
    canonical_candidate_sha256: str | None
    prompt_eval_count: int | None
    eval_count: int | None
    done_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalLLMReproducibilitySummary:
    case_id: str
    provider: str
    model: str
    requested_runs: int
    completed_runs: int

    parse_success_count: int
    parse_success_fraction: float

    accepted_count: int
    acceptance_fraction: float

    unique_raw_response_count: int
    unique_candidate_count: int

    raw_match_to_first_count: int
    raw_match_to_first_fraction: float
    candidate_match_to_first_count: int
    candidate_match_to_first_fraction: float

    all_raw_responses_identical: bool
    all_candidates_identical: bool

    wall_time_mean_seconds: float
    wall_time_median_seconds: float
    wall_time_min_seconds: float
    wall_time_max_seconds: float

    warm_run_count: int
    warm_wall_time_mean_seconds: float | None
    warm_wall_time_median_seconds: float | None

    prompt_eval_count_min: int | None
    prompt_eval_count_max: int | None
    eval_count_min: int | None
    eval_count_max: int | None

    run_rows: tuple[ReproducibilityRun, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json_sha256(
    payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


def build_reproducibility_run(
    *,
    run_index: int,
    local_run: LocalLLMRun,
    prompt_eval_count: int | None,
    eval_count: int | None,
    done_reason: str | None,
) -> ReproducibilityRun:
    audit_payload = local_run.audit

    audit_passed = bool(
        isinstance(audit_payload, dict)
        and audit_payload.get("passed") is True
    )
    issue_count = (
        int(audit_payload.get("issue_count", 0))
        if isinstance(audit_payload, dict)
        else 0
    )

    candidate_hash = (
        canonical_json_sha256(local_run.candidate)
        if isinstance(local_run.candidate, dict)
        else None
    )

    return ReproducibilityRun(
        run_index=run_index,
        accepted=bool(local_run.accepted),
        parsed=bool(local_run.parsed),
        audit_passed=audit_passed,
        issue_count=issue_count,
        wall_time_seconds=float(local_run.wall_time_seconds),
        raw_response_sha256=str(local_run.response_sha256),
        canonical_candidate_sha256=candidate_hash,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        done_reason=done_reason,
    )


def _optional_min(values: Iterable[int | None]) -> int | None:
    finite = tuple(
        int(value)
        for value in values
        if value is not None
    )
    return min(finite) if finite else None


def _optional_max(values: Iterable[int | None]) -> int | None:
    finite = tuple(
        int(value)
        for value in values
        if value is not None
    )
    return max(finite) if finite else None


def summarize_local_llm_reproducibility(
    *,
    case_id: str,
    provider: str,
    model: str,
    requested_runs: int,
    rows: Iterable[ReproducibilityRun],
) -> LocalLLMReproducibilitySummary:
    rows = tuple(rows)

    if requested_runs < 2:
        raise ValueError(
            "Reproducibility study requires at least two requested runs."
        )

    if not rows:
        raise ValueError("At least one completed run is required.")

    if len(rows) != requested_runs:
        raise ValueError(
            "Completed run count must equal requested run count."
        )

    expected_indices = tuple(range(1, requested_runs + 1))
    actual_indices = tuple(row.run_index for row in rows)

    if actual_indices != expected_indices:
        raise ValueError(
            "Run indices must be contiguous and ordered from 1."
        )

    parsed_count = sum(row.parsed for row in rows)
    accepted_count = sum(row.accepted for row in rows)

    raw_hashes = tuple(
        row.raw_response_sha256
        for row in rows
    )
    candidate_hashes = tuple(
        row.canonical_candidate_sha256
        for row in rows
        if row.canonical_candidate_sha256 is not None
    )

    first_raw = raw_hashes[0]
    raw_match_count = sum(
        value == first_raw
        for value in raw_hashes
    )

    first_candidate = rows[0].canonical_candidate_sha256

    if first_candidate is None:
        candidate_match_count = 0
    else:
        candidate_match_count = sum(
            row.canonical_candidate_sha256 == first_candidate
            for row in rows
        )

    wall_times = tuple(
        row.wall_time_seconds
        for row in rows
    )
    warm_times = wall_times[1:]

    return LocalLLMReproducibilitySummary(
        case_id=case_id,
        provider=provider,
        model=model,
        requested_runs=requested_runs,
        completed_runs=len(rows),

        parse_success_count=parsed_count,
        parse_success_fraction=float(
            parsed_count / len(rows)
        ),

        accepted_count=accepted_count,
        acceptance_fraction=float(
            accepted_count / len(rows)
        ),

        unique_raw_response_count=len(set(raw_hashes)),
        unique_candidate_count=len(set(candidate_hashes)),

        raw_match_to_first_count=raw_match_count,
        raw_match_to_first_fraction=float(
            raw_match_count / len(rows)
        ),
        candidate_match_to_first_count=candidate_match_count,
        candidate_match_to_first_fraction=float(
            candidate_match_count / len(rows)
        ),

        all_raw_responses_identical=len(set(raw_hashes)) == 1,
        all_candidates_identical=(
            len(candidate_hashes) == len(rows)
            and len(set(candidate_hashes)) == 1
        ),

        wall_time_mean_seconds=float(mean(wall_times)),
        wall_time_median_seconds=float(median(wall_times)),
        wall_time_min_seconds=float(min(wall_times)),
        wall_time_max_seconds=float(max(wall_times)),

        warm_run_count=len(warm_times),
        warm_wall_time_mean_seconds=(
            float(mean(warm_times))
            if warm_times
            else None
        ),
        warm_wall_time_median_seconds=(
            float(median(warm_times))
            if warm_times
            else None
        ),

        prompt_eval_count_min=_optional_min(
            row.prompt_eval_count
            for row in rows
        ),
        prompt_eval_count_max=_optional_max(
            row.prompt_eval_count
            for row in rows
        ),
        eval_count_min=_optional_min(
            row.eval_count
            for row in rows
        ),
        eval_count_max=_optional_max(
            row.eval_count
            for row in rows
        ),

        run_rows=rows,
    )


def write_reproducibility_outputs(
    summary: LocalLLMReproducibilitySummary,
    output_root: Path,
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "summary.json"
    csv_path = output_root / "runs.csv"

    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "local_llm_reproducibility",
                "interpretation": {
                    "exact_reproducibility": (
                        "Measured by response/candidate hashes."
                    ),
                    "factual_reproducibility": (
                        "Measured by deterministic acceptance-gate pass rate."
                    ),
                    "clinical_validation": False,
                },
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fieldnames = list(
        asdict(summary.run_rows[0]).keys()
    )

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

        for row in summary.run_rows:
            writer.writerow(asdict(row))

    return summary_path, csv_path
