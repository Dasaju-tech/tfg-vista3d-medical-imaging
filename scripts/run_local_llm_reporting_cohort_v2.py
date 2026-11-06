from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import time

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.local_llm_prompt_v2_cohort import (
    PromptV2CaseResult,
    summarize_prompt_v2_cohort,
    write_prompt_v2_cohort_outputs,
)
from vista3d_tfg.reports.llm_contract import (
    build_llm_input_packet,
    load_findings,
)
from vista3d_tfg.reports.llm_prompt_v2 import (
    PROMPT_VERSION,
    render_llm_prompt_v2,
)
from vista3d_tfg.reports.ollama_local import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaGenerationConfig,
    candidate_json_schema,
    evaluate_ollama_content,
    generate_ollama_candidate,
    probe_ollama,
    render_ollama_prompt,
    write_local_llm_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prospectively regenerate the frozen Block 26 cohort using the "
            "minimal-facts prompt v2 while keeping model, generation "
            "configuration, structured findings, and deterministic auditor "
            "unchanged."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "VISTA3D_OLLAMA_URL",
            DEFAULT_OLLAMA_URL,
        ),
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional subset of frozen Block 26 case numbers for a smoke "
            "run. Full runs use all cases from Block 26 cases.csv."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def _bool_or_none(value: str | None) -> bool | None:
    if value is None or not value.strip():
        return None

    normalized = value.strip().lower()

    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise ValueError(f"Unexpected boolean value: {value!r}")


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    model_slug = args.model.replace(":", "_")

    block26_root = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort"
        / model_slug
    )
    block26_cases_path = block26_root / "cases.csv"

    if not block26_cases_path.is_file():
        raise FileNotFoundError(
            "Block 27 requires the frozen Block 26 cohort cases.csv: "
            f"{block26_cases_path}"
        )

    frozen_rows = _read_csv(block26_cases_path)
    frozen_case_ids = tuple(
        row["case_id"]
        for row in frozen_rows
    )

    if len(frozen_case_ids) != 17 and args.cases is None:
        raise ValueError(
            "The full Block 27 protocol expects the 17-case frozen "
            f"Block 26 cohort, found {len(frozen_case_ids)} cases."
        )

    selected_case_ids = frozen_case_ids

    if args.cases is not None:
        requested = tuple(
            f"case_{value:05d}"
            for value in args.cases
        )
        missing = tuple(
            case_id
            for case_id in requested
            if case_id not in frozen_case_ids
        )

        if missing:
            raise ValueError(
                "Requested case(s) are not in the frozen Block 26 cohort: "
                + ", ".join(missing)
            )

        selected_case_ids = tuple(
            case_id
            for case_id in frozen_case_ids
            if case_id in set(requested)
        )

    baseline_path = block26_root / "reaudit_cases.csv"
    baseline_by_case: dict[str, bool] = {}

    if baseline_path.is_file():
        for row in _read_csv(baseline_path):
            baseline_by_case[row["case_id"]] = bool(
                _bool_or_none(row.get("reaudit_accepted"))
            )

    config = OllamaGenerationConfig(
        base_url=args.base_url,
        model=args.model,
    )
    runtime = probe_ollama(
        base_url=config.base_url,
        requested_model=config.model,
    )

    if not runtime.requested_model_present:
        raise RuntimeError(
            f"Model {config.model!r} is not installed in Ollama."
        )

    schema = candidate_json_schema()

    experiment_root = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort_v2"
        / model_slug
    )

    report_root = (
        project_root
        / "outputs"
        / "reports"
        / "local_llm_cohort_v2"
    )

    print("Block 27 — prospective minimal-facts prompt v2 cohort")
    print(f"Prompt:       {PROMPT_VERSION}")
    print(f"Cases:        {len(selected_case_ids)}")
    print(f"Model:        {config.model}")
    print(f"Ollama:       {runtime.version}")
    print(f"Temperature:  {config.temperature}")
    print(f"Seed:         {config.seed}")
    print(f"Thinking:     {config.think}")
    print("VISTA called: NO")
    print("Segmentation repeated: NO")
    print("Auditor changed from Block 26C: NO")
    print()
    print(
        "Only the LLM input contract changes: the model sees the intended "
        "minimal facts and no raw findings JSON."
    )
    print()

    rows: list[PromptV2CaseResult] = []

    for index, case_id in enumerate(selected_case_ids, start=1):
        findings_path = (
            project_root
            / "outputs"
            / "reports"
            / "structured"
            / case_id
            / "findings.json"
        )

        findings = load_findings(findings_path)
        packet = build_llm_input_packet(findings)

        base_prompt = render_llm_prompt_v2(packet)
        prompt = render_ollama_prompt(
            base_prompt=base_prompt,
            schema=schema,
        )

        print(
            f"[{index:02d}/{len(selected_case_ids):02d}] "
            f"{case_id} generating ..."
        )

        start = time.perf_counter()

        response = generate_ollama_candidate(
            prompt=prompt,
            config=config,
            schema=schema,
        )

        wall_time = time.perf_counter() - start

        candidate, audit, parse_error = evaluate_ollama_content(
            packet=packet,
            content=response.content,
        )

        output_root = (
            report_root
            / case_id
            / model_slug
        )

        run = write_local_llm_run(
            output_root=output_root,
            packet=packet,
            prompt=prompt,
            runtime=runtime,
            config=config,
            response=response,
            candidate=candidate,
            audit=audit,
            parse_error=parse_error,
            wall_time_seconds=wall_time,
        )

        issue_codes = (
            tuple(issue.code for issue in audit.issues)
            if audit is not None
            else ()
        )

        row = PromptV2CaseResult(
            case_id=case_id,
            v1_final_accepted=baseline_by_case.get(case_id),

            v2_parsed=candidate is not None,
            v2_accepted=run.accepted,
            v2_issue_count=(
                audit.issue_count
                if audit is not None
                else 0
            ),
            v2_issue_codes=issue_codes,

            wall_time_seconds=wall_time,
            prompt_tokens=response.prompt_eval_count,
            output_tokens=response.eval_count,

            prompt_sha256=run.prompt_sha256,
            response_sha256=run.response_sha256,
        )
        rows.append(row)

        baseline = row.v1_final_accepted

        if baseline is None:
            comparison = "v1=n/a"
        else:
            comparison = (
                f"v1={'PASS' if baseline else 'FAIL'} -> "
                f"v2={'PASS' if row.v2_accepted else 'FAIL'}"
            )

        print(
            f"  {'PASS' if row.v2_accepted else 'FAIL'}: "
            f"{comparison}, wall={wall_time:.2f} s, "
            f"prompt_tokens={row.prompt_tokens}, "
            f"output_tokens={row.output_tokens}, "
            f"issues={row.v2_issue_count}"
        )

        if audit is not None and not audit.passed:
            for issue in audit.issues:
                suffix = (
                    f" [{issue.evidence}]"
                    if issue.evidence is not None
                    else ""
                )
                print(
                    f"    {issue.code}: "
                    f"{issue.message}{suffix}"
                )
        elif parse_error is not None:
            print(f"    parse error: {parse_error}")

    summary = summarize_prompt_v2_cohort(rows)

    if args.cases is None:
        summary_path, csv_path = write_prompt_v2_cohort_outputs(
            summary,
            experiment_root,
        )
    else:
        subset_slug = "_".join(
            case_id.split("_")[-1]
            for case_id in selected_case_ids
        )
        subset_root = experiment_root / f"smoke_{subset_slug}"
        summary_path, csv_path = write_prompt_v2_cohort_outputs(
            summary,
            subset_root,
        )

    print()
    print("Prompt v2 cohort summary")
    print(
        f"JSON parse success: "
        f"{summary.v2_parse_success_count}/"
        f"{summary.case_count} "
        f"({summary.v2_parse_success_fraction:.1%})"
    )
    print(
        f"V2 acceptance:     "
        f"{summary.v2_accepted_count}/"
        f"{summary.case_count} "
        f"({summary.v2_acceptance_fraction:.1%})"
    )

    if summary.v1_accepted_count is not None:
        print(
            f"V1 final baseline: "
            f"{summary.v1_accepted_count}/"
            f"{summary.v1_comparable_case_count} "
            f"({summary.v1_acceptance_fraction:.1%})"
        )
        print(
            f"Acceptance delta:  "
            f"{summary.acceptance_delta_count:+d} cases, "
            f"{summary.acceptance_delta_fraction:+.1%}"
        )

    print(
        "Improved cases:    "
        + (
            ", ".join(summary.improved_case_ids)
            if summary.improved_case_ids
            else "none"
        )
    )
    print(
        "Regressed cases:   "
        + (
            ", ".join(summary.regressed_case_ids)
            if summary.regressed_case_ids
            else "none"
        )
    )
    print(
        "Rejected v2 cases: "
        + (
            ", ".join(summary.v2_rejected_case_ids)
            if summary.v2_rejected_case_ids
            else "none"
        )
    )
    print(
        "V2 issue codes:    "
        + (
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(
                    summary.v2_issue_code_counts.items()
                )
            )
            if summary.v2_issue_code_counts
            else "none"
        )
    )
    print(
        f"LLM wall time: mean={summary.wall_time_mean_seconds:.2f} s, "
        f"median={summary.wall_time_median_seconds:.2f} s, "
        f"range=[{summary.wall_time_min_seconds:.2f}, "
        f"{summary.wall_time_max_seconds:.2f}] s"
    )
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print()

    if summary.v2_acceptance_fraction < 1.0:
        print(
            "Block 27 COMPLETED with rejected v2 generation(s). "
            "Do not modify the frozen Block 26C auditor; inspect the new "
            "candidate(s) first."
        )
        return 2

    print(
        "Block 27 PASSED: every prospectively regenerated candidate "
        "satisfied the unchanged Block 26C deterministic gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
