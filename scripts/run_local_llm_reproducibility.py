from __future__ import annotations

import argparse
import os
import time

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.local_llm_reproducibility import (
    build_reproducibility_run,
    summarize_local_llm_reproducibility,
    write_reproducibility_outputs,
)
from vista3d_tfg.reports.llm_contract import (
    build_llm_input_packet,
    load_findings,
    render_llm_prompt,
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


DEFAULT_REPETITIONS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a frozen multi-run reproducibility study for the local "
            "Ollama report verbalizer."
        )
    )
    parser.add_argument(
        "--case",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_REPETITIONS,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.runs < 2:
        raise ValueError("--runs must be at least 2.")
    if args.runs > 20:
        raise ValueError(
            "--runs is capped at 20 for this bounded experiment."
        )

    project_root = discover_project_root()
    case_id = f"case_{args.case:05d}"

    findings_path = (
        project_root
        / "outputs"
        / "reports"
        / "structured"
        / case_id
        / "findings.json"
    )

    packet = build_llm_input_packet(
        load_findings(findings_path)
    )

    schema = candidate_json_schema()
    prompt = render_ollama_prompt(
        base_prompt=render_llm_prompt(packet),
        schema=schema,
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

    model_slug = config.model.replace(":", "_")
    output_root = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reproducibility"
        / case_id
        / model_slug
    )

    rows = []

    print("Local LLM reproducibility study")
    print(f"Case:         {case_id}")
    print(f"Provider:     Ollama local")
    print(f"Ollama:       {runtime.version}")
    print(f"Model:        {config.model}")
    print(f"Runs:         {args.runs}")
    print(f"Temperature:  {config.temperature}")
    print(f"Seed:         {config.seed}")
    print(f"Thinking:     {config.think}")
    print("Prompt/schema/configuration are frozen across runs.")
    print()

    for run_index in range(1, args.runs + 1):
        print(
            f"[{run_index:02d}/{args.runs:02d}] "
            "generating ..."
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

        run_root = (
            output_root
            / "runs"
            / f"run_{run_index:03d}"
        )

        local_run = write_local_llm_run(
            output_root=run_root,
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

        row = build_reproducibility_run(
            run_index=run_index,
            local_run=local_run,
            prompt_eval_count=response.prompt_eval_count,
            eval_count=response.eval_count,
            done_reason=response.done_reason,
        )
        rows.append(row)

        audit_label = (
            "PASS"
            if row.accepted
            else "FAIL"
        )
        candidate_hash = (
            row.canonical_candidate_sha256[:12]
            if row.canonical_candidate_sha256
            else "none"
        )

        print(
            f"  {audit_label}: "
            f"wall={row.wall_time_seconds:.2f} s, "
            f"raw={row.raw_response_sha256[:12]}, "
            f"candidate={candidate_hash}, "
            f"output_tokens={row.eval_count}"
        )

        if not row.accepted:
            if parse_error is not None:
                print(f"    parse error: {parse_error}")
            elif audit is not None:
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

    summary = summarize_local_llm_reproducibility(
        case_id=case_id,
        provider="ollama-local",
        model=config.model,
        requested_runs=args.runs,
        rows=rows,
    )

    summary_path, csv_path = write_reproducibility_outputs(
        summary,
        output_root,
    )

    print()
    print("Reproducibility summary")
    print(
        f"JSON parse success: "
        f"{summary.parse_success_count}/"
        f"{summary.completed_runs} "
        f"({summary.parse_success_fraction:.1%})"
    )
    print(
        f"Acceptance gate:   "
        f"{summary.accepted_count}/"
        f"{summary.completed_runs} "
        f"({summary.acceptance_fraction:.1%})"
    )
    print(
        f"Unique raw responses: "
        f"{summary.unique_raw_response_count}"
    )
    print(
        f"Unique canonical candidates: "
        f"{summary.unique_candidate_count}"
    )
    print(
        f"Raw exact match to run 1: "
        f"{summary.raw_match_to_first_count}/"
        f"{summary.completed_runs} "
        f"({summary.raw_match_to_first_fraction:.1%})"
    )
    print(
        f"Candidate exact match to run 1: "
        f"{summary.candidate_match_to_first_count}/"
        f"{summary.completed_runs} "
        f"({summary.candidate_match_to_first_fraction:.1%})"
    )
    print(
        f"All raw responses identical: "
        f"{summary.all_raw_responses_identical}"
    )
    print(
        f"All candidates identical: "
        f"{summary.all_candidates_identical}"
    )
    print(
        f"Wall time: mean={summary.wall_time_mean_seconds:.2f} s, "
        f"median={summary.wall_time_median_seconds:.2f} s, "
        f"range=[{summary.wall_time_min_seconds:.2f}, "
        f"{summary.wall_time_max_seconds:.2f}] s"
    )

    if summary.warm_run_count:
        print(
            f"Warm runs (2..N): "
            f"mean={summary.warm_wall_time_mean_seconds:.2f} s, "
            f"median={summary.warm_wall_time_median_seconds:.2f} s"
        )

    print(
        f"Prompt tokens range: "
        f"[{summary.prompt_eval_count_min}, "
        f"{summary.prompt_eval_count_max}]"
    )
    print(
        f"Output tokens range: "
        f"[{summary.eval_count_min}, "
        f"{summary.eval_count_max}]"
    )
    print(f"Summary: {summary_path}")
    print(f"CSV:     {csv_path}")
    print()

    if summary.acceptance_fraction < 1.0:
        print(
            "Reproducibility experiment COMPLETED with rejected "
            "generation(s). Inspect the saved audit records."
        )
        return 2

    print(
        "Reproducibility experiment PASSED: "
        "all generations satisfied the deterministic factual gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
