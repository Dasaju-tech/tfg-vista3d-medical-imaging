from __future__ import annotations

import argparse
import json
import os
import time

from vista3d_tfg.config import discover_project_root
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a local structured report with Ollama and pass the "
            "candidate through the deterministic Block 23 auditor."
        )
    )
    parser.add_argument("--case", type=int, required=True)
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

    findings = load_findings(findings_path)
    packet = build_llm_input_packet(findings)

    schema = candidate_json_schema()
    base_prompt = render_llm_prompt(packet)
    prompt = render_ollama_prompt(
        base_prompt=base_prompt,
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
            f"Model {config.model!r} is not installed in Ollama. "
            "Run prepare_local_llm_runtime.py --pull first."
        )

    print("Local constrained LLM report")
    print(f"Case:         {case_id}")
    print(f"Provider:     Ollama local")
    print(f"Ollama:       {runtime.version}")
    print(f"Model:        {config.model}")
    print(f"Temperature:  {config.temperature}")
    print(f"Seed:         {config.seed}")
    print(f"Thinking:     {config.think}")
    print("Structured output: JSON Schema")
    print()

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
        project_root
        / "outputs"
        / "reports"
        / "local_llm"
        / case_id
        / config.model.replace(":", "_")
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

    print(f"Wall time:     {wall_time:.2f} s")
    print(
        f"Prompt tokens: "
        f"{response.prompt_eval_count}"
    )
    print(
        f"Output tokens: "
        f"{response.eval_count}"
    )
    print(
        f"Done reason:   "
        f"{response.done_reason}"
    )
    print()

    if candidate is None:
        print("Candidate JSON parse: FAILED")
        print(f"Parse error: {parse_error}")
    else:
        print("Candidate JSON parse: PASSED")
        print()
        print(candidate.narrative)

    print()

    if audit is None:
        print("Deterministic audit: NOT RUN")
    else:
        print(
            f"Deterministic audit: "
            f"{'PASSED' if audit.passed else 'FAILED'}"
        )
        print(f"Audit issues: {audit.issue_count}")

        for issue in audit.issues:
            suffix = (
                f" [{issue.evidence}]"
                if issue.evidence is not None
                else ""
            )
            print(
                f"  - {issue.code}: "
                f"{issue.message}{suffix}"
            )

    print()
    print(f"Run directory: {output_root}")
    print(f"Accepted: {run.accepted}")

    if not run.accepted:
        print("Local LLM report FAILED acceptance gate.")
        return 2

    print("Local LLM report PASSED acceptance gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
