from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.local_llm_reporting_cohort import (
    ReportingCohortRow,
    summarize_reporting_cohort,
    write_reporting_cohort_outputs,
)
from vista3d_tfg.experiments.cyst_oracle_prompt import (
    load_prompt_manifest,
)
from vista3d_tfg.experiments.pilot30 import pilot_case_paths
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)
from vista3d_tfg.reports.llm_contract import (
    build_llm_input_packet,
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
from vista3d_tfg.reports.structured_findings import (
    build_structured_case_findings,
    write_structured_report,
)
from vista3d_tfg.workflows.user_guided_cyst import (
    run_user_guided_cyst_segmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 17-case cyst-positive end-to-end reporting cohort: "
            "oracle-point replay -> GT-free runtime workflow -> structured "
            "findings -> local LLM -> deterministic audit."
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
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help="Optional case-number subset for smoke/debug execution.",
    )
    parser.add_argument(
        "--force-workflow",
        action="store_true",
        help="Recompute user-guided VISTA outputs even if already saved.",
    )
    return parser.parse_args()


def canonical_candidate_sha256(
    candidate: dict | None,
) -> str | None:
    if candidate is None:
        return None

    payload = json.dumps(
        candidate,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    prompt_manifest_path = (
        project_root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_cyst_oracle_prompt"
        / "prompts.json"
    )
    prompts = load_prompt_manifest(
        prompt_manifest_path
    )

    if args.cases is not None:
        requested = {
            f"case_{number:05d}"
            for number in args.cases
        }
        prompts = tuple(
            prompt
            for prompt in prompts
            if prompt.case_id in requested
        )

        if not prompts:
            raise ValueError(
                "No prepared cyst prompts match requested cases."
            )

    nv_paths = runtime_paths(project_root)

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
    model_slug = config.model.replace(":", "_")

    cohort_root = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort"
        / model_slug
    )
    rows = []

    print("Local LLM cyst-positive reporting cohort")
    print(f"Cases:        {len(prompts)}")
    print(f"Model:        {config.model}")
    print(f"Ollama:       {runtime.version}")
    print(f"Temperature:  {config.temperature}")
    print(f"Seed:         {config.seed}")
    print()
    print(
        "IMPORTANT: click coordinates are frozen Block 16 GT-derived oracle "
        "points replayed as experimental user-click surrogates."
    )
    print(
        "Runtime VISTA workflow, structured reporting, and LLM generation "
        "do not read GT, but the overall cohort is NOT deployment-valid."
    )
    print()

    for index, prompt in enumerate(prompts, start=1):
        case_number = int(
            prompt.case_id.split("_")[-1]
        )
        paths = pilot_case_paths(
            project_root,
            case_number,
        )

        print(
            f"[{index:02d}/{len(prompts):02d}] "
            f"{prompt.case_id} point={prompt.point_xyz}"
        )

        workflow = run_user_guided_cyst_segmentation(
            project_root=project_root,
            case_number=case_number,
            point_xyz=prompt.point_xyz,
            nv_paths=nv_paths,
            environment_name=args.environment,
            force=args.force_workflow,
        )

        user_guided_root = (
            project_root
            / "outputs"
            / "predictions"
            / "nv_segment_ct_user_guided_cyst"
            / prompt.case_id
        )
        workflow_path = (
            user_guided_root / "workflow.json"
        )
        interactive_prediction_path = (
            user_guided_root / "prediction.nii.gz"
        )

        findings = build_structured_case_findings(
            case_id=prompt.case_id,
            image_path=paths.imaging_path,
            automatic_prediction_path=paths.prediction_path,
            workflow_path=workflow_path,
            interactive_prediction_path=interactive_prediction_path,
        )

        structured_root = (
            project_root
            / "outputs"
            / "reports"
            / "structured"
            / prompt.case_id
        )
        write_structured_report(
            findings,
            structured_root,
        )

        packet = build_llm_input_packet(
            findings.to_dict()
        )
        llm_prompt = render_ollama_prompt(
            base_prompt=render_llm_prompt(packet),
            schema=schema,
        )

        start = time.perf_counter()
        response = generate_ollama_candidate(
            prompt=llm_prompt,
            config=config,
            schema=schema,
        )
        llm_wall_time = time.perf_counter() - start

        candidate, audit, parse_error = (
            evaluate_ollama_content(
                packet=packet,
                content=response.content,
            )
        )

        llm_root = (
            project_root
            / "outputs"
            / "reports"
            / "local_llm_cohort"
            / prompt.case_id
            / model_slug
        )

        local_run = write_local_llm_run(
            output_root=llm_root,
            packet=packet,
            prompt=llm_prompt,
            runtime=runtime,
            config=config,
            response=response,
            candidate=candidate,
            audit=audit,
            parse_error=parse_error,
            wall_time_seconds=llm_wall_time,
        )

        issue_count = (
            int(audit.issue_count)
            if audit is not None
            else 0
        )

        row = ReportingCohortRow(
            case_id=prompt.case_id,
            point_xyz=prompt.point_xyz,
            laterality=(
                workflow.semantic_assignment.laterality
            ),
            class_label=(
                workflow.semantic_assignment.class_label
            ),

            workflow_volume_ml=workflow.foreground_volume_ml,
            selected_kidney_distance_mm=(
                workflow.semantic_assignment.selected_distance_mm
            ),
            laterality_margin_mm=(
                workflow.semantic_assignment.laterality_margin_mm
            ),

            llm_parsed=candidate is not None,
            llm_accepted=local_run.accepted,
            llm_issue_count=issue_count,
            llm_wall_time_seconds=llm_wall_time,
            llm_response_sha256=local_run.response_sha256,
            llm_candidate_sha256=canonical_candidate_sha256(
                local_run.candidate
            ),
        )
        rows.append(row)

        print(
            f"  workflow: "
            f"{row.laterality}/class {row.class_label}, "
            f"volume={row.workflow_volume_ml:.2f} mL"
        )
        print(
            f"  LLM: "
            f"{'PASS' if row.llm_accepted else 'FAIL'}, "
            f"wall={row.llm_wall_time_seconds:.2f} s, "
            f"issues={row.llm_issue_count}"
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

    summary = summarize_reporting_cohort(rows)

    if args.cases is None:
        summary_path, csv_path = (
            write_reporting_cohort_outputs(
                summary,
                cohort_root,
            )
        )
    else:
        summary_path = None
        csv_path = None

    print()
    print("Reporting cohort summary")
    print(
        f"Cases: {summary.case_count} "
        f"(left={summary.left_case_count}, "
        f"right={summary.right_case_count})"
    )
    print(
        f"Workflow cyst volume: "
        f"mean={summary.workflow_volume_mean_ml:.2f} mL, "
        f"median={summary.workflow_volume_median_ml:.2f} mL"
    )
    print(
        f"Selected-kidney distance: "
        f"mean={summary.selected_kidney_distance_mean_mm:.2f} mm, "
        f"median={summary.selected_kidney_distance_median_mm:.2f} mm, "
        f"max={summary.selected_kidney_distance_max_mm:.2f} mm"
    )
    print(
        f"Laterality margin: "
        f"mean={summary.laterality_margin_mean_mm:.2f} mm, "
        f"median={summary.laterality_margin_median_mm:.2f} mm, "
        f"min={summary.laterality_margin_min_mm:.2f} mm"
    )
    print(
        f"LLM JSON parse success: "
        f"{summary.llm_parse_success_count}/"
        f"{summary.case_count} "
        f"({summary.llm_parse_success_fraction:.1%})"
    )
    print(
        f"LLM acceptance gate: "
        f"{summary.llm_accepted_count}/"
        f"{summary.case_count} "
        f"({summary.llm_acceptance_fraction:.1%})"
    )
    print(
        f"LLM total audit issues: "
        f"{summary.llm_total_issue_count}"
    )

    if summary.llm_rejected_case_ids:
        print(
            "Rejected cases: "
            + ", ".join(summary.llm_rejected_case_ids)
        )
    else:
        print("Rejected cases: none")

    print(
        f"LLM wall time: "
        f"mean={summary.llm_wall_time_mean_seconds:.2f} s, "
        f"median={summary.llm_wall_time_median_seconds:.2f} s, "
        f"range=[{summary.llm_wall_time_min_seconds:.2f}, "
        f"{summary.llm_wall_time_max_seconds:.2f}] s"
    )

    if summary_path is not None:
        print(f"Summary: {summary_path}")
        print(f"CSV:     {csv_path}")

    print()

    if summary.llm_acceptance_fraction < 1.0:
        print(
            "Reporting cohort COMPLETED with rejected LLM case(s). "
            "Inspect saved audits before changing the protocol."
        )
        return 2

    print(
        "Reporting cohort PASSED: every case satisfied the frozen "
        "deterministic LLM acceptance gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
