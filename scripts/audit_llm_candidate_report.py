from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reports.llm_contract import (
    audit_llm_candidate,
    build_llm_input_packet,
    load_findings,
    parse_candidate_json,
    write_audit_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit an LLM-generated JSON report candidate against the "
            "authoritative structured findings."
        )
    )
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        required=True,
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

    packet = build_llm_input_packet(
        load_findings(findings_path)
    )

    candidate = parse_candidate_json(
        args.candidate.read_text(encoding="utf-8")
    )

    result = audit_llm_candidate(
        packet,
        candidate,
    )

    output_path = (
        project_root
        / "outputs"
        / "reports"
        / "llm_contract"
        / case_id
        / "candidate_audit.json"
    )
    write_audit_result(result, output_path)

    print("LLM candidate report audit")
    print(f"Case: {case_id}")
    print(f"Passed: {result.passed}")
    print(f"Issues: {result.issue_count}")

    for issue in result.issues:
        suffix = (
            f" [{issue.evidence}]"
            if issue.evidence is not None
            else ""
        )
        print(
            f"  - {issue.code}: "
            f"{issue.message}{suffix}"
        )

    print(f"Audit JSON: {output_path}")

    if not result.passed:
        print("LLM candidate report FAILED audit.")
        return 2

    print("LLM candidate report PASSED audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
