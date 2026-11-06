from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reports.llm_contract import (
    audit_llm_candidate,
    build_llm_input_packet,
    load_findings,
    parse_candidate_json,
)


DEFAULT_MODEL = "qwen3:4b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-audit the saved Block 26 local-LLM cohort without calling "
            "VISTA or the language model."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    model_slug = args.model.replace(":", "_")

    original_csv = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort"
        / model_slug
        / "cases.csv"
    )

    if not original_csv.is_file():
        raise FileNotFoundError(
            f"Block 26 cohort CSV not found: {original_csv}"
        )

    with original_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        original_rows = list(csv.DictReader(stream))

    rows = []
    issue_counter: Counter[str] = Counter()

    print("Block 26 saved-candidate re-audit")
    print(f"Model: {args.model}")
    print(f"Cases: {len(original_rows)}")
    print("VISTA called: NO")
    print("LLM called: NO")
    print()

    for original in original_rows:
        case_id = original["case_id"]

        findings_path = (
            project_root
            / "outputs"
            / "reports"
            / "structured"
            / case_id
            / "findings.json"
        )
        candidate_path = (
            project_root
            / "outputs"
            / "reports"
            / "local_llm_cohort"
            / case_id
            / model_slug
            / "candidate.json"
        )
        audit_path = (
            project_root
            / "outputs"
            / "reports"
            / "local_llm_cohort"
            / case_id
            / model_slug
            / "audit_rechecked.json"
        )

        packet = build_llm_input_packet(
            load_findings(findings_path)
        )
        candidate = parse_candidate_json(
            candidate_path.read_text(encoding="utf-8")
        )
        audit = audit_llm_candidate(
            packet,
            candidate,
        )

        audit_path.write_text(
            json.dumps(
                audit.to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        old_accepted = (
            str(original["llm_accepted"]).strip().lower()
            == "true"
        )
        status = "PASS" if audit.passed else "FAIL"

        codes = tuple(
            issue.code
            for issue in audit.issues
        )
        issue_counter.update(codes)

        rows.append(
            {
                "case_id": case_id,
                "original_accepted": old_accepted,
                "reaudit_accepted": audit.passed,
                "recovered_by_auditor_calibration": (
                    (not old_accepted) and audit.passed
                ),
                "issue_count": audit.issue_count,
                "issue_codes": ";".join(codes),
            }
        )

        print(
            f"{case_id}: {status}, "
            f"issues={audit.issue_count}, "
            f"original={'PASS' if old_accepted else 'FAIL'}"
        )

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

    passed = sum(
        row["reaudit_accepted"]
        for row in rows
    )
    recovered = sum(
        row["recovered_by_auditor_calibration"]
        for row in rows
    )
    persistent = tuple(
        row["case_id"]
        for row in rows
        if not row["reaudit_accepted"]
    )

    output_root = (
        project_root
        / "outputs"
        / "experiments"
        / "local_llm_reporting_cohort"
        / model_slug
    )
    json_path = output_root / "reaudit_summary.json"
    csv_path = output_root / "reaudit_cases.csv"

    json_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "analysis": "block26_saved_candidate_reaudit",
                "generation_repeated": False,
                "vista_repeated": False,
                "auditor_change": (
                    "Accept canonical 'selected kidney distance' as a "
                    "field-preserving alias while continuing to reject "
                    "cyst/lesion/mask-to-kidney reinterpretations."
                ),
                "case_count": len(rows),
                "accepted_count": passed,
                "acceptance_fraction": passed / len(rows),
                "recovered_case_count": recovered,
                "persistent_rejected_case_ids": persistent,
                "issue_code_counts": dict(issue_counter),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Re-audit summary")
    print(
        f"Acceptance: {passed}/{len(rows)} "
        f"({passed / len(rows):.1%})"
    )
    print(
        f"Recovered by auditor calibration: "
        f"{recovered}"
    )
    print(
        "Issue codes: "
        + (
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(issue_counter.items())
            )
            if issue_counter
            else "none"
        )
    )
    print(
        "Persistent rejected cases: "
        + (
            ", ".join(persistent)
            if persistent
            else "none"
        )
    )
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")

    if persistent:
        print(
            "Saved-candidate re-audit COMPLETED with persistent "
            "rejection(s)."
        )
        return 2

    print(
        "Saved-candidate re-audit PASSED: all stored Block 26 "
        "candidates satisfy the calibrated deterministic gate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
