from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reports.structured_findings import (
    build_structured_case_findings,
    render_deterministic_report,
    write_structured_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GT-free structured findings JSON and deterministic "
            "research report from automatic renal and user-guided cyst "
            "segmentations."
        )
    )
    parser.add_argument(
        "--case",
        type=int,
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    case_id = f"case_{args.case:05d}"

    image_path = (
        project_root
        / "data"
        / "raw"
        / "kits23"
        / case_id
        / "imaging.nii.gz"
    )
    automatic_prediction_path = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
        / case_id
        / "prediction.nii.gz"
    )
    user_guided_root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_user_guided_cyst"
        / case_id
    )
    workflow_path = user_guided_root / "workflow.json"
    interactive_prediction_path = (
        user_guided_root / "prediction.nii.gz"
    )

    findings = build_structured_case_findings(
        case_id=case_id,
        image_path=image_path,
        automatic_prediction_path=automatic_prediction_path,
        workflow_path=workflow_path,
        interactive_prediction_path=interactive_prediction_path,
    )

    output_root = (
        project_root
        / "outputs"
        / "reports"
        / "structured"
        / case_id
    )

    json_path, text_path = write_structured_report(
        findings,
        output_root,
    )

    print(render_deterministic_report(findings), end="")
    print(f"Structured JSON: {json_path}")
    print(f"Deterministic report: {text_path}")
    print("Ground truth used: NO")
    print("LLM used: NO")
    print("Structured case reporting PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
