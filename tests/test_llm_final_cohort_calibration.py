from __future__ import annotations

from vista3d_tfg.reports.llm_contract import (
    LLMCandidate,
    audit_llm_candidate,
    build_llm_input_packet,
)


LIMITATIONS = (
    "Segmentation-derived research measurements only.",
    "No ground-truth annotation is used by this reporting workflow.",
    "No diagnostic, malignancy, staging, or treatment claim is generated.",
    "Interactive cyst output depends on a user-provided spatial click.",
    "The workflow has not been clinically validated.",
)


def base_findings(case_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "ground_truth_used": False,
        "clinical_validation": False,
        "automatic_measurements": {
            "left_kidney_core_volume_ml": 334.46288108825684,
            "right_kidney_core_volume_ml": 400.9692192077637,
            "automatic_left_cyst_volume_ml": 0.11587142944335938,
            "automatic_right_cyst_volume_ml": 21.343517303466797,
            "labels_present": [0, 5, 14, 116, 117],
        },
        "interactive_cyst": {
            "user_point_xyz": [11, 313, 317],
            "laterality": "left",
            "class_label": 116,
            "class_name": "left kidney cyst",
            "laterality_margin_mm": 113.14690990260458,
            "selected_kidney_distance_mm": 0.87890625,
            "predicted_volume_ml": 5.612039566040039,
            "labels_present": [0, 116, 255],
            "contains_255_unprocessed": True,
        },
        "evidence": {
            "fixture": {
                "path": "fixture.nii.gz",
                "sha256": "a" * 64,
            }
        },
        "limitations": list(LIMITATIONS),
    }


def candidate(case_id: str, narrative: str) -> LLMCandidate:
    return LLMCandidate(
        case_id=case_id,
        narrative=narrative,
        limitations=LIMITATIONS,
    )


def test_case_00275_negative_generate_claims_is_meta_reporting() -> None:
    findings = base_findings("case_00275")
    findings["interactive_cyst"]["laterality"] = "left"
    findings["interactive_cyst"]["class_label"] = 116

    packet = build_llm_input_packet(findings)

    result = audit_llm_candidate(
        packet,
        candidate(
            "case_00275",
            "The workflow uses segmentation-derived measurements and does "
            "not rely on ground-truth annotations or generate any diagnostic, "
            "malignancy, staging, or treatment claims."
        ),
    )

    assert result.passed is True
    assert result.issue_count == 0


def test_patient_level_no_malignancy_statement_still_fails() -> None:
    packet = build_llm_input_packet(
        base_findings("case_00275")
    )

    result = audit_llm_candidate(
        packet,
        candidate(
            "case_00275",
            "There is no malignancy."
        ),
    )

    assert any(
        issue.code == "malignancy_claim"
        for issue in result.issues
    )


def test_case_00418_four_decimal_rounding_is_authoritative() -> None:
    packet = build_llm_input_packet(
        base_findings("case_00418")
    )

    result = audit_llm_candidate(
        packet,
        candidate(
            "case_00418",
            "The automatic left kidney cyst volume is 0.1159 mL, and the "
            "automatic right kidney cyst volume is 21.34 mL. The left kidney "
            "core volume is 334.46 mL, and the right kidney core volume is "
            "400.97 mL. The interactive cyst class label is 116, identified "
            "as a left kidney cyst. The interactive cyst predicted volume is "
            "5.61 mL. The selected kidney distance from the user click is "
            "0.88 mm. The interactive cyst laterality margin is 113.15 mm. "
            "The user provided spatial click occurred at voxel coordinates "
            "(11, 313, 317)."
        ),
    )

    assert result.passed is True
    assert result.issue_count == 0


def test_non_authoritative_nearby_number_still_fails() -> None:
    packet = build_llm_input_packet(
        base_findings("case_00418")
    )

    result = audit_llm_candidate(
        packet,
        candidate(
            "case_00418",
            "The automatic left kidney cyst volume is 0.1165 mL."
        ),
    )

    assert any(
        issue.code == "unsupported_numeric_value"
        for issue in result.issues
    )


def test_case_00237_alignment_overreach_still_fails() -> None:
    findings = base_findings("case_00237")
    findings["interactive_cyst"]["laterality"] = "right"
    findings["interactive_cyst"]["class_label"] = 117
    findings["interactive_cyst"]["class_name"] = "right kidney cyst"
    findings["interactive_cyst"]["selected_kidney_distance_mm"] = 0.0

    packet = build_llm_input_packet(findings)

    result = audit_llm_candidate(
        packet,
        candidate(
            "case_00237",
            "The selected kidney distance is 0.00 mm, indicating the "
            "interactive cyst is spatially aligned with the selected kidney."
        ),
    )

    assert result.passed is False
    assert any(
        issue.code == "distance_relation_mismatch"
        for issue in result.issues
    )
