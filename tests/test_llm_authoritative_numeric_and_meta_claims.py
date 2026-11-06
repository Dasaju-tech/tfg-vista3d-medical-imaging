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


def findings() -> dict:
    return {
        "schema_version": "1.0",
        "case_id": "case_00237",
        "ground_truth_used": False,
        "clinical_validation": False,
        "automatic_measurements": {
            "left_kidney_core_volume_ml": 185.40622380848856,
            "right_kidney_core_volume_ml": 137.0636190428574,
            "automatic_left_cyst_volume_ml": 104.57991446252034,
            "automatic_right_cyst_volume_ml": 106.43743120969498,
            "labels_present": [0, 5, 14, 116, 117],
        },
        "interactive_cyst": {
            "user_point_xyz": [72, 335, 203],
            "laterality": "right",
            "class_label": 117,
            "class_name": "right kidney cyst",
            "laterality_margin_mm": 81.53,
            "selected_kidney_distance_mm": 0.0,
            "predicted_volume_ml": 95.77,
            "labels_present": [0, 117, 255],
            "contains_255_unprocessed": True,
            "extra_fixture_measurement": 0.00604263676330504,
            "extra_fixture_distance": 1.184356805607788,
        },
        "evidence": {
            "fixture": {
                "path": "fixture.nii.gz",
                "sha256": "a" * 64,
            }
        },
        "limitations": list(LIMITATIONS),
    }


def candidate(narrative: str) -> LLMCandidate:
    return LLMCandidate(
        case_id="case_00237",
        narrative=narrative,
        limitations=LIMITATIONS,
    )


def test_exact_authoritative_precision_is_not_hallucination() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The automatic left kidney-core volume is "
            "185.40622380848856 mL."
        ),
    )

    assert not any(
        issue.code == "unsupported_numeric_value"
        for issue in result.issues
    )


def test_authoritative_label_255_is_not_hallucination() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The stored interactive output includes label 255."
        ),
    )

    assert not any(
        issue.code == "unsupported_numeric_value"
        for issue in result.issues
    )


def test_nested_authoritative_numeric_value_is_allowed() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The stored value is 0.00604263676330504."
        ),
    )

    assert not any(
        issue.code == "unsupported_numeric_value"
        for issue in result.issues
    )


def test_meta_clinical_limitation_is_not_a_clinical_claim() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "No diagnostic, malignancy, staging, or treatment claim "
            "is generated."
        ),
    )

    assert not any(
        issue.code in {
            "diagnostic_claim",
            "malignancy_claim",
            "staging_claim",
            "treatment_claim",
        }
        for issue in result.issues
    )


def test_no_evidence_of_malignancy_remains_prohibited() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "There is no evidence of malignancy."
        ),
    )

    assert any(
        issue.code == "malignancy_claim"
        for issue in result.issues
    )


def test_negative_diagnostic_interpretation_meta_statement_is_allowed() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "No diagnostic interpretation is provided."
        ),
    )

    assert not any(
        issue.code == "diagnostic_claim"
        for issue in result.issues
    )


def test_real_cyst_alignment_inference_remains_rejected() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The selected kidney distance is 0.00 mm, indicating the "
            "interactive cyst is spatially aligned with the selected kidney."
        ),
    )

    assert any(
        issue.code == "distance_relation_mismatch"
        for issue in result.issues
    )
