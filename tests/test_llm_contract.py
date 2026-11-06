from __future__ import annotations

import json

import pytest

from vista3d_tfg.reports.llm_contract import (
    LLMCandidate,
    audit_llm_candidate,
    build_llm_input_packet,
    parse_candidate_json,
    render_llm_prompt,
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
        "case_id": "case_00002",
        "ground_truth_used": False,
        "clinical_validation": False,
        "automatic_measurements": {
            "left_kidney_core_volume_ml": 234.68,
            "right_kidney_core_volume_ml": 231.06,
            "automatic_left_cyst_volume_ml": 23.90,
            "automatic_right_cyst_volume_ml": 22.95,
            "labels_present": [0, 5, 14, 116, 117],
        },
        "interactive_cyst": {
            "user_point_xyz": [68, 372, 151],
            "laterality": "right",
            "class_label": 117,
            "class_name": "right kidney cyst",
            "laterality_margin_mm": 123.53,
            "selected_kidney_distance_mm": 12.00,
            "predicted_volume_ml": 31.92,
            "labels_present": [0, 117, 255],
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


def valid_candidate() -> LLMCandidate:
    return LLMCandidate(
        case_id="case_00002",
        narrative=(
            "Automatic renal segmentation produced left and right "
            "kidney-core volumes of 234.68 mL and 231.06 mL, respectively. "
            "A user-guided right kidney cyst mask (class 117) had a "
            "predicted volume of 31.92 mL. The click was 12.00 mm from "
            "the selected automatic kidney with a laterality margin of "
            "123.53 mm."
        ),
        limitations=LIMITATIONS,
    )


def test_packet_contains_authoritative_facts() -> None:
    packet = build_llm_input_packet(findings())

    keys = {fact.key for fact in packet.allowed_facts}

    assert packet.case_id == "case_00002"
    assert "interactive_cyst_predicted_volume_ml" in keys
    assert "interactive_cyst_class_label" in keys
    assert packet.required_limitations == LIMITATIONS


def test_prompt_forbids_diagnostic_inference() -> None:
    packet = build_llm_input_packet(findings())
    prompt = render_llm_prompt(packet)

    assert "Do not add clinical interpretation or diagnosis" in prompt
    assert "31.92" in prompt
    assert "case_00002" in prompt


def test_valid_candidate_passes() -> None:
    packet = build_llm_input_packet(findings())
    result = audit_llm_candidate(
        packet,
        valid_candidate(),
    )

    assert result.passed is True
    assert result.issue_count == 0


def test_changed_numeric_value_fails() -> None:
    packet = build_llm_input_packet(findings())
    candidate = LLMCandidate(
        case_id="case_00002",
        narrative=(
            "The user-guided right kidney cyst volume was 45.00 mL."
        ),
        limitations=LIMITATIONS,
    )

    result = audit_llm_candidate(packet, candidate)

    assert result.passed is False
    assert any(
        issue.code == "unsupported_numeric_value"
        for issue in result.issues
    )


def test_wrong_cyst_laterality_fails() -> None:
    packet = build_llm_input_packet(findings())
    candidate = LLMCandidate(
        case_id="case_00002",
        narrative=(
            "The left kidney cyst had a predicted volume of 31.92 mL."
        ),
        limitations=LIMITATIONS,
    )

    result = audit_llm_candidate(packet, candidate)

    assert result.passed is False
    assert any(
        issue.code == "cyst_laterality_mismatch"
        for issue in result.issues
    )


def test_wrong_class_fails() -> None:
    packet = build_llm_input_packet(findings())
    candidate = LLMCandidate(
        case_id="case_00002",
        narrative=(
            "The right kidney cyst was reported as class 116."
        ),
        limitations=LIMITATIONS,
    )

    result = audit_llm_candidate(packet, candidate)

    assert result.passed is False
    assert any(
        issue.code == "cyst_class_mismatch"
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "phrase,code",
    [
        ("The lesion is malignant.", "malignancy_claim"),
        ("This represents cancer.", "cancer_claim"),
        ("Biopsy is recommended.", "biopsy_claim"),
        ("Stage 2 disease is present.", "staging_claim"),
    ],
)
def test_forbidden_clinical_claims_fail(
    phrase: str,
    code: str,
) -> None:
    packet = build_llm_input_packet(findings())
    candidate = LLMCandidate(
        case_id="case_00002",
        narrative=phrase,
        limitations=LIMITATIONS,
    )

    result = audit_llm_candidate(packet, candidate)

    assert result.passed is False
    assert any(
        issue.code == code
        for issue in result.issues
    )


def test_missing_limitation_fails() -> None:
    packet = build_llm_input_packet(findings())
    candidate = LLMCandidate(
        case_id="case_00002",
        narrative="The interactive predicted volume was 31.92 mL.",
        limitations=LIMITATIONS[:-1],
    )

    result = audit_llm_candidate(packet, candidate)

    assert result.passed is False
    assert any(
        issue.code == "limitations_mismatch"
        for issue in result.issues
    )


def test_candidate_json_requires_exact_keys() -> None:
    text = json.dumps(
        {
            "case_id": "case_00002",
            "narrative": "Volume 31.92 mL.",
            "limitations": list(LIMITATIONS),
        }
    )

    candidate = parse_candidate_json(text)

    assert candidate.case_id == "case_00002"

    with pytest.raises(ValueError, match="exactly"):
        parse_candidate_json(
            json.dumps(
                {
                    "case_id": "case_00002",
                    "narrative": "Volume 31.92 mL.",
                    "limitations": list(LIMITATIONS),
                    "diagnosis": "invented",
                }
            )
        )


def test_gt_or_clinical_findings_are_rejected() -> None:
    bad = findings()
    bad["ground_truth_used"] = True

    with pytest.raises(ValueError, match="ground_truth_used=false"):
        build_llm_input_packet(bad)

    bad = findings()
    bad["clinical_validation"] = True

    with pytest.raises(ValueError, match="clinical_validation=false"):
        build_llm_input_packet(bad)
