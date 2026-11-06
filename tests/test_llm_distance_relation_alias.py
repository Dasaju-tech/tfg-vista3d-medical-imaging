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


def findings(
    *,
    selected_distance: float = 0.0,
) -> dict:
    return {
        "schema_version": "1.0",
        "case_id": "case_00479",
        "ground_truth_used": False,
        "clinical_validation": False,
        "automatic_measurements": {
            "left_kidney_core_volume_ml": 200.0,
            "right_kidney_core_volume_ml": 210.0,
            "automatic_left_cyst_volume_ml": 10.0,
            "automatic_right_cyst_volume_ml": 20.0,
            "labels_present": [0, 5, 14, 116, 117],
        },
        "interactive_cyst": {
            "user_point_xyz": [72, 335, 203],
            "laterality": "right",
            "class_label": 117,
            "class_name": "right kidney cyst",
            "laterality_margin_mm": 81.53,
            "selected_kidney_distance_mm": selected_distance,
            "predicted_volume_ml": 95.77,
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


def candidate(narrative: str) -> LLMCandidate:
    return LLMCandidate(
        case_id="case_00479",
        narrative=narrative,
        limitations=LIMITATIONS,
    )


def test_selected_kidney_distance_field_alias_is_valid() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The selected kidney distance is 0.00 mm, and the laterality "
            "margin is 81.53 mm."
        ),
    )

    assert result.passed is True
    assert result.issue_count == 0


def test_selected_kidney_distance_alias_with_nonzero_value_is_valid() -> None:
    packet = build_llm_input_packet(
        findings(selected_distance=11.52)
    )

    result = audit_llm_candidate(
        packet,
        candidate(
            "The selected kidney distance is 11.52 mm."
        ),
    )

    assert result.passed is True


def test_explicit_click_to_kidney_relation_is_valid() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The user click is 0.00 mm from the selected automatic kidney."
        ),
    )

    assert result.passed is True


def test_cyst_to_kidney_relation_is_still_rejected() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The distance between the selected kidney and the interactive "
            "cyst is 0.00 mm."
        ),
    )

    assert result.passed is False
    assert any(
        issue.code == "distance_relation_mismatch"
        for issue in result.issues
    )


def test_cyst_mask_to_kidney_relation_is_still_rejected() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The cyst mask is 0.00 mm from the selected kidney."
        ),
    )

    assert result.passed is False
    assert any(
        issue.code == "distance_relation_mismatch"
        for issue in result.issues
    )
