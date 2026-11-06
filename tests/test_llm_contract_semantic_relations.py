from __future__ import annotations

from vista3d_tfg.reports.llm_contract import (
    LLMCandidate,
    audit_llm_candidate,
    build_llm_input_packet,
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


def candidate(narrative: str) -> LLMCandidate:
    return LLMCandidate(
        case_id="case_00002",
        narrative=narrative,
        limitations=LIMITATIONS,
    )


def test_automatic_bilateral_cyst_mentions_do_not_trigger_interactive_laterality() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The automatic left cyst volume is 23.90 mL, and the automatic "
            "right cyst volume is 22.95 mL. The interactive cyst has "
            "laterality of right and class label of 117."
        ),
    )

    assert not any(
        issue.code == "cyst_laterality_mismatch"
        for issue in result.issues
    )
    assert not any(
        issue.code == "cyst_class_mismatch"
        for issue in result.issues
    )


def test_automatic_left_cyst_measurement_is_allowed_with_right_target() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The automatic left cyst volume is 23.90 mL. "
            "The user-guided right kidney cyst uses class 117 and has "
            "a predicted volume of 31.92 mL."
        ),
    )

    assert result.passed is True


def test_wrong_explicit_interactive_laterality_still_fails() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The interactive cyst has laterality of left and class 117."
        ),
    )

    assert any(
        issue.code == "cyst_laterality_mismatch"
        for issue in result.issues
    )


def test_wrong_interactive_class_still_fails() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The user-guided right kidney cyst was reported as class 116."
        ),
    )

    assert any(
        issue.code == "cyst_class_mismatch"
        for issue in result.issues
    )


def test_click_to_kidney_distance_relation_passes() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The user click was 12.00 mm from the selected automatic kidney."
        ),
    )

    assert result.passed is True


def test_cyst_to_kidney_distance_relabel_is_rejected() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The distance between the selected kidney and the interactive "
            "cyst is 12.00 mm."
        ),
    )

    assert any(
        issue.code == "distance_relation_mismatch"
        for issue in result.issues
    )


def test_prompt_explains_distance_relation() -> None:
    packet = build_llm_input_packet(findings())
    prompt = render_llm_prompt(packet)

    assert "distance from the USER CLICK" in prompt
    assert "never describe it as a cyst-to-kidney distance" in prompt

def test_first_qwen_response_is_rejected_for_distance_semantics() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The left kidney core volume is 234.68 mL, and the right kidney "
            "core volume is 231.06 mL. The automatic left cyst volume is "
            "23.90 mL, and the automatic right cyst volume is 22.95 mL. "
            "The interactive cyst predicted volume is 31.92 mL, with a "
            "class label of 117, laterality of right, and a laterality "
            "margin of 123.53 mm. The distance between the selected kidney "
            "and the interactive cyst is 12.00 mm. The user-provided "
            "spatial click is at voxel coordinates (68, 372, 151)."
        ),
    )

    assert result.passed is False
    assert any(
        issue.code == "distance_relation_mismatch"
        for issue in result.issues
    )
    assert not any(
        issue.code == "cyst_laterality_mismatch"
        for issue in result.issues
    )


def test_second_qwen_response_passes_semantic_audit() -> None:
    packet = build_llm_input_packet(findings())

    result = audit_llm_candidate(
        packet,
        candidate(
            "The automatic left kidney core volume is 234.68 mL, and the "
            "right kidney core volume is 231.06 mL. The automatic left cyst "
            "volume is 23.90 mL, and the automatic right cyst volume is "
            "22.95 mL. The interactive cyst class label is 117, identified "
            "as a right kidney cyst. The interactive cyst predicted volume "
            "is 31.92 mL. The selected kidney distance from the user click "
            "is 12.00 mm. The interactive cyst laterality margin is "
            "123.53 mm. The user provided spatial click occurred at voxel "
            "coordinates (68, 372, 151)."
        ),
    )

    assert result.passed is True
    assert result.issue_count == 0

