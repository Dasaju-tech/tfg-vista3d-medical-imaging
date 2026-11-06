from __future__ import annotations

from vista3d_tfg.reports.llm_contract import build_llm_input_packet
from vista3d_tfg.reports.llm_prompt_v2 import (
    PROMPT_VERSION,
    build_minimal_verbalization_facts,
    render_llm_prompt_v2,
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
        },
        "evidence": {
            "input_image": {
                "path": r"C:\private\case_00237\imaging.nii.gz",
                "sha256": "a" * 64,
            }
        },
        "limitations": list(LIMITATIONS),
    }


def test_minimal_facts_are_rendered_at_reporting_precision() -> None:
    packet = build_llm_input_packet(findings())
    facts = build_minimal_verbalization_facts(packet)

    assert facts.left_kidney_core_volume_ml == "185.41"
    assert facts.automatic_left_cyst_volume_ml == "104.58"
    assert facts.selected_kidney_distance_mm == "0.00"
    assert facts.interactive_cyst_laterality == "right"
    assert facts.interactive_cyst_class_label == 117
    assert (
        facts.user_point_x,
        facts.user_point_y,
        facts.user_point_z,
    ) == (72, 335, 203)


def test_prompt_v2_does_not_expose_raw_findings_json() -> None:
    packet = build_llm_input_packet(findings())
    prompt = render_llm_prompt_v2(packet)

    assert PROMPT_VERSION in prompt
    assert "AUTHORITATIVE_FINDINGS_JSON" not in prompt
    assert "sha256" not in prompt
    assert r"C:\private" not in prompt
    assert "255" not in prompt
    assert "185.40622380848856" not in prompt


def test_prompt_v2_contains_only_intended_reporting_values() -> None:
    packet = build_llm_input_packet(findings())
    prompt = render_llm_prompt_v2(packet)

    expected = (
        "185.41",
        "137.06",
        "104.58",
        "106.44",
        "95.77",
        "0.00",
        "81.53",
        "117",
        "(72, 335, 203)",
    )

    for value in expected:
        assert value in prompt


def test_prompt_v2_explicitly_blocks_case_00237_failure_mode() -> None:
    packet = build_llm_input_packet(findings())
    prompt = render_llm_prompt_v2(packet)

    assert "USER-PROVIDED CLICK" in prompt
    assert "does NOT describe" in prompt
    assert "spatial relationship" in prompt
    assert "Never infer a cyst-to-kidney spatial relationship" in prompt


def test_prompt_v2_preserves_required_limitations() -> None:
    packet = build_llm_input_packet(findings())
    prompt = render_llm_prompt_v2(packet)

    for limitation in LIMITATIONS:
        assert limitation in prompt


def test_inconsistent_laterality_and_class_is_rejected() -> None:
    broken = findings()
    broken["interactive_cyst"]["class_label"] = 116
    packet = build_llm_input_packet(broken)

    try:
        build_minimal_verbalization_facts(packet)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError(
            "Expected inconsistent laterality/class to be rejected."
        )
