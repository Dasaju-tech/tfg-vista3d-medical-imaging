from __future__ import annotations

import json

import pytest

from vista3d_tfg.reports.llm_contract import (
    build_llm_input_packet,
)
from vista3d_tfg.reports.ollama_local import (
    DEFAULT_MODEL,
    OllamaGenerationConfig,
    build_chat_payload,
    candidate_json_schema,
    evaluate_ollama_content,
    parse_ollama_response,
    probe_ollama,
    render_ollama_prompt,
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


def good_candidate_json() -> str:
    return json.dumps(
        {
            "case_id": "case_00002",
            "narrative": (
                "The user-guided right kidney cyst mask "
                "(class 117) had a predicted volume of 31.92 mL."
            ),
            "limitations": list(LIMITATIONS),
        }
    )


def test_chat_payload_freezes_generation_settings() -> None:
    config = OllamaGenerationConfig()
    schema = candidate_json_schema()

    payload = build_chat_payload(
        prompt="fixture",
        config=config,
        schema=schema,
    )

    assert payload["model"] == DEFAULT_MODEL
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"] == schema
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["seed"] == 20260828
    assert payload["options"]["num_predict"] == 700


def test_nonzero_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature=0"):
        build_chat_payload(
            prompt="fixture",
            config=OllamaGenerationConfig(
                temperature=0.2,
            ),
        )


def test_prompt_contains_schema() -> None:
    prompt = render_ollama_prompt(
        base_prompt="base",
        schema=candidate_json_schema(),
    )

    assert "OUTPUT_JSON_SCHEMA" in prompt
    assert '"additionalProperties": false' in prompt
    assert "Do not use Markdown fences" in prompt


def test_parse_ollama_response() -> None:
    response = parse_ollama_response(
        {
            "model": DEFAULT_MODEL,
            "created_at": "2026-08-28T10:00:00Z",
            "message": {
                "role": "assistant",
                "content": good_candidate_json(),
            },
            "done": True,
            "done_reason": "stop",
            "total_duration": 100,
            "load_duration": 10,
            "prompt_eval_count": 200,
            "prompt_eval_duration": 20,
            "eval_count": 50,
            "eval_duration": 30,
        }
    )

    assert response.model == DEFAULT_MODEL
    assert response.done is True
    assert response.eval_count == 50
    assert json.loads(response.content)["case_id"] == "case_00002"


def test_missing_message_content_is_rejected() -> None:
    with pytest.raises(ValueError, match="no textual content"):
        parse_ollama_response(
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                }
            }
        )


def test_good_local_content_passes_existing_auditor() -> None:
    packet = build_llm_input_packet(findings())

    candidate, audit, parse_error = evaluate_ollama_content(
        packet=packet,
        content=good_candidate_json(),
    )

    assert parse_error is None
    assert candidate is not None
    assert audit is not None
    assert audit.passed is True


def test_hallucinated_local_content_is_rejected() -> None:
    packet = build_llm_input_packet(findings())

    content = json.dumps(
        {
            "case_id": "case_00002",
            "narrative": (
                "The right kidney cyst had a predicted volume "
                "of 45.00 mL."
            ),
            "limitations": list(LIMITATIONS),
        }
    )

    candidate, audit, parse_error = evaluate_ollama_content(
        packet=packet,
        content=content,
    )

    assert parse_error is None
    assert candidate is not None
    assert audit is not None
    assert audit.passed is False
    assert any(
        issue.code == "unsupported_numeric_value"
        for issue in audit.issues
    )


def test_invalid_json_is_recorded_as_parse_failure() -> None:
    packet = build_llm_input_packet(findings())

    candidate, audit, parse_error = evaluate_ollama_content(
        packet=packet,
        content="not json",
    )

    assert candidate is None
    assert audit is None
    assert parse_error is not None


def test_probe_ollama_uses_version_and_tags() -> None:
    calls = []

    def fake_request(
        method: str,
        url: str,
        payload,
        timeout: float,
    ):
        calls.append((method, url))

        if url.endswith("/api/version"):
            return {"version": "0.99.0"}

        if url.endswith("/api/tags"):
            return {
                "models": [
                    {
                        "name": "qwen3:4b-instruct",
                        "model": "qwen3:4b-instruct",
                    }
                ]
            }

        raise AssertionError(url)

    runtime = probe_ollama(
        requested_model="qwen3:4b-instruct",
        request_json=fake_request,
    )

    assert runtime.version == "0.99.0"
    assert runtime.requested_model_present is True
    assert len(calls) == 2
