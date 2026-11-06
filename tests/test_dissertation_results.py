from __future__ import annotations

from vista3d_tfg.reports.dissertation_results import (
    FROZEN_RESULTS,
    build_discussion_outline,
    build_results_outline,
    latex_escape,
    render_main_results_table,
    render_results_macros,
)


def by_key(key: str):
    return next(
        result
        for result in FROZEN_RESULTS
        if result.key == key
    )


def test_frozen_catalog_contains_core_segmentation_results() -> None:
    assert by_key("pilot30_renal_dice_mean").value == 0.9127
    assert by_key("automatic_cyst_positive_dice_mean").value == 0.2198
    assert by_key("single_click_oracle_dice_mean").value == 0.4417
    assert by_key("semantic_point_dice_mean").value == 0.4692


def test_frozen_catalog_contains_final_llm_results() -> None:
    assert float(by_key("llm_prompt_v1_acceptance").value) == 16 / 17
    assert float(by_key("llm_prompt_v2_acceptance").value) == 1.0


def test_main_table_is_valid_booktabs_fragment() -> None:
    rendered = render_main_results_table(
        FROZEN_RESULTS
    )

    assert r"\begin{table}" in rendered
    assert r"\toprule" in rendered
    assert r"\bottomrule" in rendered
    assert r"\label{tab:main-results}" in rendered
    assert "0.9127" in rendered
    assert "0.2198" in rendered


def test_results_macros_keep_values_centralized() -> None:
    rendered = render_results_macros(
        FROZEN_RESULTS
    )

    assert r"\newcommand{\PilotRenalDice}{0.9127}" in rendered
    assert r"\newcommand{\LLMVOneAcceptance}{94.1\%}" in rendered
    assert r"\newcommand{\LLMVTwoAcceptance}{100.0\%}" in rendered


def test_results_outline_contains_methodological_roles() -> None:
    rendered = build_results_outline(
        FROZEN_RESULTS
    )

    assert "Automatic renal segmentation" in rendered
    assert "Interactive cyst segmentation" in rendered
    assert "Local LLM reporting" in rendered
    assert "not deployment-valid" in rendered


def test_discussion_outline_preserves_clinical_boundary() -> None:
    rendered = build_discussion_outline()

    assert "does not constitute clinical validation" in rendered
    assert "oracle" in rendered
    assert "semantic overreach" in rendered


def test_latex_escape_handles_special_characters() -> None:
    assert latex_escape("A&B_1") == r"A\&B\_1"
