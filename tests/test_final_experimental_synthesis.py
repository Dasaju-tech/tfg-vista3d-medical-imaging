from __future__ import annotations

import pytest

from vista3d_tfg.analysis.final_experimental_synthesis import (
    synthesize_final_experimental_results,
)


def _artifacts() -> dict:
    return {
        "interaction": {
            "key_results": {
                "oracle_cohort": {
                    "tumor_dice_mean": 0.7557,
                    "tumor_dice_median": 0.7802,
                    "tumor_coverage_gain_mean": 0.521,
                    "hec_hd95_reduction_mean_mm": 29.31,
                    "hd95_improved_count": 9,
                },
                "click_robustness": {},
                "safe_region_generalization": {},
                "relative_depth": {
                    "paired_tumor_dice_delta_mean": -0.0236,
                    "paired_tumor_dice_delta_95_ci": [
                        -0.1014,
                        0.0444,
                    ],
                    "upper_middle_delta_mean": -0.1627,
                },
                "dual_point": {
                    "paired_tumor_dice_delta_mean": 0.0153,
                    "paired_tumor_dice_delta_95_ci": [
                        -0.0325,
                        0.0594,
                    ],
                    "upper_middle_delta_mean": -0.0285,
                },
            },
            "final_decision": {
                "oracle_prompt_engineering_status": "closed",
                "preferred_interaction_concept": (
                    "single well-interior positive tumor point"
                ),
            },
        },
        "manual_geometry": {
            "geometry_contract_valid": True,
            "tumor_hit_count": 0,
        },
        "manual_inference": {
            "case_count": 12,
            "localization_audit": {
                "any_tumor_hit_count": 0,
                "target_component_hit_count": 0,
            },
            "overall": {
                "manual": {
                    "tumor_dice": {
                        "mean": 0.0117,
                        "median": 0.0,
                    }
                },
                "deep": {
                    "tumor_dice": {
                        "mean": 0.5219,
                        "median": 0.5889,
                    }
                },
                "paired_manual_minus_deep": {
                    "tumor_dice": {
                        "mean": -0.5102,
                    }
                },
            },
        },
        "cyst": {
            "case_count": 46,
            "key_results": {
                "dice_mean": 0.1958,
                "dice_median": 0.0051,
                "recall_mean": 0.1599,
                "precision_mean": 0.3735,
                "zero_dice_count": 21,
                "fn_dominant_count": 34,
                "fp_dominant_count": 12,
                "volume_vs_dice_pearson": 0.659,
                "volume_vs_recall_pearson": 0.674,
                "any_overlap_count": 25,
                "dice_ge_050_count": 12,
                "any_overlap_volume_auroc": 0.950,
                "dice_ge_050_volume_auroc": 0.914,
                "smallest_quartile": {},
                "largest_quartile": {},
                "volume_response_bins": [],
            },
            "final_decision": {
                "cyst_branch_status": "closed",
            },
        },
    }


def test_final_synthesis_preserves_core_results() -> None:
    payload = synthesize_final_experimental_results(
        _artifacts()
    )

    assert payload["automatic_baseline"][
        "kidney_and_mass_dice_mean"
    ] == pytest.approx(0.8863)

    assert payload["tumor_interaction"][
        "oracle_tumor_dice_mean"
    ] == pytest.approx(0.7557)

    assert payload["manual_ct_only_localization"][
        "any_tumor_hit_count"
    ] == 0

    assert payload["cyst_branch"][
        "any_overlap_volume_auroc"
    ] == pytest.approx(0.950)


def test_final_synthesis_rejects_unsupported_claims() -> None:
    payload = synthesize_final_experimental_results(
        _artifacts()
    )

    statuses = {
        item["claim_id"]: item["status"]
        for item in payload["claims"]
    }

    assert statuses[
        "automatic_postprocessing"
    ] == "rejected"
    assert statuses[
        "oracle_rule_generalization"
    ] == "rejected"
    assert statuses[
        "manual_localization"
    ] == "negative_result"


def test_final_synthesis_enforces_reporting_boundaries() -> None:
    payload = synthesize_final_experimental_results(
        _artifacts()
    )

    constraints = " ".join(
        payload["reporting_constraints"]
    ).lower()

    assert "official kits23 challenge score" in constraints
    assert "clinician" in constraints
    assert "clinical cutoffs" in constraints


def test_final_synthesis_requires_valid_geometry() -> None:
    artifacts = _artifacts()
    artifacts["manual_geometry"][
        "geometry_contract_valid"
    ] = False

    with pytest.raises(
        RuntimeError,
        match="manual_geometry_contract_valid",
    ):
        synthesize_final_experimental_results(
            artifacts
        )


def test_final_synthesis_propagates_cyst_volume_response_bins() -> None:
    artifacts = _artifacts()
    expected = [
        {
            "bin": 1,
            "count": 6,
            "volume_median_ml": 0.17,
            "any_overlap_fraction": 0.0,
            "dice_ge_050_fraction": 0.0,
            "dice_mean": 0.0,
            "recall_mean": 0.0,
        }
    ]
    artifacts["cyst"]["key_results"][
        "volume_response_bins"
    ] = expected

    payload = synthesize_final_experimental_results(
        artifacts
    )

    assert payload["cyst_branch"][
        "volume_response_bins"
    ] == expected
