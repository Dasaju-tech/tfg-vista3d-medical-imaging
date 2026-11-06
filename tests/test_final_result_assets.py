from __future__ import annotations

from vista3d_tfg.reporting.final_result_assets import (
    build_claim_table,
    build_main_results_table,
    build_result_macros,
)


def _payload() -> dict:
    return {
        "automatic_baseline": {
            "case_count": 100,
            "kidney_and_mass_dice_mean": 0.8863,
            "kidney_and_mass_dice_sd": 0.1309,
            "surface_dice_mean": 0.7471,
            "hd95_median_mm": 4.53,
            "hd95_mean_mm": 12.1003,
            "cyst_positive_count": 46,
            "cyst_positive_dice_mean": 0.1958,
        },
        "hd95_tail": {
            "outlier_count": 22,
            "prediction_to_gt_dominant_count": 12,
            "gt_to_prediction_dominant_count": 9,
            "balanced_count": 1,
        },
        "tumor_interaction": {
            "oracle_tumor_dice_mean": 0.7557,
            "oracle_hec_hd95_reduction_mean_mm": 29.31,
        },
        "manual_ct_only_localization": {
            "case_count": 12,
            "any_tumor_hit_count": 0,
            "manual_tumor_dice_mean": 0.0117,
            "deep_tumor_dice_mean": 0.5219,
        },
        "cyst_branch": {
            "case_count": 46,
            "dice_mean": 0.1958,
            "dice_median": 0.0051,
            "zero_dice_count": 21,
            "fn_dominant_count": 34,
            "any_overlap_volume_auroc": 0.950,
            "dice_ge_050_volume_auroc": 0.914,
            "volume_response_bins": [
                {
                    "bin": 1,
                    "volume_median_ml": 0.17,
                    "any_overlap_fraction": 0.0,
                    "dice_ge_050_fraction": 0.0,
                }
            ],
        },
        "claims": [
            {
                "claim_id": "fixture",
                "status": "supported",
                "claim": "A_B & C",
                "evidence": "fixture",
            }
        ],
    }


def test_result_macros_include_frozen_numbers() -> None:
    text = build_result_macros(_payload())
    assert r"\newcommand{\TFGKidneyMassDice}{0.8863}" in text
    assert r"\newcommand{\TFGOracleTumorDice}{0.7557}" in text
    assert r"\newcommand{\TFGCystOverlapAUROC}{0.950}" in text


def test_main_results_table_preserves_reporting_boundaries() -> None:
    text = build_main_results_table(_payload())
    assert "ground-truth-derived localization" in text
    assert "simulated-user" in text
    assert "0/12" in text
    assert "0.1958 / 0.0051" in text


def test_claim_table_escapes_latex_special_characters() -> None:
    text = build_claim_table(_payload())
    assert r"A\_B \& C" in text
    assert "supported" in text
