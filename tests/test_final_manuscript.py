from __future__ import annotations

from vista3d_tfg.reporting.final_manuscript import (
    _conclusions_section,
    _discussion_section,
    _limitations_section,
    _results_section,
)


def _payload() -> dict:
    return {
        "automatic_baseline": {
            "case_count": 100,
            "cyst_positive_count": 46,
        },
        "hd95_tail": {
            "outlier_count": 22,
            "outlier_threshold_mm": 11.08,
            "top5_hd95_mean_share": 0.392,
            "top10_hd95_mean_share": 0.549,
            "prediction_to_gt_dominant_count": 12,
            "gt_to_prediction_dominant_count": 9,
            "balanced_count": 1,
            "prediction_only_trigger_balanced_accuracy": 0.612,
            "prediction_only_trigger_target_resolved_count": 0,
        },
        "tumor_undercoverage": {
            "undercoverage_case_count": 9,
            "tumor_dominant_missing_count": 8,
            "weighted_tumor_miss_fraction": 0.892,
        },
        "tumor_interaction": {
            "relative_depth_paired_dice_delta": -0.0236,
            "dual_point_paired_dice_delta": 0.0153,
        },
        "manual_ct_only_localization": {
            "case_count": 12,
            "any_tumor_hit_count": 0,
            "paired_manual_minus_deep_dice_mean": -0.5102,
        },
        "cyst_branch": {
            "case_count": 46,
            "dice_median": 0.0051,
            "zero_dice_count": 21,
            "fn_dominant_count": 34,
            "volume_vs_dice_pearson": 0.659,
            "volume_vs_recall_pearson": 0.674,
        },
    }


def test_results_preserve_reporting_boundaries() -> None:
    text = _results_section(_payload())
    normalized = " ".join(text.split())

    assert "ground-truth-derived" in normalized
    assert "simulated-user" in normalized
    assert "not as clinician" in normalized
    assert "descriptive" in normalized


def test_discussion_separates_capability_from_localization() -> None:
    text = _discussion_section(_payload()).lower()
    assert "segmentation capability" in text
    assert "lesion localization" in text
    assert "cannot be interpreted as an autonomous tumor" in text
    assert "expert radiologists" in text


def test_limitations_reject_overclaims() -> None:
    text = _limitations_section(_payload()).lower()
    normalized = " ".join(text.split())

    assert "official kits23 challenge submission" in normalized
    assert "oracle experiments" in normalized
    assert "neither a clinician study" in normalized
    assert "clinically validated cyst-size cutoff" in normalized


def test_conclusions_do_not_claim_complete_autonomy() -> None:
    text = _conclusions_section(_payload()).lower()
    assert "rather than as a complete autonomous" in text
    assert "independent cohort" in text
