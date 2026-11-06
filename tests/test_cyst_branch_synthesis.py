from __future__ import annotations

import pytest

from vista3d_tfg.analysis.cyst_branch_synthesis import (
    synthesize_cyst_branch,
)


def _block50() -> dict:
    def group(
        *,
        count: int,
        dice_mean: float,
        recall_mean: float,
        zero: int,
    ) -> dict:
        return {
            "count": count,
            "reference_volume_ml": {
                "mean": 1.0,
                "median": 1.0,
                "min": 0.1,
                "max": 2.0,
            },
            "dice": {
                "mean": dice_mean,
                "median": dice_mean,
                "min": 0.0,
                "max": 1.0,
            },
            "precision": {
                "mean": 0.3,
                "median": 0.3,
                "min": 0.0,
                "max": 1.0,
            },
            "recall": {
                "mean": recall_mean,
                "median": recall_mean,
                "min": 0.0,
                "max": 1.0,
            },
            "false_negative_volume_ml": {
                "mean": 1.0,
                "median": 1.0,
                "min": 0.0,
                "max": 2.0,
            },
            "false_positive_volume_ml": {
                "mean": 0.2,
                "median": 0.2,
                "min": 0.0,
                "max": 1.0,
            },
            "zero_dice_count": zero,
            "dice_ge_050_count": 0,
            "dominant_error_counts": {
                "undersegmentation_fn": count,
                "oversegmentation_fp": 0,
                "balanced": 0,
            },
        }

    overall = group(
        count=46,
        dice_mean=0.1958,
        recall_mean=0.1599,
        zero=21,
    )
    overall["dice"]["median"] = 0.0051
    overall["precision"]["mean"] = 0.3735
    overall["dominant_error_counts"] = {
        "undersegmentation_fn": 34,
        "oversegmentation_fp": 12,
        "balanced": 0,
    }

    return {
        "case_count": 46,
        "overall": overall,
        "by_reference_volume_quartile": {
            "smallest": group(
                count=12,
                dice_mean=0.0,
                recall_mean=0.0,
                zero=12,
            ),
            "lower_middle": group(
                count=12,
                dice_mean=0.0662,
                recall_mean=0.0721,
                zero=7,
            ),
            "upper_middle": group(
                count=11,
                dice_mean=0.1797,
                recall_mean=0.1212,
                zero=2,
            ),
            "largest": group(
                count=11,
                dice_mean=0.5667,
                recall_mean=0.4686,
                zero=0,
            ),
        },
        "associations": {
            "reference_volume_vs_dice_pearson": 0.659,
            "reference_volume_vs_recall_pearson": 0.674,
        },
    }


def _block51() -> dict:
    predictor = {
        "auroc": 0.950,
        "bootstrap_95_ci": {
            "lower_95": 0.886,
            "upper_95": 0.992,
        },
        "best_descriptive_threshold": {
            "threshold_ml": 5.93,
            "balanced_accuracy": 0.872,
            "tp": 21,
            "fp": 2,
            "tn": 19,
            "fn": 4,
        },
        "empirical_transition": {
            "largest_negative_ml": 8.31,
            "smallest_positive_ml": 2.15,
            "separable": False,
            "gap_ml": -6.16,
        },
    }

    useful = {
        **predictor,
        "auroc": 0.914,
        "empirical_transition": {
            "largest_negative_ml": 112.0,
            "smallest_positive_ml": 5.31,
            "separable": False,
            "gap_ml": -106.69,
        },
    }

    return {
        "event_counts": {
            "any_overlap": 25,
            "zero_overlap": 21,
            "dice_ge_050": 12,
            "dice_lt_050": 34,
        },
        "volume_as_predictor": {
            "any_overlap": predictor,
            "dice_ge_050": useful,
        },
        "volume_response_bins": [],
    }


def test_synthesis_closes_cyst_branch() -> None:
    payload = synthesize_cyst_branch(
        block50=_block50(),
        block51=_block51(),
    )

    assert payload["final_decision"][
        "cyst_branch_status"
    ] == "closed"
    assert payload["final_decision"][
        "global_postprocessing_recommended"
    ] is False


def test_synthesis_preserves_key_results() -> None:
    payload = synthesize_cyst_branch(
        block50=_block50(),
        block51=_block51(),
    )
    key = payload["key_results"]

    assert key["dice_mean"] == pytest.approx(
        0.1958
    )
    assert key["fn_dominant_count"] == 34
    assert key[
        "any_overlap_volume_auroc"
    ] == pytest.approx(0.950)
    assert key[
        "dice_ge_050_volume_auroc"
    ] == pytest.approx(0.914)


def test_no_clinical_cutoff_is_claimed() -> None:
    payload = synthesize_cyst_branch(
        block50=_block50(),
        block51=_block51(),
    )

    assert payload["final_decision"][
        "validated_clinical_cutoff_available"
    ] is False

    statuses = {
        item["hypothesis"]: item["status"]
        for item in payload["decisions"]
    }
    assert statuses[
        "A single validated cyst-size cutoff exists"
    ] == "rejected"
