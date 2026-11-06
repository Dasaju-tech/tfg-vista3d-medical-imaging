from __future__ import annotations

import pytest

from vista3d_tfg.analysis.interaction_study_synthesis import (
    synthesize_interaction_study,
)


def _sources() -> dict[str, dict]:
    return {
        "block41c_oracle": {
            "summary": {
                "interactive_target_dice_mean": 0.7557,
                "interactive_target_dice_median": 0.7802,
                "tumor_coverage_gain_mean": 0.521,
                "hec_hd95_reduction_mean_mm": 29.31,
                "hd95_improved_case_count": 9,
            }
        },
        "block42_click_robustness": {
            "variants": {
                "deep": {
                    "tumor_dice_mean": 0.7557,
                },
                "centroid": {
                    "tumor_dice_mean": 0.7676,
                },
                "shallow_5mm": {
                    "tumor_dice_mean": 0.4461,
                },
            }
        },
        "block43_depth": {
            "variants": {
                "depth_10mm": {
                    "tumor_dice_mean": 0.6595,
                },
                "depth_15mm": {
                    "tumor_dice_mean": 0.7272,
                },
            }
        },
        "block44_safe_region_outliers": {
            "thresholds": {}
        },
        "block45_safe_region_expanded100": {
            "tumor_positive_count": 100,
            "depths": {
                "15mm": {
                    "cases_with_nonempty_region": 51,
                }
            },
            "volume_quartiles": {
                "smallest_quartile": {
                    "count": 25,
                    "15mm": {
                        "cases_with_nonempty_region": 0,
                    },
                }
            },
        },
        "block46b_relative_depth": {
            "overall": {
                "paired_adaptive_minus_deep": {
                    "tumor_dice": {
                        "mean": -0.0236,
                        "bootstrap_mean_95_ci": {
                            "lower_95": -0.1014,
                            "upper_95": 0.0444,
                        },
                    }
                }
            },
            "by_quartile": {
                "upper_middle": {
                    "paired_adaptive_minus_deep": {
                        "tumor_dice": {
                            "mean": -0.1627,
                        }
                    }
                }
            },
        },
        "block47b_dual_point": {
            "overall": {
                "paired_dual_minus_deep": {
                    "tumor_dice": {
                        "mean": 0.0153,
                        "bootstrap_mean_95_ci": {
                            "lower_95": -0.0325,
                            "upper_95": 0.0594,
                        },
                    }
                }
            },
            "by_quartile": {
                "upper_middle": {
                    "paired_dual_minus_deep": {
                        "tumor_dice": {
                            "mean": -0.0285,
                        }
                    }
                }
            },
        },
    }


def test_synthesis_closes_oracle_prompt_engineering() -> None:
    payload = synthesize_interaction_study(
        _sources()
    )

    assert (
        payload["final_decision"][
            "oracle_prompt_engineering_status"
        ]
        == "closed"
    )
    assert (
        payload["final_decision"][
            "preferred_interaction_concept"
        ]
        == "single well-interior positive tumor point"
    )


def test_synthesis_records_expected_decisions() -> None:
    payload = synthesize_interaction_study(
        _sources()
    )

    decisions = {
        item["hypothesis"]: item["status"]
        for item in payload["decisions"]
    }

    assert (
        decisions["Latent interactive tumor capability"]
        == "accepted"
    )
    assert (
        decisions["Fixed 15 mm interior rule generalizes"]
        == "rejected"
    )
    assert (
        decisions["Single 70% relative-depth rule generalizes"]
        == "rejected"
    )
    assert (
        decisions[
            "Two spatially separated positive points are robustly superior"
        ]
        == "rejected"
    )


def test_synthesis_preserves_key_numeric_results() -> None:
    payload = synthesize_interaction_study(
        _sources()
    )

    key = payload["key_results"]

    assert key["oracle_cohort"][
        "tumor_dice_mean"
    ] == pytest.approx(0.7557)
    assert key["relative_depth"][
        "paired_tumor_dice_delta_mean"
    ] == pytest.approx(-0.0236)
    assert key["dual_point"][
        "paired_tumor_dice_delta_mean"
    ] == pytest.approx(0.0153)
