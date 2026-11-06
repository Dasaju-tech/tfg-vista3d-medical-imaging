from __future__ import annotations

import pytest

from vista3d_tfg.analysis.cyst_detectability_limit import (
    CystDetectabilityCase,
    best_volume_threshold,
    binary_auroc,
    empirical_transition,
    summarize_cyst_detectability,
)


def test_binary_auroc_is_one_for_perfect_size_ordering() -> None:
    result = binary_auroc(
        [1.0, 2.0, 10.0, 20.0],
        [False, False, True, True],
    )

    assert result == pytest.approx(1.0)


def test_best_threshold_finds_separation_midpoint() -> None:
    result = best_volume_threshold(
        volumes_ml=[
            1.0,
            2.0,
            10.0,
            20.0,
        ],
        labels=[
            False,
            False,
            True,
            True,
        ],
    )

    assert result["threshold_ml"] == pytest.approx(6.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["tp"] == 2
    assert result["tn"] == 2


def test_transition_reports_overlap_when_classes_cross() -> None:
    result = empirical_transition(
        volumes_ml=[
            1.0,
            10.0,
            5.0,
            20.0,
        ],
        labels=[
            False,
            False,
            True,
            True,
        ],
    )

    assert result["largest_negative_ml"] == pytest.approx(10.0)
    assert result["smallest_positive_ml"] == pytest.approx(5.0)
    assert result["separable"] is False
    assert result["gap_ml"] == pytest.approx(-5.0)


def test_full_summary_preserves_46_case_contract() -> None:
    cases = tuple(
        CystDetectabilityCase(
            case_id=f"case_{index:05d}",
            reference_volume_ml=float(index + 1),
            dice=(
                0.0
                if index < 20
                else (
                    0.30
                    if index < 34
                    else 0.60
                )
            ),
            recall=(
                0.0
                if index < 20
                else 0.4
            ),
        )
        for index in range(46)
    )

    payload = summarize_cyst_detectability(
        cases
    )

    assert payload["case_count"] == 46
    assert payload["event_counts"]["zero_overlap"] == 20
    assert payload["event_counts"]["dice_ge_050"] == 12
    assert len(
        payload["volume_response_bins"]
    ) == 8
    assert (
        payload["volume_as_predictor"][
            "any_overlap"
        ]["auroc"]
        > 0.9
    )
