from __future__ import annotations

import pytest

from vista3d_tfg.evaluation.selective_trigger_validation import (
    FEATURES,
    FilterCase,
    MetricTriplet,
    TriggerCase,
    analyze_selective_filtering,
    confusion_summary,
    fit_univariate_trigger,
    leave_one_out_decisions,
)


def make_trigger(index: int) -> TriggerCase:
    target = index < 12
    features = {
        feature: float(index)
        for feature in FEATURES
    }
    features["largest_to_second_ratio"] = (
        10.0 + index
        if target
        else float(index) / 100.0
    )

    return TriggerCase(
        case_id=f"case_{index:05d}",
        target=target,
        hd95_outlier=index < 22,
        features=features,
    )


def make_filter(index: int) -> FilterCase:
    original = MetricTriplet(
        dice=0.90,
        surface_dice=0.75,
        hd95_mm=(20.0 if index < 22 else 4.0),
    )
    improved = MetricTriplet(
        dice=0.90,
        surface_dice=0.76,
        hd95_mm=(8.0 if index < 12 else original.hd95_mm),
    )

    return FilterCase(
        case_id=f"case_{index:05d}",
        original=original,
        global_top2=improved,
        side_largest=improved,
    )


def test_fit_trigger_finds_separating_feature() -> None:
    rows = tuple(make_trigger(i) for i in range(100))

    feature, direction, _, auc, balanced = (
        fit_univariate_trigger(rows)
    )

    # Several fixture features may separate the synthetic labels
    # perfectly. The contract is perfect separability, not a specific
    # tie-broken feature name or direction.
    assert feature in FEATURES
    assert direction in {"high", "low"}
    assert auc == pytest.approx(1.0)
    assert balanced == pytest.approx(1.0)


def test_loocv_produces_100_held_out_decisions() -> None:
    rows = tuple(make_trigger(i) for i in range(100))

    decisions = leave_one_out_decisions(rows)
    summary = confusion_summary(decisions)

    assert len(decisions) == 100
    assert summary["sensitivity"] >= 0.90
    assert summary["specificity"] >= 0.90


def test_selective_analysis_requires_matching_cases() -> None:
    triggers = tuple(make_trigger(i) for i in range(100))
    filters = tuple(make_filter(i) for i in range(99))

    with pytest.raises(ValueError, match="IDs"):
        analyze_selective_filtering(
            trigger_rows=triggers,
            filter_rows=filters,
        )


def test_selective_filtering_improves_targets() -> None:
    triggers = tuple(make_trigger(i) for i in range(100))
    filters = tuple(make_filter(i) for i in range(100))

    result = analyze_selective_filtering(
        trigger_rows=triggers,
        filter_rows=filters,
    )

    trigger = result["trigger"]
    assert trigger["sensitivity"] >= 0.90

    variant = result["variants"]["global_top2"]
    assert variant["target_hd95_reduction_mean_mm"] > 0.0
    assert variant["non_target_hd95_worsened_gt_1mm"] == 0
