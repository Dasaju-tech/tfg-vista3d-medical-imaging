from __future__ import annotations

import pytest

from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    AdaptiveDepthComparison,
    summarize_adaptive_depth_results,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)


def _result(
    case_id: str,
    *,
    dice: float,
    hd95_reduction: float,
) -> TumorOracleSmokeResult:
    return TumorOracleSmokeResult(
        case_id=case_id,
        point_xyz=(1, 2, 3),
        point_label=3,
        target_volume_ml=10.0,
        total_tumor_volume_ml=10.0,
        interactive_target_dice=dice,
        interactive_full_tumor_dice=dice,
        original_tumor_coverage_fraction=0.1,
        merged_tumor_coverage_fraction=0.7,
        tumor_coverage_gain=0.6,
        original_hec_dice=0.5,
        merged_hec_dice=0.7,
        hec_dice_delta=0.2,
        original_hec_surface_dice=0.4,
        merged_hec_surface_dice=0.5,
        hec_surface_dice_delta=0.1,
        original_hec_hd95_mm=50.0,
        merged_hec_hd95_mm=(
            50.0 - hd95_reduction
        ),
        hec_hd95_reduction_mm=hd95_reduction,
        interactive_prediction_volume_ml=10.0,
        interactive_labels_present=(0, 1, 255),
        contains_255_unprocessed=True,
        wall_time_seconds=2.0,
        prediction_path="fixture.nii.gz",
    )


def test_summary_is_balanced_and_paired() -> None:
    rows = []
    case_index = 0

    for quartile in QUARTILE_NAMES:
        for _ in range(3):
            case_id = f"case_{case_index:05d}"
            rows.append(
                AdaptiveDepthComparison(
                    case_id=case_id,
                    quartile=quartile,
                    largest_component_volume_ml=10.0,
                    max_interior_depth_mm=20.0,
                    adaptive_actual_depth_mm=14.0,
                    adaptive_depth_fraction_actual=0.70,
                    adaptive=_result(
                        case_id,
                        dice=0.78,
                        hd95_reduction=29.0,
                    ),
                    deep=_result(
                        case_id,
                        dice=0.80,
                        hd95_reduction=30.0,
                    ),
                )
            )
            case_index += 1

    payload = summarize_adaptive_depth_results(
        rows
    )

    assert payload["case_count"] == 12

    overall = payload["overall"]
    assert overall["adaptive"]["tumor_dice"]["mean"] == pytest.approx(
        0.78
    )
    assert overall["deep"]["tumor_dice"]["mean"] == pytest.approx(
        0.80
    )
    assert overall["paired_adaptive_minus_deep"]["tumor_dice"][
        "mean"
    ] == pytest.approx(-0.02)
    assert overall["paired_adaptive_minus_deep"][
        "cases_within_0_05_tumor_dice"
    ] == 12

    for quartile in QUARTILE_NAMES:
        assert payload["by_quartile"][quartile]["count"] == 3


def test_summary_requires_twelve_cases() -> None:
    with pytest.raises(ValueError, match="12"):
        summarize_adaptive_depth_results([])
