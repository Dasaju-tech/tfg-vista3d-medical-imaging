from __future__ import annotations

import pytest

from vista3d_tfg.experiments.tumor_oracle_cohort import (
    TumorOracleCohortSummary,
    summarize_tumor_oracle_cohort,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)


def make_result(
    index: int,
) -> TumorOracleSmokeResult:
    return TumorOracleSmokeResult(
        case_id=f"case_{index:05d}",
        point_xyz=(1, 2, 3),
        point_label=3,
        target_volume_ml=100.0,
        total_tumor_volume_ml=100.0,
        interactive_target_dice=0.6,
        interactive_full_tumor_dice=0.6,
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
        merged_hec_hd95_mm=30.0,
        hec_hd95_reduction_mm=20.0,
        interactive_prediction_volume_ml=100.0,
        interactive_labels_present=(0, 1, 255),
        contains_255_unprocessed=True,
        wall_time_seconds=10.0,
        prediction_path=f"case_{index:05d}/prediction.nii.gz",
    )


def test_summary_requires_nine_cases() -> None:
    with pytest.raises(ValueError, match="9"):
        summarize_tumor_oracle_cohort(
            make_result(i)
            for i in range(8)
        )


def test_summary_counts_successes() -> None:
    summary = summarize_tumor_oracle_cohort(
        make_result(i)
        for i in range(9)
    )

    assert isinstance(
        summary,
        TumorOracleCohortSummary,
    )
    assert summary.case_count == 9
    assert summary.interactive_target_dice_mean == pytest.approx(0.6)
    assert summary.tumor_coverage_gain_mean == pytest.approx(0.6)
    assert summary.hec_hd95_reduction_mean_mm == pytest.approx(20.0)
    assert summary.hd95_improved_case_count == 9
    assert summary.hd95_reduced_gt_5mm_count == 9
    assert summary.target_dice_ge_050_count == 9
    assert summary.target_dice_ge_080_count == 0
    assert summary.surface_dice_improved_case_count == 9
    assert summary.dice_improved_case_count == 9
