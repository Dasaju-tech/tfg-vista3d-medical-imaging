from __future__ import annotations

from pathlib import Path

import pytest

from vista3d_tfg.evaluation.cohort import CaseCohortRow
from vista3d_tfg.evaluation.pilot_analysis import (
    analyze_pilot30,
    bootstrap_mean_ci,
    pearson_correlation,
    render_pilot_figures,
)


def row(
    case_id: str,
    *,
    dice: float,
    surface: float,
    hd95: float,
    cyst_gt: float,
    cyst_pred: float,
    cyst_dice: float,
    volume_error: float,
) -> CaseCohortRow:
    return CaseCohortRow(
        case_id=case_id,
        kidney_and_mass_dice=dice,
        kidney_and_mass_surface_dice=surface,
        kidney_and_mass_hd95_mm=hd95,
        kidney_core_dice=dice - 0.01,
        cyst_dice=cyst_dice,
        cyst_reference_volume_ml=cyst_gt,
        cyst_prediction_volume_ml=cyst_pred,
        left_renal_dice=dice,
        right_renal_dice=dice,
        left_renal_volume_error_ml=volume_error / 2,
        right_renal_volume_error_ml=volume_error / 2,
        gt_tumor_volume_ml=5.0,
        inference_wall_time_seconds=15.0,
        qc_false_positive_voxels=None,
        qc_false_negative_voxels=None,
        qc_total_disagreement_voxels=None,
    )


def fixture_rows() -> list[CaseCohortRow]:
    return [
        row(
            "case_a",
            dice=0.95,
            surface=0.90,
            hd95=2.0,
            cyst_gt=0.0,
            cyst_pred=0.0,
            cyst_dice=1.0,
            volume_error=-5.0,
        ),
        row(
            "case_b",
            dice=0.90,
            surface=0.75,
            hd95=5.0,
            cyst_gt=20.0,
            cyst_pred=10.0,
            cyst_dice=0.30,
            volume_error=-20.0,
        ),
        row(
            "case_c",
            dice=0.85,
            surface=0.60,
            hd95=9.0,
            cyst_gt=40.0,
            cyst_pred=50.0,
            cyst_dice=0.20,
            volume_error=-40.0,
        ),
    ]


def test_bootstrap_is_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0])
    second = bootstrap_mean_ci([1.0, 2.0, 3.0])

    assert first == second
    assert first.lower <= 2.0 <= first.upper


def test_pearson_correlation() -> None:
    correlation = pearson_correlation(
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
    )

    assert correlation == pytest.approx(-1.0)


def test_constant_correlation_is_undefined() -> None:
    assert pearson_correlation(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ) is None


def test_analysis_separates_cyst_status() -> None:
    analysis = analyze_pilot30(fixture_rows())

    assert analysis.case_count == 3
    assert analysis.cyst_positive_count == 2
    assert analysis.cyst_negative_count == 1
    assert analysis.cyst_positive_dice_mean == pytest.approx(0.25)
    assert (
        "low_cyst_positive_segmentation_overlap"
        in analysis.interpretation_flags
    )


def test_analysis_ranks_worst_cases() -> None:
    analysis = analyze_pilot30(fixture_rows())

    assert analysis.worst_surface_dice_case_ids[0] == "case_c"
    assert analysis.worst_hd95_case_ids[0] == "case_c"
    assert (
        analysis.largest_undersegmentation_case_ids[0]
        == "case_c"
    )


def test_render_pilot_figures(tmp_path: Path) -> None:
    paths = render_pilot_figures(
        fixture_rows(),
        tmp_path,
    )

    assert len(paths) == 3
    assert all(path.is_file() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)
