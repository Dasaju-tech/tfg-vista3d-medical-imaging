from __future__ import annotations

import numpy as np
import pytest

from vista3d_tfg.evaluation.failure_decomposition import (
    CaseFailureDecomposition,
    CystLocalizationSummary,
    ErrorComponentSummary,
    summarize_cyst_localization,
    summarize_error_components,
    summarize_failure_decomposition,
)


def test_error_component_distance_uses_physical_spacing() -> None:
    target = np.zeros((10, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    error = np.zeros_like(target)
    error[6, 1, 1] = True

    result = summarize_error_components(
        error,
        target_mask=target,
        spacing_mm=(2.0, 1.0, 1.0),
    )

    assert result.component_count == 1
    assert result.farthest_distance_mm == pytest.approx(10.0)


def test_cyst_localization_detects_near_centroids() -> None:
    gt = np.zeros((20, 10, 10), dtype=np.uint8)
    pred = np.zeros_like(gt)
    gt[5:7, 4:6, 4:6] = 3
    pred[7:9, 4:6, 4:6] = 116

    result = summarize_cyst_localization(
        ground_truth=gt,
        prediction=pred,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    )

    assert result is not None
    assert result.gt_component_count == 1
    assert result.prediction_component_count == 1
    assert result.within_10mm == 1


def test_no_gt_cyst_returns_none() -> None:
    gt = np.zeros((5, 5, 5), dtype=np.uint8)
    pred = np.zeros_like(gt)

    assert summarize_cyst_localization(
        ground_truth=gt,
        prediction=pred,
        spacing_mm=(1.0, 1.0, 1.0),
        affine=np.eye(4),
    ) is None


def make_case(
    case_id: str,
    *,
    hd95: float,
    fp_distance: float,
    cyst: CystLocalizationSummary | None,
) -> CaseFailureDecomposition:
    return CaseFailureDecomposition(
        case_id=case_id,
        kidney_and_mass_dice=0.9,
        kidney_and_mass_surface_dice=0.8,
        kidney_and_mass_hd95_mm=hd95,
        false_positive=ErrorComponentSummary(1, 1.0, 1.0, fp_distance),
        false_negative=ErrorComponentSummary(1, 1.0, 1.0, 2.0),
        cyst=cyst,
    )


def test_failure_summary_counts_distant_fp() -> None:
    summary = summarize_failure_decomposition(
        [
            make_case("a", hd95=2.0, fp_distance=5.0, cyst=None),
            make_case("b", hd95=40.0, fp_distance=60.0, cyst=None),
        ]
    )

    assert summary.cases_with_fp_farther_than_20mm == 1
    assert summary.cases_with_fp_farther_than_50mm == 1
    assert summary.worst_farthest_fp_case_ids[0] == "b"


def test_failure_summary_cyst_component_recall() -> None:
    cyst = CystLocalizationSummary(
        gt_component_count=2,
        prediction_component_count=1,
        gt_volume_ml=10.0,
        prediction_volume_ml=9.0,
        voxel_dice=0.1,
        within_10mm=1,
        within_20mm=1,
        within_30mm=2,
        nearest_centroid_median_mm=18.0,
        zero_voxel_overlap=False,
    )

    summary = summarize_failure_decomposition(
        [make_case("a", hd95=3.0, fp_distance=2.0, cyst=cyst)]
    )

    assert summary.cyst_gt_component_count == 2
    assert summary.cyst_component_localization_recall_10mm == 0.5
    assert summary.cyst_component_localization_recall_30mm == 1.0
