from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments.manual_tumor_click_inference import (
    ManualClickComparison,
    assess_manual_click_localization,
    summarize_manual_click_results,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    AdaptiveDepthCase,
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)
from vista3d_tfg.interaction.manual_tumor_clicks import (
    ManualTumorClick,
)


def _save(
    path: Path,
    array: np.ndarray,
) -> None:
    nib.save(
        nib.Nifti1Image(
            array,
            np.eye(4),
        ),
        str(path),
    )


def test_posthoc_localization_distinguishes_hit_and_miss(
    tmp_path: Path,
) -> None:
    gt = np.zeros(
        (30, 30, 30),
        dtype=np.uint8,
    )
    gt[10:20, 10:20, 10:20] = 2

    path = tmp_path / "segmentation.nii.gz"
    _save(path, gt)

    case = AdaptiveDepthCase(
        case_id="fixture",
        quartile="smallest_quartile",
        largest_component_volume_ml=1.0,
        component_id=1,
        target_voxels=1000,
        max_interior_depth_mm=5.0,
        deep_point_xyz=(14, 14, 14),
        adaptive_target_depth_mm=3.5,
        adaptive_actual_depth_mm=3.5,
        adaptive_point_xyz=(13, 13, 13),
    )

    hit = assess_manual_click_localization(
        case=case,
        click=ManualTumorClick(
            case_id="fixture",
            quartile="smallest_quartile",
            point_xyz=(14, 14, 14),
            display_point_ras_xyz=(14, 14, 14),
        ),
        segmentation_path=path,
    )

    miss = assess_manual_click_localization(
        case=case,
        click=ManualTumorClick(
            case_id="fixture",
            quartile="smallest_quartile",
            point_xyz=(5, 5, 5),
            display_point_ras_xyz=(5, 5, 5),
        ),
        segmentation_path=path,
    )

    assert hit.inside_target_component is True
    assert hit.distance_to_target_component_mm == pytest.approx(0.0)

    assert miss.inside_any_tumor is False
    assert miss.distance_to_any_tumor_mm > 0.0
    assert miss.distance_to_target_component_mm > 0.0


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


def test_summary_reports_localization_and_paired_metrics() -> None:
    rows = []
    index = 0

    for quartile in QUARTILE_NAMES:
        for local_index in range(3):
            case_id = f"case_{index:05d}"
            hit = local_index != 2

            rows.append(
                ManualClickComparison(
                    case_id=case_id,
                    quartile=quartile,
                    largest_component_volume_ml=10.0,
                    localization=(
                        __import__(
                            "vista3d_tfg.experiments.manual_tumor_click_inference",
                            fromlist=["ManualClickLocalization"],
                        ).ManualClickLocalization(
                            case_id=case_id,
                            point_xyz=(1, 2, 3),
                            gt_label_at_point=(2 if hit else 0),
                            inside_any_tumor=hit,
                            inside_target_component=hit,
                            distance_to_any_tumor_mm=(0.0 if hit else 5.0),
                            distance_to_target_component_mm=(0.0 if hit else 5.0),
                        )
                    ),
                    manual=_result(
                        case_id,
                        dice=(0.75 if hit else 0.10),
                        hd95_reduction=(10.0 if hit else 0.0),
                    ),
                    deep=_result(
                        case_id,
                        dice=0.80,
                        hd95_reduction=12.0,
                    ),
                )
            )
            index += 1

    payload = summarize_manual_click_results(
        rows
    )

    assert payload["case_count"] == 12
    assert payload["localization_audit"][
        "target_component_hit_count"
    ] == 8
    assert payload["localization_audit"][
        "miss_count"
    ] == 4
    assert payload["overall"]["manual"]["tumor_dice"][
        "mean"
    ] < payload["overall"]["deep"]["tumor_dice"]["mean"]

    for quartile in QUARTILE_NAMES:
        assert payload["by_quartile"][quartile]["count"] == 3


def test_summary_requires_twelve_cases() -> None:
    with pytest.raises(ValueError, match="12"):
        summarize_manual_click_results([])


def test_manual_miss_can_be_evaluated_without_oracle_guard(
    tmp_path: Path,
) -> None:
    from vista3d_tfg.experiments.tumor_oracle_protocol import (
        TumorOraclePrompt,
    )
    from vista3d_tfg.experiments.tumor_oracle_smoke import (
        evaluate_tumor_oracle_smoke,
    )

    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    automatic = np.zeros_like(gt)
    interactive = np.zeros_like(gt)

    gt[8:12, 8:12, 8:12] = 2
    interactive[2:5, 2:5, 2:5] = 1

    gt_path = tmp_path / "gt.nii.gz"
    auto_path = tmp_path / "auto.nii.gz"
    interactive_path = tmp_path / "interactive.nii.gz"

    _save(gt_path, gt)
    _save(auto_path, automatic)
    _save(interactive_path, interactive)

    prompt = TumorOraclePrompt(
        case_id="fixture",
        component_id=1,
        point_xyz=(3, 3, 3),
        target_voxels=4 ** 3,
        target_volume_ml=0.064,
        total_tumor_volume_ml=0.064,
        interior_radius_mm=0.0,
    )

    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=gt_path,
        automatic_prediction_path=auto_path,
        interactive_prediction_path=interactive_path,
        wall_time_seconds=1.0,
        require_prompt_inside_target=False,
    )

    assert result.point_xyz == (3, 3, 3)
    assert result.interactive_target_dice == pytest.approx(0.0)
