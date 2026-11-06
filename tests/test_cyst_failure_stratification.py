from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.analysis.cyst_failure_stratification import (
    CystCaseFailure,
    analyze_cyst_case,
    summarize_cyst_failures,
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


def test_cyst_case_metrics_are_voxel_consistent(
    tmp_path: Path,
) -> None:
    gt = np.zeros((10, 10, 10), dtype=np.uint8)
    pred = np.zeros_like(gt)

    gt[2:6, 2:6, 2:6] = 3
    pred[2:4, 2:6, 2:6] = 116
    pred[8:10, 8:10, 8:10] = 117

    gt_path = tmp_path / "gt.nii.gz"
    pred_path = tmp_path / "pred.nii.gz"
    _save(gt_path, gt)
    _save(pred_path, pred)

    result = analyze_cyst_case(
        case_id="fixture",
        ground_truth_path=gt_path,
        automatic_prediction_path=pred_path,
    )

    assert result.recall == pytest.approx(0.5)
    assert result.precision == pytest.approx(32 / 40)
    assert result.dice == pytest.approx(64 / 104)
    assert result.dominant_error == "undersegmentation_fn"


def _row(
    index: int,
    volume: float,
    dice: float,
) -> CystCaseFailure:
    return CystCaseFailure(
        case_id=f"case_{index:05d}",
        reference_volume_ml=volume,
        prediction_volume_ml=volume,
        intersection_volume_ml=volume * dice,
        dice=dice,
        precision=dice,
        recall=dice,
        false_negative_volume_ml=volume * (1.0 - dice),
        false_positive_volume_ml=0.0,
        reference_component_count=1,
        prediction_component_count=1,
        largest_reference_component_ml=volume,
        largest_prediction_component_ml=volume,
        dominant_error="undersegmentation_fn",
    )


def test_summary_stratifies_reference_volume() -> None:
    rows = tuple(
        _row(
            index,
            volume=float(index + 1),
            dice=float(index + 1) / 10.0,
        )
        for index in range(8)
    )

    payload = summarize_cyst_failures(
        rows
    )

    assert payload["case_count"] == 8
    assert payload["by_reference_volume_quartile"][
        "smallest"
    ]["count"] == 2
    assert payload["by_reference_volume_quartile"][
        "largest"
    ]["count"] == 2
    assert payload["associations"][
        "reference_volume_vs_dice_pearson"
    ] > 0.99


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="No cyst"):
        summarize_cyst_failures([])


def test_expanded100_manifest_reader_handles_nested_case_ids(
    tmp_path: Path,
) -> None:
    import importlib.util
    import json

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_nv_segment_ct_expanded100_cyst_failures.py"
    )
    spec = importlib.util.spec_from_file_location(
        "cyst_failure_script",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = {
        "selection": {
            "frozen_cases": [
                {
                    "metadata": {
                        "case_id": f"case_{index:05d}"
                    }
                }
                for index in range(100)
            ]
        }
    }

    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    case_ids = module._load_expanded100_case_ids(path)

    assert len(case_ids) == 100
    assert case_ids[0] == "case_00000"
    assert case_ids[-1] == "case_00099"
