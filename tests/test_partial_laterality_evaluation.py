from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from vista3d_tfg.evaluation.cohort import (
    load_case_cohort_row,
    summarize_cohort,
)
from vista3d_tfg.evaluation.nv_kits23 import (
    evaluate_supported_renal_prediction,
    write_supported_renal_comparison,
)
from vista3d_tfg.experiments.pilot30 import write_run_manifest


def test_unresolved_gt_laterality_preserves_global_metrics(
    tmp_path: Path,
) -> None:
    ground_truth = np.zeros((12, 12, 12), dtype=np.uint8)
    ground_truth[2:10, 2:10, 2:10] = 1

    prediction = np.zeros_like(ground_truth)
    prediction[2:10, 2:10, 2:10] = 14

    gt_path = tmp_path / "segmentation.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"

    nib.save(nib.Nifti1Image(ground_truth, np.eye(4)), gt_path)
    nib.save(nib.Nifti1Image(prediction, np.eye(4)), pred_path)

    result = evaluate_supported_renal_prediction(
        case_id="unilateral_fixture",
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert result.ground_truth_laterality_resolved is False
    assert result.ground_truth_laterality_reason
    assert result.kidney_and_mass_hec.dice == 1.0
    assert result.kidney_core_diagnostic.dice == 1.0
    assert result.left_renal_envelope is None
    assert result.right_renal_envelope is None
    assert result.left_cyst is None
    assert result.right_cyst is None

    evaluation_path = tmp_path / "supported_renal_metrics.json"
    write_supported_renal_comparison(result, evaluation_path)

    payload = json.loads(
        evaluation_path.read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "1.1"

    row = load_case_cohort_row(
        case_id="unilateral_fixture",
        evaluation_path=evaluation_path,
    )

    assert row.ground_truth_laterality_resolved is False
    assert row.kidney_and_mass_dice == 1.0
    assert row.left_renal_dice is None
    assert row.right_renal_dice is None

    summary = summarize_cohort([row])

    assert summary.case_count == 1
    assert summary.kidney_and_mass_dice.count == 1
    assert summary.kidney_and_mass_dice.mean == 1.0
    assert summary.laterality_resolved_case_count == 0
    assert summary.laterality_unresolved_case_count == 1
    assert summary.left_renal_dice is None
    assert summary.right_renal_dice is None


def test_schema10_artifacts_remain_backward_compatible(
    tmp_path: Path,
) -> None:
    evaluation_path = tmp_path / "legacy.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": {
                    "kidney_and_mass_hec": {
                        "dice": 0.9,
                        "surface_dice": 0.8,
                        "diagnostic_hd95_mm": 3.0,
                    },
                    "kidney_core_diagnostic": {"dice": 0.89},
                    "cyst": {
                        "dice": 1.0,
                        "reference_volume_ml": 0.0,
                        "prediction_volume_ml": 0.0,
                    },
                    "left_renal_envelope": {
                        "dice": 0.91,
                        "signed_volume_error_ml": -1.0,
                    },
                    "right_renal_envelope": {
                        "dice": 0.92,
                        "signed_volume_error_ml": -2.0,
                    },
                    "gt_tumor_volume_ml": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )

    row = load_case_cohort_row(
        case_id="legacy",
        evaluation_path=evaluation_path,
    )

    assert row.ground_truth_laterality_resolved is True
    assert row.left_renal_dice == 0.91
    assert row.right_renal_dice == 0.92


def test_run_manifest_accepts_explicit_experiment_name(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run_manifest.json"

    write_run_manifest(
        output_path=output,
        requested_case_ids=("case_a",),
        successful_case_ids=("case_a",),
        failed_cases={},
        worst_qc_case_ids=(),
        experiment="nv_segment_ct_expanded100",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "nv_segment_ct_expanded100"
