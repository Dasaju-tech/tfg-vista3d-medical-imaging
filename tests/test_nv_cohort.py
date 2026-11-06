from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vista3d_tfg.evaluation.cohort import (
    CaseCohortRow,
    summarize_cohort,
    summarize_values,
    write_cohort_csv,
    write_cohort_json,
)


def row(
    case_id: str,
    *,
    kidney_dice: float,
    surface_dice: float,
    hd95: float,
    cyst_dice: float,
    cyst_gt: float,
    cyst_pred: float,
    runtime: float | None = None,
) -> CaseCohortRow:
    return CaseCohortRow(
        case_id=case_id,
        kidney_and_mass_dice=kidney_dice,
        kidney_and_mass_surface_dice=surface_dice,
        kidney_and_mass_hd95_mm=hd95,
        kidney_core_dice=kidney_dice - 0.01,
        cyst_dice=cyst_dice,
        cyst_reference_volume_ml=cyst_gt,
        cyst_prediction_volume_ml=cyst_pred,
        left_renal_dice=kidney_dice - 0.005,
        right_renal_dice=kidney_dice + 0.005,
        left_renal_volume_error_ml=-5.0,
        right_renal_volume_error_ml=-4.0,
        gt_tumor_volume_ml=8.0,
        inference_wall_time_seconds=runtime,
        qc_false_positive_voxels=100,
        qc_false_negative_voxels=200,
        qc_total_disagreement_voxels=300,
    )


def test_summarize_values() -> None:
    summary = summarize_values([1.0, 2.0, 3.0])

    assert summary.count == 3
    assert summary.mean == pytest.approx(2.0)
    assert summary.median == pytest.approx(2.0)
    assert summary.std == pytest.approx(1.0)
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0


def test_empty_summary_is_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_values([])


def test_cyst_positive_summary_excludes_empty_perfect_cases() -> None:
    summary = summarize_cohort(
        [
            row(
                "case_00000",
                kidney_dice=0.95,
                surface_dice=0.88,
                hd95=1.8,
                cyst_dice=1.0,
                cyst_gt=0.0,
                cyst_pred=0.0,
            ),
            row(
                "case_00001",
                kidney_dice=0.94,
                surface_dice=0.84,
                hd95=5.8,
                cyst_dice=1.0,
                cyst_gt=0.0,
                cyst_pred=0.0,
            ),
            row(
                "case_00002",
                kidney_dice=0.94,
                surface_dice=0.78,
                hd95=4.2,
                cyst_dice=0.54,
                cyst_gt=40.0,
                cyst_pred=47.0,
            ),
        ]
    )

    assert summary.cyst_positive_case_count == 1
    assert summary.cyst_dice_all_cases.mean > 0.8
    assert summary.cyst_dice_cyst_positive_cases is not None
    assert summary.cyst_dice_cyst_positive_cases.mean == pytest.approx(
        0.54
    )


def test_duplicate_case_ids_are_rejected() -> None:
    fixture = row(
        "case_00000",
        kidney_dice=0.95,
        surface_dice=0.88,
        hd95=1.8,
        cyst_dice=1.0,
        cyst_gt=0.0,
        cyst_pred=0.0,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        summarize_cohort([fixture, fixture])


def test_runtime_summary_uses_available_values_only() -> None:
    summary = summarize_cohort(
        [
            row(
                "case_00000",
                kidney_dice=0.95,
                surface_dice=0.88,
                hd95=1.8,
                cyst_dice=1.0,
                cyst_gt=0.0,
                cyst_pred=0.0,
                runtime=20.0,
            ),
            row(
                "case_00001",
                kidney_dice=0.94,
                surface_dice=0.84,
                hd95=5.8,
                cyst_dice=1.0,
                cyst_gt=0.0,
                cyst_pred=0.0,
                runtime=None,
            ),
        ]
    )

    assert summary.inference_wall_time_seconds is not None
    assert summary.inference_wall_time_seconds.count == 1
    assert summary.inference_wall_time_seconds.mean == 20.0


def test_json_and_csv_writers(tmp_path: Path) -> None:
    summary = summarize_cohort(
        [
            row(
                "case_00000",
                kidney_dice=0.95,
                surface_dice=0.88,
                hd95=1.8,
                cyst_dice=1.0,
                cyst_gt=0.0,
                cyst_pred=0.0,
                runtime=20.0,
            )
        ]
    )

    json_path = tmp_path / "summary.json"
    csv_path = tmp_path / "cases.csv"

    write_cohort_json(summary, json_path)
    write_cohort_csv(summary, csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["case_count"] == 1
    assert payload["full_kits23_score_available"] is False

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["case_id"] == "case_00000"
