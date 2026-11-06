from __future__ import annotations

import json
from pathlib import Path

import pytest

from vista3d_tfg.evaluation.cohort import CaseCohortRow
from vista3d_tfg.experiments.pilot30 import (
    PILOT30_CASES,
    PILOT30_INVALID_V1_CASES,
    PILOT30_REPLACEMENT_CASES,
    PILOT30_SEED,
    PILOT30_SELECTION_REVISION,
    deterministic_pilot30_cases,
    pilot_case_paths,
    select_worst_surface_dice_cases,
    selection_manifest_payload,
    write_selection_manifest,
)


EXPECTED_PILOT30 = (
    0, 1, 2, 17, 47, 63, 109, 116, 142, 143,
    171, 190, 193, 200, 237, 255, 275, 293, 297, 299,
    416, 418, 463, 471, 473, 474, 479, 480, 481, 482,
)


def make_row(
    case_id: str,
    *,
    surface_dice: float,
) -> CaseCohortRow:
    return CaseCohortRow(
        case_id=case_id,
        kidney_and_mass_dice=0.94,
        kidney_and_mass_surface_dice=surface_dice,
        kidney_and_mass_hd95_mm=4.0,
        kidney_core_dice=0.93,
        cyst_dice=1.0,
        cyst_reference_volume_ml=0.0,
        cyst_prediction_volume_ml=0.0,
        left_renal_dice=0.94,
        right_renal_dice=0.94,
        left_renal_volume_error_ml=-5.0,
        right_renal_volume_error_ml=-4.0,
        gt_tumor_volume_ml=8.0,
        inference_wall_time_seconds=20.0,
        qc_false_positive_voxels=None,
        qc_false_negative_voxels=None,
        qc_total_disagreement_voxels=None,
    )


def test_pilot30_selection_is_frozen_and_reproducible() -> None:
    assert PILOT30_SEED == 20260828
    assert PILOT30_CASES == EXPECTED_PILOT30
    assert deterministic_pilot30_cases() == EXPECTED_PILOT30


def test_pilot30_selection_is_valid() -> None:
    assert len(PILOT30_CASES) == 30
    assert len(set(PILOT30_CASES)) == 30
    assert PILOT30_CASES[:3] == (0, 1, 2)
    assert min(PILOT30_CASES) >= 0
    assert not set(PILOT30_CASES) & set(PILOT30_INVALID_V1_CASES)
    assert set(PILOT30_REPLACEMENT_CASES).issubset(PILOT30_CASES)


def test_pilot_case_paths_are_deterministic(
    tmp_path: Path,
) -> None:
    paths = pilot_case_paths(tmp_path, 17)

    assert paths.case_id == "case_00017"
    assert paths.imaging_path == (
        tmp_path.resolve()
        / "data"
        / "raw"
        / "kits23"
        / "case_00017"
        / "imaging.nii.gz"
    )
    assert paths.prediction_path.name == "prediction.nii.gz"
    assert paths.evaluation_json.name == "supported_renal_metrics.json"


def test_worst_surface_dice_selection() -> None:
    rows = [
        make_row("case_a", surface_dice=0.90),
        make_row("case_b", surface_dice=0.70),
        make_row("case_c", surface_dice=0.80),
    ]

    assert select_worst_surface_dice_cases(
        rows,
        count=2,
    ) == ("case_b", "case_c")


def test_negative_worst_case_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        select_worst_surface_dice_cases(
            [],
            count=-1,
        )


def test_selection_manifest_records_interpretation_boundary() -> None:
    payload = selection_manifest_payload()

    assert payload["experiment"] == "nv_segment_ct_pilot30"
    assert payload["selection"]["requested_case_count"] == 30
    assert payload["selection"]["seed"] == 20260828
    assert payload["selection"]["revision"] == PILOT30_SELECTION_REVISION
    assert payload["selection"]["invalid_revision_1_case_numbers"] == [
        324, 345, 396, 398
    ]
    assert payload["selection"]["replacement_case_numbers"] == [
        474, 479, 481, 482
    ]
    assert "not a population-representative" in (
        payload["interpretation_boundary"]
    )


def test_write_selection_manifest(tmp_path: Path) -> None:
    output = tmp_path / "selection.json"

    write_selection_manifest(output)

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["selection"]["case_numbers"] == list(
        EXPECTED_PILOT30
    )
