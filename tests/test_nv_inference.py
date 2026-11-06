from __future__ import annotations

import json
from pathlib import Path
import subprocess

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.integration.nv_inference import (
    EXPECTED_RENAL_LABELS,
    RENAL_LABEL_PROMPT,
    audit_renal_prediction,
    build_renal_inference_command,
    discover_single_prediction,
    load_upstream_label_dict,
    validate_renal_label_contract,
)
from vista3d_tfg.integration.nv_segment import runtime_paths


def test_renal_label_prompt_is_explicit_and_stable() -> None:
    assert RENAL_LABEL_PROMPT == (5, 14, 116, 117)
    assert EXPECTED_RENAL_LABELS == {
        "right kidney": 5,
        "left kidney": 14,
        "left kidney cyst": 116,
        "right kidney cyst": 117,
    }


def test_validate_renal_label_contract() -> None:
    validate_renal_label_contract(EXPECTED_RENAL_LABELS.copy())


def test_changed_upstream_label_is_rejected() -> None:
    labels = EXPECTED_RENAL_LABELS.copy()
    labels["left kidney"] = 999

    with pytest.raises(RuntimeError, match="contract changed"):
        validate_renal_label_contract(labels)


def test_load_upstream_label_dict(tmp_path: Path) -> None:
    path = tmp_path / "label_dict.json"
    path.write_text(
        json.dumps(EXPECTED_RENAL_LABELS),
        encoding="utf-8",
    )

    assert load_upstream_label_dict(path) == EXPECTED_RENAL_LABELS


def test_command_is_windows_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_conda = tmp_path / "conda.exe"
    fake_conda.write_bytes(b"fixture")

    monkeypatch.setattr(
        "vista3d_tfg.integration.nv_inference.resolve_conda_executable",
        lambda: str(fake_conda),
    )

    command = build_renal_inference_command(
        environment_name="vista3d-nv",
        image_path=tmp_path / "a folder" / "imaging.nii.gz",
        output_dir=tmp_path / "output folder",
    )

    assert command[0] == str(fake_conda)
    assert command[:5] == [
        str(fake_conda),
        "run",
        "--no-capture-output",
        "-n",
        "vista3d-nv",
    ]
    assert all("\n" not in str(value) for value in command)
    assert "'label_prompt': [5, 14, 116, 117]" in command[-3]
    assert "\\" not in command[-3]


def test_discover_single_prediction(tmp_path: Path) -> None:
    output = tmp_path / "imaging" / "imaging_trans.nii.gz"
    output.parent.mkdir()
    output.write_bytes(b"fixture")

    assert discover_single_prediction(tmp_path) == output


def test_multiple_predictions_are_rejected(tmp_path: Path) -> None:
    for name in ("a_trans.nii.gz", "b_trans.nii.gz"):
        path = tmp_path / name
        path.write_bytes(b"fixture")

    with pytest.raises(RuntimeError, match="exactly one"):
        discover_single_prediction(tmp_path)


def write_pair(
    tmp_path: Path,
    *,
    prediction: np.ndarray,
    prediction_affine: np.ndarray | None = None,
) -> tuple[Path, Path]:
    image = np.zeros(prediction.shape, dtype=np.int16)

    input_path = tmp_path / "input.nii.gz"
    prediction_path = tmp_path / "prediction.nii.gz"

    nib.save(
        nib.Nifti1Image(image, np.eye(4)),
        input_path,
    )
    nib.save(
        nib.Nifti1Image(
            prediction,
            (
                np.eye(4)
                if prediction_affine is None
                else prediction_affine
            ),
        ),
        prediction_path,
    )

    return input_path, prediction_path


def test_prediction_audit_accepts_supported_labels(
    tmp_path: Path,
) -> None:
    prediction = np.zeros((8, 8, 8), dtype=np.float32)
    prediction[1:3, 1:3, 1:3] = 5
    prediction[5:7, 1:3, 1:3] = 14
    prediction[1, 5, 5] = 116
    prediction[6, 5, 5] = 117

    input_path, prediction_path = write_pair(
        tmp_path,
        prediction=prediction,
    )

    audit = audit_renal_prediction(
        input_path=input_path,
        prediction_path=prediction_path,
        labels=EXPECTED_RENAL_LABELS,
    )

    assert audit.is_valid
    assert audit.labels_present == (0, 5, 14, 116, 117)
    assert audit.unexpected_labels == ()
    assert audit.affine_matches_input
    assert audit.shape_matches_input


def test_prediction_audit_rejects_unexpected_label(
    tmp_path: Path,
) -> None:
    prediction = np.zeros((4, 4, 4), dtype=np.float32)
    prediction[1, 1, 1] = 23

    input_path, prediction_path = write_pair(
        tmp_path,
        prediction=prediction,
    )

    audit = audit_renal_prediction(
        input_path=input_path,
        prediction_path=prediction_path,
        labels=EXPECTED_RENAL_LABELS,
    )

    assert not audit.is_valid
    assert audit.unexpected_labels == (23,)


def test_prediction_audit_rejects_fractional_values(
    tmp_path: Path,
) -> None:
    prediction = np.zeros((4, 4, 4), dtype=np.float32)
    prediction[1, 1, 1] = 5.5

    input_path, prediction_path = write_pair(
        tmp_path,
        prediction=prediction,
    )

    audit = audit_renal_prediction(
        input_path=input_path,
        prediction_path=prediction_path,
        labels=EXPECTED_RENAL_LABELS,
    )

    assert not audit.is_valid
    assert not audit.integer_valued


def test_prediction_audit_detects_affine_mismatch(
    tmp_path: Path,
) -> None:
    prediction = np.zeros((4, 4, 4), dtype=np.float32)

    affine = np.eye(4)
    affine[0, 3] = 25.0

    input_path, prediction_path = write_pair(
        tmp_path,
        prediction=prediction,
        prediction_affine=affine,
    )

    audit = audit_renal_prediction(
        input_path=input_path,
        prediction_path=prediction_path,
        labels=EXPECTED_RENAL_LABELS,
    )

    assert not audit.is_valid
    assert not audit.affine_matches_input
