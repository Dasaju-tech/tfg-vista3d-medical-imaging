from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.reports.structured_findings import (
    LEFT_CYST_LABEL,
    RIGHT_CYST_LABEL,
    build_structured_case_findings,
    measure_automatic_renal_prediction,
    render_deterministic_report,
    sha256_file,
    validate_structured_findings,
)


def save_nifti(
    path: Path,
    array: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    affine = np.diag(
        [spacing[0], spacing[1], spacing[2], 1.0]
    )
    image = nib.Nifti1Image(array, affine)
    image.header.set_zooms(spacing)
    nib.save(image, path)


def write_workflow(
    path: Path,
    *,
    ground_truth_used: bool = False,
    laterality: str = "right",
    class_label: int = RIGHT_CYST_LABEL,
) -> None:
    payload = {
        "ground_truth_used": ground_truth_used,
        "clinical_validation": False,
        "result": {
            "case_id": "case_00002",
            "point_xyz": [4, 5, 6],
            "semantic_assignment": {
                "laterality": laterality,
                "class_label": class_label,
                "class_name": f"{laterality} kidney cyst",
                "distance_to_left_kidney_mm": 100.0,
                "distance_to_right_kidney_mm": 5.0,
                "selected_distance_mm": 5.0,
                "laterality_margin_mm": 95.0,
            },
            "labels_present": [0, class_label, 255],
            "contains_255_unprocessed": True,
            "foreground_voxels": 100,
            "foreground_volume_ml": 0.1,
            "wall_time_seconds": 1.0,
            "prediction_path": "fixture.nii.gz",
        },
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_measurement_uses_physical_voxel_volume(
    tmp_path: Path,
) -> None:
    labels = np.zeros((5, 5, 5), dtype=np.uint8)
    labels[0:2, 0:2, 0:2] = 14
    labels[2:4, 0:2, 0:2] = 5
    labels[0, 4, 4] = 116
    labels[4, 4, 4] = 117

    path = tmp_path / "prediction.nii.gz"
    save_nifti(path, labels, spacing=(2.0, 1.0, 1.0))

    result = measure_automatic_renal_prediction(path)

    assert result.left_kidney_core_volume_ml == pytest.approx(0.016)
    assert result.right_kidney_core_volume_ml == pytest.approx(0.016)
    assert result.automatic_left_cyst_volume_ml == pytest.approx(0.002)
    assert result.automatic_right_cyst_volume_ml == pytest.approx(0.002)


def test_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"vista3d")

    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_ground_truth_workflow_is_rejected(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "workflow.json"
    write_workflow(
        workflow,
        ground_truth_used=True,
    )

    from vista3d_tfg.reports.structured_findings import (
        load_user_guided_finding,
    )

    with pytest.raises(ValueError, match="ground_truth_used=false"):
        load_user_guided_finding(workflow)


def test_laterality_class_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "workflow.json"
    write_workflow(
        workflow,
        laterality="left",
        class_label=RIGHT_CYST_LABEL,
    )

    from vista3d_tfg.reports.structured_findings import (
        load_user_guided_finding,
    )

    with pytest.raises(ValueError, match="disagree"):
        load_user_guided_finding(workflow)


def test_structured_report_is_gt_free_and_deterministic(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.nii.gz"
    automatic = tmp_path / "automatic.nii.gz"
    interactive = tmp_path / "interactive.nii.gz"
    workflow = tmp_path / "workflow.json"

    save_nifti(
        image,
        np.zeros((10, 10, 10), dtype=np.int16),
    )

    automatic_array = np.zeros((10, 10, 10), dtype=np.uint8)
    automatic_array[1:4, 1:4, 1:4] = 14
    automatic_array[6:9, 1:4, 1:4] = 5
    automatic_array[7:8, 2:3, 2:3] = 117
    save_nifti(automatic, automatic_array)

    interactive_array = np.zeros((10, 10, 10), dtype=np.uint8)
    interactive_array[7:9, 2:4, 2:4] = 117
    save_nifti(interactive, interactive_array)

    write_workflow(workflow)

    findings = build_structured_case_findings(
        case_id="case_00002",
        image_path=image,
        automatic_prediction_path=automatic,
        workflow_path=workflow,
        interactive_prediction_path=interactive,
    )

    validate_structured_findings(findings)

    first = render_deterministic_report(findings)
    second = render_deterministic_report(findings)

    assert first == second
    assert "No ground-truth annotation was used" in first
    assert "right / class 117" in first
    assert "malignancy" in first
    assert findings.ground_truth_used is False


def test_missing_evidence_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        build_structured_case_findings(
            case_id="case_00002",
            image_path=tmp_path / "missing_image.nii.gz",
            automatic_prediction_path=tmp_path / "missing_auto.nii.gz",
            workflow_path=tmp_path / "missing_workflow.json",
            interactive_prediction_path=tmp_path / "missing_interactive.nii.gz",
        )
