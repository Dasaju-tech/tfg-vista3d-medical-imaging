from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.analysis.kits23_baseline import (
    analyze_kits23_case,
    cohort_summary,
    write_case_baselines,
    write_case_table_csv,
    write_cohort_summary,
)


def write_nifti_pair(
    root: Path,
    *,
    image: np.ndarray,
    segmentation: np.ndarray,
    affine: np.ndarray | None = None,
) -> tuple[Path, Path]:
    affine = np.eye(4) if affine is None else affine

    imaging_path = root / "imaging.nii.gz"
    segmentation_path = root / "segmentation.nii.gz"

    nib.save(
        nib.Nifti1Image(image, affine),
        imaging_path,
    )
    nib.save(
        nib.Nifti1Image(segmentation, affine),
        segmentation_path,
    )

    return imaging_path, segmentation_path


def make_fixture() -> tuple[np.ndarray, np.ndarray]:
    image = np.arange(5 * 6 * 7, dtype=np.int16).reshape(5, 6, 7)
    segmentation = np.zeros((5, 6, 7), dtype=np.uint8)

    segmentation[1:4, 1:5, 1:6] = 1
    segmentation[2:4, 2:4, 2:4] = 2
    segmentation[1:2, 3:5, 4:6] = 3

    return image, segmentation


def test_analyze_case_computes_real_volumes_and_presence(
    tmp_path: Path,
) -> None:
    image, segmentation = make_fixture()

    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    imaging_path, segmentation_path = write_nifti_pair(
        tmp_path,
        image=image,
        segmentation=segmentation,
        affine=affine,
    )

    result = analyze_kits23_case(
        case_id="case_fixture",
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
    )

    assert result.shape == (5, 6, 7)
    assert result.spacing_mm == pytest.approx((2.0, 3.0, 4.0))

    tumor = result.region("tumor")
    cyst = result.region("cyst")

    assert tumor.present
    assert cyst.present
    assert tumor.voxel_count == 8
    assert tumor.volume_ml == pytest.approx(8 * 24 / 1000)
    assert tumor.component_count == 1

    assert cyst.voxel_count == 4
    assert cyst.volume_ml == pytest.approx(4 * 24 / 1000)


def test_world_centroid_uses_affine(
    tmp_path: Path,
) -> None:
    image = np.zeros((4, 4, 4), dtype=np.int16)
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1, 2, 3] = 2

    affine = np.diag([2.0, 3.0, 4.0, 1.0])

    imaging_path, segmentation_path = write_nifti_pair(
        tmp_path,
        image=image,
        segmentation=segmentation,
        affine=affine,
    )

    result = analyze_kits23_case(
        case_id="case_fixture",
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
    )

    assert result.region("tumor").centroid_voxel == pytest.approx(
        (1.0, 2.0, 3.0)
    )
    assert result.region("tumor").centroid_world_mm == pytest.approx(
        (2.0, 6.0, 12.0)
    )


def test_absent_region_has_zero_volume_and_no_centroid(
    tmp_path: Path,
) -> None:
    image = np.zeros((4, 4, 4), dtype=np.int16)
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1:3, 1:3, 1:3] = 1

    imaging_path, segmentation_path = write_nifti_pair(
        tmp_path,
        image=image,
        segmentation=segmentation,
    )

    result = analyze_kits23_case(
        case_id="case_fixture",
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
    )

    tumor = result.region("tumor")

    assert not tumor.present
    assert tumor.volume_ml == 0.0
    assert tumor.component_count == 0
    assert tumor.centroid_voxel is None
    assert tumor.centroid_world_mm is None
    assert result.tumor_burden_fraction == 0.0


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 4), dtype=np.int16)
    segmentation = np.zeros((5, 4, 4), dtype=np.uint8)

    imaging_path, segmentation_path = write_nifti_pair(
        tmp_path,
        image=image,
        segmentation=segmentation,
    )

    with pytest.raises(ValueError, match="shapes do not match"):
        analyze_kits23_case(
            case_id="case_fixture",
            imaging_path=imaging_path,
            segmentation_path=segmentation_path,
        )


def test_affine_mismatch_is_rejected(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 4), dtype=np.int16)
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)

    imaging_path = tmp_path / "imaging.nii.gz"
    segmentation_path = tmp_path / "segmentation.nii.gz"

    nib.save(
        nib.Nifti1Image(image, np.eye(4)),
        imaging_path,
    )

    different_affine = np.eye(4)
    different_affine[0, 3] = 10.0

    nib.save(
        nib.Nifti1Image(segmentation, different_affine),
        segmentation_path,
    )

    with pytest.raises(ValueError, match="affines do not match"):
        analyze_kits23_case(
            case_id="case_fixture",
            imaging_path=imaging_path,
            segmentation_path=segmentation_path,
        )


def test_unexpected_label_is_rejected(tmp_path: Path) -> None:
    image = np.zeros((4, 4, 4), dtype=np.int16)
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1, 1, 1] = 4

    imaging_path, segmentation_path = write_nifti_pair(
        tmp_path,
        image=image,
        segmentation=segmentation,
    )

    with pytest.raises(ValueError, match="Unexpected KiTS23 labels"):
        analyze_kits23_case(
            case_id="case_fixture",
            imaging_path=imaging_path,
            segmentation_path=segmentation_path,
        )


def test_cohort_summary_aggregates_cases(tmp_path: Path) -> None:
    image, segmentation = make_fixture()

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()

    first_paths = write_nifti_pair(
        first_dir,
        image=image,
        segmentation=segmentation,
    )
    second_paths = write_nifti_pair(
        second_dir,
        image=image,
        segmentation=segmentation,
    )

    cases = [
        analyze_kits23_case(
            case_id="case_a",
            imaging_path=first_paths[0],
            segmentation_path=first_paths[1],
        ),
        analyze_kits23_case(
            case_id="case_b",
            imaging_path=second_paths[0],
            segmentation_path=second_paths[1],
        ),
    ]

    summary = cohort_summary(cases)

    assert summary["number_of_cases"] == 2
    assert summary["regions"]["tumor"]["present_cases"] == 2
    assert summary["regions"]["tumor"]["prevalence"] == pytest.approx(1.0)


def test_empty_cohort_is_rejected() -> None:
    with pytest.raises(ValueError):
        cohort_summary([])


def test_writers_create_json_and_csv(tmp_path: Path) -> None:
    image, segmentation = make_fixture()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    imaging_path, segmentation_path = write_nifti_pair(
        data_dir,
        image=image,
        segmentation=segmentation,
    )

    case = analyze_kits23_case(
        case_id="case_fixture",
        imaging_path=imaging_path,
        segmentation_path=segmentation_path,
    )

    cases_json = tmp_path / "out" / "cases.json"
    cohort_json = tmp_path / "out" / "cohort.json"
    table_csv = tmp_path / "out" / "table.csv"

    write_case_baselines([case], cases_json)
    write_cohort_summary([case], cohort_json)
    write_case_table_csv([case], table_csv)

    cases_payload = json.loads(cases_json.read_text(encoding="utf-8"))
    cohort_payload = json.loads(cohort_json.read_text(encoding="utf-8"))

    assert cases_payload["dataset"] == "KiTS23"
    assert cases_payload["cases"][0]["case_id"] == "case_fixture"
    assert cohort_payload["number_of_cases"] == 1

    with table_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 1
    assert rows[0]["case_id"] == "case_fixture"
    assert float(rows[0]["tumor_volume_ml"]) > 0.0
