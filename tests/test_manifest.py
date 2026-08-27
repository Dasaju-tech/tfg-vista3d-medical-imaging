import json
from pathlib import Path

from vista3d_tfg.data.kits23 import KiTS23Case
from vista3d_tfg.data.manifest import build_manifest, case_to_record, write_manifest


def _case(root: Path, case_id: str, number: int, with_instances: bool) -> KiTS23Case:
    directory = root / case_id
    instances = directory / "instances" if with_instances else None

    return KiTS23Case(
        case_id=case_id,
        case_number=number,
        directory=directory,
        imaging_path=directory / "imaging.nii.gz",
        segmentation_path=directory / "segmentation.nii.gz",
        instances_path=instances,
    )


def test_case_to_record(tmp_path: Path) -> None:
    case = _case(tmp_path, "case_00007", 7, True)

    record = case_to_record(case)

    assert record["case_id"] == "case_00007"
    assert record["case_number"] == 7
    assert record["has_instances"] is True
    assert record["instances_path"].endswith("instances")


def test_manifest_build_and_write(tmp_path: Path) -> None:
    cases = (
        _case(tmp_path, "case_00001", 1, False),
        _case(tmp_path, "case_00002", 2, True),
    )

    manifest = build_manifest(cases)

    assert manifest["dataset"] == "KiTS23"
    assert manifest["number_of_cases"] == 2
    assert manifest["cases"][0]["instances_path"] is None

    output_path = tmp_path / "nested" / "manifest.json"
    write_manifest(manifest, output_path)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"
    assert loaded["number_of_cases"] == 2
