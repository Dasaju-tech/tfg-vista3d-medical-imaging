import json
from pathlib import Path

from vista3d_tfg.data.acquisition import DownloadedFile
from vista3d_tfg.data.provenance import (
    acquisition_provenance_to_dict,
    write_acquisition_provenance,
)


def make_record(tmp_path: Path) -> DownloadedFile:
    return DownloadedFile(
        case_id="case_00000",
        kind="imaging",
        source_url="https://example.invalid/case_00000.nii.gz",
        path=tmp_path / "case_00000" / "imaging.nii.gz",
        sha256="a" * 64,
        size_bytes=123,
        status="downloaded",
        etag='"fixture"',
    )


def test_acquisition_provenance_contains_reproducibility_fields(
    tmp_path: Path,
) -> None:
    record = make_record(tmp_path)

    payload = acquisition_provenance_to_dict(
        [record],
        retrieved_at_utc="2026-08-27T12:00:00+00:00",
    )

    assert payload["dataset"] == "KiTS23"
    assert payload["dataset_version"] == "0.1.4"
    assert payload["data_license"] == "CC BY-NC-SA 4.0"
    assert payload["number_of_files"] == 1
    assert payload["files"][0]["sha256"] == "a" * 64
    assert payload["files"][0]["etag"] == '"fixture"'


def test_write_acquisition_provenance(
    tmp_path: Path,
) -> None:
    output = tmp_path / "outputs" / "provenance.json"

    write_acquisition_provenance(
        [make_record(tmp_path)],
        output,
        retrieved_at_utc="2026-08-27T12:00:00+00:00",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "1.0"
    assert payload["retrieved_at_utc"] == "2026-08-27T12:00:00+00:00"
    assert payload["files"][0]["case_id"] == "case_00000"
