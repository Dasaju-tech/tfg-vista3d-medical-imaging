from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from vista3d_tfg.data.acquisition import (
    RemoteFile,
    build_case_sources,
    download_remote_file,
    format_case_id,
    sha256_file,
)


class FakeResponse(BytesIO):
    def __init__(self, payload: bytes, etag: str | None = None):
        super().__init__(payload)
        self.headers = {}
        if etag is not None:
            self.headers["ETag"] = etag


def test_format_case_id() -> None:
    assert format_case_id(0) == "case_00000"
    assert format_case_id(42) == "case_00042"


@pytest.mark.parametrize("value", [-1, -42])
def test_negative_case_number_is_rejected(value: int) -> None:
    with pytest.raises(ValueError):
        format_case_id(value)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_non_integer_case_number_is_rejected(value) -> None:
    with pytest.raises(TypeError):
        format_case_id(value)


def test_build_case_sources_matches_official_layout(
    tmp_path: Path,
) -> None:
    image, segmentation = build_case_sources(7, tmp_path)

    assert image.case_id == "case_00007"
    assert image.url.endswith("/images/case_00007.nii.gz")
    assert image.destination == (
        tmp_path.resolve()
        / "case_00007"
        / "imaging.nii.gz"
    )

    assert segmentation.url.endswith(
        "/dataset/case_00007/segmentation.nii.gz"
    )
    assert segmentation.destination == (
        tmp_path.resolve()
        / "case_00007"
        / "segmentation.nii.gz"
    )


def test_download_remote_file_is_atomic_and_hashed(
    tmp_path: Path,
) -> None:
    payload = b"synthetic-nifti-payload"

    source = RemoteFile(
        case_id="case_00000",
        kind="imaging",
        url="https://example.invalid/imaging.nii.gz",
        destination=tmp_path / "case_00000" / "imaging.nii.gz",
    )

    def opener(request, timeout):
        assert request.full_url == source.url
        assert timeout == 10.0
        return FakeResponse(payload, etag='"fixture-etag"')

    record = download_remote_file(
        source,
        timeout_seconds=10.0,
        retries=0,
        opener=opener,
    )

    assert record.status == "downloaded"
    assert record.size_bytes == len(payload)
    assert record.etag == '"fixture-etag"'
    assert source.destination.read_bytes() == payload
    assert record.sha256 == sha256_file(source.destination)
    assert not (
        source.destination.parent / ".partial.imaging.nii.gz"
    ).exists()


def test_existing_non_empty_file_is_reused(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "case_00000" / "imaging.nii.gz"
    destination.parent.mkdir()
    destination.write_bytes(b"existing")

    source = RemoteFile(
        case_id="case_00000",
        kind="imaging",
        url="https://example.invalid/imaging.nii.gz",
        destination=destination,
    )

    def opener(*args, **kwargs):
        raise AssertionError("Network must not be used for a reused file.")

    record = download_remote_file(
        source,
        opener=opener,
    )

    assert record.status == "reused"
    assert record.size_bytes == len(b"existing")


def test_force_redownload_replaces_existing_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "case_00000" / "imaging.nii.gz"
    destination.parent.mkdir()
    destination.write_bytes(b"old")

    source = RemoteFile(
        case_id="case_00000",
        kind="imaging",
        url="https://example.invalid/imaging.nii.gz",
        destination=destination,
    )

    def opener(*args, **kwargs):
        return FakeResponse(b"new")

    record = download_remote_file(
        source,
        force=True,
        retries=0,
        opener=opener,
    )

    assert record.status == "downloaded"
    assert destination.read_bytes() == b"new"


def test_failed_download_removes_partial_file(
    tmp_path: Path,
) -> None:
    source = RemoteFile(
        case_id="case_00000",
        kind="imaging",
        url="https://example.invalid/imaging.nii.gz",
        destination=tmp_path / "case_00000" / "imaging.nii.gz",
    )

    def opener(*args, **kwargs):
        raise OSError("network fixture failure")

    with pytest.raises(OSError):
        download_remote_file(
            source,
            retries=0,
            opener=opener,
        )

    assert not source.destination.exists()
    assert not (
        source.destination.parent / ".partial.imaging.nii.gz"
    ).exists()


def test_retry_succeeds_on_second_attempt(
    tmp_path: Path,
) -> None:
    source = RemoteFile(
        case_id="case_00000",
        kind="imaging",
        url="https://example.invalid/imaging.nii.gz",
        destination=tmp_path / "case_00000" / "imaging.nii.gz",
    )

    attempts = 0
    sleeps = []

    def opener(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary failure")
        return FakeResponse(b"success")

    record = download_remote_file(
        source,
        retries=1,
        opener=opener,
        sleep_fn=sleeps.append,
    )

    assert record.status == "downloaded"
    assert attempts == 2
    assert sleeps == [1.0]


def test_negative_retry_count_is_rejected(
    tmp_path: Path,
) -> None:
    source = RemoteFile(
        case_id="case_00000",
        kind="imaging",
        url="https://example.invalid/imaging.nii.gz",
        destination=tmp_path / "imaging.nii.gz",
    )

    with pytest.raises(ValueError):
        download_remote_file(source, retries=-1)
