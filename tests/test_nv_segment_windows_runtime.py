from pathlib import Path

import pytest

from vista3d_tfg.integration.nv_segment import resolve_conda_executable


def test_resolve_conda_prefers_conda_exe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = tmp_path / "Scripts" / "conda.exe"
    fake.parent.mkdir()
    fake.write_bytes(b"fixture")

    monkeypatch.setenv("CONDA_EXE", str(fake))
    monkeypatch.delenv("CONDA_PREFIX", raising=False)

    assert resolve_conda_executable() == str(fake.resolve())


def test_resolve_conda_from_named_environment_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "miniconda"
    active = root / "envs" / "vista3d-tfg-311"
    active.mkdir(parents=True)

    fake = root / "Scripts" / "conda.exe"
    fake.parent.mkdir()
    fake.write_bytes(b"fixture")

    monkeypatch.delenv("CONDA_EXE", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(active))
    monkeypatch.setattr(
        "vista3d_tfg.integration.nv_segment.shutil.which",
        lambda _: None,
    )

    assert resolve_conda_executable() == str(fake.resolve())


def test_probe_command_contains_no_newline_argument() -> None:
    from vista3d_tfg.integration.nv_segment import probe_runtime

    captured = {}

    def runner(command, **kwargs):
        import json
        import subprocess

        captured["command"] = command
        payload = {
            "python_version": "3.11.16",
            "torch_version": "2.8.0+cu128",
            "monai_version": "1.5.0",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "Fixture GPU",
        }
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    probe = probe_runtime(runner=runner)

    assert probe.cuda_available is True
    assert all("\n" not in str(argument) for argument in captured["command"])
