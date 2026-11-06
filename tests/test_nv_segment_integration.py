from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NV_SEGMENT_REPOSITORY_URL,
    RuntimeManifest,
    RuntimeProbe,
    build_single_case_command,
    ensure_repository,
    ensure_runtime_environment,
    list_conda_environments,
    probe_runtime,
    runtime_paths,
    validate_runtime_probe,
    write_runtime_manifest,
)


def completed(
    stdout: str = "",
    *,
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_runtime_paths_stay_in_ignored_data_external(
    tmp_path: Path,
) -> None:
    paths = runtime_paths(tmp_path)

    assert paths.checkout_root == (
        tmp_path.resolve()
        / "data"
        / "external"
        / "NV-Segment-CTMR"
    )
    assert paths.ct_bundle_root.name == "NV-Segment-CT"
    assert paths.requirements_path.parts[-2:] == (
        "NV-Segment-CTMR",
        "requirements.txt",
    )


def test_list_conda_environments_extracts_names() -> None:
    def runner(*args, **kwargs):
        return completed(
            json.dumps(
                {
                    "envs": [
                        r"C:\Users\fixture\miniconda3",
                        r"C:\Users\fixture\miniconda3\envs\vista3d-tfg-311",
                        r"C:\Users\fixture\miniconda3\envs\vista3d-nv",
                    ]
                }
            )
        )

    environments = list_conda_environments(runner=runner)

    assert "vista3d-nv" in environments
    assert "vista3d-tfg-311" in environments


def test_existing_environment_is_reused() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[:4] == ["conda", "env", "list", "--json"]:
            return completed(
                json.dumps(
                    {"envs": [r"C:\envs\vista3d-nv"]}
                )
            )
        raise AssertionError("Environment creation must not run.")

    created = ensure_runtime_environment(runner=runner)

    assert created is False
    assert len(calls) == 1


def test_missing_environment_is_created() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        if command[:4] == ["conda", "env", "list", "--json"]:
            return completed(json.dumps({"envs": []}))
        if command[:3] == ["conda", "create", "-y"]:
            return completed()
        raise AssertionError(command)

    created = ensure_runtime_environment(runner=runner)

    assert created is True
    assert any(
        f"python=3.11" in command
        for command in calls
    )


def test_existing_checkout_is_not_pulled(
    tmp_path: Path,
) -> None:
    paths = runtime_paths(tmp_path)

    (paths.checkout_root / ".git").mkdir(parents=True)
    paths.ct_bundle_root.mkdir()
    paths.requirements_path.parent.mkdir(exist_ok=True)
    paths.requirements_path.write_text("", encoding="utf-8")

    calls = []

    def runner(command, **kwargs):
        calls.append(command)

        if command[:3] == ["git", "-C", str(paths.checkout_root)]:
            return completed("deadbeef\n")

        raise AssertionError(command)

    commit = ensure_repository(paths, runner=runner)

    assert commit == "deadbeef"
    assert not any("pull" in command for command in calls)


def test_probe_runtime_parses_json() -> None:
    payload = {
        "python_version": "3.11.16",
        "torch_version": "2.8.0+cu128",
        "monai_version": "1.5.0",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": "Fixture GPU",
    }

    def runner(*args, **kwargs):
        return completed(json.dumps(payload) + "\n")

    probe = probe_runtime(runner=runner)

    assert probe.python_version == "3.11.16"
    assert probe.cuda_available
    assert probe.cuda_device_name == "Fixture GPU"


def test_valid_runtime_probe_passes() -> None:
    validate_runtime_probe(
        RuntimeProbe(
            python_version="3.11.16",
            torch_version="2.8.0+cu128",
            monai_version="1.5.0",
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="Fixture GPU",
        )
    )


def test_cpu_only_runtime_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_runtime_probe(
            RuntimeProbe(
                python_version="3.11.16",
                torch_version="2.8.0",
                monai_version="1.5.0",
                cuda_available=False,
                cuda_device_count=0,
                cuda_device_name=None,
            )
        )


def test_wrong_python_runtime_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="Python 3.11"):
        validate_runtime_probe(
            RuntimeProbe(
                python_version="3.12.1",
                torch_version="2.8.0",
                monai_version="1.5.0",
                cuda_available=True,
                cuda_device_count=1,
                cuda_device_name="Fixture GPU",
            )
        )


def test_single_case_command_targets_maintained_bundle(
    tmp_path: Path,
) -> None:
    paths = runtime_paths(tmp_path)
    image = tmp_path / "case_00000" / "imaging.nii.gz"

    command = build_single_case_command(
        paths,
        image_path=image,
    )

    assert command[:5] == [
        "conda",
        "run",
        "-n",
        DEFAULT_RUNTIME_ENV,
        "python",
    ]
    assert "monai.bundle" in command
    assert "configs/inference.json" in command
    #assert str(image.resolve()) in command[-1]
    # windows fuck me up
    #assert str(image.resolve()).replace("\\", "\\\\") in command[-1]
    assert image.resolve().as_posix() in command[-1].replace("\\\\", "/")

def test_label_prompt_is_encoded_in_input_dict(
    tmp_path: Path,
) -> None:
    paths = runtime_paths(tmp_path)

    command = build_single_case_command(
        paths,
        image_path=tmp_path / "image.nii.gz",
        label_prompt=(3, 7),
    )

    assert "'label_prompt': [3, 7]" in command[-1]


def test_empty_label_prompt_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        build_single_case_command(
            runtime_paths(tmp_path),
            image_path=tmp_path / "image.nii.gz",
            label_prompt=(),
        )


def test_runtime_manifest_is_written(
    tmp_path: Path,
) -> None:
    manifest = RuntimeManifest(
        repository_url=NV_SEGMENT_REPOSITORY_URL,
        repository_commit="abc123",
        environment_name="vista3d-nv",
        runtime_python_requested="3.11",
        probe=RuntimeProbe(
            python_version="3.11.16",
            torch_version="2.8.0+cu128",
            monai_version="1.5.0",
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_name="Fixture GPU",
        ),
    )

    output = tmp_path / "runtime.json"
    write_runtime_manifest(manifest, output)

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["repository_commit"] == "abc123"
    assert payload["probe"]["cuda_available"] is True
    assert payload["probe"]["cuda_device_name"] == "Fixture GPU"
