from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


NV_SEGMENT_REPOSITORY_URL = (
    "https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git"
)
DEFAULT_RUNTIME_ENV = "vista3d-nv"
RUNTIME_PYTHON_VERSION = "3.11"


@dataclass(frozen=True, slots=True)
class NVSegmentPaths:
    project_root: Path
    checkout_root: Path
    ct_bundle_root: Path
    requirements_path: Path
    provenance_path: Path


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    python_version: str
    torch_version: str
    monai_version: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    repository_url: str
    repository_commit: str
    environment_name: str
    runtime_python_requested: str
    probe: RuntimeProbe

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


Runner = Callable[..., subprocess.CompletedProcess[str]]


def runtime_paths(project_root: Path) -> NVSegmentPaths:
    root = project_root.resolve()
    checkout = root / "data" / "external" / "NV-Segment-CTMR"

    return NVSegmentPaths(
        project_root=root,
        checkout_root=checkout,
        ct_bundle_root=checkout / "NV-Segment-CT",
        requirements_path=checkout / "NV-Segment-CTMR" / "requirements.txt",
        provenance_path=(
            root
            / "outputs"
            / "runtime"
            / "nv_segment_ct_runtime.json"
        ),
    )


def resolve_conda_executable() -> str:
    """
    Resolve the actual Conda executable without relying on PowerShell/CMD.

    Windows often exposes ``conda`` as a shell function even though
    CreateProcess cannot resolve a program literally named ``conda``.
    """

    candidates: list[Path] = []

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(Path(conda_exe))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).resolve()

        if prefix.parent.name.lower() == "envs":
            candidates.append(
                prefix.parent.parent / "Scripts" / "conda.exe"
            )

        candidates.append(prefix / "Scripts" / "conda.exe")

    executable = Path(sys.executable).resolve()
    if executable.parent.parent.name.lower() == "envs":
        candidates.append(
            executable.parent.parent.parent
            / "Scripts"
            / "conda.exe"
        )

    for name in ("conda.exe", "conda"):
        located = shutil.which(name)
        if located:
            candidates.append(Path(located))

    seen: set[str] = set()

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        key = str(candidate)

        if key in seen:
            continue

        seen.add(key)

        if candidate.is_file():
            return key

    raise FileNotFoundError(
        "Could not locate Conda. Expected CONDA_EXE or "
        "<conda-root>\\Scripts\\conda.exe. "
        "Check PowerShell with: $env:CONDA_EXE"
    )


def run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    resolved_command = list(command)

    if (
        runner is subprocess.run
        and resolved_command
        and resolved_command[0].lower() == "conda"
    ):
        resolved_command[0] = resolve_conda_executable()

    result = runner(
        resolved_command,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        capture_output=True,
        shell=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {resolved_command!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result

def ensure_repository(
    paths: NVSegmentPaths,
    *,
    runner: Runner = subprocess.run,
) -> str:
    """
    Clone the maintained NVIDIA runtime when missing.

    Existing checkouts are intentionally not pulled automatically. This keeps
    a previously recorded model runtime reproducible.
    """

    git_directory = paths.checkout_root / ".git"

    if not git_directory.is_dir():
        paths.checkout_root.parent.mkdir(parents=True, exist_ok=True)

        run_checked(
            [
                "git",
                "clone",
                NV_SEGMENT_REPOSITORY_URL,
                str(paths.checkout_root),
            ],
            runner=runner,
        )

    if not paths.ct_bundle_root.is_dir():
        raise RuntimeError(
            "NV-Segment checkout is missing the NV-Segment-CT bundle: "
            f"{paths.ct_bundle_root}"
        )

    if not paths.requirements_path.is_file():
        raise RuntimeError(
            "NV-Segment checkout is missing the expected requirements file: "
            f"{paths.requirements_path}"
        )

    commit = run_checked(
        [
            "git",
            "-C",
            str(paths.checkout_root),
            "rev-parse",
            "HEAD",
        ],
        runner=runner,
    ).stdout.strip()

    if not commit:
        raise RuntimeError("Could not determine NV-Segment git commit.")

    return commit


def list_conda_environments(
    *,
    runner: Runner = subprocess.run,
) -> set[str]:
    result = run_checked(
        ["conda", "env", "list", "--json"],
        runner=runner,
    )

    payload = json.loads(result.stdout)
    environments = payload.get("envs", [])

    return {
        Path(value).name
        for value in environments
        if value
    }


def ensure_runtime_environment(
    *,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    runner: Runner = subprocess.run,
) -> bool:
    """
    Ensure the isolated Python 3.11 model runtime exists.

    Returns True when a new environment was created.
    """

    environments = list_conda_environments(runner=runner)

    if environment_name in environments:
        return False

    run_checked(
        [
            "conda",
            "create",
            "-y",
            "-n",
            environment_name,
            f"python={RUNTIME_PYTHON_VERSION}",
        ],
        runner=runner,
    )

    return True


def install_runtime_requirements(
    paths: NVSegmentPaths,
    *,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    runner: Runner = subprocess.run,
) -> None:
    if not paths.requirements_path.is_file():
        raise FileNotFoundError(paths.requirements_path)

    run_checked(
        [
            "conda",
            "run",
            "-n",
            environment_name,
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            str(paths.requirements_path),
        ],
        cwd=paths.checkout_root / "NV-Segment-CTMR",
        runner=runner,
    )


def probe_runtime(
    *,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    runner: Runner = subprocess.run,
) -> RuntimeProbe:
    # Conda 26.x on Windows does not support executable arguments that
    # contain newline characters. Keep the probe as one physical command-line
    # argument instead of passing a multiline ``python -c`` program.
    code = (
        "import json, platform, torch, monai; "
        "cuda_available=bool(torch.cuda.is_available()); "
        "device_count=int(torch.cuda.device_count()) if cuda_available else 0; "
        "device_name=torch.cuda.get_device_name(0) "
        "if cuda_available and device_count > 0 else None; "
        "print(json.dumps({"
        "'python_version': platform.python_version(), "
        "'torch_version': torch.__version__, "
        "'monai_version': monai.__version__, "
        "'cuda_available': cuda_available, "
        "'cuda_device_count': device_count, "
        "'cuda_device_name': device_name"
        "}))"
    )

    result = run_checked(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            environment_name,
            "python",
            "-c",
            code,
        ],
        runner=runner,
    )

    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    payload = None

    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue

        if isinstance(candidate, dict) and "python_version" in candidate:
            payload = candidate
            break

    if payload is None:
        raise RuntimeError(
            "Runtime probe did not return the expected JSON payload.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return RuntimeProbe(
        python_version=str(payload["python_version"]),
        torch_version=str(payload["torch_version"]),
        monai_version=str(payload["monai_version"]),
        cuda_available=bool(payload["cuda_available"]),
        cuda_device_count=int(payload["cuda_device_count"]),
        cuda_device_name=(
            str(payload["cuda_device_name"])
            if payload["cuda_device_name"] is not None
            else None
        ),
    )


def validate_runtime_probe(probe: RuntimeProbe) -> None:
    major_minor = ".".join(probe.python_version.split(".")[:2])

    if major_minor != RUNTIME_PYTHON_VERSION:
        raise RuntimeError(
            "NV-Segment runtime must use Python "
            f"{RUNTIME_PYTHON_VERSION}, got {probe.python_version}."
        )

    if not probe.cuda_available:
        raise RuntimeError(
            "NV-Segment runtime did not detect CUDA. "
            "The upstream inference path is GPU-oriented."
        )

    if probe.cuda_device_count < 1 or not probe.cuda_device_name:
        raise RuntimeError(
            "CUDA was reported as available but no GPU device was resolved."
        )


def build_single_case_command(
    paths: NVSegmentPaths,
    *,
    image_path: Path,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    label_prompt: tuple[int, ...] | None = None,
) -> list[str]:
    image = image_path.resolve()

    input_dict = {"image": str(image)}

    if label_prompt is not None:
        if not label_prompt:
            raise ValueError("label_prompt must not be empty.")

        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in label_prompt
        ):
            raise TypeError("label_prompt values must be integers.")

        input_dict["label_prompt"] = list(label_prompt)

    # MONAI Bundle accepts a Python-dict-like string for input_dict.
    input_dict_text = repr(input_dict)

    return [
        "conda",
        "run",
        "-n",
        environment_name,
        "python",
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        "configs/inference.json",
        "--input_dict",
        input_dict_text,
    ]


def write_runtime_manifest(
    manifest: RuntimeManifest,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
