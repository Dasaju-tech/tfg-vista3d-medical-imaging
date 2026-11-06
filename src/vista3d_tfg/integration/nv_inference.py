from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable, Sequence

import nibabel as nib
import numpy as np

from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
    resolve_conda_executable,
)


RENAL_LABEL_PROMPT: tuple[int, ...] = (5, 14, 116, 117)

EXPECTED_RENAL_LABELS: dict[str, int] = {
    "right kidney": 5,
    "left kidney": 14,
    "left kidney cyst": 116,
    "right kidney cyst": 117,
}


@dataclass(frozen=True, slots=True)
class PredictionAudit:
    prediction_path: str
    sha256: str
    size_bytes: int
    shape: tuple[int, int, int]
    affine_matches_input: bool
    shape_matches_input: bool
    integer_valued: bool
    finite: bool
    labels_present: tuple[int, ...]
    named_labels_present: tuple[str, ...]
    unexpected_labels: tuple[int, ...]
    contains_unprocessed_255: bool

    @property
    def is_valid(self) -> bool:
        return (
            self.affine_matches_input
            and self.shape_matches_input
            and self.integer_valued
            and self.finite
            and not self.unexpected_labels
            and not self.contains_unprocessed_255
        )


@dataclass(frozen=True, slots=True)
class NVInferenceResult:
    case_id: str
    environment_name: str
    repository_commit: str
    input_path: str
    output_root: str
    raw_prediction_path: str
    prediction_path: str
    label_prompt: tuple[int, ...]
    checkpoint_present_before: bool
    wall_time_seconds: float
    audit: PredictionAudit

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["audit"]["is_valid"] = self.audit.is_valid
        return payload


Runner = Callable[..., subprocess.CompletedProcess[str]]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()

    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def load_upstream_label_dict(label_dict_path: Path) -> dict[str, int]:
    payload = json.loads(label_dict_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise TypeError("Upstream label_dict.json must contain a JSON object.")

    labels: dict[str, int] = {}

    for name, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Invalid upstream label index for {name!r}: {value!r}."
            )
        labels[str(name)] = int(value)

    return labels


def validate_renal_label_contract(labels: dict[str, int]) -> None:
    mismatches = {
        name: {
            "expected": expected,
            "actual": labels.get(name),
        }
        for name, expected in EXPECTED_RENAL_LABELS.items()
        if labels.get(name) != expected
    }

    if mismatches:
        raise RuntimeError(
            "Upstream renal label contract changed: "
            f"{mismatches!r}"
        )


def build_renal_inference_command(
    *,
    environment_name: str,
    image_path: Path,
    output_dir: Path,
) -> list[str]:
    """
    Build a Windows-safe MONAI Bundle command.

    Forward-slash absolute paths avoid Python-literal backslash escaping inside
    MONAI's ``input_dict`` expression.
    """

    image = image_path.resolve().as_posix()
    output = output_dir.resolve().as_posix()

    input_dict = {
        "image": image,
        "label_prompt": list(RENAL_LABEL_PROMPT),
    }

    return [
        resolve_conda_executable(),
        "run",
        "--no-capture-output",
        "-n",
        environment_name,
        "python",
        "-m",
        "monai.bundle",
        "run",
        "--config_file",
        "configs/inference.json",
        "--input_dict",
        repr(input_dict),
        "--output_dir",
        output,
    ]


def discover_single_prediction(raw_output_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in raw_output_dir.rglob("*_trans.nii.gz")
        if path.is_file() and path.stat().st_size > 0
    )

    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one NV-Segment prediction under "
            f"{raw_output_dir}, found {len(candidates)}: "
            f"{[str(path) for path in candidates]}"
        )

    return candidates[0]


def audit_renal_prediction(
    *,
    input_path: Path,
    prediction_path: Path,
    labels: dict[str, int],
) -> PredictionAudit:
    input_nii = nib.load(str(input_path))
    prediction_nii = nib.load(str(prediction_path))

    if len(input_nii.shape) != 3 or len(prediction_nii.shape) != 3:
        raise ValueError("Input and prediction must both be 3D NIfTI volumes.")

    data = np.asanyarray(prediction_nii.dataobj)

    finite = bool(np.all(np.isfinite(data)))

    if finite:
        rounded = np.rint(data)
        integer_valued = bool(
            np.allclose(data, rounded, rtol=0.0, atol=1e-5)
        )
    else:
        rounded = np.zeros_like(data)
        integer_valued = False

    if integer_valued:
        label_values = tuple(
            sorted(int(value) for value in np.unique(rounded))
        )
    else:
        label_values = ()

    allowed = {0, *RENAL_LABEL_PROMPT}
    unexpected = tuple(
        value
        for value in label_values
        if value not in allowed and value != 255
    )

    inverse_labels = {
        int(index): str(name)
        for name, index in labels.items()
    }

    named = tuple(
        inverse_labels.get(value, f"unknown:{value}")
        for value in label_values
        if value != 0
    )

    shape_matches = tuple(input_nii.shape) == tuple(prediction_nii.shape)
    affine_matches = bool(
        np.allclose(
            input_nii.affine,
            prediction_nii.affine,
            rtol=0.0,
            atol=1e-4,
        )
    )

    return PredictionAudit(
        prediction_path=str(prediction_path),
        sha256=sha256_file(prediction_path),
        size_bytes=prediction_path.stat().st_size,
        shape=tuple(int(value) for value in prediction_nii.shape),
        affine_matches_input=affine_matches,
        shape_matches_input=shape_matches,
        integer_valued=integer_valued,
        finite=finite,
        labels_present=label_values,
        named_labels_present=named,
        unexpected_labels=unexpected,
        contains_unprocessed_255=255 in label_values,
    )


def run_renal_inference(
    *,
    case_id: str,
    image_path: Path,
    paths: NVSegmentPaths,
    repository_commit: str,
    project_output_root: Path,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> NVInferenceResult:
    label_dict_path = paths.ct_bundle_root / "configs" / "label_dict.json"
    labels = load_upstream_label_dict(label_dict_path)
    validate_renal_label_contract(labels)

    case_root = project_output_root.resolve() / case_id
    raw_output = case_root / "raw"
    stable_prediction = case_root / "prediction.nii.gz"

    if case_root.exists():
        if not force:
            raise FileExistsError(
                f"Inference output already exists: {case_root}. "
                "Use --force to replace it."
            )
        shutil.rmtree(case_root)

    raw_output.mkdir(parents=True, exist_ok=True)

    checkpoint_path = paths.ct_bundle_root / "models" / "model.pt"
    checkpoint_present_before = (
        checkpoint_path.is_file()
        and checkpoint_path.stat().st_size > 0
    )

    command = build_renal_inference_command(
        environment_name=environment_name,
        image_path=image_path,
        output_dir=raw_output,
    )

    start = time.perf_counter()

    result = runner(
        command,
        cwd=str(paths.ct_bundle_root),
        check=False,
        text=True,
        shell=False,
    )

    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise RuntimeError(
            "NV-Segment inference failed with exit code "
            f"{result.returncode}."
        )

    raw_prediction = discover_single_prediction(raw_output)
    shutil.copy2(raw_prediction, stable_prediction)

    audit = audit_renal_prediction(
        input_path=image_path,
        prediction_path=stable_prediction,
        labels=labels,
    )

    if not audit.is_valid:
        raise RuntimeError(
            "NV-Segment produced a prediction that failed the project audit: "
            f"{audit!r}"
        )

    return NVInferenceResult(
        case_id=case_id,
        environment_name=environment_name,
        repository_commit=repository_commit,
        input_path=str(image_path.resolve()),
        output_root=str(case_root),
        raw_prediction_path=str(raw_prediction),
        prediction_path=str(stable_prediction),
        label_prompt=RENAL_LABEL_PROMPT,
        checkpoint_present_before=checkpoint_present_before,
        wall_time_seconds=float(elapsed),
        audit=audit,
    )


def write_inference_result(
    result: NVInferenceResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model": "NV-Segment-CT",
                "task": "supported_renal_structure_smoke_inference",
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
