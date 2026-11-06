from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.evaluation.nv_kits23 import (
    compare_binary_masks,
    evaluate_kidney_and_mass_hec,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    EXPECTED_AUTOMATIC_RENAL_LABELS,
    SPECIAL_TUMOR_POSITIVE_POINT_LABEL,
    TumorOraclePrompt,
)
from vista3d_tfg.integration.nv_inference import (
    discover_single_prediction,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
    resolve_conda_executable,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class TumorOracleSmokeResult:
    case_id: str
    point_xyz: tuple[int, int, int]
    point_label: int
    target_volume_ml: float
    total_tumor_volume_ml: float
    interactive_target_dice: float
    interactive_full_tumor_dice: float
    original_tumor_coverage_fraction: float
    merged_tumor_coverage_fraction: float
    tumor_coverage_gain: float
    original_hec_dice: float
    merged_hec_dice: float
    hec_dice_delta: float
    original_hec_surface_dice: float
    merged_hec_surface_dice: float
    hec_surface_dice_delta: float
    original_hec_hd95_mm: float
    merged_hec_hd95_mm: float
    hec_hd95_reduction_mm: float
    interactive_prediction_volume_ml: float
    interactive_labels_present: tuple[int, ...]
    contains_255_unprocessed: bool
    wall_time_seconds: float
    prediction_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_frozen_prompt(
    protocol_path: Path,
    *,
    case_id: str,
) -> TumorOraclePrompt:
    payload = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )

    if payload.get("oracle") is not True:
        raise ValueError("Tumor protocol is not marked as oracle.")
    if payload.get("deployment_valid") is not False:
        raise ValueError(
            "Tumor oracle protocol must remain deployment-invalid."
        )

    input_protocol = payload.get("input_protocol")
    if not isinstance(input_protocol, dict):
        raise ValueError("Protocol has no input_protocol object.")

    if input_protocol.get("label_prompt") is not None:
        raise ValueError("Frozen tumor oracle must not use label_prompt.")
    if input_protocol.get("point_labels") != [
        SPECIAL_TUMOR_POSITIVE_POINT_LABEL
    ]:
        raise ValueError("Frozen tumor oracle point-label contract changed.")

    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("Protocol has no prompt list.")

    matches = [
        item
        for item in prompts
        if item.get("case_id") == case_id
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one frozen prompt for {case_id}, got {len(matches)}."
        )

    item = matches[0]
    return TumorOraclePrompt(
        case_id=str(item["case_id"]),
        component_id=int(item["component_id"]),
        point_xyz=tuple(
            int(value)
            for value in item["point_xyz"]
        ),
        target_voxels=int(item["target_voxels"]),
        target_volume_ml=float(item["target_volume_ml"]),
        total_tumor_volume_ml=float(
            item["total_tumor_volume_ml"]
        ),
        interior_radius_mm=float(
            item["interior_radius_mm"]
        ),
    )


def build_tumor_special_point_command(
    *,
    environment_name: str,
    image_path: Path,
    output_dir: Path,
    point_xyz: tuple[int, int, int],
) -> list[str]:
    input_dict = {
        "image": image_path.resolve().as_posix(),
        "points": [list(point_xyz)],
        "point_labels": [
            SPECIAL_TUMOR_POSITIVE_POINT_LABEL
        ],
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
        output_dir.resolve().as_posix(),
    ]


def _integer_labels(array: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise ValueError("Label map contains non-finite values.")
    if not np.allclose(
        array,
        np.rint(array),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError("Label map must be integer-valued.")

    return np.rint(array).astype(
        np.int16,
        copy=False,
    )


def _target_component(
    ground_truth: np.ndarray,
    prompt: TumorOraclePrompt,
    *,
    require_prompt_inside_target: bool = True,
) -> np.ndarray:
    tumor = ground_truth == 2
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(
        tumor,
        structure=structure,
    )

    if prompt.component_id < 1 or prompt.component_id > count:
        raise RuntimeError(
            f"{prompt.case_id}: frozen tumor component no longer exists."
        )

    target = labels == prompt.component_id

    if int(np.count_nonzero(target)) != prompt.target_voxels:
        raise RuntimeError(
            f"{prompt.case_id}: frozen tumor component size changed."
        )
    if (
        require_prompt_inside_target
        and not target[prompt.point_xyz]
    ):
        raise RuntimeError(
            f"{prompt.case_id}: frozen oracle point is outside target."
        )

    return target


def _interactive_foreground(
    prediction: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], bool]:
    integer = _integer_labels(prediction)
    labels_present = tuple(
        sorted(int(value) for value in np.unique(integer))
    )
    contains_255 = 255 in labels_present

    # The maintained bundle uses 255 for unprocessed/NaN output after
    # post-processing. Point-only inference is treated as a single binary
    # object, so every other non-zero value is foreground.
    foreground = (integer != 0) & (integer != 255)

    return foreground, labels_present, contains_255


def _tumor_coverage(
    tumor: np.ndarray,
    envelope: np.ndarray,
) -> float:
    denominator = int(np.count_nonzero(tumor))
    if denominator == 0:
        raise ValueError("Tumor coverage requires tumor-positive GT.")

    return (
        int(np.count_nonzero(tumor & envelope))
        / denominator
    )


def evaluate_tumor_oracle_smoke(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
    automatic_prediction_path: Path,
    interactive_prediction_path: Path,
    wall_time_seconds: float,
    require_prompt_inside_target: bool = True,
) -> TumorOracleSmokeResult:
    gt_nii = nib.load(str(ground_truth_path))
    automatic_nii = nib.load(
        str(automatic_prediction_path)
    )
    interactive_nii = nib.load(
        str(interactive_prediction_path)
    )

    if not (
        gt_nii.shape
        == automatic_nii.shape
        == interactive_nii.shape
    ):
        raise ValueError(
            "GT, automatic, and interactive prediction shapes differ."
        )

    if not np.allclose(
        gt_nii.affine,
        automatic_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("Automatic prediction affine differs from GT.")
    if not np.allclose(
        gt_nii.affine,
        interactive_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("Interactive prediction affine differs from GT.")

    ground_truth = _integer_labels(
        np.asanyarray(gt_nii.dataobj)
    )
    automatic = _integer_labels(
        np.asanyarray(automatic_nii.dataobj)
    )
    interactive = np.asanyarray(
        interactive_nii.dataobj
    )

    target = _target_component(
        ground_truth,
        prompt,
        require_prompt_inside_target=(
            require_prompt_inside_target
        ),
    )
    full_tumor = ground_truth == 2

    automatic_renal = np.isin(
        automatic,
        tuple(
            EXPECTED_AUTOMATIC_RENAL_LABELS.values()
        ),
    )

    (
        interactive_foreground,
        interactive_labels,
        contains_255,
    ) = _interactive_foreground(interactive)

    merged_envelope = (
        automatic_renal | interactive_foreground
    )
    hec_reference = np.isin(
        ground_truth,
        (1, 2, 3),
    )

    spacing = tuple(
        float(value)
        for value in gt_nii.header.get_zooms()[:3]
    )

    target_comparison = compare_binary_masks(
        target,
        interactive_foreground,
        spacing_mm=spacing,
    )
    full_tumor_comparison = compare_binary_masks(
        full_tumor,
        interactive_foreground,
        spacing_mm=spacing,
    )

    original_hec = evaluate_kidney_and_mass_hec(
        hec_reference,
        automatic_renal,
        spacing_mm=spacing,
    )
    merged_hec = evaluate_kidney_and_mass_hec(
        hec_reference,
        merged_envelope,
        spacing_mm=spacing,
    )

    original_coverage = _tumor_coverage(
        full_tumor,
        automatic_renal,
    )
    merged_coverage = _tumor_coverage(
        full_tumor,
        merged_envelope,
    )

    return TumorOracleSmokeResult(
        case_id=prompt.case_id,
        point_xyz=prompt.point_xyz,
        point_label=SPECIAL_TUMOR_POSITIVE_POINT_LABEL,
        target_volume_ml=prompt.target_volume_ml,
        total_tumor_volume_ml=(
            prompt.total_tumor_volume_ml
        ),
        interactive_target_dice=(
            target_comparison.dice
        ),
        interactive_full_tumor_dice=(
            full_tumor_comparison.dice
        ),
        original_tumor_coverage_fraction=(
            original_coverage
        ),
        merged_tumor_coverage_fraction=(
            merged_coverage
        ),
        tumor_coverage_gain=(
            merged_coverage - original_coverage
        ),
        original_hec_dice=original_hec.dice,
        merged_hec_dice=merged_hec.dice,
        hec_dice_delta=(
            merged_hec.dice - original_hec.dice
        ),
        original_hec_surface_dice=(
            original_hec.surface_dice
        ),
        merged_hec_surface_dice=(
            merged_hec.surface_dice
        ),
        hec_surface_dice_delta=(
            merged_hec.surface_dice
            - original_hec.surface_dice
        ),
        original_hec_hd95_mm=(
            original_hec.diagnostic_hd95_mm
        ),
        merged_hec_hd95_mm=(
            merged_hec.diagnostic_hd95_mm
        ),
        hec_hd95_reduction_mm=(
            original_hec.diagnostic_hd95_mm
            - merged_hec.diagnostic_hd95_mm
        ),
        interactive_prediction_volume_ml=(
            target_comparison.prediction_volume_ml
        ),
        interactive_labels_present=(
            interactive_labels
        ),
        contains_255_unprocessed=contains_255,
        wall_time_seconds=float(
            wall_time_seconds
        ),
        prediction_path=str(
            interactive_prediction_path
        ),
    )


def _load_cached_result(
    path: Path,
) -> TumorOracleSmokeResult:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    result = payload["result"]

    return TumorOracleSmokeResult(
        case_id=str(result["case_id"]),
        point_xyz=tuple(
            int(value)
            for value in result["point_xyz"]
        ),
        point_label=int(result["point_label"]),
        target_volume_ml=float(
            result["target_volume_ml"]
        ),
        total_tumor_volume_ml=float(
            result["total_tumor_volume_ml"]
        ),
        interactive_target_dice=float(
            result["interactive_target_dice"]
        ),
        interactive_full_tumor_dice=float(
            result["interactive_full_tumor_dice"]
        ),
        original_tumor_coverage_fraction=float(
            result[
                "original_tumor_coverage_fraction"
            ]
        ),
        merged_tumor_coverage_fraction=float(
            result[
                "merged_tumor_coverage_fraction"
            ]
        ),
        tumor_coverage_gain=float(
            result["tumor_coverage_gain"]
        ),
        original_hec_dice=float(
            result["original_hec_dice"]
        ),
        merged_hec_dice=float(
            result["merged_hec_dice"]
        ),
        hec_dice_delta=float(
            result["hec_dice_delta"]
        ),
        original_hec_surface_dice=float(
            result["original_hec_surface_dice"]
        ),
        merged_hec_surface_dice=float(
            result["merged_hec_surface_dice"]
        ),
        hec_surface_dice_delta=float(
            result["hec_surface_dice_delta"]
        ),
        original_hec_hd95_mm=float(
            result["original_hec_hd95_mm"]
        ),
        merged_hec_hd95_mm=float(
            result["merged_hec_hd95_mm"]
        ),
        hec_hd95_reduction_mm=float(
            result["hec_hd95_reduction_mm"]
        ),
        interactive_prediction_volume_ml=float(
            result["interactive_prediction_volume_ml"]
        ),
        interactive_labels_present=tuple(
            int(value)
            for value in result[
                "interactive_labels_present"
            ]
        ),
        contains_255_unprocessed=bool(
            result["contains_255_unprocessed"]
        ),
        wall_time_seconds=float(
            result["wall_time_seconds"]
        ),
        prediction_path=str(
            result["prediction_path"]
        ),
    )


def run_tumor_oracle_smoke_case(
    *,
    prompt: TumorOraclePrompt,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> TumorOracleSmokeResult:
    case_root = (
        project_root
        / "data"
        / "raw"
        / "kits23"
        / prompt.case_id
    )
    image_path = case_root / "imaging.nii.gz"
    ground_truth_path = (
        case_root / "segmentation.nii.gz"
    )
    automatic_prediction_path = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
        / prompt.case_id
        / "prediction.nii.gz"
    )

    output_root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_interactive_tumor_oracle"
        / prompt.case_id
    )
    raw_output = output_root / "raw"
    stable_prediction = (
        output_root / "prediction.nii.gz"
    )
    result_json = output_root / "result.json"

    if (
        stable_prediction.is_file()
        and result_json.is_file()
        and not force
    ):
        return _load_cached_result(result_json)

    for required in (
        image_path,
        ground_truth_path,
        automatic_prediction_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    if output_root.exists() and force:
        shutil.rmtree(output_root)

    raw_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = build_tumor_special_point_command(
        environment_name=environment_name,
        image_path=image_path,
        output_dir=raw_output,
        point_xyz=prompt.point_xyz,
    )

    start = time.perf_counter()
    completed = runner(
        command,
        cwd=str(nv_paths.ct_bundle_root),
        check=False,
        text=True,
        shell=False,
    )
    elapsed = time.perf_counter() - start

    if completed.returncode != 0:
        raise RuntimeError(
            f"Tumor oracle inference failed for "
            f"{prompt.case_id} with exit code "
            f"{completed.returncode}."
        )

    raw_prediction = discover_single_prediction(
        raw_output
    )
    shutil.copy2(
        raw_prediction,
        stable_prediction,
    )

    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=ground_truth_path,
        automatic_prediction_path=(
            automatic_prediction_path
        ),
        interactive_prediction_path=(
            stable_prediction
        ),
        wall_time_seconds=elapsed,
    )

    result_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": (
                    "nv_segment_ct_tumor_oracle_special_point_smoke"
                ),
                "oracle": True,
                "deployment_valid": False,
                "input_protocol": {
                    "label_prompt": None,
                    "point_labels": [
                        SPECIAL_TUMOR_POSITIVE_POINT_LABEL
                    ],
                },
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return result
