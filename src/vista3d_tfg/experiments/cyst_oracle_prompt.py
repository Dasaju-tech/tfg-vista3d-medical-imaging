from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable, Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.evaluation.nv_kits23 import compare_binary_masks
from vista3d_tfg.experiments.pilot30 import PILOT30_CASES, pilot_case_paths
from vista3d_tfg.integration.nv_inference import discover_single_prediction
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
    resolve_conda_executable,
)


NV_CYST_LABELS = (116, 117)
Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class OracleCystPrompt:
    case_id: str
    component_id: int
    point_xyz: tuple[int, int, int]
    target_voxels: int
    target_volume_ml: float
    interior_radius_mm: float


@dataclass(frozen=True, slots=True)
class InteractiveCystResult:
    case_id: str
    point_xyz: tuple[int, int, int]
    target_volume_ml: float
    automatic_target_dice: float
    interactive_target_dice: float
    dice_delta: float
    interactive_prediction_volume_ml: float
    interactive_signed_volume_error_ml: float
    interactive_labels_present: tuple[int, ...]
    contains_255_unprocessed: bool
    wall_time_seconds: float
    prediction_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InteractiveCystCohortSummary:
    case_count: int
    automatic_target_dice_mean: float
    automatic_target_dice_median: float
    interactive_target_dice_mean: float
    interactive_target_dice_median: float
    dice_delta_mean: float
    dice_delta_median: float
    improved_case_count: int
    improved_case_fraction: float
    interactive_dice_ge_050_count: int
    interactive_dice_ge_080_count: int
    automatic_zero_overlap_case_count: int
    zero_overlap_rescued_case_count: int
    rows: tuple[InteractiveCystResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, count = ndimage.label(
        mask.astype(bool, copy=False),
        structure=structure,
    )
    return labels.astype(np.int32, copy=False), int(count)


def select_largest_cyst_oracle_prompt(
    *,
    case_id: str,
    segmentation: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> OracleCystPrompt | None:
    cyst = segmentation == 3

    if not np.any(cyst):
        return None

    labels, count = _label_components(cyst)
    sizes = np.bincount(labels.ravel())[1:]

    if count == 0:
        return None

    component_id = int(np.argmax(sizes)) + 1
    target = labels == component_id

    # Pick the deepest voxel, not the arithmetic centroid, because a centroid
    # can fall outside a non-convex component.
    interior_distance = ndimage.distance_transform_edt(
        target,
        sampling=spacing_mm,
    )
    flat_index = int(np.argmax(interior_distance))
    point = tuple(
        int(value)
        for value in np.unravel_index(flat_index, target.shape)
    )

    voxel_ml = float(np.prod(spacing_mm) / 1000.0)

    return OracleCystPrompt(
        case_id=case_id,
        component_id=component_id,
        point_xyz=point,
        target_voxels=int(np.count_nonzero(target)),
        target_volume_ml=float(np.count_nonzero(target) * voxel_ml),
        interior_radius_mm=float(interior_distance[point]),
    )


def prepare_oracle_prompts(
    project_root: Path,
    case_numbers: Iterable[int] = PILOT30_CASES,
) -> tuple[OracleCystPrompt, ...]:
    prompts = []

    for case_number in case_numbers:
        paths = pilot_case_paths(project_root, case_number)

        segmentation_nii = nib.load(str(paths.segmentation_path))
        segmentation = np.asanyarray(segmentation_nii.dataobj)

        spacing = tuple(
            float(value)
            for value in segmentation_nii.header.get_zooms()[:3]
        )

        prompt = select_largest_cyst_oracle_prompt(
            case_id=paths.case_id,
            segmentation=segmentation,
            spacing_mm=spacing,
        )

        if prompt is not None:
            prompts.append(prompt)

    return tuple(prompts)


def write_prompt_manifest(
    prompts: Iterable[OracleCystPrompt],
    output_path: Path,
) -> None:
    prompts = tuple(prompts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "experiment": "largest_cyst_single_positive_oracle_click",
                "prompt_source": (
                    "ground-truth-derived deepest interior voxel of the "
                    "largest connected KiTS23 cyst component"
                ),
                "deployment_valid": False,
                "interpretation": (
                    "Oracle localization experiment / upper-bound probe. "
                    "Not a deployable automatic segmentation protocol."
                ),
                "point_coordinate_space": "original NIfTI voxel x,y,z",
                "point_labels": [1],
                "prompts": [asdict(prompt) for prompt in prompts],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_prompt_manifest(path: Path) -> tuple[OracleCystPrompt, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_prompts = payload.get("prompts")

    if not isinstance(raw_prompts, list):
        raise ValueError("Prompt manifest does not contain a prompt list.")

    return tuple(
        OracleCystPrompt(
            case_id=str(item["case_id"]),
            component_id=int(item["component_id"]),
            point_xyz=tuple(int(v) for v in item["point_xyz"]),
            target_voxels=int(item["target_voxels"]),
            target_volume_ml=float(item["target_volume_ml"]),
            interior_radius_mm=float(item["interior_radius_mm"]),
        )
        for item in raw_prompts
    )


def build_interactive_point_command(
    *,
    environment_name: str,
    image_path: Path,
    output_dir: Path,
    point_xyz: tuple[int, int, int],
) -> list[str]:
    input_dict = {
        "image": image_path.resolve().as_posix(),
        "points": [list(point_xyz)],
        "point_labels": [1],
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


def _target_component_from_prompt(
    segmentation: np.ndarray,
    prompt: OracleCystPrompt,
) -> np.ndarray:
    labels, count = _label_components(segmentation == 3)

    if prompt.component_id < 1 or prompt.component_id > count:
        raise RuntimeError(
            f"Prompt component {prompt.component_id} no longer exists."
        )

    target = labels == prompt.component_id

    if not target[prompt.point_xyz]:
        raise RuntimeError(
            f"Frozen oracle point is no longer inside target: {prompt.case_id}."
        )

    return target


def _interactive_foreground(
    prediction: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], bool]:
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Interactive output contains non-finite values.")

    if not np.allclose(
        prediction,
        np.rint(prediction),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError("Interactive output must be integer-valued.")

    values = tuple(
        sorted(int(value) for value in np.unique(np.rint(prediction)))
    )
    contains_255 = 255 in values

    # The bundle documents 255 as unprocessed/NaN after postprocessing.
    # Any other non-zero output value is treated as foreground because
    # point-only inference is a single-object binary task.
    foreground = (prediction != 0) & (prediction != 255)

    return foreground, values, contains_255


def evaluate_interactive_prediction(
    *,
    prompt: OracleCystPrompt,
    ground_truth_path: Path,
    automatic_prediction_path: Path,
    interactive_prediction_path: Path,
    wall_time_seconds: float,
) -> InteractiveCystResult:
    gt_nii = nib.load(str(ground_truth_path))
    auto_nii = nib.load(str(automatic_prediction_path))
    interactive_nii = nib.load(str(interactive_prediction_path))

    if not (
        gt_nii.shape == auto_nii.shape == interactive_nii.shape
    ):
        raise ValueError("GT, automatic, and interactive shapes differ.")

    if not np.allclose(
        gt_nii.affine,
        interactive_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("Interactive prediction affine differs from GT.")

    gt = np.asanyarray(gt_nii.dataobj)
    automatic = np.asanyarray(auto_nii.dataobj)
    interactive = np.asanyarray(interactive_nii.dataobj)

    target = _target_component_from_prompt(gt, prompt)
    auto_cyst = np.isin(
        np.rint(automatic).astype(np.int16, copy=False),
        NV_CYST_LABELS,
    )
    interactive_fg, labels_present, contains_255 = _interactive_foreground(
        interactive
    )

    spacing = tuple(
        float(value) for value in gt_nii.header.get_zooms()[:3]
    )

    automatic_comparison = compare_binary_masks(
        target,
        auto_cyst,
        spacing_mm=spacing,
    )
    interactive_comparison = compare_binary_masks(
        target,
        interactive_fg,
        spacing_mm=spacing,
    )

    return InteractiveCystResult(
        case_id=prompt.case_id,
        point_xyz=prompt.point_xyz,
        target_volume_ml=prompt.target_volume_ml,
        automatic_target_dice=automatic_comparison.dice,
        interactive_target_dice=interactive_comparison.dice,
        dice_delta=(
            interactive_comparison.dice - automatic_comparison.dice
        ),
        interactive_prediction_volume_ml=(
            interactive_comparison.prediction_volume_ml
        ),
        interactive_signed_volume_error_ml=(
            interactive_comparison.signed_volume_error_ml
        ),
        interactive_labels_present=labels_present,
        contains_255_unprocessed=contains_255,
        wall_time_seconds=float(wall_time_seconds),
        prediction_path=str(interactive_prediction_path),
    )


def run_interactive_case(
    *,
    prompt: OracleCystPrompt,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> InteractiveCystResult:
    case_number = int(prompt.case_id.split("_")[-1])
    paths = pilot_case_paths(project_root, case_number)

    output_root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_interactive_cyst"
        / prompt.case_id
    )
    raw_output = output_root / "raw"
    stable_prediction = output_root / "prediction.nii.gz"
    result_json = output_root / "result.json"

    if stable_prediction.is_file() and result_json.is_file() and not force:
        payload = json.loads(result_json.read_text(encoding="utf-8"))
        result = payload["result"]
        return InteractiveCystResult(
            case_id=str(result["case_id"]),
            point_xyz=tuple(int(v) for v in result["point_xyz"]),
            target_volume_ml=float(result["target_volume_ml"]),
            automatic_target_dice=float(result["automatic_target_dice"]),
            interactive_target_dice=float(result["interactive_target_dice"]),
            dice_delta=float(result["dice_delta"]),
            interactive_prediction_volume_ml=float(
                result["interactive_prediction_volume_ml"]
            ),
            interactive_signed_volume_error_ml=float(
                result["interactive_signed_volume_error_ml"]
            ),
            interactive_labels_present=tuple(
                int(v) for v in result["interactive_labels_present"]
            ),
            contains_255_unprocessed=bool(
                result["contains_255_unprocessed"]
            ),
            wall_time_seconds=float(result["wall_time_seconds"]),
            prediction_path=str(result["prediction_path"]),
        )

    if output_root.exists() and force:
        shutil.rmtree(output_root)

    raw_output.mkdir(parents=True, exist_ok=True)

    command = build_interactive_point_command(
        environment_name=environment_name,
        image_path=paths.imaging_path,
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
            f"Interactive inference failed for {prompt.case_id} "
            f"with exit code {completed.returncode}."
        )

    raw_prediction = discover_single_prediction(raw_output)
    shutil.copy2(raw_prediction, stable_prediction)

    result = evaluate_interactive_prediction(
        prompt=prompt,
        ground_truth_path=paths.segmentation_path,
        automatic_prediction_path=paths.prediction_path,
        interactive_prediction_path=stable_prediction,
        wall_time_seconds=elapsed,
    )

    result_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "largest_cyst_single_positive_oracle_click",
                "deployment_valid": False,
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return result


def summarize_interactive_results(
    rows: Iterable[InteractiveCystResult],
) -> InteractiveCystCohortSummary:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one interactive result is required.")

    automatic = np.asarray(
        [row.automatic_target_dice for row in rows],
        dtype=float,
    )
    interactive = np.asarray(
        [row.interactive_target_dice for row in rows],
        dtype=float,
    )
    delta = interactive - automatic

    automatic_zero = automatic == 0.0

    return InteractiveCystCohortSummary(
        case_count=len(rows),
        automatic_target_dice_mean=float(np.mean(automatic)),
        automatic_target_dice_median=float(np.median(automatic)),
        interactive_target_dice_mean=float(np.mean(interactive)),
        interactive_target_dice_median=float(np.median(interactive)),
        dice_delta_mean=float(np.mean(delta)),
        dice_delta_median=float(np.median(delta)),
        improved_case_count=int(np.count_nonzero(delta > 0.0)),
        improved_case_fraction=float(np.mean(delta > 0.0)),
        interactive_dice_ge_050_count=int(
            np.count_nonzero(interactive >= 0.5)
        ),
        interactive_dice_ge_080_count=int(
            np.count_nonzero(interactive >= 0.8)
        ),
        automatic_zero_overlap_case_count=int(
            np.count_nonzero(automatic_zero)
        ),
        zero_overlap_rescued_case_count=int(
            np.count_nonzero(automatic_zero & (interactive > 0.0))
        ),
        rows=rows,
    )


def write_interactive_summary(
    summary: InteractiveCystCohortSummary,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "largest_cyst_single_positive_oracle_click",
                "deployment_valid": False,
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    fields = list(asdict(summary.rows[0]).keys())

    with (output_root / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(asdict(row))
