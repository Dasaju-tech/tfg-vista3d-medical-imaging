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

from vista3d_tfg.evaluation.interaction_stability import (
    BootstrapInterval,
    bootstrap_mean_difference_ci,
)
from vista3d_tfg.evaluation.nv_kits23 import compare_binary_masks
from vista3d_tfg.experiments.cyst_oracle_prompt import (
    OracleCystPrompt,
    _interactive_foreground,
    _target_component_from_prompt,
)
from vista3d_tfg.experiments.pilot30 import pilot_case_paths
from vista3d_tfg.integration.nv_inference import discover_single_prediction
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
    resolve_conda_executable,
)


LEFT_CYST_LABEL = 116
RIGHT_CYST_LABEL = 117

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class SemanticCystTarget:
    case_id: str
    laterality: str
    class_label: int
    class_name: str
    target_centroid_world_mm: tuple[float, float, float]
    left_anchor_world_x_mm: float
    right_anchor_world_x_mm: float


@dataclass(frozen=True, slots=True)
class SemanticPointResult:
    case_id: str
    laterality: str
    class_label: int
    class_name: str
    point_xyz: tuple[int, int, int]
    target_volume_ml: float

    automatic_side_class_dice: float
    point_only_dice: float
    semantic_plus_point_dice: float

    semantic_plus_point_minus_point_only: float
    semantic_plus_point_minus_automatic: float

    semantic_plus_point_volume_ml: float
    semantic_plus_point_signed_volume_error_ml: float

    labels_present: tuple[int, ...]
    contains_255_unprocessed: bool
    wall_time_seconds: float
    prediction_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticPointSummary:
    case_count: int

    automatic_side_class_dice_mean: float
    automatic_side_class_dice_median: float

    point_only_dice_mean: float
    point_only_dice_median: float

    semantic_plus_point_dice_mean: float
    semantic_plus_point_dice_median: float

    semantic_plus_point_minus_point_only_mean: float
    semantic_plus_point_minus_point_only_median: float
    semantic_plus_point_minus_point_only_bootstrap_95ci: BootstrapInterval

    improved_over_point_only_count: int
    regressed_vs_point_only_count: int

    semantic_plus_point_dice_ge_050_count: int
    semantic_plus_point_dice_ge_080_count: int

    point_only_zero_overlap_count: int
    point_only_zero_overlap_rescued_count: int

    left_target_count: int
    right_target_count: int

    rows: tuple[SemanticPointResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, count = ndimage.label(
        mask.astype(bool, copy=False),
        structure=structure,
    )
    return labels.astype(np.int32, copy=False), int(count)


def _world_centroid(
    mask: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        raise ValueError("Cannot compute centroid of an empty mask.")

    centroid_voxel = coordinates.mean(axis=0)

    return np.asarray(
        nib.affines.apply_affine(affine, centroid_voxel),
        dtype=float,
    )


def infer_oracle_cyst_laterality(
    *,
    case_id: str,
    segmentation: np.ndarray,
    affine: np.ndarray,
    prompt: OracleCystPrompt,
) -> SemanticCystTarget:
    target = _target_component_from_prompt(
        segmentation,
        prompt,
    )

    renal_envelope = np.isin(segmentation, (1, 2, 3))
    renal_labels, renal_count = _label_components(renal_envelope)

    if renal_count < 2:
        raise RuntimeError(
            f"{case_id}: fewer than two renal-envelope components; "
            "cannot derive bilateral semantic cyst class."
        )

    sizes = np.bincount(renal_labels.ravel())[1:]
    dominant_ids = (
        np.argsort(sizes)[-2:] + 1
    )

    dominant_centroids = [
        _world_centroid(
            renal_labels == int(component_id),
            affine,
        )
        for component_id in dominant_ids
    ]

    dominant_x = sorted(
        float(centroid[0])
        for centroid in dominant_centroids
    )

    left_anchor_x = dominant_x[0]
    right_anchor_x = dominant_x[1]

    if np.isclose(left_anchor_x, right_anchor_x):
        raise RuntimeError(
            f"{case_id}: bilateral renal anchors have indistinguishable "
            "world-x coordinates."
        )

    target_centroid = _world_centroid(target, affine)
    midpoint_x = 0.5 * (left_anchor_x + right_anchor_x)

    # NIfTI world coordinates follow RAS convention:
    # smaller x is patient left, larger x is patient right.
    if float(target_centroid[0]) < midpoint_x:
        laterality = "left"
        class_label = LEFT_CYST_LABEL
        class_name = "left kidney cyst"
    else:
        laterality = "right"
        class_label = RIGHT_CYST_LABEL
        class_name = "right kidney cyst"

    return SemanticCystTarget(
        case_id=case_id,
        laterality=laterality,
        class_label=class_label,
        class_name=class_name,
        target_centroid_world_mm=tuple(
            float(value)
            for value in target_centroid
        ),
        left_anchor_world_x_mm=left_anchor_x,
        right_anchor_world_x_mm=right_anchor_x,
    )


def build_semantic_point_command(
    *,
    environment_name: str,
    image_path: Path,
    output_dir: Path,
    class_label: int,
    point_xyz: tuple[int, int, int],
) -> list[str]:
    if class_label not in {LEFT_CYST_LABEL, RIGHT_CYST_LABEL}:
        raise ValueError(
            "Semantic cyst class must be 116 (left) or 117 (right)."
        )

    input_dict = {
        "image": image_path.resolve().as_posix(),
        "label_prompt": [class_label],
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


def _load_block16_point_only_result(
    *,
    project_root: Path,
    case_id: str,
) -> float:
    result_path = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_interactive_cyst"
        / case_id
        / "result.json"
    )

    if not result_path.is_file():
        raise FileNotFoundError(
            f"Block 16 point-only result missing: {result_path}"
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload.get("result")

    if not isinstance(result, dict):
        raise ValueError(
            f"Invalid Block 16 result JSON: {result_path}"
        )

    return float(result["interactive_target_dice"])


def _automatic_side_class_mask(
    automatic_prediction: np.ndarray,
    class_label: int,
) -> np.ndarray:
    rounded = np.rint(automatic_prediction).astype(
        np.int16,
        copy=False,
    )
    return rounded == class_label


def evaluate_semantic_point_prediction(
    *,
    prompt: OracleCystPrompt,
    semantic_target: SemanticCystTarget,
    project_root: Path,
    interactive_prediction_path: Path,
    wall_time_seconds: float,
) -> SemanticPointResult:
    case_number = int(prompt.case_id.split("_")[-1])
    paths = pilot_case_paths(project_root, case_number)

    gt_nii = nib.load(str(paths.segmentation_path))
    automatic_nii = nib.load(str(paths.prediction_path))
    semantic_point_nii = nib.load(
        str(interactive_prediction_path)
    )

    if not (
        gt_nii.shape
        == automatic_nii.shape
        == semantic_point_nii.shape
    ):
        raise ValueError(
            "GT, automatic, and semantic+point prediction shapes differ."
        )

    if not np.allclose(
        gt_nii.affine,
        semantic_point_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Semantic+point prediction affine differs from GT."
        )

    gt = np.asanyarray(gt_nii.dataobj)
    automatic = np.asanyarray(automatic_nii.dataobj)
    semantic_point = np.asanyarray(
        semantic_point_nii.dataobj
    )

    target = _target_component_from_prompt(gt, prompt)

    semantic_fg, labels_present, contains_255 = (
        _interactive_foreground(semantic_point)
    )

    spacing = tuple(
        float(value)
        for value in gt_nii.header.get_zooms()[:3]
    )

    automatic_comparison = compare_binary_masks(
        target,
        _automatic_side_class_mask(
            automatic,
            semantic_target.class_label,
        ),
        spacing_mm=spacing,
    )

    semantic_comparison = compare_binary_masks(
        target,
        semantic_fg,
        spacing_mm=spacing,
    )

    point_only_dice = _load_block16_point_only_result(
        project_root=project_root,
        case_id=prompt.case_id,
    )

    return SemanticPointResult(
        case_id=prompt.case_id,
        laterality=semantic_target.laterality,
        class_label=semantic_target.class_label,
        class_name=semantic_target.class_name,
        point_xyz=prompt.point_xyz,
        target_volume_ml=prompt.target_volume_ml,

        automatic_side_class_dice=automatic_comparison.dice,
        point_only_dice=point_only_dice,
        semantic_plus_point_dice=semantic_comparison.dice,

        semantic_plus_point_minus_point_only=(
            semantic_comparison.dice - point_only_dice
        ),
        semantic_plus_point_minus_automatic=(
            semantic_comparison.dice - automatic_comparison.dice
        ),

        semantic_plus_point_volume_ml=(
            semantic_comparison.prediction_volume_ml
        ),
        semantic_plus_point_signed_volume_error_ml=(
            semantic_comparison.signed_volume_error_ml
        ),

        labels_present=labels_present,
        contains_255_unprocessed=contains_255,
        wall_time_seconds=float(wall_time_seconds),
        prediction_path=str(interactive_prediction_path),
    )


def _load_saved_result(
    path: Path,
) -> SemanticPointResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["result"]

    return SemanticPointResult(
        case_id=str(raw["case_id"]),
        laterality=str(raw["laterality"]),
        class_label=int(raw["class_label"]),
        class_name=str(raw["class_name"]),
        point_xyz=tuple(int(v) for v in raw["point_xyz"]),
        target_volume_ml=float(raw["target_volume_ml"]),
        automatic_side_class_dice=float(
            raw["automatic_side_class_dice"]
        ),
        point_only_dice=float(raw["point_only_dice"]),
        semantic_plus_point_dice=float(
            raw["semantic_plus_point_dice"]
        ),
        semantic_plus_point_minus_point_only=float(
            raw["semantic_plus_point_minus_point_only"]
        ),
        semantic_plus_point_minus_automatic=float(
            raw["semantic_plus_point_minus_automatic"]
        ),
        semantic_plus_point_volume_ml=float(
            raw["semantic_plus_point_volume_ml"]
        ),
        semantic_plus_point_signed_volume_error_ml=float(
            raw["semantic_plus_point_signed_volume_error_ml"]
        ),
        labels_present=tuple(
            int(v) for v in raw["labels_present"]
        ),
        contains_255_unprocessed=bool(
            raw["contains_255_unprocessed"]
        ),
        wall_time_seconds=float(raw["wall_time_seconds"]),
        prediction_path=str(raw["prediction_path"]),
    )


def run_semantic_point_case(
    *,
    prompt: OracleCystPrompt,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> SemanticPointResult:
    case_number = int(prompt.case_id.split("_")[-1])
    paths = pilot_case_paths(project_root, case_number)

    gt_nii = nib.load(str(paths.segmentation_path))
    gt = np.asanyarray(gt_nii.dataobj)

    semantic_target = infer_oracle_cyst_laterality(
        case_id=prompt.case_id,
        segmentation=gt,
        affine=gt_nii.affine,
        prompt=prompt,
    )

    output_root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_semantic_point_cyst"
        / prompt.case_id
    )
    raw_output = output_root / "raw"
    stable_prediction = output_root / "prediction.nii.gz"
    result_json = output_root / "result.json"

    if stable_prediction.is_file() and result_json.is_file() and not force:
        return _load_saved_result(result_json)

    if output_root.exists() and force:
        shutil.rmtree(output_root)

    raw_output.mkdir(parents=True, exist_ok=True)

    command = build_semantic_point_command(
        environment_name=environment_name,
        image_path=paths.imaging_path,
        output_dir=raw_output,
        class_label=semantic_target.class_label,
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
            f"Semantic+point inference failed for {prompt.case_id} "
            f"with exit code {completed.returncode}."
        )

    raw_prediction = discover_single_prediction(raw_output)
    shutil.copy2(raw_prediction, stable_prediction)

    result = evaluate_semantic_point_prediction(
        prompt=prompt,
        semantic_target=semantic_target,
        project_root=project_root,
        interactive_prediction_path=stable_prediction,
        wall_time_seconds=elapsed,
    )

    result_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "cyst_semantic_plus_oracle_point",
                "deployment_valid": False,
                "semantic_target": asdict(semantic_target),
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return result


def summarize_semantic_point_results(
    rows: Iterable[SemanticPointResult],
) -> SemanticPointSummary:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one result is required.")

    automatic = np.asarray(
        [row.automatic_side_class_dice for row in rows],
        dtype=float,
    )
    point_only = np.asarray(
        [row.point_only_dice for row in rows],
        dtype=float,
    )
    semantic = np.asarray(
        [row.semantic_plus_point_dice for row in rows],
        dtype=float,
    )

    delta_point = semantic - point_only
    delta_auto = semantic - automatic

    return SemanticPointSummary(
        case_count=len(rows),

        automatic_side_class_dice_mean=float(np.mean(automatic)),
        automatic_side_class_dice_median=float(np.median(automatic)),

        point_only_dice_mean=float(np.mean(point_only)),
        point_only_dice_median=float(np.median(point_only)),

        semantic_plus_point_dice_mean=float(np.mean(semantic)),
        semantic_plus_point_dice_median=float(np.median(semantic)),

        semantic_plus_point_minus_point_only_mean=float(
            np.mean(delta_point)
        ),
        semantic_plus_point_minus_point_only_median=float(
            np.median(delta_point)
        ),
        semantic_plus_point_minus_point_only_bootstrap_95ci=(
            bootstrap_mean_difference_ci(delta_point)
        ),

        improved_over_point_only_count=int(
            np.count_nonzero(delta_point > 1e-12)
        ),
        regressed_vs_point_only_count=int(
            np.count_nonzero(delta_point < -1e-12)
        ),

        semantic_plus_point_dice_ge_050_count=int(
            np.count_nonzero(semantic >= 0.5)
        ),
        semantic_plus_point_dice_ge_080_count=int(
            np.count_nonzero(semantic >= 0.8)
        ),

        point_only_zero_overlap_count=int(
            np.count_nonzero(point_only == 0.0)
        ),
        point_only_zero_overlap_rescued_count=int(
            np.count_nonzero(
                (point_only == 0.0) & (semantic > 0.0)
            )
        ),

        left_target_count=sum(row.laterality == "left" for row in rows),
        right_target_count=sum(row.laterality == "right" for row in rows),

        rows=rows,
    )


def write_semantic_point_summary(
    summary: SemanticPointSummary,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "cyst_semantic_plus_oracle_point",
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
