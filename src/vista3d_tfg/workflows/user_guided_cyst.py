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

from vista3d_tfg.experiments.cyst_oracle_prompt import (
    _interactive_foreground,
)
from vista3d_tfg.experiments.cyst_semantic_point import (
    LEFT_CYST_LABEL,
    RIGHT_CYST_LABEL,
    build_semantic_point_command,
)
from vista3d_tfg.experiments.pilot30 import pilot_case_paths
from vista3d_tfg.integration.nv_inference import discover_single_prediction
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
)


LEFT_KIDNEY_LABEL = 14
RIGHT_KIDNEY_LABEL = 5

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class AutomaticCystClassAssignment:
    laterality: str
    class_label: int
    class_name: str
    distance_to_left_kidney_mm: float
    distance_to_right_kidney_mm: float
    selected_distance_mm: float
    laterality_margin_mm: float


@dataclass(frozen=True, slots=True)
class UserGuidedCystResult:
    case_id: str
    point_xyz: tuple[int, int, int]
    semantic_assignment: AutomaticCystClassAssignment
    labels_present: tuple[int, ...]
    contains_255_unprocessed: bool
    foreground_voxels: int
    foreground_volume_ml: float
    wall_time_seconds: float
    prediction_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _distance_to_mask(
    mask: np.ndarray,
    *,
    point_xyz: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> float:
    if not np.any(mask):
        return float("inf")

    distances = ndimage.distance_transform_edt(
        ~mask.astype(bool, copy=False),
        sampling=spacing_mm,
    )
    return float(distances[point_xyz])


def assign_cyst_class_from_automatic_kidneys(
    *,
    automatic_prediction: np.ndarray,
    spacing_mm: tuple[float, float, float],
    point_xyz: tuple[int, int, int],
) -> AutomaticCystClassAssignment:
    if automatic_prediction.ndim != 3:
        raise ValueError("Automatic prediction must be 3D.")

    if any(
        coordinate < 0
        or coordinate >= automatic_prediction.shape[axis]
        for axis, coordinate in enumerate(point_xyz)
    ):
        raise ValueError("User click lies outside prediction volume.")

    rounded = np.rint(automatic_prediction).astype(
        np.int16,
        copy=False,
    )

    left_mask = rounded == LEFT_KIDNEY_LABEL
    right_mask = rounded == RIGHT_KIDNEY_LABEL

    if not np.any(left_mask) and not np.any(right_mask):
        raise RuntimeError(
            "Automatic renal segmentation contains neither left nor "
            "right kidney."
        )

    left_distance = _distance_to_mask(
        left_mask,
        point_xyz=point_xyz,
        spacing_mm=spacing_mm,
    )
    right_distance = _distance_to_mask(
        right_mask,
        point_xyz=point_xyz,
        spacing_mm=spacing_mm,
    )

    if np.isclose(
        left_distance,
        right_distance,
        rtol=0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "User click is laterally ambiguous: equal physical distance "
            "to automatic left and right kidneys."
        )

    if left_distance < right_distance:
        laterality = "left"
        class_label = LEFT_CYST_LABEL
        class_name = "left kidney cyst"
        selected_distance = left_distance
    else:
        laterality = "right"
        class_label = RIGHT_CYST_LABEL
        class_name = "right kidney cyst"
        selected_distance = right_distance

    if np.isfinite(left_distance) and np.isfinite(right_distance):
        margin = abs(left_distance - right_distance)
    else:
        margin = float("inf")

    return AutomaticCystClassAssignment(
        laterality=laterality,
        class_label=class_label,
        class_name=class_name,
        distance_to_left_kidney_mm=float(left_distance),
        distance_to_right_kidney_mm=float(right_distance),
        selected_distance_mm=float(selected_distance),
        laterality_margin_mm=float(margin),
    )


def assign_from_prediction_file(
    *,
    automatic_prediction_path: Path,
    point_xyz: tuple[int, int, int],
) -> AutomaticCystClassAssignment:
    prediction_nii = nib.load(str(automatic_prediction_path))
    prediction = np.asanyarray(prediction_nii.dataobj)

    spacing = tuple(
        float(value)
        for value in prediction_nii.header.get_zooms()[:3]
    )

    return assign_cyst_class_from_automatic_kidneys(
        automatic_prediction=prediction,
        spacing_mm=spacing,
        point_xyz=point_xyz,
    )


def _load_saved_result(
    path: Path,
) -> UserGuidedCystResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["result"]
    assignment = raw["semantic_assignment"]

    return UserGuidedCystResult(
        case_id=str(raw["case_id"]),
        point_xyz=tuple(int(v) for v in raw["point_xyz"]),
        semantic_assignment=AutomaticCystClassAssignment(
            laterality=str(assignment["laterality"]),
            class_label=int(assignment["class_label"]),
            class_name=str(assignment["class_name"]),
            distance_to_left_kidney_mm=float(
                assignment["distance_to_left_kidney_mm"]
            ),
            distance_to_right_kidney_mm=float(
                assignment["distance_to_right_kidney_mm"]
            ),
            selected_distance_mm=float(
                assignment["selected_distance_mm"]
            ),
            laterality_margin_mm=float(
                assignment["laterality_margin_mm"]
            ),
        ),
        labels_present=tuple(
            int(value) for value in raw["labels_present"]
        ),
        contains_255_unprocessed=bool(
            raw["contains_255_unprocessed"]
        ),
        foreground_voxels=int(raw["foreground_voxels"]),
        foreground_volume_ml=float(raw["foreground_volume_ml"]),
        wall_time_seconds=float(raw["wall_time_seconds"]),
        prediction_path=str(raw["prediction_path"]),
    )


def run_user_guided_cyst_segmentation(
    *,
    project_root: Path,
    case_number: int,
    point_xyz: tuple[int, int, int],
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    runner: Runner = subprocess.run,
    automatic_prediction_path: Path | None = None,
    output_directory: Path | None = None,
) -> UserGuidedCystResult:
    paths = pilot_case_paths(project_root, case_number)
    automatic_prediction = automatic_prediction_path or paths.prediction_path

    if not automatic_prediction.is_file():
        raise FileNotFoundError(
            "Automatic renal prediction is required before the user-guided "
            f"workflow: {automatic_prediction}"
        )

    assignment = assign_from_prediction_file(
        automatic_prediction_path=automatic_prediction,
        point_xyz=point_xyz,
    )

    output_root = output_directory or (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_user_guided_cyst"
        / paths.case_id
    )
    raw_output = output_root / "raw"
    stable_prediction = output_root / "prediction.nii.gz"
    manifest_path = output_root / "workflow.json"

    if stable_prediction.is_file() and manifest_path.is_file() and not force:
        saved = _load_saved_result(manifest_path)

        if saved.point_xyz != point_xyz:
            raise RuntimeError(
                f"{paths.case_id} already has a saved user-guided result "
                f"for point {saved.point_xyz}; use --force to replace it."
            )

        return saved

    if output_root.exists() and force:
        shutil.rmtree(output_root)

    raw_output.mkdir(parents=True, exist_ok=True)

    command = build_semantic_point_command(
        environment_name=environment_name,
        image_path=paths.imaging_path,
        output_dir=raw_output,
        class_label=assignment.class_label,
        point_xyz=point_xyz,
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
            f"User-guided cyst inference failed for {paths.case_id} "
            f"with exit code {completed.returncode}."
        )

    raw_prediction = discover_single_prediction(raw_output)
    shutil.copy2(raw_prediction, stable_prediction)

    result_nii = nib.load(str(stable_prediction))
    result_array = np.asanyarray(result_nii.dataobj)

    foreground, labels_present, contains_255 = _interactive_foreground(
        result_array
    )

    spacing = tuple(
        float(value)
        for value in result_nii.header.get_zooms()[:3]
    )
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    foreground_voxels = int(np.count_nonzero(foreground))

    result = UserGuidedCystResult(
        case_id=paths.case_id,
        point_xyz=point_xyz,
        semantic_assignment=assignment,
        labels_present=labels_present,
        contains_255_unprocessed=contains_255,
        foreground_voxels=foreground_voxels,
        foreground_volume_ml=float(
            foreground_voxels * voxel_volume_ml
        ),
        wall_time_seconds=float(elapsed),
        prediction_path=str(stable_prediction),
    )

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "workflow": "user_guided_renal_cyst_segmentation",
                "ground_truth_used": False,
                "clinical_validation": False,
                "workflow_type": "interactive_research_prototype",
                "input": {
                    "case_id": paths.case_id,
                    "image_path": str(paths.imaging_path),
                    "automatic_renal_prediction_path": str(
                        automatic_prediction
                    ),
                    "user_point_xyz": list(point_xyz),
                },
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return result
