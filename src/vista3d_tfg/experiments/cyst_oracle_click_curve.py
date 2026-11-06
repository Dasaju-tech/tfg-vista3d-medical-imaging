from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable, Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.evaluation.nv_kits23 import compare_binary_masks
from vista3d_tfg.experiments.cyst_oracle_prompt import (
    OracleCystPrompt,
    _interactive_foreground,
    _target_component_from_prompt,
    load_prompt_manifest,
)
from vista3d_tfg.experiments.pilot30 import pilot_case_paths
from vista3d_tfg.integration.nv_inference import discover_single_prediction
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
    resolve_conda_executable,
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class OracleCorrectionClick:
    index: int
    point_xyz: tuple[int, int, int]
    point_label: int
    source: str
    error_component_voxels: int
    interior_radius_mm: float


@dataclass(frozen=True, slots=True)
class OracleClickRound:
    click_count: int
    dice: float
    prediction_volume_ml: float
    signed_volume_error_ml: float
    labels_present: tuple[int, ...]
    contains_255_unprocessed: bool
    wall_time_seconds: float
    prediction_path: str


@dataclass(frozen=True, slots=True)
class OracleClickTrajectory:
    case_id: str
    target_volume_ml: float
    automatic_target_dice: float
    clicks: tuple[OracleCorrectionClick, ...]
    rounds: tuple[OracleClickRound, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClickCountSummary:
    click_count: int
    case_count: int
    dice_mean: float
    dice_median: float
    dice_minimum: float
    dice_maximum: float
    dice_ge_050_count: int
    dice_ge_080_count: int


@dataclass(frozen=True, slots=True)
class OracleClickCurveSummary:
    case_count: int
    maximum_clicks: int
    round_summaries: tuple[ClickCountSummary, ...]
    mean_delta_click1_to_final: float
    median_delta_click1_to_final: float
    improved_from_click1_count: int
    regressed_from_click1_count: int
    automatic_zero_overlap_case_count: int
    automatic_zero_overlap_rescued_by_final_count: int
    trajectories: tuple[OracleClickTrajectory, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _largest_component_deepest_point(
    mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> tuple[tuple[int, int, int], int, float] | None:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, count = ndimage.label(
        mask.astype(bool, copy=False),
        structure=structure,
    )

    if count == 0:
        return None

    sizes = np.bincount(labels.ravel())[1:]
    component_id = int(np.argmax(sizes)) + 1
    component = labels == component_id

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing_mm,
    )
    flat_index = int(np.argmax(distance))
    point = tuple(
        int(value)
        for value in np.unravel_index(flat_index, component.shape)
    )

    return (
        point,
        int(np.count_nonzero(component)),
        float(distance[point]),
    )


def select_next_oracle_correction(
    *,
    index: int,
    target: np.ndarray,
    prediction_foreground: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> OracleCorrectionClick | None:
    false_negative = target & ~prediction_foreground
    false_positive = prediction_foreground & ~target

    fn_count = int(np.count_nonzero(false_negative))
    fp_count = int(np.count_nonzero(false_positive))

    if fn_count == 0 and fp_count == 0:
        return None

    if fn_count >= fp_count:
        selected = false_negative
        label = 1
        source = "largest_false_negative_component"
    else:
        selected = false_positive
        label = 0
        source = "largest_false_positive_component"

    deepest = _largest_component_deepest_point(
        selected,
        spacing_mm=spacing_mm,
    )

    if deepest is None:
        raise RuntimeError("Selected oracle error mask has no component.")

    point, component_voxels, interior_radius_mm = deepest

    return OracleCorrectionClick(
        index=index,
        point_xyz=point,
        point_label=label,
        source=source,
        error_component_voxels=component_voxels,
        interior_radius_mm=interior_radius_mm,
    )


def build_multi_point_command(
    *,
    environment_name: str,
    image_path: Path,
    output_dir: Path,
    clicks: Iterable[OracleCorrectionClick],
) -> list[str]:
    clicks = tuple(clicks)

    if not clicks:
        raise ValueError("At least one click is required.")

    input_dict = {
        "image": image_path.resolve().as_posix(),
        "points": [list(click.point_xyz) for click in clicks],
        "point_labels": [click.point_label for click in clicks],
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


def _load_prediction_foreground(
    prediction_path: Path,
) -> tuple[
    np.ndarray,
    tuple[int, ...],
    bool,
]:
    nii = nib.load(str(prediction_path))
    prediction = np.asanyarray(nii.dataobj)
    foreground, labels_present, contains_255 = _interactive_foreground(
        prediction
    )
    return foreground, labels_present, contains_255


def _evaluate_round(
    *,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
    prediction_path: Path,
    click_count: int,
    wall_time_seconds: float,
) -> OracleClickRound:
    foreground, labels_present, contains_255 = _load_prediction_foreground(
        prediction_path
    )

    comparison = compare_binary_masks(
        target,
        foreground,
        spacing_mm=spacing_mm,
    )

    return OracleClickRound(
        click_count=click_count,
        dice=comparison.dice,
        prediction_volume_ml=comparison.prediction_volume_ml,
        signed_volume_error_ml=comparison.signed_volume_error_ml,
        labels_present=labels_present,
        contains_255_unprocessed=contains_255,
        wall_time_seconds=float(wall_time_seconds),
        prediction_path=str(prediction_path),
    )


def _load_block16_first_round(
    *,
    project_root: Path,
    prompt: OracleCystPrompt,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> tuple[OracleCorrectionClick, OracleClickRound, float]:
    root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_interactive_cyst"
        / prompt.case_id
    )
    result_path = root / "result.json"
    prediction_path = root / "prediction.nii.gz"

    if not result_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError(
            f"Block 16 result is missing for {prompt.case_id}."
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    result = payload["result"]

    click = OracleCorrectionClick(
        index=1,
        point_xyz=prompt.point_xyz,
        point_label=1,
        source="largest_gt_cyst_deepest_interior",
        error_component_voxels=prompt.target_voxels,
        interior_radius_mm=prompt.interior_radius_mm,
    )

    round_one = _evaluate_round(
        target=target,
        spacing_mm=spacing_mm,
        prediction_path=prediction_path,
        click_count=1,
        wall_time_seconds=float(result["wall_time_seconds"]),
    )

    return click, round_one, float(result["automatic_target_dice"])


def run_oracle_click_trajectory(
    *,
    prompt: OracleCystPrompt,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    maximum_clicks: int = 3,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> OracleClickTrajectory:
    if maximum_clicks < 1:
        raise ValueError("maximum_clicks must be at least 1.")

    case_number = int(prompt.case_id.split("_")[-1])
    paths = pilot_case_paths(project_root, case_number)

    gt_nii = nib.load(str(paths.segmentation_path))
    gt = np.asanyarray(gt_nii.dataobj)
    spacing = tuple(
        float(value)
        for value in gt_nii.header.get_zooms()[:3]
    )
    target = _target_component_from_prompt(gt, prompt)

    click_one, round_one, automatic_target_dice = _load_block16_first_round(
        project_root=project_root,
        prompt=prompt,
        target=target,
        spacing_mm=spacing,
    )

    clicks = [click_one]
    rounds = [round_one]

    if maximum_clicks == 1:
        return OracleClickTrajectory(
            case_id=prompt.case_id,
            target_volume_ml=prompt.target_volume_ml,
            automatic_target_dice=automatic_target_dice,
            clicks=tuple(clicks),
            rounds=tuple(rounds),
        )

    trajectory_root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_interactive_cyst_curve"
        / prompt.case_id
    )

    foreground, _, _ = _load_prediction_foreground(
        Path(round_one.prediction_path)
    )

    for click_index in range(2, maximum_clicks + 1):
        correction = select_next_oracle_correction(
            index=click_index,
            target=target,
            prediction_foreground=foreground,
            spacing_mm=spacing,
        )

        if correction is None:
            # Perfect segmentation: subsequent clicks are unnecessary.
            break

        clicks.append(correction)

        round_root = trajectory_root / f"click_{click_index:02d}"
        raw_output = round_root / "raw"
        stable_prediction = round_root / "prediction.nii.gz"
        round_json = round_root / "round.json"

        if stable_prediction.is_file() and round_json.is_file() and not force:
            payload = json.loads(round_json.read_text(encoding="utf-8"))
            stored = payload["result"]
            round_result = OracleClickRound(
                click_count=int(stored["click_count"]),
                dice=float(stored["dice"]),
                prediction_volume_ml=float(stored["prediction_volume_ml"]),
                signed_volume_error_ml=float(
                    stored["signed_volume_error_ml"]
                ),
                labels_present=tuple(
                    int(value) for value in stored["labels_present"]
                ),
                contains_255_unprocessed=bool(
                    stored["contains_255_unprocessed"]
                ),
                wall_time_seconds=float(stored["wall_time_seconds"]),
                prediction_path=str(stored["prediction_path"]),
            )
        else:
            if round_root.exists() and force:
                shutil.rmtree(round_root)

            raw_output.mkdir(parents=True, exist_ok=True)

            command = build_multi_point_command(
                environment_name=environment_name,
                image_path=paths.imaging_path,
                output_dir=raw_output,
                clicks=clicks,
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
                    f"Oracle click-curve inference failed for "
                    f"{prompt.case_id}, click {click_index}."
                )

            raw_prediction = discover_single_prediction(raw_output)
            shutil.copy2(raw_prediction, stable_prediction)

            round_result = _evaluate_round(
                target=target,
                spacing_mm=spacing,
                prediction_path=stable_prediction,
                click_count=click_index,
                wall_time_seconds=elapsed,
            )

            round_json.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "experiment": "oracle_cyst_iterative_click_curve",
                        "deployment_valid": False,
                        "clicks": [asdict(click) for click in clicks],
                        "result": asdict(round_result),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        rounds.append(round_result)
        foreground, _, _ = _load_prediction_foreground(
            Path(round_result.prediction_path)
        )

    trajectory = OracleClickTrajectory(
        case_id=prompt.case_id,
        target_volume_ml=prompt.target_volume_ml,
        automatic_target_dice=automatic_target_dice,
        clicks=tuple(clicks),
        rounds=tuple(rounds),
    )

    trajectory_root.mkdir(parents=True, exist_ok=True)
    (trajectory_root / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "oracle_cyst_iterative_click_curve",
                "deployment_valid": False,
                "maximum_clicks_requested": maximum_clicks,
                "result": trajectory.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return trajectory


def _dice_at_or_before(
    trajectory: OracleClickTrajectory,
    click_count: int,
) -> float:
    eligible = [
        round_result
        for round_result in trajectory.rounds
        if round_result.click_count <= click_count
    ]

    if not eligible:
        raise RuntimeError("Trajectory contains no eligible round.")

    return eligible[-1].dice


def summarize_click_curve(
    trajectories: Iterable[OracleClickTrajectory],
    *,
    maximum_clicks: int,
) -> OracleClickCurveSummary:
    trajectories = tuple(trajectories)

    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    round_summaries = []

    for click_count in range(1, maximum_clicks + 1):
        dice = np.asarray(
            [
                _dice_at_or_before(trajectory, click_count)
                for trajectory in trajectories
            ],
            dtype=float,
        )

        round_summaries.append(
            ClickCountSummary(
                click_count=click_count,
                case_count=len(trajectories),
                dice_mean=float(np.mean(dice)),
                dice_median=float(np.median(dice)),
                dice_minimum=float(np.min(dice)),
                dice_maximum=float(np.max(dice)),
                dice_ge_050_count=int(np.count_nonzero(dice >= 0.5)),
                dice_ge_080_count=int(np.count_nonzero(dice >= 0.8)),
            )
        )

    first = np.asarray(
        [_dice_at_or_before(t, 1) for t in trajectories],
        dtype=float,
    )
    final = np.asarray(
        [
            _dice_at_or_before(t, maximum_clicks)
            for t in trajectories
        ],
        dtype=float,
    )
    delta = final - first

    automatic = np.asarray(
        [t.automatic_target_dice for t in trajectories],
        dtype=float,
    )
    automatic_zero = automatic == 0.0

    return OracleClickCurveSummary(
        case_count=len(trajectories),
        maximum_clicks=maximum_clicks,
        round_summaries=tuple(round_summaries),
        mean_delta_click1_to_final=float(np.mean(delta)),
        median_delta_click1_to_final=float(np.median(delta)),
        improved_from_click1_count=int(np.count_nonzero(delta > 0.0)),
        regressed_from_click1_count=int(np.count_nonzero(delta < 0.0)),
        automatic_zero_overlap_case_count=int(
            np.count_nonzero(automatic_zero)
        ),
        automatic_zero_overlap_rescued_by_final_count=int(
            np.count_nonzero(automatic_zero & (final > 0.0))
        ),
        trajectories=trajectories,
    )


def write_click_curve_summary(
    summary: OracleClickCurveSummary,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "oracle_cyst_iterative_click_curve",
                "deployment_valid": False,
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with (output_root / "trajectories.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fieldnames = [
            "case_id",
            "automatic_target_dice",
            "click_count",
            "dice",
            "dice_delta_from_click1",
            "last_click_label",
            "last_click_source",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for trajectory in summary.trajectories:
            click1_dice = _dice_at_or_before(trajectory, 1)

            for click_count in range(1, summary.maximum_clicks + 1):
                dice = _dice_at_or_before(trajectory, click_count)
                click = (
                    trajectory.clicks[
                        min(click_count, len(trajectory.clicks)) - 1
                    ]
                )

                writer.writerow(
                    {
                        "case_id": trajectory.case_id,
                        "automatic_target_dice": (
                            trajectory.automatic_target_dice
                        ),
                        "click_count": click_count,
                        "dice": dice,
                        "dice_delta_from_click1": dice - click1_dice,
                        "last_click_label": click.point_label,
                        "last_click_source": click.source,
                    }
                )


def render_click_curve(
    summary: OracleClickCurveSummary,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(1, summary.maximum_clicks + 1))

    figure, axis = plt.subplots(figsize=(8, 6))

    for trajectory in summary.trajectories:
        y = [
            _dice_at_or_before(trajectory, click_count)
            for click_count in x
        ]
        axis.plot(x, y, marker="o", alpha=0.35)

    median_curve = [
        next(
            round_summary.dice_median
            for round_summary in summary.round_summaries
            if round_summary.click_count == click_count
        )
        for click_count in x
    ]
    axis.plot(x, median_curve, marker="o", linewidth=3)

    axis.set_xticks(x)
    axis.set_xlabel("Cumulative oracle clicks")
    axis.set_ylabel("Target cyst Dice")
    axis.set_title("Interactive cyst Dice versus cumulative oracle clicks")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
