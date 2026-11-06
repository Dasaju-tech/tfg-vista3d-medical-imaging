from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable

import nibabel as nib
import numpy as np

from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickVariant,
    run_click_variant,
)
from vista3d_tfg.experiments.tumor_dual_point_protocol import (
    DualPointCase,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
    evaluate_tumor_oracle_smoke,
)
from vista3d_tfg.integration.nv_inference import (
    discover_single_prediction,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
    resolve_conda_executable,
)


@dataclass(frozen=True, slots=True)
class DualPointComparison:
    case_id: str
    quartile: str
    largest_component_volume_ml: float
    point_separation_mm: float
    second_point_relative_depth: float
    dual: TumorOracleSmokeResult
    deep: TumorOracleSmokeResult


def load_dual_point_protocol(
    path: Path,
) -> tuple[DualPointCase, ...]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("case_count") != 12:
        raise ValueError(
            f"Expected frozen dual-point protocol N=12, got "
            f"{payload.get('case_count')}."
        )
    if payload.get("deployment_valid") is not False:
        raise ValueError(
            "Dual-point protocol must remain deployment-invalid."
        )
    if payload.get("oracle_simulated_clicks") is not True:
        raise ValueError(
            "Dual-point protocol must remain oracle-simulated."
        )

    protocol = payload.get("point_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Dual-point protocol has no point_protocol.")

    if protocol.get("point_labels") != [3, 3]:
        raise ValueError(
            "Frozen dual-point labels must remain [3, 3]."
        )
    if protocol.get("label_prompt") is not None:
        raise ValueError(
            "Frozen dual-point protocol must not use label_prompt."
        )

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Dual-point protocol has no cases list.")

    rows = tuple(
        DualPointCase(
            case_id=str(item["case_id"]),
            quartile=str(item["quartile"]),
            largest_component_volume_ml=float(
                item["largest_component_volume_ml"]
            ),
            component_id=int(item["component_id"]),
            target_voxels=int(item["target_voxels"]),
            max_interior_depth_mm=float(
                item["max_interior_depth_mm"]
            ),
            first_point_xyz=tuple(
                int(value)
                for value in item["first_point_xyz"]
            ),
            first_point_depth_mm=float(
                item["first_point_depth_mm"]
            ),
            second_point_xyz=tuple(
                int(value)
                for value in item["second_point_xyz"]
            ),
            second_point_depth_mm=float(
                item["second_point_depth_mm"]
            ),
            second_point_relative_depth=float(
                item["second_point_relative_depth"]
            ),
            point_separation_mm=float(
                item["point_separation_mm"]
            ),
        )
        for item in cases
    )

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 dual-point cases, got {len(rows)}."
        )

    counts = {
        quartile: sum(
            row.quartile == quartile
            for row in rows
        )
        for quartile in QUARTILE_NAMES
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError(
            f"Dual-point quartiles are unbalanced: {counts}."
        )

    return rows


def build_dual_point_command(
    *,
    environment_name: str,
    image_path: Path,
    output_dir: Path,
    first_point_xyz: tuple[int, int, int],
    second_point_xyz: tuple[int, int, int],
) -> list[str]:
    input_dict = {
        "image": image_path.resolve().as_posix(),
        "points": [
            list(first_point_xyz),
            list(second_point_xyz),
        ],
        "point_labels": [3, 3],
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


def _deep_prompt(
    row: DualPointCase,
) -> TumorOraclePrompt:
    return TumorOraclePrompt(
        case_id=row.case_id,
        component_id=row.component_id,
        point_xyz=row.first_point_xyz,
        target_voxels=row.target_voxels,
        target_volume_ml=row.largest_component_volume_ml,
        total_tumor_volume_ml=row.largest_component_volume_ml,
        interior_radius_mm=row.first_point_depth_mm,
    )


def _load_result(
    path: Path,
) -> TumorOracleSmokeResult:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )["result"]

    return TumorOracleSmokeResult(
        case_id=str(payload["case_id"]),
        point_xyz=tuple(
            int(value)
            for value in payload["point_xyz"]
        ),
        point_label=int(payload["point_label"]),
        target_volume_ml=float(
            payload["target_volume_ml"]
        ),
        total_tumor_volume_ml=float(
            payload["total_tumor_volume_ml"]
        ),
        interactive_target_dice=float(
            payload["interactive_target_dice"]
        ),
        interactive_full_tumor_dice=float(
            payload["interactive_full_tumor_dice"]
        ),
        original_tumor_coverage_fraction=float(
            payload["original_tumor_coverage_fraction"]
        ),
        merged_tumor_coverage_fraction=float(
            payload["merged_tumor_coverage_fraction"]
        ),
        tumor_coverage_gain=float(
            payload["tumor_coverage_gain"]
        ),
        original_hec_dice=float(
            payload["original_hec_dice"]
        ),
        merged_hec_dice=float(
            payload["merged_hec_dice"]
        ),
        hec_dice_delta=float(
            payload["hec_dice_delta"]
        ),
        original_hec_surface_dice=float(
            payload["original_hec_surface_dice"]
        ),
        merged_hec_surface_dice=float(
            payload["merged_hec_surface_dice"]
        ),
        hec_surface_dice_delta=float(
            payload["hec_surface_dice_delta"]
        ),
        original_hec_hd95_mm=float(
            payload["original_hec_hd95_mm"]
        ),
        merged_hec_hd95_mm=float(
            payload["merged_hec_hd95_mm"]
        ),
        hec_hd95_reduction_mm=float(
            payload["hec_hd95_reduction_mm"]
        ),
        interactive_prediction_volume_ml=float(
            payload["interactive_prediction_volume_ml"]
        ),
        interactive_labels_present=tuple(
            int(value)
            for value in payload["interactive_labels_present"]
        ),
        contains_255_unprocessed=bool(
            payload["contains_255_unprocessed"]
        ),
        wall_time_seconds=float(
            payload["wall_time_seconds"]
        ),
        prediction_path=str(
            payload["prediction_path"]
        ),
    )


def run_dual_point_case(
    *,
    row: DualPointCase,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
) -> TumorOracleSmokeResult:
    case_root = (
        project_root
        / "data"
        / "raw"
        / "kits23"
        / row.case_id
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
        / row.case_id
        / "prediction.nii.gz"
    )

    output_root = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_interactive_tumor_dual_point"
        / row.case_id
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
        return _load_result(result_json)

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

    command = build_dual_point_command(
        environment_name=environment_name,
        image_path=image_path,
        output_dir=raw_output,
        first_point_xyz=row.first_point_xyz,
        second_point_xyz=row.second_point_xyz,
    )

    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(nv_paths.ct_bundle_root),
        check=False,
        text=True,
        shell=False,
    )
    elapsed = time.perf_counter() - start

    if completed.returncode != 0:
        raise RuntimeError(
            f"Dual-point tumor inference failed for "
            f"{row.case_id} with exit code "
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
        prompt=_deep_prompt(row),
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
                    "nv_segment_ct_tumor_dual_interior_point"
                ),
                "oracle_simulated_clicks": True,
                "deployment_valid": False,
                "input_protocol": {
                    "points": [
                        list(row.first_point_xyz),
                        list(row.second_point_xyz),
                    ],
                    "point_labels": [3, 3],
                    "label_prompt": None,
                    "point_separation_mm": (
                        row.point_separation_mm
                    ),
                    "second_point_relative_depth": (
                        row.second_point_relative_depth
                    ),
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


def run_dual_vs_deep_comparison(
    *,
    row: DualPointCase,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
) -> DualPointComparison:
    deep_prompt = _deep_prompt(row)

    deep = run_click_variant(
        base_prompt=deep_prompt,
        variant=ClickVariant(
            case_id=row.case_id,
            variant="deep",
            point_xyz=row.first_point_xyz,
            interior_radius_mm=row.first_point_depth_mm,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=False,
    )

    dual = run_dual_point_case(
        row=row,
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=force,
    )

    return DualPointComparison(
        case_id=row.case_id,
        quartile=row.quartile,
        largest_component_volume_ml=(
            row.largest_component_volume_ml
        ),
        point_separation_mm=row.point_separation_mm,
        second_point_relative_depth=(
            row.second_point_relative_depth
        ),
        dual=dual,
        deep=deep,
    )


def _summary(
    values: Iterable[float],
) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize empty values.")

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _paired_bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 5000,
) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Bootstrap requires a non-empty 1D array."
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    means = values[indices].mean(axis=1)

    return {
        "lower_95": float(
            np.quantile(means, 0.025)
        ),
        "upper_95": float(
            np.quantile(means, 0.975)
        ),
    }


def _group_payload(
    rows: tuple[DualPointComparison, ...],
) -> dict[str, object]:
    dual_dice = np.asarray(
        [row.dual.interactive_target_dice for row in rows],
        dtype=float,
    )
    deep_dice = np.asarray(
        [row.deep.interactive_target_dice for row in rows],
        dtype=float,
    )
    dice_delta = dual_dice - deep_dice

    dual_coverage = np.asarray(
        [row.dual.tumor_coverage_gain for row in rows],
        dtype=float,
    )
    deep_coverage = np.asarray(
        [row.deep.tumor_coverage_gain for row in rows],
        dtype=float,
    )
    coverage_delta = dual_coverage - deep_coverage

    dual_hd95 = np.asarray(
        [row.dual.hec_hd95_reduction_mm for row in rows],
        dtype=float,
    )
    deep_hd95 = np.asarray(
        [row.deep.hec_hd95_reduction_mm for row in rows],
        dtype=float,
    )
    hd95_delta = dual_hd95 - deep_hd95

    dual_hec_dice = np.asarray(
        [row.dual.hec_dice_delta for row in rows],
        dtype=float,
    )
    deep_hec_dice = np.asarray(
        [row.deep.hec_dice_delta for row in rows],
        dtype=float,
    )
    hec_dice_delta = (
        dual_hec_dice - deep_hec_dice
    )

    dual_surface = np.asarray(
        [
            row.dual.hec_surface_dice_delta
            for row in rows
        ],
        dtype=float,
    )
    deep_surface = np.asarray(
        [
            row.deep.hec_surface_dice_delta
            for row in rows
        ],
        dtype=float,
    )
    surface_delta = (
        dual_surface - deep_surface
    )

    return {
        "count": len(rows),
        "dual": {
            "tumor_dice": _summary(dual_dice),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(
                    dual_dice >= 0.50
                )
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(
                    dual_dice >= 0.70
                )
            ),
            "tumor_coverage_gain": _summary(
                dual_coverage
            ),
            "hec_hd95_reduction_mm": _summary(
                dual_hd95
            ),
            "hd95_improved_count": int(
                np.count_nonzero(
                    dual_hd95 > 0.0
                )
            ),
            "hec_dice_delta": _summary(
                dual_hec_dice
            ),
            "hec_surface_dice_delta": _summary(
                dual_surface
            ),
        },
        "deep": {
            "tumor_dice": _summary(deep_dice),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(
                    deep_dice >= 0.50
                )
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(
                    deep_dice >= 0.70
                )
            ),
            "tumor_coverage_gain": _summary(
                deep_coverage
            ),
            "hec_hd95_reduction_mm": _summary(
                deep_hd95
            ),
            "hd95_improved_count": int(
                np.count_nonzero(
                    deep_hd95 > 0.0
                )
            ),
            "hec_dice_delta": _summary(
                deep_hec_dice
            ),
            "hec_surface_dice_delta": _summary(
                deep_surface
            ),
        },
        "paired_dual_minus_deep": {
            "tumor_dice": {
                **_summary(dice_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        dice_delta,
                        seed=20260905,
                    )
                ),
            },
            "tumor_coverage_gain": {
                **_summary(coverage_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        coverage_delta,
                        seed=20260906,
                    )
                ),
            },
            "hec_hd95_reduction_mm": {
                **_summary(hd95_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        hd95_delta,
                        seed=20260907,
                    )
                ),
            },
            "hec_dice_delta": _summary(
                hec_dice_delta
            ),
            "hec_surface_dice_delta": _summary(
                surface_delta
            ),
            "cases_within_0_05_tumor_dice": int(
                np.count_nonzero(
                    np.abs(dice_delta)
                    <= (0.05 + 1e-12)
                )
            ),
            "cases_within_0_10_tumor_dice": int(
                np.count_nonzero(
                    np.abs(dice_delta)
                    <= (0.10 + 1e-12)
                )
            ),
            "dual_tumor_dice_better_count": int(
                np.count_nonzero(
                    dice_delta > 0.0
                )
            ),
        },
    }


def summarize_dual_point_results(
    rows: Iterable[DualPointComparison],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 dual-point comparisons, got {len(rows)}."
        )

    counts = {
        quartile: sum(
            row.quartile == quartile
            for row in rows
        )
        for quartile in QUARTILE_NAMES
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError(
            f"Dual-point comparison quartiles "
            f"are unbalanced: {counts}."
        )

    separation = np.asarray(
        [row.point_separation_mm for row in rows],
        dtype=float,
    )
    dice_gain = np.asarray(
        [
            row.dual.interactive_target_dice
            - row.deep.interactive_target_dice
            for row in rows
        ],
        dtype=float,
    )

    if (
        separation.size >= 2
        and np.std(separation) > 0.0
        and np.std(dice_gain) > 0.0
    ):
        separation_vs_dice_gain = float(
            np.corrcoef(
                separation,
                dice_gain,
            )[0, 1]
        )
    else:
        separation_vs_dice_gain = 0.0

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_tumor_dual_interior_point_inference"
        ),
        "case_count": 12,
        "oracle_simulated_clicks": True,
        "deployment_valid": False,
        "overall": _group_payload(rows),
        "by_quartile": {
            quartile: _group_payload(
                tuple(
                    row
                    for row in rows
                    if row.quartile == quartile
                )
            )
            for quartile in QUARTILE_NAMES
        },
        "geometry_association": {
            "point_separation_mm": _summary(
                separation
            ),
            "separation_vs_dual_minus_deep_tumor_dice_pearson": (
                separation_vs_dice_gain
            ),
        },
        "case_rows": [
            {
                "case_id": row.case_id,
                "quartile": row.quartile,
                "largest_component_volume_ml": (
                    row.largest_component_volume_ml
                ),
                "point_separation_mm": (
                    row.point_separation_mm
                ),
                "second_point_relative_depth": (
                    row.second_point_relative_depth
                ),
                "dual": row.dual.to_dict(),
                "deep": row.deep.to_dict(),
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "Dual and deep interaction points are derived from "
            "ground-truth tumor geometry. This paired experiment "
            "tests whether a second spatially separated interior "
            "positive point improves interactive segmentation "
            "robustness; it does not validate autonomous localization."
        ),
    }


def write_dual_point_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_root / "summary.json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "quartile",
        "largest_component_volume_ml",
        "point_separation_mm",
        "second_point_relative_depth",
        "variant",
        "tumor_dice",
        "tumor_coverage_gain",
        "hec_dice_delta",
        "hec_surface_dice_delta",
        "hec_hd95_reduction_mm",
        "wall_time_seconds",
    ]

    with (output_root / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            assert isinstance(row, dict)

            for variant in ("dual", "deep"):
                result = row[variant]
                assert isinstance(result, dict)

                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "quartile": row["quartile"],
                        "largest_component_volume_ml": row[
                            "largest_component_volume_ml"
                        ],
                        "point_separation_mm": row[
                            "point_separation_mm"
                        ],
                        "second_point_relative_depth": row[
                            "second_point_relative_depth"
                        ],
                        "variant": variant,
                        "tumor_dice": result[
                            "interactive_target_dice"
                        ],
                        "tumor_coverage_gain": result[
                            "tumor_coverage_gain"
                        ],
                        "hec_dice_delta": result[
                            "hec_dice_delta"
                        ],
                        "hec_surface_dice_delta": result[
                            "hec_surface_dice_delta"
                        ],
                        "hec_hd95_reduction_mm": result[
                            "hec_hd95_reduction_mm"
                        ],
                        "wall_time_seconds": result[
                            "wall_time_seconds"
                        ],
                    }
                )
