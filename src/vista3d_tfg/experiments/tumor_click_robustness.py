from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.experiments.tumor_oracle_cohort import (
    load_all_frozen_prompts,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
    build_tumor_special_point_command,
    evaluate_tumor_oracle_smoke,
    run_tumor_oracle_smoke_case,
)
from vista3d_tfg.integration.nv_inference import (
    discover_single_prediction,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
)


VARIANTS = ("deep", "centroid", "shallow_5mm")


@dataclass(frozen=True, slots=True)
class ClickVariant:
    case_id: str
    variant: str
    point_xyz: tuple[int, int, int]
    interior_radius_mm: float


@dataclass(frozen=True, slots=True)
class ClickRobustnessCase:
    case_id: str
    deep: TumorOracleSmokeResult
    centroid: TumorOracleSmokeResult
    shallow_5mm: TumorOracleSmokeResult


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
    return np.rint(array).astype(np.int16, copy=False)


def _largest_component(
    tumor: np.ndarray,
    component_id: int,
) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(tumor, structure=structure)

    if component_id < 1 or component_id > count:
        raise ValueError("Frozen tumor component no longer exists.")

    return labels == component_id


def derive_click_variants(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
) -> tuple[ClickVariant, ...]:
    nii = nib.load(str(ground_truth_path))
    gt = _integer_labels(np.asanyarray(nii.dataobj))
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])

    component = _largest_component(
        gt == 2,
        prompt.component_id,
    )

    if int(np.count_nonzero(component)) != prompt.target_voxels:
        raise ValueError(
            f"{prompt.case_id}: frozen target size changed."
        )

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing,
    )

    deep_point = prompt.point_xyz
    deep_radius = float(distance[deep_point])

    coords = np.argwhere(component)
    centroid = coords.mean(axis=0)
    centroid_index = int(
        np.argmin(
            np.sum(
                (coords.astype(float) - centroid[None, :]) ** 2,
                axis=1,
            )
        )
    )
    centroid_point = tuple(
        int(v) for v in coords[centroid_index]
    )
    centroid_radius = float(distance[centroid_point])

    component_distances = distance[component]
    target_depth_mm = 5.0
    shallow_index = int(
        np.argmin(
            np.abs(component_distances - target_depth_mm)
        )
    )
    shallow_point = tuple(
        int(v) for v in coords[shallow_index]
    )
    shallow_radius = float(distance[shallow_point])

    return (
        ClickVariant(
            case_id=prompt.case_id,
            variant="deep",
            point_xyz=deep_point,
            interior_radius_mm=deep_radius,
        ),
        ClickVariant(
            case_id=prompt.case_id,
            variant="centroid",
            point_xyz=centroid_point,
            interior_radius_mm=centroid_radius,
        ),
        ClickVariant(
            case_id=prompt.case_id,
            variant="shallow_5mm",
            point_xyz=shallow_point,
            interior_radius_mm=shallow_radius,
        ),
    )


def _prompt_for_variant(
    base: TumorOraclePrompt,
    variant: ClickVariant,
) -> TumorOraclePrompt:
    return TumorOraclePrompt(
        case_id=base.case_id,
        component_id=base.component_id,
        point_xyz=variant.point_xyz,
        target_voxels=base.target_voxels,
        target_volume_ml=base.target_volume_ml,
        total_tumor_volume_ml=base.total_tumor_volume_ml,
        interior_radius_mm=variant.interior_radius_mm,
    )


def run_click_variant(
    *,
    base_prompt: TumorOraclePrompt,
    variant: ClickVariant,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
    require_prompt_inside_target: bool = True,
) -> TumorOracleSmokeResult:
    if variant.variant == "deep":
        return run_tumor_oracle_smoke_case(
            prompt=base_prompt,
            project_root=project_root,
            nv_paths=nv_paths,
            environment_name=environment_name,
            force=False,
        )

    prompt = _prompt_for_variant(base_prompt, variant)

    case_root = (
        project_root
        / "data"
        / "raw"
        / "kits23"
        / prompt.case_id
    )
    image_path = case_root / "imaging.nii.gz"
    ground_truth_path = case_root / "segmentation.nii.gz"
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
        / "nv_segment_ct_interactive_tumor_click_robustness"
        / variant.variant
        / prompt.case_id
    )
    raw_output = output_root / "raw"
    stable_prediction = output_root / "prediction.nii.gz"
    result_json = output_root / "result.json"

    if stable_prediction.is_file() and result_json.is_file() and not force:
        payload = json.loads(
            result_json.read_text(encoding="utf-8")
        )["result"]
        return TumorOracleSmokeResult(
            case_id=str(payload["case_id"]),
            point_xyz=tuple(int(v) for v in payload["point_xyz"]),
            point_label=int(payload["point_label"]),
            target_volume_ml=float(payload["target_volume_ml"]),
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
            original_hec_dice=float(payload["original_hec_dice"]),
            merged_hec_dice=float(payload["merged_hec_dice"]),
            hec_dice_delta=float(payload["hec_dice_delta"]),
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
                int(v) for v in payload["interactive_labels_present"]
            ),
            contains_255_unprocessed=bool(
                payload["contains_255_unprocessed"]
            ),
            wall_time_seconds=float(payload["wall_time_seconds"]),
            prediction_path=str(payload["prediction_path"]),
        )

    if output_root.exists() and force:
        shutil.rmtree(output_root)
    raw_output.mkdir(parents=True, exist_ok=True)

    command = build_tumor_special_point_command(
        environment_name=environment_name,
        image_path=image_path,
        output_dir=raw_output,
        point_xyz=prompt.point_xyz,
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
            f"{variant.variant} inference failed for "
            f"{prompt.case_id}: exit {completed.returncode}."
        )

    raw_prediction = discover_single_prediction(raw_output)
    shutil.copy2(raw_prediction, stable_prediction)

    result = evaluate_tumor_oracle_smoke(
        prompt=prompt,
        ground_truth_path=ground_truth_path,
        automatic_prediction_path=automatic_prediction_path,
        interactive_prediction_path=stable_prediction,
        wall_time_seconds=elapsed,
        require_prompt_inside_target=(
            require_prompt_inside_target
        ),
    )

    result_json.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": (
                    "nv_segment_ct_tumor_click_robustness"
                ),
                "variant": variant.variant,
                "oracle_simulated_click": True,
                "deployment_valid": False,
                "interior_radius_mm": variant.interior_radius_mm,
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return result


def summarize_click_robustness(
    rows: Iterable[ClickRobustnessCase],
) -> dict[str, object]:
    rows = tuple(rows)
    if len(rows) != 9:
        raise ValueError(
            f"Expected 9 click-robustness cases, got {len(rows)}."
        )

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "experiment": "nv_segment_ct_tumor_click_robustness",
        "case_count": 9,
        "oracle_simulated_clicks": True,
        "deployment_valid": False,
        "variants": {},
    }

    variants_payload = payload["variants"]
    assert isinstance(variants_payload, dict)

    for variant_name in VARIANTS:
        results = tuple(
            getattr(row, variant_name)
            for row in rows
        )

        dice = np.asarray(
            [row.interactive_target_dice for row in results],
            dtype=float,
        )
        coverage = np.asarray(
            [row.tumor_coverage_gain for row in results],
            dtype=float,
        )
        hd95 = np.asarray(
            [row.hec_hd95_reduction_mm for row in results],
            dtype=float,
        )
        hec_dice = np.asarray(
            [row.hec_dice_delta for row in results],
            dtype=float,
        )
        surface = np.asarray(
            [row.hec_surface_dice_delta for row in results],
            dtype=float,
        )

        variants_payload[variant_name] = {
            "tumor_dice_mean": float(dice.mean()),
            "tumor_dice_median": float(np.median(dice)),
            "tumor_dice_ge_050": int(np.count_nonzero(dice >= 0.50)),
            "tumor_coverage_gain_mean": float(coverage.mean()),
            "tumor_coverage_gain_median": float(
                np.median(coverage)
            ),
            "hec_hd95_reduction_mean_mm": float(hd95.mean()),
            "hec_hd95_reduction_median_mm": float(
                np.median(hd95)
            ),
            "hd95_improved_count": int(
                np.count_nonzero(hd95 > 0.0)
            ),
            "hec_dice_delta_mean": float(hec_dice.mean()),
            "hec_surface_dice_delta_mean": float(
                surface.mean()
            ),
        }

    deep_dice = np.asarray(
        [row.deep.interactive_target_dice for row in rows],
        dtype=float,
    )
    centroid_dice = np.asarray(
        [row.centroid.interactive_target_dice for row in rows],
        dtype=float,
    )
    shallow_dice = np.asarray(
        [row.shallow_5mm.interactive_target_dice for row in rows],
        dtype=float,
    )

    payload["robustness_vs_deep"] = {
        "centroid_tumor_dice_delta_mean": float(
            (centroid_dice - deep_dice).mean()
        ),
        "centroid_cases_within_0_05_dice": int(
            np.count_nonzero(
                np.abs(centroid_dice - deep_dice) <= 0.05
            )
        ),
        "shallow_5mm_tumor_dice_delta_mean": float(
            (shallow_dice - deep_dice).mean()
        ),
        "shallow_5mm_cases_within_0_05_dice": int(
            np.count_nonzero(
                np.abs(shallow_dice - deep_dice) <= 0.05
            )
        ),
    }

    payload["case_rows"] = [
        {
            "case_id": row.case_id,
            "deep": row.deep.to_dict(),
            "centroid": row.centroid.to_dict(),
            "shallow_5mm": row.shallow_5mm.to_dict(),
        }
        for row in rows
    ]

    return payload


def write_click_robustness_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "variant",
        "point_xyz",
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            assert isinstance(row, dict)
            for variant in VARIANTS:
                result = row[variant]
                assert isinstance(result, dict)
                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "variant": variant,
                        "point_xyz": ";".join(
                            str(v) for v in result["point_xyz"]
                        ),
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
