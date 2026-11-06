from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)


DEPTH_THRESHOLDS_MM = (5.0, 10.0, 15.0)


@dataclass(frozen=True, slots=True)
class SafeRegionDepth:
    depth_mm: float
    voxel_count: int
    volume_ml: float
    fraction_of_target: float
    nonempty: bool


@dataclass(frozen=True, slots=True)
class SafeRegionCase:
    case_id: str
    target_voxels: int
    target_volume_ml: float
    max_interior_depth_mm: float
    safe_regions: tuple[SafeRegionDepth, ...]


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


def _frozen_target_component(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(ground_truth_path))
    gt = _integer_labels(np.asanyarray(nii.dataobj))

    spacing = tuple(
        float(value)
        for value in nii.header.get_zooms()[:3]
    )

    structure = ndimage.generate_binary_structure(3, 1)
    labels, component_count = ndimage.label(
        gt == 2,
        structure=structure,
    )

    if not (1 <= prompt.component_id <= component_count):
        raise ValueError(
            f"{prompt.case_id}: frozen tumor component is unavailable."
        )

    component = labels == prompt.component_id
    target_voxels = int(np.count_nonzero(component))

    if target_voxels != prompt.target_voxels:
        raise ValueError(
            f"{prompt.case_id}: frozen tumor target size changed "
            f"({target_voxels} != {prompt.target_voxels})."
        )

    return component, spacing


def evaluate_safe_click_region(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
    thresholds_mm: tuple[float, ...] = DEPTH_THRESHOLDS_MM,
) -> SafeRegionCase:
    if not thresholds_mm:
        raise ValueError("At least one depth threshold is required.")
    if any(value <= 0.0 for value in thresholds_mm):
        raise ValueError("Depth thresholds must be positive.")
    if tuple(sorted(thresholds_mm)) != thresholds_mm:
        raise ValueError("Depth thresholds must be sorted ascending.")

    component, spacing = _frozen_target_component(
        prompt=prompt,
        ground_truth_path=ground_truth_path,
    )

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing,
    )

    component_distances = distance[component]
    if component_distances.size != prompt.target_voxels:
        raise RuntimeError(
            f"{prompt.case_id}: distance-map target size mismatch."
        )

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)
    max_depth = float(component_distances.max())

    safe_regions = []
    previous_count = prompt.target_voxels + 1

    for depth_mm in thresholds_mm:
        safe_voxels = int(
            np.count_nonzero(
                component_distances >= (depth_mm - 1e-12)
            )
        )

        if safe_voxels > previous_count:
            raise RuntimeError(
                f"{prompt.case_id}: safe-region volume is not monotonic."
            )
        previous_count = safe_voxels

        safe_regions.append(
            SafeRegionDepth(
                depth_mm=float(depth_mm),
                voxel_count=safe_voxels,
                volume_ml=safe_voxels * voxel_volume_ml,
                fraction_of_target=(
                    safe_voxels / prompt.target_voxels
                ),
                nonempty=(safe_voxels > 0),
            )
        )

    return SafeRegionCase(
        case_id=prompt.case_id,
        target_voxels=prompt.target_voxels,
        target_volume_ml=prompt.target_volume_ml,
        max_interior_depth_mm=max_depth,
        safe_regions=tuple(safe_regions),
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


def summarize_safe_click_regions(
    rows: Iterable[SafeRegionCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 9:
        raise ValueError(
            f"Expected 9 frozen tumor cases, got {len(rows)}."
        )

    threshold_order = tuple(
        item.depth_mm
        for item in rows[0].safe_regions
    )

    for row in rows[1:]:
        if tuple(
            item.depth_mm
            for item in row.safe_regions
        ) != threshold_order:
            raise ValueError(
                "Safe-region thresholds differ across cases."
            )

    thresholds_payload: dict[str, object] = {}

    for index, depth_mm in enumerate(threshold_order):
        items = tuple(
            row.safe_regions[index]
            for row in rows
        )

        fractions = np.asarray(
            [item.fraction_of_target for item in items],
            dtype=float,
        )
        volumes = np.asarray(
            [item.volume_ml for item in items],
            dtype=float,
        )

        thresholds_payload[f"{depth_mm:g}mm"] = {
            "depth_mm": depth_mm,
            "cases_with_nonempty_region": int(
                np.count_nonzero(
                    [item.nonempty for item in items]
                )
            ),
            "safe_fraction": _summary(fractions),
            "safe_volume_ml": _summary(volumes),
            "cases_safe_fraction_ge_0_05": int(
                np.count_nonzero(fractions >= 0.05)
            ),
            "cases_safe_fraction_ge_0_10": int(
                np.count_nonzero(fractions >= 0.10)
            ),
            "cases_safe_fraction_ge_0_25": int(
                np.count_nonzero(fractions >= 0.25)
            ),
        }

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_tumor_safe_click_region_audit"
        ),
        "case_count": len(rows),
        "deployment_valid": False,
        "gt_derived_geometry": True,
        "target": (
            "largest frozen KiTS23 tumor component per Block 41A"
        ),
        "max_interior_depth_mm": _summary(
            row.max_interior_depth_mm
            for row in rows
        ),
        "thresholds": thresholds_payload,
        "case_rows": [
            {
                "case_id": row.case_id,
                "target_voxels": row.target_voxels,
                "target_volume_ml": row.target_volume_ml,
                "max_interior_depth_mm": row.max_interior_depth_mm,
                "safe_regions": [
                    asdict(item)
                    for item in row.safe_regions
                ],
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "Safe regions are derived from ground-truth tumor geometry. "
            "They quantify whether the click depths tested in Block 43 "
            "occupy a practically available portion of these nine tumors; "
            "they are not an autonomous localization method or a universal "
            "clinical distance threshold."
        ),
    }


def attach_block43_performance(
    *,
    payload: dict[str, object],
    block43_summary_path: Path,
) -> None:
    block43 = json.loads(
        block43_summary_path.read_text(encoding="utf-8")
    )
    variants = block43.get("variants")

    if not isinstance(variants, dict):
        raise ValueError("Block 43 summary has no variants object.")

    mapping = {
        "5mm": "shallow_5mm",
        "10mm": "depth_10mm",
        "15mm": "depth_15mm",
    }

    performance = {}

    for threshold_name, variant_name in mapping.items():
        item = variants.get(variant_name)
        if not isinstance(item, dict):
            raise ValueError(
                f"Block 43 summary is missing {variant_name}."
            )

        performance[threshold_name] = {
            "source_variant": variant_name,
            "tumor_dice_mean": float(
                item["tumor_dice_mean"]
            ),
            "tumor_dice_median": float(
                item["tumor_dice_median"]
            ),
            "tumor_dice_ge_050_count": int(
                item["tumor_dice_ge_050_count"]
            ),
            "tumor_dice_ge_070_count": int(
                item["tumor_dice_ge_070_count"]
            ),
            "tumor_coverage_gain_mean": float(
                item["tumor_coverage_gain_mean"]
            ),
            "hec_hd95_reduction_mean_mm": float(
                item["hec_hd95_reduction_mean_mm"]
            ),
            "hd95_improved_count": int(
                item["hd95_improved_count"]
            ),
        }

    payload["block43_depth_performance"] = performance


def write_safe_click_region_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
