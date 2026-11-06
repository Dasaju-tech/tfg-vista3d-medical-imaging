from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    load_adaptive_protocol,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    AdaptiveDepthCase,
    QUARTILE_NAMES,
)


SECOND_POINT_MIN_RELATIVE_DEPTH = 0.50


@dataclass(frozen=True, slots=True)
class DualPointCase:
    case_id: str
    quartile: str
    largest_component_volume_ml: float
    component_id: int
    target_voxels: int
    max_interior_depth_mm: float
    first_point_xyz: tuple[int, int, int]
    first_point_depth_mm: float
    second_point_xyz: tuple[int, int, int]
    second_point_depth_mm: float
    second_point_relative_depth: float
    point_separation_mm: float


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
    *,
    case: AdaptiveDepthCase,
    segmentation_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
]:
    nii = nib.load(str(segmentation_path))
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

    if not (1 <= case.component_id <= component_count):
        raise ValueError(
            f"{case.case_id}: frozen tumor component is unavailable."
        )

    component = labels == case.component_id
    target_voxels = int(np.count_nonzero(component))

    if target_voxels != case.target_voxels:
        raise ValueError(
            f"{case.case_id}: frozen tumor component size changed "
            f"({target_voxels} != {case.target_voxels})."
        )

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing,
    )

    return component, distance, spacing


def derive_dual_point_case(
    *,
    case: AdaptiveDepthCase,
    segmentation_path: Path,
    second_point_min_relative_depth: float = (
        SECOND_POINT_MIN_RELATIVE_DEPTH
    ),
) -> DualPointCase:
    if not (0.0 < second_point_min_relative_depth < 1.0):
        raise ValueError(
            "second_point_min_relative_depth must lie in (0, 1)."
        )

    component, distance, spacing = _largest_component(
        case=case,
        segmentation_path=segmentation_path,
    )

    coords = np.argwhere(component)
    component_depths = distance[component]

    deepest_index = int(np.argmax(component_depths))
    first_point = coords[deepest_index]
    first_depth = float(component_depths[deepest_index])

    if not np.isclose(
        first_depth,
        case.max_interior_depth_mm,
        rtol=0.0,
        atol=max(1e-6, min(spacing)),
    ):
        raise ValueError(
            f"{case.case_id}: maximum interior depth changed."
        )

    min_second_depth = (
        second_point_min_relative_depth * first_depth
    )
    candidate_mask = (
        component_depths
        >= (min_second_depth - 1e-12)
    )
    candidate_coords = coords[candidate_mask]
    candidate_depths = component_depths[candidate_mask]

    if candidate_coords.shape[0] < 2:
        raise ValueError(
            f"{case.case_id}: fewer than two voxels satisfy the "
            f"{second_point_min_relative_depth:.2f} relative-depth rule."
        )

    physical_delta = (
        candidate_coords.astype(float)
        - first_point[None, :].astype(float)
    ) * np.asarray(spacing, dtype=float)[None, :]
    physical_distance = np.linalg.norm(
        physical_delta,
        axis=1,
    )

    nonzero = physical_distance > 1e-12
    if not np.any(nonzero):
        raise ValueError(
            f"{case.case_id}: no spatially separated second-point candidate."
        )

    candidate_indices = np.flatnonzero(nonzero)
    farthest_local = int(
        candidate_indices[
            np.argmax(physical_distance[candidate_indices])
        ]
    )

    second_point = candidate_coords[farthest_local]
    second_depth = float(candidate_depths[farthest_local])
    separation = float(physical_distance[farthest_local])

    first_xyz = tuple(
        int(value) for value in first_point
    )
    second_xyz = tuple(
        int(value) for value in second_point
    )

    if first_xyz == second_xyz:
        raise RuntimeError(
            f"{case.case_id}: dual points are identical."
        )

    if not component[first_xyz] or not component[second_xyz]:
        raise RuntimeError(
            f"{case.case_id}: dual point lies outside target."
        )

    return DualPointCase(
        case_id=case.case_id,
        quartile=case.quartile,
        largest_component_volume_ml=(
            case.largest_component_volume_ml
        ),
        component_id=case.component_id,
        target_voxels=case.target_voxels,
        max_interior_depth_mm=first_depth,
        first_point_xyz=first_xyz,
        first_point_depth_mm=first_depth,
        second_point_xyz=second_xyz,
        second_point_depth_mm=second_depth,
        second_point_relative_depth=(
            second_depth / first_depth
        ),
        point_separation_mm=separation,
    )


def prepare_dual_point_protocol(
    *,
    block46a_protocol_path: Path,
    dataset_root: Path,
) -> tuple[DualPointCase, ...]:
    source_cases = load_adaptive_protocol(
        block46a_protocol_path
    )

    rows = tuple(
        derive_dual_point_case(
            case=case,
            segmentation_path=(
                dataset_root
                / case.case_id
                / "segmentation.nii.gz"
            ),
        )
        for case in source_cases
    )

    if len(rows) != 12:
        raise RuntimeError(
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
        raise RuntimeError(
            f"Dual-point protocol quartiles are unbalanced: {counts}."
        )

    return rows


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


def write_dual_point_protocol(
    *,
    rows: Iterable[DualPointCase],
    path: Path,
) -> None:
    rows = tuple(rows)

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 protocol rows, got {len(rows)}."
        )

    counts = {
        quartile: sum(
            row.quartile == quartile
            for row in rows
        )
        for quartile in QUARTILE_NAMES
    }

    payload = {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_tumor_dual_interior_point"
        ),
        "source_protocol": (
            "Block 46A 12-case tumor-size-stratified cohort"
        ),
        "case_count": 12,
        "quartile_counts": counts,
        "point_protocol": {
            "first_point": "deepest-interior tumor voxel",
            "second_point": (
                "farthest physical tumor voxel from first point "
                "among voxels with interior depth >= 50% of "
                "the case-specific maximum depth"
            ),
            "second_point_min_relative_depth": (
                SECOND_POINT_MIN_RELATIVE_DEPTH
            ),
            "point_labels": [3, 3],
            "label_prompt": None,
        },
        "oracle_simulated_clicks": True,
        "deployment_valid": False,
        "geometry_summary": {
            "second_point_relative_depth": _summary(
                row.second_point_relative_depth
                for row in rows
            ),
            "point_separation_mm": _summary(
                row.point_separation_mm
                for row in rows
            ),
        },
        "cases": [
            asdict(row)
            for row in rows
        ],
        "interpretation_boundary": (
            "Both positive points are derived from the ground-truth "
            "largest tumor component. The experiment tests whether "
            "spatially distributed interior interaction improves "
            "single-point robustness across tumor sizes; it is not "
            "deployment-valid localization."
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
