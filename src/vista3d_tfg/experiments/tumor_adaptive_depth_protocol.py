from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


ADAPTIVE_DEPTH_FRACTION = 0.70
CASES_PER_QUARTILE = 3
QUARTILE_NAMES = (
    "smallest_quartile",
    "lower_middle",
    "upper_middle",
    "largest_quartile",
)


@dataclass(frozen=True, slots=True)
class AdaptiveDepthCase:
    case_id: str
    quartile: str
    largest_component_volume_ml: float
    component_id: int
    target_voxels: int
    max_interior_depth_mm: float
    deep_point_xyz: tuple[int, int, int]
    adaptive_target_depth_mm: float
    adaptive_actual_depth_mm: float
    adaptive_point_xyz: tuple[int, int, int]


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


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


def read_tumor_positive_geometry(
    path: Path,
) -> tuple[dict[str, str], ...]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = tuple(csv.DictReader(handle))

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 geometry N=100, got {len(rows)}."
        )

    positive = tuple(
        row
        for row in rows
        if _parse_bool(row["tumor_positive"])
    )

    if not positive:
        raise ValueError("No tumor-positive cases found.")

    return positive


def assign_volume_quartiles(
    rows: Iterable[dict[str, str]],
) -> dict[str, tuple[dict[str, str], ...]]:
    rows = tuple(rows)
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                float(row["largest_component_volume_ml"]),
                row["case_id"],
            ),
        )
    )

    n = len(ordered)
    if n < 4:
        raise ValueError("Need at least four tumor-positive cases.")

    splits = np.array_split(
        np.arange(n),
        4,
    )

    return {
        name: tuple(
            ordered[int(index)]
            for index in indices
        )
        for name, indices in zip(
            QUARTILE_NAMES,
            splits,
        )
    }


def _representative_indices(count: int) -> tuple[int, int, int]:
    if count < CASES_PER_QUARTILE:
        raise ValueError(
            f"Need at least {CASES_PER_QUARTILE} cases per quartile."
        )

    positions = np.linspace(
        0.25,
        0.75,
        CASES_PER_QUARTILE,
    )
    indices = tuple(
        int(round(position * (count - 1)))
        for position in positions
    )

    if len(set(indices)) != CASES_PER_QUARTILE:
        indices = (
            0,
            count // 2,
            count - 1,
        )

    return indices


def select_stratified_cases(
    rows: Iterable[dict[str, str]],
) -> tuple[tuple[str, dict[str, str]], ...]:
    quartiles = assign_volume_quartiles(rows)

    selected = []

    for quartile_name in QUARTILE_NAMES:
        group = quartiles[quartile_name]
        indices = _representative_indices(len(group))

        for index in indices:
            selected.append(
                (quartile_name, group[index])
            )

    if len(selected) != 12:
        raise RuntimeError(
            f"Expected 12 selected cases, got {len(selected)}."
        )

    case_ids = [
        row["case_id"]
        for _, row in selected
    ]
    if len(set(case_ids)) != 12:
        raise RuntimeError("Selected case IDs are not unique.")

    return tuple(selected)


def derive_adaptive_case(
    *,
    case_id: str,
    quartile: str,
    largest_component_volume_ml: float,
    segmentation_path: Path,
    adaptive_fraction: float = ADAPTIVE_DEPTH_FRACTION,
) -> AdaptiveDepthCase:
    if not (0.0 < adaptive_fraction <= 1.0):
        raise ValueError(
            "adaptive_fraction must lie in (0, 1]."
        )

    nii = nib.load(str(segmentation_path))
    gt = _integer_labels(
        np.asanyarray(nii.dataobj)
    )
    spacing = tuple(
        float(value)
        for value in nii.header.get_zooms()[:3]
    )

    tumor = gt == 2
    structure = ndimage.generate_binary_structure(3, 1)
    labels, component_count = ndimage.label(
        tumor,
        structure=structure,
    )

    if component_count == 0:
        raise ValueError(
            f"{case_id}: selected case has no tumor."
        )

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    component_id = int(np.argmax(counts))
    component = labels == component_id
    target_voxels = int(counts[component_id])

    voxel_volume_ml = float(
        np.prod(spacing) / 1000.0
    )
    measured_volume_ml = (
        target_voxels * voxel_volume_ml
    )

    if not np.isclose(
        measured_volume_ml,
        largest_component_volume_ml,
        rtol=0.0,
        atol=max(1e-6, voxel_volume_ml),
    ):
        raise ValueError(
            f"{case_id}: largest-component volume changed "
            f"({measured_volume_ml:.6f} vs "
            f"{largest_component_volume_ml:.6f} mL)."
        )

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing,
    )

    component_distances = distance[component]
    coords = np.argwhere(component)

    deep_index = int(
        np.argmax(component_distances)
    )
    deep_point = tuple(
        int(value)
        for value in coords[deep_index]
    )
    max_depth = float(
        component_distances[deep_index]
    )

    target_depth = (
        adaptive_fraction * max_depth
    )
    adaptive_index = int(
        np.argmin(
            np.abs(
                component_distances
                - target_depth
            )
        )
    )
    adaptive_point = tuple(
        int(value)
        for value in coords[adaptive_index]
    )
    actual_depth = float(
        component_distances[adaptive_index]
    )

    return AdaptiveDepthCase(
        case_id=case_id,
        quartile=quartile,
        largest_component_volume_ml=measured_volume_ml,
        component_id=component_id,
        target_voxels=target_voxels,
        max_interior_depth_mm=max_depth,
        deep_point_xyz=deep_point,
        adaptive_target_depth_mm=target_depth,
        adaptive_actual_depth_mm=actual_depth,
        adaptive_point_xyz=adaptive_point,
    )


def prepare_adaptive_protocol(
    *,
    safe_region_cases_csv: Path,
    dataset_root: Path,
) -> tuple[AdaptiveDepthCase, ...]:
    positive = read_tumor_positive_geometry(
        safe_region_cases_csv
    )
    selected = select_stratified_cases(
        positive
    )

    rows = tuple(
        derive_adaptive_case(
            case_id=source["case_id"],
            quartile=quartile,
            largest_component_volume_ml=float(
                source[
                    "largest_component_volume_ml"
                ]
            ),
            segmentation_path=(
                dataset_root
                / source["case_id"]
                / "segmentation.nii.gz"
            ),
        )
        for quartile, source in selected
    )

    if len(rows) != 12:
        raise RuntimeError("Adaptive protocol must contain 12 cases.")

    return rows


def write_adaptive_protocol(
    *,
    rows: Iterable[AdaptiveDepthCase],
    path: Path,
) -> None:
    rows = tuple(rows)

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 protocol rows, got {len(rows)}."
        )

    counts = {
        name: sum(
            row.quartile == name
            for row in rows
        )
        for name in QUARTILE_NAMES
    }

    if any(
        value != CASES_PER_QUARTILE
        for value in counts.values()
    ):
        raise ValueError(
            "Adaptive protocol is not balanced across quartiles."
        )

    payload = {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_tumor_adaptive_relative_depth"
        ),
        "source_cohort": "expanded100",
        "case_count": len(rows),
        "cases_per_quartile": CASES_PER_QUARTILE,
        "quartile_counts": counts,
        "adaptive_depth_fraction": ADAPTIVE_DEPTH_FRACTION,
        "candidate_protocol": {
            "adaptive_point": (
                "tumor voxel closest to "
                "0.70 * maximum interior depth"
            ),
            "reference_point": (
                "deepest-interior tumor voxel"
            ),
            "point_label": 3,
            "label_prompt": None,
        },
        "oracle_simulated_clicks": True,
        "deployment_valid": False,
        "selection": (
            "three deterministic volume-ranked representatives "
            "per tumor-size quartile; no segmentation-performance "
            "metric is used for case selection"
        ),
        "cases": [
            asdict(row)
            for row in rows
        ],
        "interpretation_boundary": (
            "Both adaptive and deep points are derived from ground-truth "
            "tumor geometry. This protocol tests whether relative depth "
            "is a size-adaptive interaction principle across small and "
            "large tumors; it is not an autonomous locator."
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
