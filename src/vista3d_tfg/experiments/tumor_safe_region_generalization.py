from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


DEPTH_THRESHOLDS_MM = (5.0, 10.0, 15.0)


@dataclass(frozen=True, slots=True)
class TumorSafeRegionCase:
    case_id: str
    tumor_positive: bool
    tumor_voxels: int
    tumor_volume_ml: float
    component_count: int
    largest_component_voxels: int
    largest_component_volume_ml: float
    max_interior_depth_mm: float | None
    safe_fraction_5mm: float | None
    safe_fraction_10mm: float | None
    safe_fraction_15mm: float | None
    safe_volume_5mm_ml: float | None
    safe_volume_10mm_ml: float | None
    safe_volume_15mm_ml: float | None


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


def read_expanded100_case_ids(path: Path) -> tuple[str, ...]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = tuple(csv.DictReader(handle))

    case_ids = tuple(row["case_id"] for row in rows)

    if len(case_ids) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(case_ids)}."
        )
    if len(set(case_ids)) != 100:
        raise ValueError("Duplicate expanded100 case IDs.")

    return case_ids


def evaluate_case_safe_region(
    *,
    case_id: str,
    segmentation_path: Path,
) -> TumorSafeRegionCase:
    nii = nib.load(str(segmentation_path))
    gt = _integer_labels(np.asanyarray(nii.dataobj))
    spacing = tuple(
        float(value)
        for value in nii.header.get_zooms()[:3]
    )
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)

    tumor = gt == 2
    tumor_voxels = int(np.count_nonzero(tumor))
    tumor_volume_ml = tumor_voxels * voxel_volume_ml

    if tumor_voxels == 0:
        return TumorSafeRegionCase(
            case_id=case_id,
            tumor_positive=False,
            tumor_voxels=0,
            tumor_volume_ml=0.0,
            component_count=0,
            largest_component_voxels=0,
            largest_component_volume_ml=0.0,
            max_interior_depth_mm=None,
            safe_fraction_5mm=None,
            safe_fraction_10mm=None,
            safe_fraction_15mm=None,
            safe_volume_5mm_ml=None,
            safe_volume_10mm_ml=None,
            safe_volume_15mm_ml=None,
        )

    structure = ndimage.generate_binary_structure(3, 1)
    labels, component_count = ndimage.label(
        tumor,
        structure=structure,
    )
    counts = np.bincount(labels.ravel())
    counts[0] = 0

    largest_component_id = int(np.argmax(counts))
    largest_component_voxels = int(
        counts[largest_component_id]
    )
    component = labels == largest_component_id

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing,
    )
    component_distances = distance[component]
    max_depth = float(component_distances.max())

    fractions: dict[float, float] = {}
    volumes: dict[float, float] = {}

    previous_count = largest_component_voxels + 1

    for depth_mm in DEPTH_THRESHOLDS_MM:
        safe_voxels = int(
            np.count_nonzero(
                component_distances >= (depth_mm - 1e-12)
            )
        )

        if safe_voxels > previous_count:
            raise RuntimeError(
                f"{case_id}: safe-region volume is not monotonic."
            )
        previous_count = safe_voxels

        fractions[depth_mm] = (
            safe_voxels / largest_component_voxels
        )
        volumes[depth_mm] = (
            safe_voxels * voxel_volume_ml
        )

    return TumorSafeRegionCase(
        case_id=case_id,
        tumor_positive=True,
        tumor_voxels=tumor_voxels,
        tumor_volume_ml=tumor_volume_ml,
        component_count=int(component_count),
        largest_component_voxels=largest_component_voxels,
        largest_component_volume_ml=(
            largest_component_voxels * voxel_volume_ml
        ),
        max_interior_depth_mm=max_depth,
        safe_fraction_5mm=fractions[5.0],
        safe_fraction_10mm=fractions[10.0],
        safe_fraction_15mm=fractions[15.0],
        safe_volume_5mm_ml=volumes[5.0],
        safe_volume_10mm_ml=volumes[10.0],
        safe_volume_15mm_ml=volumes[15.0],
    )


def build_expanded100_safe_region_rows(
    *,
    cohort_csv: Path,
    dataset_root: Path,
) -> tuple[TumorSafeRegionCase, ...]:
    case_ids = read_expanded100_case_ids(cohort_csv)

    return tuple(
        evaluate_case_safe_region(
            case_id=case_id,
            segmentation_path=(
                dataset_root
                / case_id
                / "segmentation.nii.gz"
            ),
        )
        for case_id in case_ids
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


def _quantile_groups(
    rows: tuple[TumorSafeRegionCase, ...],
) -> dict[str, tuple[TumorSafeRegionCase, ...]]:
    volumes = np.asarray(
        [row.largest_component_volume_ml for row in rows],
        dtype=float,
    )

    q1, q2, q3 = np.quantile(
        volumes,
        [0.25, 0.50, 0.75],
    )

    return {
        "smallest_quartile": tuple(
            row
            for row in rows
            if row.largest_component_volume_ml <= q1
        ),
        "lower_middle": tuple(
            row
            for row in rows
            if q1 < row.largest_component_volume_ml <= q2
        ),
        "upper_middle": tuple(
            row
            for row in rows
            if q2 < row.largest_component_volume_ml <= q3
        ),
        "largest_quartile": tuple(
            row
            for row in rows
            if row.largest_component_volume_ml > q3
        ),
    }


def _depth_payload(
    rows: tuple[TumorSafeRegionCase, ...],
    *,
    fraction_attr: str,
    volume_attr: str,
    depth_attr_threshold: float,
) -> dict[str, object]:
    fractions = np.asarray(
        [float(getattr(row, fraction_attr)) for row in rows],
        dtype=float,
    )
    volumes = np.asarray(
        [float(getattr(row, volume_attr)) for row in rows],
        dtype=float,
    )
    max_depth = np.asarray(
        [float(row.max_interior_depth_mm) for row in rows],
        dtype=float,
    )

    return {
        "cases_with_nonempty_region": int(
            np.count_nonzero(
                max_depth >= (depth_attr_threshold - 1e-12)
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


def summarize_expanded100_safe_regions(
    rows: Iterable[TumorSafeRegionCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(rows)}."
        )

    positive = tuple(
        row for row in rows if row.tumor_positive
    )
    negative = tuple(
        row for row in rows if not row.tumor_positive
    )

    if not positive:
        raise ValueError("Expanded100 contains no tumor-positive cases.")

    depths = {
        "5mm": _depth_payload(
            positive,
            fraction_attr="safe_fraction_5mm",
            volume_attr="safe_volume_5mm_ml",
            depth_attr_threshold=5.0,
        ),
        "10mm": _depth_payload(
            positive,
            fraction_attr="safe_fraction_10mm",
            volume_attr="safe_volume_10mm_ml",
            depth_attr_threshold=10.0,
        ),
        "15mm": _depth_payload(
            positive,
            fraction_attr="safe_fraction_15mm",
            volume_attr="safe_volume_15mm_ml",
            depth_attr_threshold=15.0,
        ),
    }

    quartiles = _quantile_groups(positive)
    quartile_payload: dict[str, object] = {}

    for group_name, group_rows in quartiles.items():
        if not group_rows:
            continue

        quartile_payload[group_name] = {
            "count": len(group_rows),
            "largest_component_volume_ml": _summary(
                row.largest_component_volume_ml
                for row in group_rows
            ),
            "max_interior_depth_mm": _summary(
                float(row.max_interior_depth_mm)
                for row in group_rows
            ),
            "15mm": _depth_payload(
                group_rows,
                fraction_attr="safe_fraction_15mm",
                volume_attr="safe_volume_15mm_ml",
                depth_attr_threshold=15.0,
            ),
        }

    positive_sorted = tuple(
        sorted(
            positive,
            key=lambda row: (
                row.largest_component_volume_ml,
                row.case_id,
            ),
        )
    )

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_tumor_safe_region_generalization"
        ),
        "case_count": 100,
        "tumor_positive_count": len(positive),
        "tumor_negative_count": len(negative),
        "largest_component_volume_ml": _summary(
            row.largest_component_volume_ml
            for row in positive
        ),
        "max_interior_depth_mm": _summary(
            float(row.max_interior_depth_mm)
            for row in positive
        ),
        "depths": depths,
        "volume_quartiles": quartile_payload,
        "smallest_tumor_positive_cases": [
            asdict(row)
            for row in positive_sorted[:10]
        ],
        "case_rows": [
            asdict(row)
            for row in rows
        ],
        "interpretation_boundary": (
            "This audit generalizes safe-region geometry from the nine "
            "severe GT-to-prediction outliers to all tumor-positive cases "
            "in the frozen expanded100 cohort. Geometry is still derived "
            "from ground truth, so it evaluates interaction feasibility "
            "rather than autonomous localization. The cohort is not a "
            "population-level clinical sample."
        ),
    }


def write_summary(
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
