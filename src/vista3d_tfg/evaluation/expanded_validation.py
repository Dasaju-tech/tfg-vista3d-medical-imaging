from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


BOOTSTRAP_SEED = 20260901
BOOTSTRAP_SAMPLES = 5000


@dataclass(frozen=True)
class CaseRow:
    case_id: str
    kidney_and_mass_dice: float
    kidney_and_mass_surface_dice: float
    kidney_and_mass_hd95_mm: float
    kidney_core_dice: float
    cyst_dice: float
    cyst_reference_volume_ml: float
    inference_wall_time_seconds: float | None
    ground_truth_laterality_resolved: bool


@dataclass(frozen=True)
class BootstrapCI:
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class MetricEstimate:
    count: int
    mean: float
    std: float | None
    median: float
    bootstrap_mean_95ci: BootstrapCI


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _bool_or_default(value: str | None, default: bool = True) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def read_case_rows(path: Path) -> tuple[CaseRow, ...]:
    rows: list[CaseRow] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            "case_id",
            "kidney_and_mass_dice",
            "kidney_and_mass_surface_dice",
            "kidney_and_mass_hd95_mm",
            "kidney_core_dice",
            "cyst_dice",
            "cyst_reference_volume_ml",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        for raw in reader:
            rows.append(
                CaseRow(
                    case_id=raw["case_id"],
                    kidney_and_mass_dice=float(
                        raw["kidney_and_mass_dice"]
                    ),
                    kidney_and_mass_surface_dice=float(
                        raw["kidney_and_mass_surface_dice"]
                    ),
                    kidney_and_mass_hd95_mm=float(
                        raw["kidney_and_mass_hd95_mm"]
                    ),
                    kidney_core_dice=float(raw["kidney_core_dice"]),
                    cyst_dice=float(raw["cyst_dice"]),
                    cyst_reference_volume_ml=float(
                        raw["cyst_reference_volume_ml"]
                    ),
                    inference_wall_time_seconds=_float_or_none(
                        raw.get("inference_wall_time_seconds")
                    ),
                    ground_truth_laterality_resolved=_bool_or_default(
                        raw.get("ground_truth_laterality_resolved"),
                        default=True,
                    ),
                )
            )

    if not rows:
        raise ValueError(f"No cohort rows found in {path}.")

    ids = [row.case_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case IDs found in {path}.")

    return tuple(rows)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> BootstrapCI:
    array = np.asarray(tuple(values), dtype=float)

    if array.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    if samples <= 0:
        raise ValueError("samples must be positive.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        array.size,
        size=(samples, array.size),
    )
    means = array[indices].mean(axis=1)

    lower, upper = np.percentile(means, [2.5, 97.5])
    return BootstrapCI(
        lower=float(lower),
        upper=float(upper),
    )


def estimate_metric(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> MetricEstimate:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        raise ValueError("Cannot estimate an empty metric.")

    return MetricEstimate(
        count=int(array.size),
        mean=float(array.mean()),
        std=(
            float(array.std(ddof=1))
            if array.size > 1
            else None
        ),
        median=float(np.median(array)),
        bootstrap_mean_95ci=bootstrap_mean_ci(
            array,
            seed=seed,
        ),
    )


def _metric_payload(
    pilot_values: Iterable[float],
    expanded_values: Iterable[float],
    additional_values: Iterable[float],
    *,
    seed: int,
) -> dict[str, object]:
    pilot = estimate_metric(pilot_values, seed=seed)
    expanded = estimate_metric(expanded_values, seed=seed)
    additional = estimate_metric(additional_values, seed=seed)

    pilot_width = pilot.bootstrap_mean_95ci.width
    expanded_width = expanded.bootstrap_mean_95ci.width

    width_reduction_fraction = (
        1.0 - expanded_width / pilot_width
        if pilot_width > 0.0
        else None
    )

    return {
        "pilot30": asdict(pilot),
        "expanded100": asdict(expanded),
        "additional70": asdict(additional),
        "expanded_minus_pilot_mean": expanded.mean - pilot.mean,
        "additional_minus_pilot_mean": additional.mean - pilot.mean,
        "expanded_ci_width_reduction_fraction": (
            width_reduction_fraction
        ),
        "pilot_mean_inside_expanded_95ci": (
            expanded.bootstrap_mean_95ci.lower
            <= pilot.mean
            <= expanded.bootstrap_mean_95ci.upper
        ),
    }


def analyze_nested_cohorts(
    pilot_rows: Iterable[CaseRow],
    expanded_rows: Iterable[CaseRow],
    *,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    pilot = tuple(pilot_rows)
    expanded = tuple(expanded_rows)

    pilot_ids = {row.case_id for row in pilot}
    expanded_ids = {row.case_id for row in expanded}

    if len(pilot) != 30:
        raise ValueError(
            f"Expected frozen pilot30 N=30, got {len(pilot)}."
        )
    if len(expanded) != 100:
        raise ValueError(
            f"Expected frozen expanded100 N=100, got {len(expanded)}."
        )
    if not pilot_ids.issubset(expanded_ids):
        missing = sorted(pilot_ids - expanded_ids)
        raise ValueError(
            "pilot30 is not a subset of expanded100; missing "
            + ", ".join(missing)
        )

    additional = tuple(
        row for row in expanded
        if row.case_id not in pilot_ids
    )
    if len(additional) != 70:
        raise ValueError(
            f"Expected 70 additional cases, got {len(additional)}."
        )

    metrics = {
        "kidney_and_mass_dice": _metric_payload(
            (r.kidney_and_mass_dice for r in pilot),
            (r.kidney_and_mass_dice for r in expanded),
            (r.kidney_and_mass_dice for r in additional),
            seed=seed,
        ),
        "surface_dice": _metric_payload(
            (r.kidney_and_mass_surface_dice for r in pilot),
            (r.kidney_and_mass_surface_dice for r in expanded),
            (r.kidney_and_mass_surface_dice for r in additional),
            seed=seed + 1,
        ),
        "hd95_mm": _metric_payload(
            (r.kidney_and_mass_hd95_mm for r in pilot),
            (r.kidney_and_mass_hd95_mm for r in expanded),
            (r.kidney_and_mass_hd95_mm for r in additional),
            seed=seed + 2,
        ),
        "kidney_core_dice": _metric_payload(
            (r.kidney_core_dice for r in pilot),
            (r.kidney_core_dice for r in expanded),
            (r.kidney_core_dice for r in additional),
            seed=seed + 3,
        ),
    }

    pilot_cyst = tuple(
        r.cyst_dice for r in pilot
        if r.cyst_reference_volume_ml > 0.0
    )
    expanded_cyst = tuple(
        r.cyst_dice for r in expanded
        if r.cyst_reference_volume_ml > 0.0
    )
    additional_cyst = tuple(
        r.cyst_dice for r in additional
        if r.cyst_reference_volume_ml > 0.0
    )

    if pilot_cyst and expanded_cyst and additional_cyst:
        metrics["cyst_positive_dice"] = _metric_payload(
            pilot_cyst,
            expanded_cyst,
            additional_cyst,
            seed=seed + 4,
        )

    worst_surface = sorted(
        expanded,
        key=lambda row: (
            row.kidney_and_mass_surface_dice,
            row.case_id,
        ),
    )[:10]
    worst_hd95 = sorted(
        expanded,
        key=lambda row: (
            -row.kidney_and_mass_hd95_mm,
            row.case_id,
        ),
    )[:10]

    runtime = tuple(
        r.inference_wall_time_seconds
        for r in expanded
        if r.inference_wall_time_seconds is not None
    )

    return {
        "schema_version": "1.0",
        "experiment": "nv_segment_ct_expanded100_statistical_validation",
        "bootstrap": {
            "seed": seed,
            "samples": BOOTSTRAP_SAMPLES,
            "method": "percentile bootstrap of the mean",
        },
        "design": {
            "pilot30_case_count": len(pilot),
            "expanded100_case_count": len(expanded),
            "additional_case_count": len(additional),
            "nested_cohorts": True,
            "inferential_boundary": (
                "Pilot30 is a strict subset of expanded100. "
                "Reported deltas are descriptive stability diagnostics, "
                "not independent-sample hypothesis tests."
            ),
        },
        "cyst_positive_counts": {
            "pilot30": len(pilot_cyst),
            "expanded100": len(expanded_cyst),
            "additional70": len(additional_cyst),
        },
        "laterality": {
            "expanded100_resolved": sum(
                r.ground_truth_laterality_resolved
                for r in expanded
            ),
            "expanded100_unresolved": sum(
                not r.ground_truth_laterality_resolved
                for r in expanded
            ),
        },
        "metrics": metrics,
        "expanded100_worst_surface_dice": [
            {
                "case_id": r.case_id,
                "surface_dice": r.kidney_and_mass_surface_dice,
            }
            for r in worst_surface
        ],
        "expanded100_worst_hd95": [
            {
                "case_id": r.case_id,
                "hd95_mm": r.kidney_and_mass_hd95_mm,
            }
            for r in worst_hd95
        ],
        "expanded100_runtime_seconds": (
            asdict(estimate_metric(runtime, seed=seed + 5))
            if runtime
            else None
        ),
    }
