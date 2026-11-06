from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    AdaptiveDepthCase,
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickVariant,
    run_click_variant,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
)


@dataclass(frozen=True, slots=True)
class AdaptiveDepthComparison:
    case_id: str
    quartile: str
    largest_component_volume_ml: float
    max_interior_depth_mm: float
    adaptive_actual_depth_mm: float
    adaptive_depth_fraction_actual: float
    adaptive: TumorOracleSmokeResult
    deep: TumorOracleSmokeResult


def load_adaptive_protocol(
    path: Path,
) -> tuple[AdaptiveDepthCase, ...]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("case_count") != 12:
        raise ValueError(
            f"Expected frozen adaptive protocol N=12, got "
            f"{payload.get('case_count')}."
        )
    if payload.get("deployment_valid") is not False:
        raise ValueError(
            "Adaptive protocol must remain deployment-invalid."
        )
    if payload.get("oracle_simulated_clicks") is not True:
        raise ValueError(
            "Adaptive protocol must remain oracle-simulated."
        )

    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Adaptive protocol has no cases list.")

    rows = tuple(
        AdaptiveDepthCase(
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
            deep_point_xyz=tuple(
                int(value)
                for value in item["deep_point_xyz"]
            ),
            adaptive_target_depth_mm=float(
                item["adaptive_target_depth_mm"]
            ),
            adaptive_actual_depth_mm=float(
                item["adaptive_actual_depth_mm"]
            ),
            adaptive_point_xyz=tuple(
                int(value)
                for value in item["adaptive_point_xyz"]
            ),
        )
        for item in cases
    )

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 adaptive protocol cases, got {len(rows)}."
        )
    if len({row.case_id for row in rows}) != 12:
        raise ValueError("Adaptive protocol case IDs are not unique.")

    counts = {
        quartile: sum(
            row.quartile == quartile
            for row in rows
        )
        for quartile in QUARTILE_NAMES
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError(
            f"Adaptive protocol quartiles are unbalanced: {counts}."
        )

    return rows


def _base_prompt(
    row: AdaptiveDepthCase,
    *,
    point_xyz: tuple[int, int, int],
    interior_radius_mm: float,
) -> TumorOraclePrompt:
    return TumorOraclePrompt(
        case_id=row.case_id,
        component_id=row.component_id,
        point_xyz=point_xyz,
        target_voxels=row.target_voxels,
        target_volume_ml=row.largest_component_volume_ml,
        total_tumor_volume_ml=row.largest_component_volume_ml,
        interior_radius_mm=interior_radius_mm,
    )


def run_adaptive_depth_comparison(
    *,
    row: AdaptiveDepthCase,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
) -> AdaptiveDepthComparison:
    adaptive_prompt = _base_prompt(
        row,
        point_xyz=row.adaptive_point_xyz,
        interior_radius_mm=row.adaptive_actual_depth_mm,
    )
    deep_prompt = _base_prompt(
        row,
        point_xyz=row.deep_point_xyz,
        interior_radius_mm=row.max_interior_depth_mm,
    )

    adaptive = run_click_variant(
        base_prompt=adaptive_prompt,
        variant=ClickVariant(
            case_id=row.case_id,
            variant="adaptive_70pct",
            point_xyz=row.adaptive_point_xyz,
            interior_radius_mm=row.adaptive_actual_depth_mm,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=force,
    )

    deep = run_click_variant(
        base_prompt=deep_prompt,
        variant=ClickVariant(
            case_id=row.case_id,
            variant="deep",
            point_xyz=row.deep_point_xyz,
            interior_radius_mm=row.max_interior_depth_mm,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=force,
    )

    return AdaptiveDepthComparison(
        case_id=row.case_id,
        quartile=row.quartile,
        largest_component_volume_ml=(
            row.largest_component_volume_ml
        ),
        max_interior_depth_mm=row.max_interior_depth_mm,
        adaptive_actual_depth_mm=row.adaptive_actual_depth_mm,
        adaptive_depth_fraction_actual=(
            row.adaptive_actual_depth_mm
            / row.max_interior_depth_mm
        ),
        adaptive=adaptive,
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
    seed: int = 20260902,
    samples: int = 5000,
) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap requires a non-empty 1D array.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    means = values[indices].mean(axis=1)

    return {
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _group_payload(
    rows: tuple[AdaptiveDepthComparison, ...],
) -> dict[str, object]:
    adaptive_dice = np.asarray(
        [row.adaptive.interactive_target_dice for row in rows],
        dtype=float,
    )
    deep_dice = np.asarray(
        [row.deep.interactive_target_dice for row in rows],
        dtype=float,
    )
    dice_delta = adaptive_dice - deep_dice

    adaptive_coverage = np.asarray(
        [row.adaptive.tumor_coverage_gain for row in rows],
        dtype=float,
    )
    deep_coverage = np.asarray(
        [row.deep.tumor_coverage_gain for row in rows],
        dtype=float,
    )
    coverage_delta = adaptive_coverage - deep_coverage

    adaptive_hd95 = np.asarray(
        [row.adaptive.hec_hd95_reduction_mm for row in rows],
        dtype=float,
    )
    deep_hd95 = np.asarray(
        [row.deep.hec_hd95_reduction_mm for row in rows],
        dtype=float,
    )
    hd95_delta = adaptive_hd95 - deep_hd95

    adaptive_hec_dice = np.asarray(
        [row.adaptive.hec_dice_delta for row in rows],
        dtype=float,
    )
    deep_hec_dice = np.asarray(
        [row.deep.hec_dice_delta for row in rows],
        dtype=float,
    )
    hec_dice_delta = adaptive_hec_dice - deep_hec_dice

    adaptive_surface = np.asarray(
        [
            row.adaptive.hec_surface_dice_delta
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
    surface_delta = adaptive_surface - deep_surface

    return {
        "count": len(rows),
        "adaptive": {
            "tumor_dice": _summary(adaptive_dice),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(adaptive_dice >= 0.50)
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(adaptive_dice >= 0.70)
            ),
            "tumor_coverage_gain": _summary(
                adaptive_coverage
            ),
            "hec_hd95_reduction_mm": _summary(
                adaptive_hd95
            ),
            "hd95_improved_count": int(
                np.count_nonzero(adaptive_hd95 > 0.0)
            ),
            "hec_dice_delta": _summary(
                adaptive_hec_dice
            ),
            "hec_surface_dice_delta": _summary(
                adaptive_surface
            ),
        },
        "deep": {
            "tumor_dice": _summary(deep_dice),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(deep_dice >= 0.50)
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(deep_dice >= 0.70)
            ),
            "tumor_coverage_gain": _summary(
                deep_coverage
            ),
            "hec_hd95_reduction_mm": _summary(
                deep_hd95
            ),
            "hd95_improved_count": int(
                np.count_nonzero(deep_hd95 > 0.0)
            ),
            "hec_dice_delta": _summary(
                deep_hec_dice
            ),
            "hec_surface_dice_delta": _summary(
                deep_surface
            ),
        },
        "paired_adaptive_minus_deep": {
            "tumor_dice": {
                **_summary(dice_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(dice_delta)
                ),
            },
            "tumor_coverage_gain": {
                **_summary(coverage_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        coverage_delta,
                        seed=20260903,
                    )
                ),
            },
            "hec_hd95_reduction_mm": {
                **_summary(hd95_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        hd95_delta,
                        seed=20260904,
                    )
                ),
            },
            "hec_dice_delta": {
                **_summary(hec_dice_delta),
            },
            "hec_surface_dice_delta": {
                **_summary(surface_delta),
            },
            "cases_within_0_05_tumor_dice": int(
                np.count_nonzero(
                    np.abs(dice_delta) <= (0.05 + 1e-12)
                )
            ),
            "cases_within_0_10_tumor_dice": int(
                np.count_nonzero(
                    np.abs(dice_delta) <= (0.10 + 1e-12)
                )
            ),
            "adaptive_tumor_dice_better_count": int(
                np.count_nonzero(dice_delta > 0.0)
            ),
        },
    }


def summarize_adaptive_depth_results(
    rows: Iterable[AdaptiveDepthComparison],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 adaptive comparisons, got {len(rows)}."
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
            f"Comparison quartiles are unbalanced: {counts}."
        )

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_tumor_adaptive_relative_depth_inference"
        ),
        "case_count": 12,
        "oracle_simulated_clicks": True,
        "deployment_valid": False,
        "adaptive_depth_fraction_nominal": 0.70,
        "actual_adaptive_depth_fraction": _summary(
            row.adaptive_depth_fraction_actual
            for row in rows
        ),
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
        "case_rows": [
            {
                "case_id": row.case_id,
                "quartile": row.quartile,
                "largest_component_volume_ml": (
                    row.largest_component_volume_ml
                ),
                "max_interior_depth_mm": (
                    row.max_interior_depth_mm
                ),
                "adaptive_actual_depth_mm": (
                    row.adaptive_actual_depth_mm
                ),
                "adaptive_depth_fraction_actual": (
                    row.adaptive_depth_fraction_actual
                ),
                "adaptive": row.adaptive.to_dict(),
                "deep": row.deep.to_dict(),
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "Both clicks are derived from ground-truth tumor geometry. "
            "This paired experiment tests whether a pre-specified 70% "
            "relative-depth click preserves deep-oracle capability across "
            "tumor-size strata; it does not validate autonomous tumor "
            "localization."
        ),
    }


def write_adaptive_depth_outputs(
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

    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "quartile",
        "largest_component_volume_ml",
        "max_interior_depth_mm",
        "adaptive_actual_depth_mm",
        "adaptive_depth_fraction_actual",
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

            for variant in ("adaptive", "deep"):
                result = row[variant]
                assert isinstance(result, dict)

                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "quartile": row["quartile"],
                        "largest_component_volume_ml": row[
                            "largest_component_volume_ml"
                        ],
                        "max_interior_depth_mm": row[
                            "max_interior_depth_mm"
                        ],
                        "adaptive_actual_depth_mm": row[
                            "adaptive_actual_depth_mm"
                        ],
                        "adaptive_depth_fraction_actual": row[
                            "adaptive_depth_fraction_actual"
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
