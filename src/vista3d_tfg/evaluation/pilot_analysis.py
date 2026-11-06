from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from statistics import mean, median
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vista3d_tfg.evaluation.cohort import CaseCohortRow


BOOTSTRAP_SEED = 20260828
BOOTSTRAP_SAMPLES = 5000


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class GroupComparison:
    group_a_name: str
    group_b_name: str
    group_a_count: int
    group_b_count: int
    group_a_mean: float
    group_b_mean: float
    mean_difference_a_minus_b: float


@dataclass(frozen=True, slots=True)
class Pilot30StatisticalAnalysis:
    case_count: int
    cyst_positive_count: int
    cyst_negative_count: int

    kidney_and_mass_dice_mean: float
    kidney_and_mass_dice_median: float
    kidney_and_mass_dice_bootstrap_95ci: ConfidenceInterval

    surface_dice_mean: float
    surface_dice_median: float
    surface_dice_bootstrap_95ci: ConfidenceInterval

    hd95_mean_mm: float
    hd95_median_mm: float
    hd95_bootstrap_95ci_mm: ConfidenceInterval

    cyst_positive_dice_mean: float | None
    cyst_positive_dice_median: float | None
    cyst_positive_dice_bootstrap_95ci: ConfidenceInterval | None

    kidney_dice_cyst_status_comparison: GroupComparison | None
    surface_dice_cyst_status_comparison: GroupComparison | None

    worst_surface_dice_case_ids: tuple[str, ...]
    worst_hd95_case_ids: tuple[str, ...]
    largest_undersegmentation_case_ids: tuple[str, ...]

    correlation_surface_dice_hd95: float | None
    correlation_cyst_gt_vs_prediction_volume: float | None

    interpretation_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    values = np.asarray(tuple(float(value) for value in values), dtype=float)

    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence.")

    if samples <= 0:
        raise ValueError("samples must be positive.")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    sample_means = values[indices].mean(axis=1)

    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))

    return ConfidenceInterval(lower=lower, upper=upper)


def pearson_correlation(
    x: Iterable[float],
    y: Iterable[float],
) -> float | None:
    x_values = np.asarray(tuple(float(value) for value in x), dtype=float)
    y_values = np.asarray(tuple(float(value) for value in y), dtype=float)

    if x_values.size != y_values.size:
        raise ValueError("Correlation inputs must have equal length.")

    if x_values.size < 2:
        return None

    if np.isclose(np.std(x_values), 0.0) or np.isclose(
        np.std(y_values),
        0.0,
    ):
        return None

    return float(np.corrcoef(x_values, y_values)[0, 1])


def compare_groups(
    *,
    group_a_name: str,
    group_a_values: Iterable[float],
    group_b_name: str,
    group_b_values: Iterable[float],
) -> GroupComparison | None:
    a = tuple(float(value) for value in group_a_values)
    b = tuple(float(value) for value in group_b_values)

    if not a or not b:
        return None

    a_mean = float(mean(a))
    b_mean = float(mean(b))

    return GroupComparison(
        group_a_name=group_a_name,
        group_b_name=group_b_name,
        group_a_count=len(a),
        group_b_count=len(b),
        group_a_mean=a_mean,
        group_b_mean=b_mean,
        mean_difference_a_minus_b=a_mean - b_mean,
    )


def analyze_pilot30(
    rows: Iterable[CaseCohortRow],
) -> Pilot30StatisticalAnalysis:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one case is required.")

    cyst_positive = tuple(
        row for row in rows if row.cyst_reference_volume_ml > 0.0
    )
    cyst_negative = tuple(
        row for row in rows if row.cyst_reference_volume_ml == 0.0
    )

    kidney_values = tuple(
        row.kidney_and_mass_dice for row in rows
    )
    surface_values = tuple(
        row.kidney_and_mass_surface_dice for row in rows
    )
    hd95_values = tuple(
        row.kidney_and_mass_hd95_mm for row in rows
    )

    cyst_positive_dice = tuple(
        row.cyst_dice for row in cyst_positive
    )

    worst_surface = tuple(
        row.case_id
        for row in sorted(
            rows,
            key=lambda row: (
                row.kidney_and_mass_surface_dice,
                row.case_id,
            ),
        )[:5]
    )

    worst_hd95 = tuple(
        row.case_id
        for row in sorted(
            rows,
            key=lambda row: (
                -row.kidney_and_mass_hd95_mm,
                row.case_id,
            ),
        )[:5]
    )

    laterality_rows = tuple(
        row
        for row in rows
        if row.left_renal_volume_error_ml is not None
        and row.right_renal_volume_error_ml is not None
    )

    largest_undersegmentation = tuple(
        row.case_id
        for row in sorted(
            laterality_rows,
            key=lambda row: (
                row.left_renal_volume_error_ml
                + row.right_renal_volume_error_ml,
                row.case_id,
            ),
        )[:5]
    )

    interpretation_flags: list[str] = []

    if cyst_positive:
        cyst_mean = float(mean(cyst_positive_dice))
        cyst_median = float(median(cyst_positive_dice))
        cyst_ci = bootstrap_mean_ci(cyst_positive_dice)

        if cyst_mean < 0.5:
            interpretation_flags.append(
                "low_cyst_positive_segmentation_overlap"
            )
    else:
        cyst_mean = None
        cyst_median = None
        cyst_ci = None

    kidney_cyst_status = compare_groups(
        group_a_name="cyst_positive",
        group_a_values=(
            row.kidney_and_mass_dice
            for row in cyst_positive
        ),
        group_b_name="cyst_negative",
        group_b_values=(
            row.kidney_and_mass_dice
            for row in cyst_negative
        ),
    )

    surface_cyst_status = compare_groups(
        group_a_name="cyst_positive",
        group_a_values=(
            row.kidney_and_mass_surface_dice
            for row in cyst_positive
        ),
        group_b_name="cyst_negative",
        group_b_values=(
            row.kidney_and_mass_surface_dice
            for row in cyst_negative
        ),
    )

    surface_hd95_corr = pearson_correlation(
        surface_values,
        hd95_values,
    )

    cyst_volume_corr = pearson_correlation(
        (
            row.cyst_reference_volume_ml
            for row in cyst_positive
        ),
        (
            row.cyst_prediction_volume_ml
            for row in cyst_positive
        ),
    )

    if surface_hd95_corr is not None and surface_hd95_corr < -0.5:
        interpretation_flags.append(
            "surface_dice_and_hd95_show_expected_inverse_association"
        )

    return Pilot30StatisticalAnalysis(
        case_count=len(rows),
        cyst_positive_count=len(cyst_positive),
        cyst_negative_count=len(cyst_negative),
        kidney_and_mass_dice_mean=float(mean(kidney_values)),
        kidney_and_mass_dice_median=float(median(kidney_values)),
        kidney_and_mass_dice_bootstrap_95ci=bootstrap_mean_ci(
            kidney_values
        ),
        surface_dice_mean=float(mean(surface_values)),
        surface_dice_median=float(median(surface_values)),
        surface_dice_bootstrap_95ci=bootstrap_mean_ci(
            surface_values
        ),
        hd95_mean_mm=float(mean(hd95_values)),
        hd95_median_mm=float(median(hd95_values)),
        hd95_bootstrap_95ci_mm=bootstrap_mean_ci(
            hd95_values
        ),
        cyst_positive_dice_mean=cyst_mean,
        cyst_positive_dice_median=cyst_median,
        cyst_positive_dice_bootstrap_95ci=cyst_ci,
        kidney_dice_cyst_status_comparison=kidney_cyst_status,
        surface_dice_cyst_status_comparison=surface_cyst_status,
        worst_surface_dice_case_ids=worst_surface,
        worst_hd95_case_ids=worst_hd95,
        largest_undersegmentation_case_ids=largest_undersegmentation,
        correlation_surface_dice_hd95=surface_hd95_corr,
        correlation_cyst_gt_vs_prediction_volume=cyst_volume_corr,
        interpretation_flags=tuple(interpretation_flags),
    )


def write_analysis_json(
    analysis: Pilot30StatisticalAnalysis,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "model": "NV-Segment-CT",
                "analysis": "pilot30_statistical_audit",
                "bootstrap": {
                    "seed": BOOTSTRAP_SEED,
                    "samples": BOOTSTRAP_SAMPLES,
                    "interval": "percentile_95",
                },
                "result": analysis.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_ranked_cases_csv(
    rows: Iterable[CaseCohortRow],
    output_path: Path,
) -> None:
    rows = tuple(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranked = sorted(
        rows,
        key=lambda row: (
            row.kidney_and_mass_surface_dice,
            row.case_id,
        ),
    )

    fieldnames = [
        "surface_dice_rank",
        "case_id",
        "kidney_and_mass_dice",
        "kidney_and_mass_surface_dice",
        "kidney_and_mass_hd95_mm",
        "cyst_reference_volume_ml",
        "cyst_prediction_volume_ml",
        "cyst_dice",
        "left_renal_volume_error_ml",
        "right_renal_volume_error_ml",
        "inference_wall_time_seconds",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "surface_dice_rank": rank,
                    "case_id": row.case_id,
                    "kidney_and_mass_dice": row.kidney_and_mass_dice,
                    "kidney_and_mass_surface_dice": (
                        row.kidney_and_mass_surface_dice
                    ),
                    "kidney_and_mass_hd95_mm": (
                        row.kidney_and_mass_hd95_mm
                    ),
                    "cyst_reference_volume_ml": (
                        row.cyst_reference_volume_ml
                    ),
                    "cyst_prediction_volume_ml": (
                        row.cyst_prediction_volume_ml
                    ),
                    "cyst_dice": row.cyst_dice,
                    "left_renal_volume_error_ml": (
                        row.left_renal_volume_error_ml
                    ),
                    "right_renal_volume_error_ml": (
                        row.right_renal_volume_error_ml
                    ),
                    "inference_wall_time_seconds": (
                        row.inference_wall_time_seconds
                    ),
                }
            )


def render_pilot_figures(
    rows: Iterable[CaseCohortRow],
    output_directory: Path,
) -> tuple[Path, ...]:
    rows = tuple(rows)

    if not rows:
        raise ValueError("At least one row is required.")

    output_directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    surface = np.asarray(
        [row.kidney_and_mass_surface_dice for row in rows],
        dtype=float,
    )
    hd95 = np.asarray(
        [row.kidney_and_mass_hd95_mm for row in rows],
        dtype=float,
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(surface, bins=min(10, len(rows)))
    axis.set_xlabel("Kidney-and-mass Surface Dice")
    axis.set_ylabel("Cases")
    axis.set_title("Pilot30 Surface Dice distribution")
    figure.tight_layout()

    path = output_directory / "surface_dice_distribution.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(surface, hd95)
    axis.set_xlabel("Kidney-and-mass Surface Dice")
    axis.set_ylabel("HD95 [mm]")
    axis.set_title("Surface Dice versus HD95")
    figure.tight_layout()

    path = output_directory / "surface_dice_vs_hd95.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    cyst_positive = [
        row
        for row in rows
        if row.cyst_reference_volume_ml > 0.0
    ]

    if cyst_positive:
        gt = np.asarray(
            [row.cyst_reference_volume_ml for row in cyst_positive],
            dtype=float,
        )
        pred = np.asarray(
            [row.cyst_prediction_volume_ml for row in cyst_positive],
            dtype=float,
        )

        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(gt, pred)

        lower = min(float(gt.min()), float(pred.min()))
        upper = max(float(gt.max()), float(pred.max()))
        axis.plot([lower, upper], [lower, upper])

        axis.set_xlabel("KiTS23 cyst volume [mL]")
        axis.set_ylabel("NV-Segment cyst volume [mL]")
        axis.set_title("Cyst-positive cases: predicted versus reference volume")
        figure.tight_layout()

        path = output_directory / "cyst_volume_reference_vs_prediction.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        outputs.append(path)

    return tuple(outputs)
