from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import csv
import json

import numpy as np

from vista3d_tfg.analysis.anatomy import analyze_kits23_anatomy_nifti


@dataclass(frozen=True)
class FailureCase:
    case_id: str
    kidney_and_mass_dice: float
    kidney_and_mass_surface_dice: float
    kidney_and_mass_hd95_mm: float
    kidney_core_dice: float
    cyst_dice: float
    cyst_reference_volume_ml: float
    cyst_prediction_volume_ml: float
    gt_tumor_volume_ml: float
    left_renal_volume_error_ml: float | None
    right_renal_volume_error_ml: float | None
    ground_truth_laterality_resolved: bool
    kidney_and_mass_components: int
    tumor_components: int
    cyst_components: int
    anatomy_qc_flags: tuple[str, ...]

    @property
    def bilateral_renal_volume_error_ml(self) -> float | None:
        if (
            self.left_renal_volume_error_ml is None
            or self.right_renal_volume_error_ml is None
        ):
            return None
        return (
            self.left_renal_volume_error_ml
            + self.right_renal_volume_error_ml
        )


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def _bool_value(value: str | None, *, default: bool = True) -> bool:
    if value is None or not value.strip():
        return default

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_failure_cases(
    *,
    cohort_csv: Path,
    dataset_root: Path,
) -> tuple[FailureCase, ...]:
    with cohort_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 cases.csv to contain 100 rows, got {len(rows)}."
        )

    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate case IDs detected in expanded100 cohort.")

    output: list[FailureCase] = []

    for row in rows:
        case_id = row["case_id"]
        segmentation_path = (
            dataset_root
            / case_id
            / "segmentation.nii.gz"
        )

        if not segmentation_path.is_file():
            raise FileNotFoundError(segmentation_path)

        anatomy = analyze_kits23_anatomy_nifti(
            case_id=case_id,
            segmentation_path=segmentation_path,
        )

        output.append(
            FailureCase(
                case_id=case_id,
                kidney_and_mass_dice=float(
                    row["kidney_and_mass_dice"]
                ),
                kidney_and_mass_surface_dice=float(
                    row["kidney_and_mass_surface_dice"]
                ),
                kidney_and_mass_hd95_mm=float(
                    row["kidney_and_mass_hd95_mm"]
                ),
                kidney_core_dice=float(
                    row["kidney_core_dice"]
                ),
                cyst_dice=float(row["cyst_dice"]),
                cyst_reference_volume_ml=float(
                    row["cyst_reference_volume_ml"]
                ),
                cyst_prediction_volume_ml=float(
                    row["cyst_prediction_volume_ml"]
                ),
                gt_tumor_volume_ml=float(
                    row["gt_tumor_volume_ml"]
                ),
                left_renal_volume_error_ml=_optional_float(
                    row.get("left_renal_volume_error_ml")
                ),
                right_renal_volume_error_ml=_optional_float(
                    row.get("right_renal_volume_error_ml")
                ),
                ground_truth_laterality_resolved=_bool_value(
                    row.get("ground_truth_laterality_resolved"),
                    default=True,
                ),
                kidney_and_mass_components=(
                    anatomy.kidney_and_mass_components
                ),
                tumor_components=len(anatomy.tumor_components),
                cyst_components=len(anatomy.cyst_components),
                anatomy_qc_flags=tuple(anatomy.qc_flags),
            )
        )

    return tuple(output)


def pearson_or_none(
    x: Iterable[float],
    y: Iterable[float],
) -> float | None:
    x_arr = np.asarray(tuple(x), dtype=float)
    y_arr = np.asarray(tuple(y), dtype=float)

    if x_arr.size != y_arr.size:
        raise ValueError("Correlation inputs must have equal length.")
    if x_arr.size < 2:
        return None
    if np.allclose(x_arr, x_arr[0]):
        return None
    if np.allclose(y_arr, y_arr[0]):
        return None

    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _mean(values: Iterable[float]) -> float:
    arr = np.asarray(tuple(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty group.")
    return float(arr.mean())


def _median(values: Iterable[float]) -> float:
    arr = np.asarray(tuple(values), dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty group.")
    return float(np.median(arr))


def _group_summary(rows: Iterable[FailureCase]) -> dict[str, object]:
    rows = tuple(rows)
    if not rows:
        return {
            "count": 0,
            "kidney_and_mass_dice_mean": None,
            "surface_dice_mean": None,
            "hd95_median_mm": None,
            "cyst_dice_mean": None,
        }

    return {
        "count": len(rows),
        "kidney_and_mass_dice_mean": _mean(
            row.kidney_and_mass_dice for row in rows
        ),
        "surface_dice_mean": _mean(
            row.kidney_and_mass_surface_dice for row in rows
        ),
        "hd95_median_mm": _median(
            row.kidney_and_mass_hd95_mm for row in rows
        ),
        "cyst_dice_mean": _mean(
            row.cyst_dice for row in rows
        ),
    }


def analyze_failure_stratification(
    rows: Iterable[FailureCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 100:
        raise ValueError(
            f"Expected N=100 for failure stratification, got {len(rows)}."
        )

    worst_hd95 = sorted(
        rows,
        key=lambda row: (
            -row.kidney_and_mass_hd95_mm,
            row.case_id,
        ),
    )[:15]

    worst_surface = sorted(
        rows,
        key=lambda row: (
            row.kidney_and_mass_surface_dice,
            row.case_id,
        ),
    )[:15]

    worst_dice = sorted(
        rows,
        key=lambda row: (
            row.kidney_and_mass_dice,
            row.case_id,
        ),
    )[:15]

    volume_eligible = tuple(
        row
        for row in rows
        if row.bilateral_renal_volume_error_ml is not None
    )

    largest_under = sorted(
        volume_eligible,
        key=lambda row: (
            row.bilateral_renal_volume_error_ml,
            row.case_id,
        ),
    )[:15]

    largest_over = sorted(
        volume_eligible,
        key=lambda row: (
            -row.bilateral_renal_volume_error_ml,
            row.case_id,
        ),
    )[:15]

    cyst_positive = tuple(
        row
        for row in rows
        if row.cyst_reference_volume_ml > 0.0
    )
    cyst_negative = tuple(
        row
        for row in rows
        if row.cyst_reference_volume_ml == 0.0
    )

    tumor_positive = tuple(
        row
        for row in rows
        if row.gt_tumor_volume_ml > 0.0
    )
    tumor_negative = tuple(
        row
        for row in rows
        if row.gt_tumor_volume_ml == 0.0
    )

    multi_component = tuple(
        row
        for row in rows
        if row.kidney_and_mass_components > 2
    )
    qc_flagged = tuple(
        row
        for row in rows
        if row.anatomy_qc_flags
    )

    correlations = {
        "hd95_vs_dice": pearson_or_none(
            (row.kidney_and_mass_hd95_mm for row in rows),
            (row.kidney_and_mass_dice for row in rows),
        ),
        "hd95_vs_surface_dice": pearson_or_none(
            (row.kidney_and_mass_hd95_mm for row in rows),
            (
                row.kidney_and_mass_surface_dice
                for row in rows
            ),
        ),
        "hd95_vs_tumor_volume_ml": pearson_or_none(
            (row.kidney_and_mass_hd95_mm for row in rows),
            (row.gt_tumor_volume_ml for row in rows),
        ),
        "hd95_vs_cyst_volume_ml": pearson_or_none(
            (row.kidney_and_mass_hd95_mm for row in rows),
            (row.cyst_reference_volume_ml for row in rows),
        ),
        "surface_dice_vs_cyst_volume_ml": pearson_or_none(
            (
                row.kidney_and_mass_surface_dice
                for row in rows
            ),
            (row.cyst_reference_volume_ml for row in rows),
        ),
        "cyst_dice_vs_reference_volume_ml": (
            pearson_or_none(
                (
                    row.cyst_reference_volume_ml
                    for row in cyst_positive
                ),
                (row.cyst_dice for row in cyst_positive),
            )
            if cyst_positive
            else None
        ),
    }

    hd95_values = np.asarray(
        [row.kidney_and_mass_hd95_mm for row in rows],
        dtype=float,
    )
    hd95_q1, hd95_q3 = np.percentile(hd95_values, [25.0, 75.0])
    hd95_iqr = hd95_q3 - hd95_q1
    hd95_outlier_threshold = float(
        hd95_q3 + 1.5 * hd95_iqr
    )

    hd95_outliers = tuple(
        row
        for row in rows
        if row.kidney_and_mass_hd95_mm > hd95_outlier_threshold
    )

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_failure_stratification"
        ),
        "case_count": len(rows),
        "cyst_positive_count": len(cyst_positive),
        "tumor_positive_count": len(tumor_positive),
        "laterality_unresolved_count": sum(
            not row.ground_truth_laterality_resolved
            for row in rows
        ),
        "hd95_outlier_rule": {
            "method": "Q3 + 1.5*IQR",
            "q1_mm": float(hd95_q1),
            "q3_mm": float(hd95_q3),
            "iqr_mm": float(hd95_iqr),
            "threshold_mm": hd95_outlier_threshold,
            "outlier_count": len(hd95_outliers),
            "outlier_case_ids": [
                row.case_id for row in hd95_outliers
            ],
        },
        "correlations": correlations,
        "subgroups": {
            "cyst_positive": _group_summary(cyst_positive),
            "cyst_negative": _group_summary(cyst_negative),
            "tumor_positive": _group_summary(tumor_positive),
            "tumor_negative": _group_summary(tumor_negative),
            "kidney_components_gt_2": _group_summary(
                multi_component
            ),
            "anatomy_qc_flagged": _group_summary(qc_flagged),
        },
        "rankings": {
            "worst_hd95": [asdict(row) for row in worst_hd95],
            "worst_surface_dice": [
                asdict(row) for row in worst_surface
            ],
            "worst_dice": [asdict(row) for row in worst_dice],
            "largest_undersegmentation": [
                asdict(row) for row in largest_under
            ],
            "largest_oversegmentation": [
                asdict(row) for row in largest_over
            ],
        },
        "interpretation_boundary": (
            "This is a descriptive post-hoc failure analysis of the "
            "frozen expanded100 cohort. Correlations are exploratory and "
            "must not be interpreted as causal or confirmatory tests."
        ),
    }


def write_ranked_cases_csv(
    rows: Iterable[FailureCase],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_id",
        "kidney_and_mass_dice",
        "kidney_and_mass_surface_dice",
        "kidney_and_mass_hd95_mm",
        "kidney_core_dice",
        "cyst_dice",
        "cyst_reference_volume_ml",
        "cyst_prediction_volume_ml",
        "gt_tumor_volume_ml",
        "bilateral_renal_volume_error_ml",
        "ground_truth_laterality_resolved",
        "kidney_and_mass_components",
        "tumor_components",
        "cyst_components",
        "anatomy_qc_flags",
    ]

    ranked = sorted(
        tuple(rows),
        key=lambda row: (
            -row.kidney_and_mass_hd95_mm,
            row.case_id,
        ),
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for row in ranked:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "kidney_and_mass_dice": (
                        row.kidney_and_mass_dice
                    ),
                    "kidney_and_mass_surface_dice": (
                        row.kidney_and_mass_surface_dice
                    ),
                    "kidney_and_mass_hd95_mm": (
                        row.kidney_and_mass_hd95_mm
                    ),
                    "kidney_core_dice": row.kidney_core_dice,
                    "cyst_dice": row.cyst_dice,
                    "cyst_reference_volume_ml": (
                        row.cyst_reference_volume_ml
                    ),
                    "cyst_prediction_volume_ml": (
                        row.cyst_prediction_volume_ml
                    ),
                    "gt_tumor_volume_ml": (
                        row.gt_tumor_volume_ml
                    ),
                    "bilateral_renal_volume_error_ml": (
                        row.bilateral_renal_volume_error_ml
                    ),
                    "ground_truth_laterality_resolved": (
                        row.ground_truth_laterality_resolved
                    ),
                    "kidney_and_mass_components": (
                        row.kidney_and_mass_components
                    ),
                    "tumor_components": row.tumor_components,
                    "cyst_components": row.cyst_components,
                    "anatomy_qc_flags": ";".join(
                        row.anatomy_qc_flags
                    ),
                }
            )


def write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
