from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from vista3d_tfg.evaluation.cohort import CaseCohortRow


PILOT30_SEED = 20260828
PILOT30_SIZE = 30
SMOKE_CASES = (0, 1, 2)
PILOT30_SELECTION_REVISION = 2

# Revision 1 incorrectly assumed that 489 KiTS23 cases implied contiguous
# numeric identifiers 0..488. Official KiTS23 metadata contains gaps in the
# numeric IDs. Revision 2 was frozen before any pilot30 model inference:
# all already acquired valid candidates are retained, while the four
# nonexistent candidates 324, 345, 396, and 398 are replaced by confirmed
# official case IDs 474, 479, 481, and 482. Replacement is based only on
# dataset availability, never on model results or segmentation difficulty.
PILOT30_INVALID_V1_CASES = (324, 345, 396, 398)
PILOT30_REPLACEMENT_CASES = (474, 479, 481, 482)

PILOT30_CASES: tuple[int, ...] = (
    0, 1, 2, 17, 47, 63, 109, 116, 142, 143,
    171, 190, 193, 200, 237, 255, 275, 293, 297, 299,
    416, 418, 463, 471, 473, 474, 479, 480, 481, 482,
)


def deterministic_pilot30_cases() -> tuple[int, ...]:
    """Return the frozen revision-2 engineering pilot cohort."""

    return PILOT30_CASES


@dataclass(frozen=True, slots=True)
class PilotCasePaths:
    case_id: str
    data_directory: Path
    imaging_path: Path
    segmentation_path: Path
    prediction_path: Path
    inference_json: Path
    evaluation_json: Path
    comparison_qc_json: Path


def pilot_case_paths(
    project_root: Path,
    case_number: int,
) -> PilotCasePaths:
    case_id = f"case_{case_number:05d}"
    root = project_root.resolve()

    data_directory = (
        root
        / "data"
        / "raw"
        / "kits23"
        / case_id
    )
    prediction_root = (
        root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
        / case_id
    )

    return PilotCasePaths(
        case_id=case_id,
        data_directory=data_directory,
        imaging_path=data_directory / "imaging.nii.gz",
        segmentation_path=data_directory / "segmentation.nii.gz",
        prediction_path=prediction_root / "prediction.nii.gz",
        inference_json=prediction_root / "inference.json",
        evaluation_json=(
            root
            / "outputs"
            / "evaluations"
            / "nv_segment_ct"
            / case_id
            / "supported_renal_metrics.json"
        ),
        comparison_qc_json=(
            root
            / "outputs"
            / "qc"
            / "nv_segment_ct_comparison"
            / case_id
            / "comparison_qc.json"
        ),
    )


def select_worst_surface_dice_cases(
    rows: Iterable[CaseCohortRow],
    *,
    count: int,
) -> tuple[str, ...]:
    if count < 0:
        raise ValueError("count must be zero or greater.")

    rows = tuple(rows)

    ordered = sorted(
        rows,
        key=lambda row: (
            row.kidney_and_mass_surface_dice,
            row.case_id,
        ),
    )

    return tuple(
        row.case_id
        for row in ordered[:count]
    )


def selection_manifest_payload(
    case_numbers: Iterable[int] = PILOT30_CASES,
) -> dict[str, object]:
    case_numbers = tuple(int(value) for value in case_numbers)

    return {
        "schema_version": "1.0",
        "dataset": "KiTS23",
        "experiment": "nv_segment_ct_pilot30",
        "selection": {
            "revision": PILOT30_SELECTION_REVISION,
            "method": (
                "revision-2 frozen pilot: retain all valid revision-1 "
                "candidates and replace nonexistent numeric IDs before any "
                "pilot30 inference"
            ),
            "seed": PILOT30_SEED,
            "requested_case_count": len(case_numbers),
            "invalid_revision_1_case_numbers": list(
                PILOT30_INVALID_V1_CASES
            ),
            "replacement_case_numbers": list(
                PILOT30_REPLACEMENT_CASES
            ),
            "case_numbers": list(case_numbers),
            "case_ids": [
                f"case_{number:05d}"
                for number in case_numbers
            ],
        },
        "selection_correction_reason": (
            "KiTS23 contains 489 cases but its numeric case identifiers are "
            "not a contiguous 0..488 range. Revision 1 sampled numeric IDs "
            "instead of official case IDs. Revision 2 was frozen before "
            "pilot30 inference and changes availability only, not outcomes."
        ),
        "interpretation_boundary": (
            "Engineering pilot cohort for pipeline scaling; not a "
            "population-representative or statistically powered sample."
        ),
    }


def write_selection_manifest(
    output_path: Path,
    case_numbers: Iterable[int] = PILOT30_CASES,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            selection_manifest_payload(case_numbers),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def write_run_manifest(
    *,
    output_path: Path,
    requested_case_ids: Iterable[str],
    successful_case_ids: Iterable[str],
    failed_cases: dict[str, str],
    worst_qc_case_ids: Iterable[str],
    experiment: str = "nv_segment_ct_pilot30",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": experiment,
                "requested_case_ids": list(requested_case_ids),
                "successful_case_ids": list(successful_case_ids),
                "failed_cases": dict(sorted(failed_cases.items())),
                "worst_surface_dice_qc_case_ids": list(
                    worst_qc_case_ids
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
