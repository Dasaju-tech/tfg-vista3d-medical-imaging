from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
    load_frozen_prompt,
)


@dataclass(frozen=True, slots=True)
class TumorOracleCohortSummary:
    case_count: int
    interactive_target_dice_mean: float
    interactive_target_dice_median: float
    interactive_full_tumor_dice_mean: float
    interactive_full_tumor_dice_median: float
    tumor_coverage_gain_mean: float
    tumor_coverage_gain_median: float
    hec_dice_delta_mean: float
    hec_dice_delta_median: float
    hec_surface_dice_delta_mean: float
    hec_surface_dice_delta_median: float
    hec_hd95_reduction_mean_mm: float
    hec_hd95_reduction_median_mm: float
    hd95_improved_case_count: int
    hd95_reduced_gt_5mm_count: int
    target_dice_ge_050_count: int
    target_dice_ge_080_count: int
    surface_dice_improved_case_count: int
    dice_improved_case_count: int


def load_all_frozen_prompts(
    protocol_path: Path,
) -> tuple[TumorOraclePrompt, ...]:
    payload = json.loads(
        protocol_path.read_text(encoding="utf-8")
    )
    prompts_payload = payload.get("prompts")

    if not isinstance(prompts_payload, list):
        raise ValueError("Protocol has no prompt list.")

    case_ids = [
        str(item["case_id"])
        for item in prompts_payload
    ]
    if len(case_ids) != 9:
        raise ValueError(
            f"Expected 9 frozen tumor prompts, got {len(case_ids)}."
        )
    if len(set(case_ids)) != 9:
        raise ValueError("Duplicate case IDs in frozen tumor protocol.")

    return tuple(
        load_frozen_prompt(
            protocol_path,
            case_id=case_id,
        )
        for case_id in case_ids
    )


def summarize_tumor_oracle_cohort(
    rows: Iterable[TumorOracleSmokeResult],
) -> TumorOracleCohortSummary:
    rows = tuple(rows)

    if len(rows) != 9:
        raise ValueError(
            f"Expected 9 tumor-oracle results, got {len(rows)}."
        )

    target_dice = np.asarray(
        [row.interactive_target_dice for row in rows],
        dtype=float,
    )
    full_tumor_dice = np.asarray(
        [row.interactive_full_tumor_dice for row in rows],
        dtype=float,
    )
    coverage_gain = np.asarray(
        [row.tumor_coverage_gain for row in rows],
        dtype=float,
    )
    hec_dice_delta = np.asarray(
        [row.hec_dice_delta for row in rows],
        dtype=float,
    )
    hec_surface_delta = np.asarray(
        [row.hec_surface_dice_delta for row in rows],
        dtype=float,
    )
    hec_hd95_reduction = np.asarray(
        [row.hec_hd95_reduction_mm for row in rows],
        dtype=float,
    )

    return TumorOracleCohortSummary(
        case_count=len(rows),
        interactive_target_dice_mean=float(target_dice.mean()),
        interactive_target_dice_median=float(
            np.median(target_dice)
        ),
        interactive_full_tumor_dice_mean=float(
            full_tumor_dice.mean()
        ),
        interactive_full_tumor_dice_median=float(
            np.median(full_tumor_dice)
        ),
        tumor_coverage_gain_mean=float(
            coverage_gain.mean()
        ),
        tumor_coverage_gain_median=float(
            np.median(coverage_gain)
        ),
        hec_dice_delta_mean=float(
            hec_dice_delta.mean()
        ),
        hec_dice_delta_median=float(
            np.median(hec_dice_delta)
        ),
        hec_surface_dice_delta_mean=float(
            hec_surface_delta.mean()
        ),
        hec_surface_dice_delta_median=float(
            np.median(hec_surface_delta)
        ),
        hec_hd95_reduction_mean_mm=float(
            hec_hd95_reduction.mean()
        ),
        hec_hd95_reduction_median_mm=float(
            np.median(hec_hd95_reduction)
        ),
        hd95_improved_case_count=int(
            np.count_nonzero(
                hec_hd95_reduction > 0.0
            )
        ),
        hd95_reduced_gt_5mm_count=int(
            np.count_nonzero(
                hec_hd95_reduction > 5.0
            )
        ),
        target_dice_ge_050_count=int(
            np.count_nonzero(
                target_dice >= 0.50
            )
        ),
        target_dice_ge_080_count=int(
            np.count_nonzero(
                target_dice >= 0.80
            )
        ),
        surface_dice_improved_case_count=int(
            np.count_nonzero(
                hec_surface_delta > 0.0
            )
        ),
        dice_improved_case_count=int(
            np.count_nonzero(
                hec_dice_delta > 0.0
            )
        ),
    )


def write_cohort_outputs(
    *,
    rows: Iterable[TumorOracleSmokeResult],
    summary: TumorOracleCohortSummary,
    output_root: Path,
) -> None:
    rows = tuple(rows)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": (
                    "nv_segment_ct_tumor_oracle_special_point_cohort"
                ),
                "oracle": True,
                "deployment_valid": False,
                "summary": asdict(summary),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fields = [
        "case_id",
        "point_xyz",
        "point_label",
        "target_volume_ml",
        "total_tumor_volume_ml",
        "interactive_target_dice",
        "interactive_full_tumor_dice",
        "original_tumor_coverage_fraction",
        "merged_tumor_coverage_fraction",
        "tumor_coverage_gain",
        "original_hec_dice",
        "merged_hec_dice",
        "hec_dice_delta",
        "original_hec_surface_dice",
        "merged_hec_surface_dice",
        "hec_surface_dice_delta",
        "original_hec_hd95_mm",
        "merged_hec_hd95_mm",
        "hec_hd95_reduction_mm",
        "interactive_prediction_volume_ml",
        "interactive_labels_present",
        "contains_255_unprocessed",
        "wall_time_seconds",
        "prediction_path",
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
            payload = row.to_dict()
            payload["point_xyz"] = ";".join(
                str(value)
                for value in row.point_xyz
            )
            payload["interactive_labels_present"] = ";".join(
                str(value)
                for value in row.interactive_labels_present
            )
            writer.writerow(payload)


def write_markdown_report(
    *,
    rows: Iterable[TumorOracleSmokeResult],
    summary: TumorOracleCohortSummary,
    path: Path,
) -> None:
    rows = tuple(rows)

    lines = [
        "# Tumor oracle special-point cohort",
        "",
        "## Scope",
        "",
        (
            "Nine frozen GT-derived tumor points are evaluated as an "
            "oracle upper-bound capability experiment. Results are not "
            "deployment-valid."
        ),
        "",
        "## Aggregate result",
        "",
        (
            f"- Largest-component tumor Dice: "
            f"mean {summary.interactive_target_dice_mean:.4f}, "
            f"median {summary.interactive_target_dice_median:.4f}."
        ),
        (
            f"- Full-tumor Dice: "
            f"mean {summary.interactive_full_tumor_dice_mean:.4f}, "
            f"median {summary.interactive_full_tumor_dice_median:.4f}."
        ),
        (
            f"- Tumor coverage gain: "
            f"mean {100.0 * summary.tumor_coverage_gain_mean:+.1f} pp, "
            f"median {100.0 * summary.tumor_coverage_gain_median:+.1f} pp."
        ),
        (
            f"- HEC Dice delta: "
            f"mean {summary.hec_dice_delta_mean:+.4f}, "
            f"median {summary.hec_dice_delta_median:+.4f}."
        ),
        (
            f"- HEC Surface Dice delta: "
            f"mean {summary.hec_surface_dice_delta_mean:+.4f}, "
            f"median {summary.hec_surface_dice_delta_median:+.4f}."
        ),
        (
            f"- HEC HD95 reduction: "
            f"mean {summary.hec_hd95_reduction_mean_mm:+.2f} mm, "
            f"median {summary.hec_hd95_reduction_median_mm:+.2f} mm."
        ),
        (
            f"- HD95 improved: "
            f"{summary.hd95_improved_case_count}/9."
        ),
        (
            f"- HD95 reduced by >5 mm: "
            f"{summary.hd95_reduced_gt_5mm_count}/9."
        ),
        "",
        "## Per-case results",
        "",
        "| Case | Tumor Dice | Coverage gain | HEC Dice Δ | Surface Dice Δ | HD95 reduction (mm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            f"| {row.case_id} "
            f"| {row.interactive_target_dice:.4f} "
            f"| {100.0 * row.tumor_coverage_gain:+.1f} pp "
            f"| {row.hec_dice_delta:+.4f} "
            f"| {row.hec_surface_dice_delta:+.4f} "
            f"| {row.hec_hd95_reduction_mm:+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "GT-derived tumor localization is an oracle input. "
                "A positive capability result shows latent interactive "
                "tumor-segmentation capacity, not an autonomous tumor "
                "segmentation system."
            ),
            "",
        ]
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
