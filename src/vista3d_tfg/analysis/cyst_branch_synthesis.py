from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object: {path}"
        )
    return payload


def synthesize_cyst_branch(
    *,
    block50: dict[str, Any],
    block51: dict[str, Any],
) -> dict[str, Any]:
    overall = block50["overall"]
    quartiles = block50[
        "by_reference_volume_quartile"
    ]
    associations = block50["associations"]

    events = block51["event_counts"]
    predictors = block51[
        "volume_as_predictor"
    ]
    bins = block51[
        "volume_response_bins"
    ]

    smallest = quartiles["smallest"]
    largest = quartiles["largest"]

    overlap_predictor = predictors[
        "any_overlap"
    ]
    useful_predictor = predictors[
        "dice_ge_050"
    ]

    decisions = [
        {
            "hypothesis": (
                "Cyst failure is primarily "
                "false-negative undersegmentation"
            ),
            "status": "accepted",
            "evidence": (
                f"{overall['dominant_error_counts']['undersegmentation_fn']}/"
                f"{block50['case_count']} cyst-positive cases were "
                "FN-dominant."
            ),
        },
        {
            "hypothesis": (
                "Small cysts show a detectability collapse"
            ),
            "status": "accepted",
            "evidence": (
                f"Smallest quartile: "
                f"{smallest['zero_dice_count']}/{smallest['count']} "
                f"zero-Dice, mean recall "
                f"{smallest['recall']['mean']:.4f}."
            ),
        },
        {
            "hypothesis": (
                "Large cysts remain universally poor"
            ),
            "status": "rejected",
            "evidence": (
                f"Largest quartile mean Dice "
                f"{largest['dice']['mean']:.4f}, mean recall "
                f"{largest['recall']['mean']:.4f}, "
                f"zero-Dice {largest['zero_dice_count']}/"
                f"{largest['count']}."
            ),
        },
        {
            "hypothesis": (
                "Reference cyst volume strongly orders detectability"
            ),
            "status": "accepted",
            "evidence": (
                f"Volume AUROC={overlap_predictor['auroc']:.3f} "
                f"for any overlap and "
                f"{useful_predictor['auroc']:.3f} for Dice>=0.50."
            ),
        },
        {
            "hypothesis": (
                "A single validated cyst-size cutoff exists"
            ),
            "status": "rejected",
            "evidence": (
                "Success and failure volumes overlap; "
                f"largest any-overlap failure="
                f"{overlap_predictor['empirical_transition']['largest_negative_ml']:.2f} mL, "
                f"smallest success="
                f"{overlap_predictor['empirical_transition']['smallest_positive_ml']:.2f} mL."
            ),
        },
        {
            "hypothesis": (
                "Global postprocessing is the primary next remedy"
            ),
            "status": "rejected",
            "evidence": (
                "The dominant pattern is missing cyst voxels/cases "
                "rather than excessive false-positive volume."
            ),
        },
    ]

    return {
        "schema_version": "1.0",
        "analysis": "cyst_branch_synthesis",
        "source_blocks": [
            "50",
            "51",
        ],
        "case_count": int(
            block50["case_count"]
        ),
        "key_results": {
            "dice_mean": float(
                overall["dice"]["mean"]
            ),
            "dice_median": float(
                overall["dice"]["median"]
            ),
            "recall_mean": float(
                overall["recall"]["mean"]
            ),
            "precision_mean": float(
                overall["precision"]["mean"]
            ),
            "zero_dice_count": int(
                overall["zero_dice_count"]
            ),
            "fn_dominant_count": int(
                overall[
                    "dominant_error_counts"
                ]["undersegmentation_fn"]
            ),
            "fp_dominant_count": int(
                overall[
                    "dominant_error_counts"
                ]["oversegmentation_fp"]
            ),
            "volume_vs_dice_pearson": float(
                associations[
                    "reference_volume_vs_dice_pearson"
                ]
            ),
            "volume_vs_recall_pearson": float(
                associations[
                    "reference_volume_vs_recall_pearson"
                ]
            ),
            "any_overlap_count": int(
                events["any_overlap"]
            ),
            "dice_ge_050_count": int(
                events["dice_ge_050"]
            ),
            "any_overlap_volume_auroc": float(
                overlap_predictor["auroc"]
            ),
            "dice_ge_050_volume_auroc": float(
                useful_predictor["auroc"]
            ),
            "smallest_quartile": {
                "count": int(
                    smallest["count"]
                ),
                "dice_mean": float(
                    smallest["dice"]["mean"]
                ),
                "recall_mean": float(
                    smallest["recall"]["mean"]
                ),
                "zero_dice_count": int(
                    smallest["zero_dice_count"]
                ),
            },
            "largest_quartile": {
                "count": int(
                    largest["count"]
                ),
                "dice_mean": float(
                    largest["dice"]["mean"]
                ),
                "recall_mean": float(
                    largest["recall"]["mean"]
                ),
                "zero_dice_count": int(
                    largest["zero_dice_count"]
                ),
            },
            "volume_response_bins": bins,
        },
        "decisions": decisions,
        "final_decision": {
            "cyst_branch_status": "closed",
            "primary_failure_mode": (
                "size-dependent false-negative undersegmentation"
            ),
            "validated_clinical_cutoff_available": False,
            "global_postprocessing_recommended": False,
            "recommended_reporting": (
                "Report cyst segmentation as a size-dependent "
                "detectability limitation of the maintained automatic "
                "renal output. Preserve descriptive volume thresholds "
                "as in-sample analysis only."
            ),
        },
        "interpretation_boundary": (
            "Blocks 50-51 are retrospective analyses of the same frozen "
            "expanded100 automatic predictions. The strong size-response "
            "association does not establish a prospective or clinical "
            "cyst-size cutoff."
        ),
    }


def load_and_synthesize(
    *,
    project_root: Path,
) -> dict[str, Any]:
    block50 = _read_json(
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_cyst_failures"
        / "summary.json"
    )
    block51 = _read_json(
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_cyst_detectability_limit"
        / "summary.json"
    )

    return synthesize_cyst_branch(
        block50=block50,
        block51=block51,
    )


def write_outputs(
    *,
    payload: dict[str, Any],
    output_root: Path,
) -> None:
    import csv

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_root / "summary.json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = payload["decisions"]

    with (
        output_root
        / "decision_matrix.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "hypothesis",
                "status",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(decisions)


def write_report(
    *,
    payload: dict[str, Any],
    path: Path,
) -> None:
    key = payload["key_results"]
    smallest = key["smallest_quartile"]
    largest = key["largest_quartile"]
    final = payload["final_decision"]

    lines = [
        "# Cyst branch synthesis and decision gate",
        "",
        "## Core result",
        "",
        (
            f"Cyst-positive Dice remained low overall "
            f"({key['dice_mean']:.4f} mean; "
            f"{key['dice_median']:.4f} median), but the "
            "error pattern was strongly size-dependent."
        ),
        "",
        "## Failure mode",
        "",
        (
            f"- FN-dominant cases: "
            f"{key['fn_dominant_count']}/{payload['case_count']}."
        ),
        (
            f"- FP-dominant cases: "
            f"{key['fp_dominant_count']}/{payload['case_count']}."
        ),
        (
            f"- Zero-Dice cases: "
            f"{key['zero_dice_count']}/{payload['case_count']}."
        ),
        "",
        "## Size response",
        "",
        (
            f"- Smallest quartile: Dice "
            f"{smallest['dice_mean']:.4f}, recall "
            f"{smallest['recall_mean']:.4f}, "
            f"zero Dice {smallest['zero_dice_count']}/"
            f"{smallest['count']}."
        ),
        (
            f"- Largest quartile: Dice "
            f"{largest['dice_mean']:.4f}, recall "
            f"{largest['recall_mean']:.4f}, "
            f"zero Dice {largest['zero_dice_count']}/"
            f"{largest['count']}."
        ),
        (
            f"- Volume vs Dice Pearson: "
            f"{key['volume_vs_dice_pearson']:+.3f}."
        ),
        (
            f"- Volume vs recall Pearson: "
            f"{key['volume_vs_recall_pearson']:+.3f}."
        ),
        (
            f"- Volume AUROC for any overlap: "
            f"{key['any_overlap_volume_auroc']:.3f}."
        ),
        (
            f"- Volume AUROC for Dice>=0.50: "
            f"{key['dice_ge_050_volume_auroc']:.3f}."
        ),
        "",
        "## Decision matrix",
        "",
        "| Hypothesis | Status | Evidence |",
        "|---|---|---|",
    ]

    for decision in payload["decisions"]:
        lines.append(
            f"| {decision['hypothesis']} "
            f"| {decision['status']} "
            f"| {decision['evidence']} |"
        )

    lines.extend(
        [
            "",
            "## Final decision",
            "",
            (
                f"- Branch status: "
                f"**{final['cyst_branch_status']}**."
            ),
            (
                f"- Primary failure mode: "
                f"**{final['primary_failure_mode']}**."
            ),
            (
                "- Validated clinical cyst-size cutoff: **no**."
            ),
            (
                "- Global postprocessing recommended: **no**."
            ),
            "",
            str(final["recommended_reporting"]),
            "",
            "## Interpretation boundary",
            "",
            str(payload["interpretation_boundary"]),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
