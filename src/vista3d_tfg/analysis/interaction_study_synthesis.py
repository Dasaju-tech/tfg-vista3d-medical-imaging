from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InteractionDecision:
    hypothesis: str
    status: str
    evidence: str
    consequence: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _require_dict(
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing object {key!r}.")
    return value


def _cohort_summary(payload: dict[str, Any]) -> dict[str, Any]:
    # Block 41C writes the aggregate metrics under top-level "summary".
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return summary
    return payload


def load_interaction_sources(
    *,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    experiments = project_root / "outputs" / "experiments"

    paths = {
        "block41c_oracle": (
            experiments
            / "nv_segment_ct_tumor_oracle_point"
            / "cohort"
            / "summary.json"
        ),
        "block42_click_robustness": (
            experiments
            / "nv_segment_ct_tumor_click_robustness"
            / "summary.json"
        ),
        "block43_depth": (
            experiments
            / "nv_segment_ct_tumor_click_depth_audit"
            / "summary.json"
        ),
        "block44_safe_region_outliers": (
            experiments
            / "nv_segment_ct_tumor_safe_click_region"
            / "summary.json"
        ),
        "block45_safe_region_expanded100": (
            experiments
            / "nv_segment_ct_expanded100_tumor_safe_regions"
            / "summary.json"
        ),
        "block46b_relative_depth": (
            experiments
            / "nv_segment_ct_tumor_adaptive_relative_depth"
            / "inference"
            / "summary.json"
        ),
        "block47b_dual_point": (
            experiments
            / "nv_segment_ct_tumor_dual_interior_point"
            / "inference"
            / "summary.json"
        ),
    }

    return {
        name: _read_json(path)
        for name, path in paths.items()
    }


def synthesize_interaction_study(
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    oracle = _cohort_summary(
        sources["block41c_oracle"]
    )
    click = sources["block42_click_robustness"]
    depth = sources["block43_depth"]
    safe9 = sources["block44_safe_region_outliers"]
    safe100 = sources["block45_safe_region_expanded100"]
    adaptive = sources["block46b_relative_depth"]
    dual = sources["block47b_dual_point"]

    click_variants = _require_dict(click, "variants")
    depth_variants = _require_dict(depth, "variants")
    safe100_depths = _require_dict(safe100, "depths")
    safe100_quartiles = _require_dict(
        safe100,
        "volume_quartiles",
    )
    adaptive_overall = _require_dict(adaptive, "overall")
    adaptive_quartiles = _require_dict(
        adaptive,
        "by_quartile",
    )
    dual_overall = _require_dict(dual, "overall")
    dual_quartiles = _require_dict(
        dual,
        "by_quartile",
    )

    deep_click = _require_dict(
        click_variants,
        "deep",
    )
    centroid_click = _require_dict(
        click_variants,
        "centroid",
    )
    shallow_click = _require_dict(
        click_variants,
        "shallow_5mm",
    )
    depth10 = _require_dict(
        depth_variants,
        "depth_10mm",
    )
    depth15 = _require_dict(
        depth_variants,
        "depth_15mm",
    )

    safe15 = _require_dict(
        safe100_depths,
        "15mm",
    )
    smallest_quartile = _require_dict(
        safe100_quartiles,
        "smallest_quartile",
    )
    smallest15 = _require_dict(
        smallest_quartile,
        "15mm",
    )

    adaptive_paired = _require_dict(
        adaptive_overall,
        "paired_adaptive_minus_deep",
    )
    adaptive_dice_delta = _require_dict(
        adaptive_paired,
        "tumor_dice",
    )
    adaptive_ci = _require_dict(
        adaptive_dice_delta,
        "bootstrap_mean_95_ci",
    )
    adaptive_upper = _require_dict(
        adaptive_quartiles,
        "upper_middle",
    )
    adaptive_upper_paired = _require_dict(
        adaptive_upper,
        "paired_adaptive_minus_deep",
    )
    adaptive_upper_dice = _require_dict(
        adaptive_upper_paired,
        "tumor_dice",
    )

    dual_paired = _require_dict(
        dual_overall,
        "paired_dual_minus_deep",
    )
    dual_dice_delta = _require_dict(
        dual_paired,
        "tumor_dice",
    )
    dual_ci = _require_dict(
        dual_dice_delta,
        "bootstrap_mean_95_ci",
    )
    dual_upper = _require_dict(
        dual_quartiles,
        "upper_middle",
    )
    dual_upper_paired = _require_dict(
        dual_upper,
        "paired_dual_minus_deep",
    )
    dual_upper_dice = _require_dict(
        dual_upper_paired,
        "tumor_dice",
    )

    oracle_dice = float(
        oracle["interactive_target_dice_mean"]
    )
    oracle_hd95_improved = int(
        oracle["hd95_improved_case_count"]
    )

    centroid_dice = float(
        centroid_click["tumor_dice_mean"]
    )
    deep_dice = float(
        deep_click["tumor_dice_mean"]
    )
    shallow_dice = float(
        shallow_click["tumor_dice_mean"]
    )

    fixed_15_nonempty = int(
        safe15["cases_with_nonempty_region"]
    )
    tumor_positive_count = int(
        safe100["tumor_positive_count"]
    )
    smallest_15_nonempty = int(
        smallest15["cases_with_nonempty_region"]
    )
    smallest_count = int(
        smallest_quartile["count"]
    )

    adaptive_delta = float(
        adaptive_dice_delta["mean"]
    )
    adaptive_upper_delta = float(
        adaptive_upper_dice["mean"]
    )

    dual_delta = float(
        dual_dice_delta["mean"]
    )
    dual_upper_delta = float(
        dual_upper_dice["mean"]
    )

    decisions = [
        InteractionDecision(
            hypothesis="Latent interactive tumor capability",
            status="accepted",
            evidence=(
                f"Block 41C oracle cohort: mean tumor Dice "
                f"{oracle_dice:.4f}; HD95 improved "
                f"{oracle_hd95_improved}/9."
            ),
            consequence=(
                "VISTA3D contains useful tumor-segmentation capability "
                "when tumor location is provided."
            ),
        ),
        InteractionDecision(
            hypothesis="Exact deepest-interior click is required",
            status="rejected",
            evidence=(
                f"Block 42 centroid mean Dice {centroid_dice:.4f} "
                f"vs deep {deep_dice:.4f}; centroid preserved HD95 "
                f"improvement in 9/9."
            ),
            consequence=(
                "A central, well-interior click is sufficient in the "
                "nine severe undercoverage cases."
            ),
        ),
        InteractionDecision(
            hypothesis="Boundary-near click is robust",
            status="rejected",
            evidence=(
                f"Block 42 shallow-5mm mean Dice "
                f"{shallow_dice:.4f} vs deep {deep_dice:.4f}."
            ),
            consequence=(
                "Clicks should avoid the lesion boundary; interaction "
                "quality is depth-sensitive."
            ),
        ),
        InteractionDecision(
            hypothesis="Fixed 15 mm interior rule generalizes",
            status="rejected",
            evidence=(
                f"Block 45: 15 mm safe region exists in "
                f"{fixed_15_nonempty}/{tumor_positive_count} "
                f"tumor-positive expanded100 cases and "
                f"{smallest_15_nonempty}/{smallest_count} in the "
                f"smallest tumor quartile."
            ),
            consequence=(
                "A universal physical-depth rule is inappropriate "
                "across tumor sizes."
            ),
        ),
        InteractionDecision(
            hypothesis="Single 70% relative-depth rule generalizes",
            status="rejected",
            evidence=(
                f"Block 46B paired mean Dice delta "
                f"{adaptive_delta:+.4f}, bootstrap 95% CI "
                f"[{float(adaptive_ci['lower_95']):+.4f}, "
                f"{float(adaptive_ci['upper_95']):+.4f}], "
                f"upper-middle delta "
                f"{adaptive_upper_delta:+.4f}."
            ),
            consequence=(
                "One global relative-depth percentage is not "
                "sufficiently stable across tumor-size strata."
            ),
        ),
        InteractionDecision(
            hypothesis="Two spatially separated positive points are robustly superior",
            status="rejected",
            evidence=(
                f"Block 47B paired mean Dice delta "
                f"{dual_delta:+.4f}, bootstrap 95% CI "
                f"[{float(dual_ci['lower_95']):+.4f}, "
                f"{float(dual_ci['upper_95']):+.4f}], "
                f"upper-middle delta "
                f"{dual_upper_delta:+.4f}."
            ),
            consequence=(
                "A second oracle point offers inconsistent benefit "
                "and does not justify further prompt tuning."
            ),
        ),
    ]

    primary_recommendation = (
        "Stop oracle prompt optimization. Preserve the strongest result "
        "as a capability finding: one well-interior clinician-style point "
        "can expose useful tumor segmentation, but point placement remains "
        "GT-derived in this study. The next validation step should use "
        "deployment-plausible human or independently localized clicks, "
        "not additional retrospectively tuned oracle rules."
    )

    return {
        "schema_version": "1.0",
        "analysis": "tumor_interaction_study_synthesis",
        "blocks_included": [
            "41C",
            "42",
            "43",
            "44",
            "45",
            "46B",
            "47B",
        ],
        "key_results": {
            "oracle_cohort": {
                "tumor_dice_mean": oracle_dice,
                "tumor_dice_median": float(
                    oracle["interactive_target_dice_median"]
                ),
                "tumor_coverage_gain_mean": float(
                    oracle["tumor_coverage_gain_mean"]
                ),
                "hec_hd95_reduction_mean_mm": float(
                    oracle["hec_hd95_reduction_mean_mm"]
                ),
                "hd95_improved_count": oracle_hd95_improved,
            },
            "click_robustness": {
                "deep_tumor_dice_mean": deep_dice,
                "centroid_tumor_dice_mean": centroid_dice,
                "shallow_5mm_tumor_dice_mean": shallow_dice,
                "depth_10mm_tumor_dice_mean": float(
                    depth10["tumor_dice_mean"]
                ),
                "depth_15mm_tumor_dice_mean": float(
                    depth15["tumor_dice_mean"]
                ),
            },
            "safe_region_generalization": {
                "tumor_positive_count": tumor_positive_count,
                "15mm_nonempty_count": fixed_15_nonempty,
                "15mm_nonempty_fraction": (
                    fixed_15_nonempty / tumor_positive_count
                ),
                "smallest_quartile_15mm_nonempty_count": (
                    smallest_15_nonempty
                ),
                "smallest_quartile_count": smallest_count,
            },
            "relative_depth": {
                "paired_tumor_dice_delta_mean": adaptive_delta,
                "paired_tumor_dice_delta_95_ci": [
                    float(adaptive_ci["lower_95"]),
                    float(adaptive_ci["upper_95"]),
                ],
                "upper_middle_delta_mean": adaptive_upper_delta,
            },
            "dual_point": {
                "paired_tumor_dice_delta_mean": dual_delta,
                "paired_tumor_dice_delta_95_ci": [
                    float(dual_ci["lower_95"]),
                    float(dual_ci["upper_95"]),
                ],
                "upper_middle_delta_mean": dual_upper_delta,
            },
        },
        "decisions": [
            asdict(decision)
            for decision in decisions
        ],
        "final_decision": {
            "oracle_prompt_engineering_status": "closed",
            "preferred_interaction_concept": (
                "single well-interior positive tumor point"
            ),
            "deployment_valid": False,
            "next_validation_priority": (
                "human or independently localized click validation"
            ),
            "recommendation": primary_recommendation,
        },
        "interpretation_boundary": (
            "All tumor interaction points in Blocks 41-47 were derived "
            "from ground-truth tumor geometry. Results establish latent "
            "interactive capability and localization sensitivity, not "
            "autonomous tumor detection or clinical deployment validity."
        ),
    }


def write_synthesis_outputs(
    *,
    payload: dict[str, Any],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "interaction_study_summary.json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = payload["decisions"]
    if not isinstance(decisions, list):
        raise TypeError("decisions must be a list.")

    with (output_root / "decision_matrix.csv").open(
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
                "consequence",
            ],
        )
        writer.writeheader()
        writer.writerows(decisions)


def write_markdown_report(
    *,
    payload: dict[str, Any],
    path: Path,
) -> None:
    key = _require_dict(payload, "key_results")
    oracle = _require_dict(key, "oracle_cohort")
    click = _require_dict(key, "click_robustness")
    safe = _require_dict(key, "safe_region_generalization")
    relative = _require_dict(key, "relative_depth")
    dual = _require_dict(key, "dual_point")
    final_decision = _require_dict(
        payload,
        "final_decision",
    )

    lines = [
        "# Tumor interaction study synthesis",
        "",
        "## Executive result",
        "",
        (
            "The oracle interaction branch demonstrates clear latent "
            "VISTA3D tumor-segmentation capability when tumor location "
            "is supplied, but increasingly complex oracle point rules "
            "do not produce a robust, size-generalizable advantage."
        ),
        "",
        "## Key results",
        "",
        (
            f"- One-point oracle cohort: mean tumor Dice "
            f"{float(oracle['tumor_dice_mean']):.4f}, mean HEC HD95 "
            f"reduction {float(oracle['hec_hd95_reduction_mean_mm']):+.2f} mm, "
            f"HD95 improved {oracle['hd95_improved_count']}/9."
        ),
        (
            f"- Click location: centroid mean Dice "
            f"{float(click['centroid_tumor_dice_mean']):.4f}, "
            f"deep {float(click['deep_tumor_dice_mean']):.4f}, "
            f"5 mm {float(click['shallow_5mm_tumor_dice_mean']):.4f}."
        ),
        (
            f"- Fixed 15 mm safe region: "
            f"{safe['15mm_nonempty_count']}/{safe['tumor_positive_count']} "
            f"expanded100 tumors; "
            f"{safe['smallest_quartile_15mm_nonempty_count']}/"
            f"{safe['smallest_quartile_count']} in the smallest quartile."
        ),
        (
            f"- 70% relative-depth rule: paired mean tumor-Dice delta "
            f"{float(relative['paired_tumor_dice_delta_mean']):+.4f}; "
            f"upper-middle delta "
            f"{float(relative['upper_middle_delta_mean']):+.4f}."
        ),
        (
            f"- Dual-point rule: paired mean tumor-Dice delta "
            f"{float(dual['paired_tumor_dice_delta_mean']):+.4f}; "
            f"upper-middle delta "
            f"{float(dual['upper_middle_delta_mean']):+.4f}."
        ),
        "",
        "## Decision matrix",
        "",
        "| Hypothesis | Status | Consequence |",
        "|---|---|---|",
    ]

    for decision in payload["decisions"]:
        lines.append(
            f"| {decision['hypothesis']} "
            f"| {decision['status']} "
            f"| {decision['consequence']} |"
        )

    lines.extend(
        [
            "",
            "## Final branch decision",
            "",
            (
                f"- Oracle prompt engineering: "
                f"**{final_decision['oracle_prompt_engineering_status']}**."
            ),
            (
                f"- Preferred interaction concept: "
                f"**{final_decision['preferred_interaction_concept']}**."
            ),
            (
                f"- Deployment-valid: "
                f"**{str(final_decision['deployment_valid']).lower()}**."
            ),
            (
                f"- Next validation priority: "
                f"**{final_decision['next_validation_priority']}**."
            ),
            "",
            str(final_decision["recommendation"]),
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
