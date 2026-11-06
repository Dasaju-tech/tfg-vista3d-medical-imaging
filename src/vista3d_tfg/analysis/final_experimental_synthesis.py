from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FROZEN_AUTOMATIC_BASELINE = {
    "case_count": 100,
    "kidney_and_mass_dice_mean": 0.8863,
    "kidney_and_mass_dice_sd": 0.1309,
    "surface_dice_mean": 0.7471,
    "hd95_median_mm": 4.53,
    "hd95_mean_mm": 12.1003,
    "cyst_positive_count": 46,
    "cyst_positive_dice_mean": 0.1958,
    "inference_wall_time_mean_s": 15.24,
    "laterality_resolved_count": 99,
    "laterality_unresolved_count": 1,
    "official_kits23_challenge_score_available": False,
}

FROZEN_HD95_TAIL = {
    "outlier_count": 22,
    "outlier_threshold_mm": 11.08,
    "top5_hd95_mean_share": 0.392,
    "top10_hd95_mean_share": 0.549,
    "prediction_to_gt_dominant_count": 12,
    "gt_to_prediction_dominant_count": 9,
    "balanced_count": 1,
    "high_tumor_outlier_risk_ratio": 6.90,
    "prediction_only_trigger_status": "rejected_after_loocv",
    "prediction_only_trigger_balanced_accuracy": 0.612,
    "prediction_only_trigger_target_resolved_count": 0,
}

FROZEN_TUMOR_UNDERCOVERAGE = {
    "undercoverage_case_count": 9,
    "tumor_dominant_missing_count": 8,
    "kidney_dominant_missing_count": 1,
    "cyst_dominant_missing_count": 0,
    "weighted_tumor_miss_fraction": 0.892,
    "renal_expansion_status": "rejected",
}


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


def _load_project_artifacts(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    return {
        "interaction": _read_json(
            project_root
            / "outputs"
            / "analyses"
            / "nv_segment_ct_tumor_interaction_study"
            / "interaction_study_summary.json"
        ),
        "manual_geometry": _read_json(
            project_root
            / "outputs"
            / "analyses"
            / "nv_segment_ct_manual_click_geometry_qc"
            / "summary.json"
        ),
        "manual_inference": _read_json(
            project_root
            / "outputs"
            / "experiments"
            / "nv_segment_ct_manual_tumor_clicks"
            / "inference"
            / "summary.json"
        ),
        "cyst": _read_json(
            project_root
            / "outputs"
            / "analyses"
            / "nv_segment_ct_cyst_branch_synthesis"
            / "summary.json"
        ),
    }


def synthesize_final_experimental_results(
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    interaction = artifacts["interaction"]
    manual_geometry = artifacts["manual_geometry"]
    manual_inference = artifacts["manual_inference"]
    cyst = artifacts["cyst"]

    interaction_key = interaction["key_results"]
    interaction_final = interaction["final_decision"]

    oracle = interaction_key["oracle_cohort"]
    dual = interaction_key["dual_point"]
    relative = interaction_key["relative_depth"]

    manual_localization = manual_inference["localization_audit"]
    manual_overall = manual_inference["overall"]
    manual_dice = manual_overall["manual"]["tumor_dice"]
    deep_dice = manual_overall["deep"]["tumor_dice"]
    manual_delta = manual_overall[
        "paired_manual_minus_deep"
    ]["tumor_dice"]

    cyst_key = cyst["key_results"]
    cyst_final = cyst["final_decision"]

    checks = {
        "interaction_branch_closed": (
            interaction_final[
                "oracle_prompt_engineering_status"
            ]
            == "closed"
        ),
        "manual_geometry_contract_valid": bool(
            manual_geometry[
                "geometry_contract_valid"
            ]
        ),
        "manual_tumor_hits_zero": (
            int(
                manual_localization[
                    "any_tumor_hit_count"
                ]
            )
            == 0
        ),
        "cyst_branch_closed": (
            cyst_final[
                "cyst_branch_status"
            ]
            == "closed"
        ),
    }

    if not all(checks.values()):
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]
        raise RuntimeError(
            "Final synthesis preconditions failed: "
            + ", ".join(failed)
        )

    claims = [
        {
            "claim_id": "automatic_baseline",
            "status": "supported",
            "claim": (
                "The frozen automatic renal output achieved strong "
                "kidney-and-mass overlap on expanded100, but its mean "
                "HD95 was affected by a concentrated failure tail."
            ),
            "evidence": (
                f"Dice={FROZEN_AUTOMATIC_BASELINE['kidney_and_mass_dice_mean']:.4f} "
                f"±{FROZEN_AUTOMATIC_BASELINE['kidney_and_mass_dice_sd']:.4f}; "
                f"surface Dice={FROZEN_AUTOMATIC_BASELINE['surface_dice_mean']:.4f}; "
                f"HD95 median={FROZEN_AUTOMATIC_BASELINE['hd95_median_mm']:.2f} mm, "
                f"mean={FROZEN_AUTOMATIC_BASELINE['hd95_mean_mm']:.2f} mm."
            ),
        },
        {
            "claim_id": "hd95_tail",
            "status": "supported",
            "claim": (
                "The HD95 weakness is concentrated rather than uniform "
                "and comprises both prediction-side distant geometry and "
                "reference-side undercoverage mechanisms."
            ),
            "evidence": (
                f"{FROZEN_HD95_TAIL['outlier_count']}/100 outliers; "
                f"top 5 contribute {100*FROZEN_HD95_TAIL['top5_hd95_mean_share']:.1f}% "
                f"of HD95-mean burden; mechanisms "
                f"{FROZEN_HD95_TAIL['prediction_to_gt_dominant_count']}/"
                f"{FROZEN_HD95_TAIL['gt_to_prediction_dominant_count']}/"
                f"{FROZEN_HD95_TAIL['balanced_count']} "
                f"(prediction→GT / GT→prediction / balanced)."
            ),
        },
        {
            "claim_id": "automatic_postprocessing",
            "status": "rejected",
            "claim": (
                "Prediction-only selective postprocessing is a robust "
                "solution for the HD95 tail."
            ),
            "evidence": (
                f"LOOCV balanced accuracy="
                f"{FROZEN_HD95_TAIL['prediction_only_trigger_balanced_accuracy']:.3f}; "
                f"resolved targets="
                f"{FROZEN_HD95_TAIL['prediction_only_trigger_target_resolved_count']}/12."
            ),
        },
        {
            "claim_id": "tumor_oracle_capability",
            "status": "supported_with_oracle_boundary",
            "claim": (
                "VISTA3D exposes substantial latent tumor-segmentation "
                "capability when lesion location is supplied by a well-interior "
                "positive point."
            ),
            "evidence": (
                f"Oracle mean tumor Dice={oracle['tumor_dice_mean']:.4f}; "
                f"mean HEC HD95 reduction="
                f"{oracle['hec_hd95_reduction_mean_mm']:+.2f} mm; "
                f"HD95 improved {oracle['hd95_improved_count']}/9."
            ),
        },
        {
            "claim_id": "oracle_rule_generalization",
            "status": "rejected",
            "claim": (
                "A single relative-depth or dual-point oracle rule "
                "provides a robust size-generalizable improvement."
            ),
            "evidence": (
                f"70% relative-depth paired Dice delta="
                f"{relative['paired_tumor_dice_delta_mean']:+.4f}; "
                f"dual-point paired Dice delta="
                f"{dual['paired_tumor_dice_delta_mean']:+.4f}."
            ),
        },
        {
            "claim_id": "manual_localization",
            "status": "negative_result",
            "claim": (
                "Unguided non-expert CT-only manual localization is an "
                "adequate substitute for correct lesion localization."
            ),
            "evidence": (
                f"Any-tumor hits="
                f"{manual_localization['any_tumor_hit_count']}/12; "
                f"manual tumor Dice mean={manual_dice['mean']:.4f} "
                f"vs deep-oracle {deep_dice['mean']:.4f}; "
                f"paired delta={manual_delta['mean']:+.4f}; "
                "geometry QC valid with zero coordinate round-trip error."
            ),
        },
        {
            "claim_id": "cyst_detectability",
            "status": "supported",
            "claim": (
                "Automatic cyst segmentation is limited primarily by "
                "size-dependent false-negative undersegmentation."
            ),
            "evidence": (
                f"Dice mean/median={cyst_key['dice_mean']:.4f}/"
                f"{cyst_key['dice_median']:.4f}; "
                f"FN-dominant={cyst_key['fn_dominant_count']}/46; "
                f"volume AUROC for any overlap="
                f"{cyst_key['any_overlap_volume_auroc']:.3f}; "
                f"for Dice≥0.50="
                f"{cyst_key['dice_ge_050_volume_auroc']:.3f}."
            ),
        },
    ]

    return {
        "schema_version": "1.0",
        "analysis": "final_experimental_synthesis",
        "automatic_baseline": FROZEN_AUTOMATIC_BASELINE,
        "hd95_tail": FROZEN_HD95_TAIL,
        "tumor_undercoverage": FROZEN_TUMOR_UNDERCOVERAGE,
        "tumor_interaction": {
            "oracle_tumor_dice_mean": float(
                oracle["tumor_dice_mean"]
            ),
            "oracle_tumor_dice_median": float(
                oracle["tumor_dice_median"]
            ),
            "oracle_hec_hd95_reduction_mean_mm": float(
                oracle[
                    "hec_hd95_reduction_mean_mm"
                ]
            ),
            "relative_depth_paired_dice_delta": float(
                relative[
                    "paired_tumor_dice_delta_mean"
                ]
            ),
            "dual_point_paired_dice_delta": float(
                dual[
                    "paired_tumor_dice_delta_mean"
                ]
            ),
            "oracle_prompt_engineering_status": (
                interaction_final[
                    "oracle_prompt_engineering_status"
                ]
            ),
        },
        "manual_ct_only_localization": {
            "case_count": int(
                manual_inference["case_count"]
            ),
            "any_tumor_hit_count": int(
                manual_localization[
                    "any_tumor_hit_count"
                ]
            ),
            "target_component_hit_count": int(
                manual_localization[
                    "target_component_hit_count"
                ]
            ),
            "manual_tumor_dice_mean": float(
                manual_dice["mean"]
            ),
            "deep_tumor_dice_mean": float(
                deep_dice["mean"]
            ),
            "paired_manual_minus_deep_dice_mean": float(
                manual_delta["mean"]
            ),
            "geometry_contract_valid": bool(
                manual_geometry[
                    "geometry_contract_valid"
                ]
            ),
            "interpretation": (
                "negative simulated-user localization result; "
                "not clinician validation"
            ),
        },
        "cyst_branch": {
            "case_count": int(
                cyst["case_count"]
            ),
            "dice_mean": float(
                cyst_key["dice_mean"]
            ),
            "dice_median": float(
                cyst_key["dice_median"]
            ),
            "zero_dice_count": int(
                cyst_key["zero_dice_count"]
            ),
            "fn_dominant_count": int(
                cyst_key["fn_dominant_count"]
            ),
            "volume_vs_dice_pearson": float(
                cyst_key[
                    "volume_vs_dice_pearson"
                ]
            ),
            "volume_vs_recall_pearson": float(
                cyst_key[
                    "volume_vs_recall_pearson"
                ]
            ),
            "any_overlap_volume_auroc": float(
                cyst_key[
                    "any_overlap_volume_auroc"
                ]
            ),
            "dice_ge_050_volume_auroc": float(
                cyst_key[
                    "dice_ge_050_volume_auroc"
                ]
            ),
            "volume_response_bins": (
                cyst_key[
                    "volume_response_bins"
                ]
            ),
            "branch_status": (
                cyst_final[
                    "cyst_branch_status"
                ]
            ),
        },
        "precondition_checks": checks,
        "claims": claims,
        "final_experimental_conclusion": (
            "The maintained NV-Segment-CT renal pipeline provides strong "
            "global kidney-and-mass overlap on the frozen expanded100 cohort, "
            "but boundary-tail failures remain concentrated in anatomically "
            "and semantically difficult cases. Tumor undercoverage reveals "
            "substantial latent VISTA3D interactive capability when lesion "
            "location is supplied, yet neither oracle prompt refinement nor "
            "unguided non-expert CT-only localization yielded a robust "
            "deployment-valid workflow. Cyst segmentation remains a distinct "
            "size-dependent detectability limitation dominated by false "
            "negatives in small lesions."
        ),
        "reporting_constraints": [
            (
                "Do not report an official KiTS23 challenge score; it is "
                "unavailable under the project protocol."
            ),
            (
                "Do not describe GT-derived tumor interaction experiments "
                "as deployment-valid localization."
            ),
            (
                "Do not describe project-author CT-only clicks as clinician "
                "or radiologist validation."
            ),
            (
                "Do not present descriptive cyst-volume thresholds as "
                "validated clinical cutoffs."
            ),
            (
                "Do not claim postprocessing solved the HD95 tail; tested "
                "prediction-only strategies were rejected."
            ),
        ],
    }


def load_and_synthesize(
    *,
    project_root: Path,
) -> dict[str, Any]:
    return synthesize_final_experimental_results(
        _load_project_artifacts(
            project_root
        )
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

    claims = payload["claims"]

    with (
        output_root
        / "claim_matrix.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "claim_id",
                "status",
                "claim",
                "evidence",
            ],
        )
        writer.writeheader()
        writer.writerows(claims)


def write_report(
    *,
    payload: dict[str, Any],
    path: Path,
) -> None:
    baseline = payload[
        "automatic_baseline"
    ]
    tail = payload["hd95_tail"]
    interaction = payload[
        "tumor_interaction"
    ]
    manual = payload[
        "manual_ct_only_localization"
    ]
    cyst = payload["cyst_branch"]

    lines = [
        "# Final experimental synthesis",
        "",
        "## 1. Automatic expanded100 baseline",
        "",
        (
            f"- Kidney-and-mass Dice: "
            f"{baseline['kidney_and_mass_dice_mean']:.4f} "
            f"± {baseline['kidney_and_mass_dice_sd']:.4f}."
        ),
        (
            f"- Surface Dice: "
            f"{baseline['surface_dice_mean']:.4f}."
        ),
        (
            f"- HD95: median "
            f"{baseline['hd95_median_mm']:.2f} mm, mean "
            f"{baseline['hd95_mean_mm']:.2f} mm."
        ),
        (
            f"- Mean inference wall time: "
            f"{baseline['inference_wall_time_mean_s']:.2f} s."
        ),
        "",
        "## 2. HD95 failure tail",
        "",
        (
            f"- Outliers: "
            f"{tail['outlier_count']}/100 above "
            f"{tail['outlier_threshold_mm']:.2f} mm."
        ),
        (
            f"- Top 5 / top 10 share of mean-H95 burden: "
            f"{100*tail['top5_hd95_mean_share']:.1f}% / "
            f"{100*tail['top10_hd95_mean_share']:.1f}%."
        ),
        (
            f"- Geometry mechanism split: "
            f"{tail['prediction_to_gt_dominant_count']} prediction→GT, "
            f"{tail['gt_to_prediction_dominant_count']} GT→prediction, "
            f"{tail['balanced_count']} balanced."
        ),
        "",
        "## 3. Tumor interaction",
        "",
        (
            f"- Oracle tumor Dice mean: "
            f"{interaction['oracle_tumor_dice_mean']:.4f}."
        ),
        (
            f"- Oracle HEC HD95 reduction mean: "
            f"{interaction['oracle_hec_hd95_reduction_mean_mm']:+.2f} mm."
        ),
        (
            f"- Relative-depth paired Dice delta: "
            f"{interaction['relative_depth_paired_dice_delta']:+.4f}."
        ),
        (
            f"- Dual-point paired Dice delta: "
            f"{interaction['dual_point_paired_dice_delta']:+.4f}."
        ),
        "",
        "## 4. Manual CT-only localization",
        "",
        (
            f"- Any-tumor hits: "
            f"{manual['any_tumor_hit_count']}/{manual['case_count']}."
        ),
        (
            f"- Manual vs deep tumor Dice mean: "
            f"{manual['manual_tumor_dice_mean']:.4f} vs "
            f"{manual['deep_tumor_dice_mean']:.4f}."
        ),
        (
            f"- Paired manual-minus-deep Dice: "
            f"{manual['paired_manual_minus_deep_dice_mean']:+.4f}."
        ),
        (
            f"- Geometry contract valid: "
            f"{str(manual['geometry_contract_valid']).lower()}."
        ),
        "",
        "## 5. Cyst limitation",
        "",
        (
            f"- Cyst-positive Dice mean / median: "
            f"{cyst['dice_mean']:.4f} / "
            f"{cyst['dice_median']:.4f}."
        ),
        (
            f"- Zero-Dice cyst cases: "
            f"{cyst['zero_dice_count']}/{cyst['case_count']}."
        ),
        (
            f"- FN-dominant cases: "
            f"{cyst['fn_dominant_count']}/{cyst['case_count']}."
        ),
        (
            f"- Volume AUROC for any overlap / Dice>=0.50: "
            f"{cyst['any_overlap_volume_auroc']:.3f} / "
            f"{cyst['dice_ge_050_volume_auroc']:.3f}."
        ),
        "",
        "## Claim matrix",
        "",
        "| Claim | Status | Evidence |",
        "|---|---|---|",
    ]

    for claim in payload["claims"]:
        lines.append(
            f"| {claim['claim']} "
            f"| {claim['status']} "
            f"| {claim['evidence']} |"
        )

    lines.extend(
        [
            "",
            "## Final experimental conclusion",
            "",
            str(
                payload[
                    "final_experimental_conclusion"
                ]
            ),
            "",
            "## Reporting constraints",
            "",
        ]
    )

    for constraint in payload[
        "reporting_constraints"
    ]:
        lines.append(
            f"- {constraint}"
        )

    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
