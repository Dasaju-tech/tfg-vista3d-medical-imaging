from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_final_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    if payload.get("analysis") != "final_experimental_synthesis":
        raise ValueError("Expected Block 53 final experimental synthesis.")
    return payload


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def build_result_macros(payload: dict[str, Any]) -> str:
    baseline = payload["automatic_baseline"]
    interaction = payload["tumor_interaction"]
    manual = payload["manual_ct_only_localization"]
    cyst = payload["cyst_branch"]
    tail = payload["hd95_tail"]

    values = {
        "ExpandedN": baseline["case_count"],
        "KidneyMassDice": f"{baseline['kidney_and_mass_dice_mean']:.4f}",
        "KidneyMassDiceSD": f"{baseline['kidney_and_mass_dice_sd']:.4f}",
        "SurfaceDice": f"{baseline['surface_dice_mean']:.4f}",
        "HDMedian": f"{baseline['hd95_median_mm']:.2f}",
        "HDMean": f"{baseline['hd95_mean_mm']:.2f}",
        "HDOutliers": tail["outlier_count"],
        "OracleTumorDice": f"{interaction['oracle_tumor_dice_mean']:.4f}",
        "OracleHDReduction": f"{interaction['oracle_hec_hd95_reduction_mean_mm']:.2f}",
        "ManualTumorHits": manual["any_tumor_hit_count"],
        "ManualTumorDice": f"{manual['manual_tumor_dice_mean']:.4f}",
        "DeepTumorDice": f"{manual['deep_tumor_dice_mean']:.4f}",
        "CystN": cyst["case_count"],
        "CystDice": f"{cyst['dice_mean']:.4f}",
        "CystZeroDice": cyst["zero_dice_count"],
        "CystFNDominant": cyst["fn_dominant_count"],
        "CystOverlapAUROC": f"{cyst['any_overlap_volume_auroc']:.3f}",
        "CystUsefulAUROC": f"{cyst['dice_ge_050_volume_auroc']:.3f}",
    }

    lines = [
        "% Auto-generated from Block 53. Do not edit numeric values manually."
    ]
    for name, value in values.items():
        lines.append(f"\\newcommand{{\\TFG{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def build_main_results_table(payload: dict[str, Any]) -> str:
    baseline = payload["automatic_baseline"]
    interaction = payload["tumor_interaction"]
    manual = payload["manual_ct_only_localization"]
    cyst = payload["cyst_branch"]

    rows = [
        ("Automatic expanded100", "Kidney-and-mass Dice",
         f"{baseline['kidney_and_mass_dice_mean']:.4f} $\\pm$ {baseline['kidney_and_mass_dice_sd']:.4f}"),
        ("Automatic expanded100", "Surface Dice",
         f"{baseline['surface_dice_mean']:.4f}"),
        ("Automatic expanded100", "HD95 median / mean",
         f"{baseline['hd95_median_mm']:.2f} / {baseline['hd95_mean_mm']:.2f} mm"),
        ("Oracle interaction", "Tumor Dice",
         f"{interaction['oracle_tumor_dice_mean']:.4f}"),
        ("Oracle interaction", "Mean HEC HD95 reduction",
         f"{interaction['oracle_hec_hd95_reduction_mean_mm']:+.2f} mm"),
        ("Manual CT-only", "Tumor hits",
         f"{manual['any_tumor_hit_count']}/{manual['case_count']}"),
        ("Manual CT-only", "Tumor Dice",
         f"{manual['manual_tumor_dice_mean']:.4f}"),
        ("Cyst-positive cohort", "Cyst Dice mean / median",
         f"{cyst['dice_mean']:.4f} / {cyst['dice_median']:.4f}"),
        ("Cyst-positive cohort", "Volume AUROC: any overlap",
         f"{cyst['any_overlap_volume_auroc']:.3f}"),
    ]

    body = "\n".join(
        f"{_latex_escape(group)} & {_latex_escape(metric)} & {value} \\\\"
        for group, metric, value in rows
    )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Final experimental summary. Oracle interaction results use "
        "ground-truth-derived localization and the manual CT-only experiment "
        "represents simulated-user, not clinician, localization.}\n"
        "\\label{tab:final-results}\n"
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Experiment & Metric & Result \\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def build_claim_table(payload: dict[str, Any]) -> str:
    body = "\n".join(
        f"{_latex_escape(item['claim'])} & "
        f"{_latex_escape(str(item['status']).replace('_', ' '))} \\\\"
        for item in payload["claims"]
    )

    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Experimental claim matrix used to constrain interpretation.}\n"
        "\\label{tab:claim-matrix}\n"
        "\\begin{tabular}{p{0.76\\textwidth}p{0.16\\textwidth}}\n"
        "\\toprule\n"
        "Claim & Status \\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_baseline_overview(payload: dict[str, Any], *, output_stem: Path) -> None:
    baseline = payload["automatic_baseline"]
    labels = ["Kidney + mass\nDice", "Surface Dice", "Cyst-positive\nDice"]
    values = [
        baseline["kidney_and_mass_dice_mean"],
        baseline["surface_dice_mean"],
        baseline["cyst_positive_dice_mean"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Frozen expanded100 automatic segmentation")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width()/2, value + 0.025,
                  f"{value:.3f}", ha="center", va="bottom")
    _save_figure(fig, output_stem)


def plot_hd95_failure_mechanisms(payload: dict[str, Any], *, output_stem: Path) -> None:
    tail = payload["hd95_tail"]
    labels = ["Prediction → GT", "GT → prediction", "Balanced"]
    values = [
        tail["prediction_to_gt_dominant_count"],
        tail["gt_to_prediction_dominant_count"],
        tail["balanced_count"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar(labels, values)
    axis.set_ylabel("HD95 outlier cases")
    axis.set_title("Residual-geometry mechanism in the HD95 tail")
    axis.set_ylim(0, max(values) + 3)
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width()/2, value + 0.3,
                  str(value), ha="center", va="bottom")
    _save_figure(fig, output_stem)


def plot_interaction_comparison(payload: dict[str, Any], *, output_stem: Path) -> None:
    interaction = payload["tumor_interaction"]
    manual = payload["manual_ct_only_localization"]
    labels = [
        "Oracle\n9-case cohort",
        "Deep oracle\n12-case reference",
        "Manual CT-only\n12 cases",
    ]
    values = [
        interaction["oracle_tumor_dice_mean"],
        manual["deep_tumor_dice_mean"],
        manual["manual_tumor_dice_mean"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    bars = axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Tumor Dice")
    axis.set_title("Interactive tumor capability versus localization setting")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width()/2, value + 0.025,
                  f"{value:.3f}", ha="center", va="bottom")
    _save_figure(fig, output_stem)


def plot_cyst_volume_response(payload: dict[str, Any], *, output_stem: Path) -> None:
    bins = payload["cyst_branch"].get("volume_response_bins", [])
    if not bins:
        raise ValueError("Block 53 summary does not contain cyst volume-response bins.")

    x = np.asarray([row["volume_median_ml"] for row in bins], dtype=float)
    overlap = np.asarray([row["any_overlap_fraction"] for row in bins], dtype=float)
    useful = np.asarray([row["dice_ge_050_fraction"] for row in bins], dtype=float)

    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(x, overlap, marker="o", label="Any overlap")
    axis.plot(x, useful, marker="s", label=r"Dice $\geq$ 0.50")
    axis.set_xscale("log")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Median reference cyst volume per bin (mL, log scale)")
    axis.set_ylabel("Fraction of cases")
    axis.set_title("Size-dependent cyst detectability")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(fig, output_stem)


def build_all_assets(*, payload: dict[str, Any], output_root: Path) -> None:
    latex_root = output_root / "latex"
    figures_root = output_root / "figures"
    latex_root.mkdir(parents=True, exist_ok=True)

    (latex_root / "results_macros.tex").write_text(
        build_result_macros(payload), encoding="utf-8"
    )
    (latex_root / "table_final_results.tex").write_text(
        build_main_results_table(payload), encoding="utf-8"
    )
    (latex_root / "table_claim_matrix.tex").write_text(
        build_claim_table(payload), encoding="utf-8"
    )

    plot_baseline_overview(payload, output_stem=figures_root / "fig01_automatic_baseline")
    plot_hd95_failure_mechanisms(payload, output_stem=figures_root / "fig02_hd95_failure_mechanisms")
    plot_interaction_comparison(payload, output_stem=figures_root / "fig03_tumor_interaction_comparison")
    plot_cyst_volume_response(payload, output_stem=figures_root / "fig04_cyst_volume_response")
