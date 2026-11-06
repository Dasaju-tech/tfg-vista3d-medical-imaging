from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.selective_trigger_validation import (
    analyze_selective_filtering,
    read_filter_cases,
    read_trigger_cases,
)


def _write_decisions(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["fold_decisions"]
    assert isinstance(rows, list)

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    trigger = payload["trigger"]
    original = payload["original"]
    variants = payload["variants"]

    assert isinstance(trigger, dict)
    assert isinstance(original, dict)
    assert isinstance(variants, dict)

    lines = [
        "# Expanded100 LOOCV selective-trigger validation",
        "",
        "## Trigger validation",
        "",
        (
            f"- TP/FP/TN/FN: "
            f"{trigger['tp']}/{trigger['fp']}/"
            f"{trigger['tn']}/{trigger['fn']}."
        ),
        (
            f"- Sensitivity: "
            f"{100.0 * float(trigger['sensitivity']):.1f}%."
        ),
        (
            f"- Specificity: "
            f"{100.0 * float(trigger['specificity']):.1f}%."
        ),
        (
            f"- Precision: "
            f"{100.0 * float(trigger['precision']):.1f}%."
        ),
        (
            f"- Balanced accuracy: "
            f"{float(trigger['balanced_accuracy']):.3f}."
        ),
        "",
        "## Selective post-processing",
        "",
        (
            f"Original mean HD95: "
            f"{float(original['hd95_mean_mm']):.2f} mm."
        ),
        "",
        "| Variant | Mean HD95 | Median HD95 | Remaining outliers | Changed cases | Target resolved | Target mean reduction | Non-target HD95 harm | Non-target Dice harm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, values in variants.items():
        assert isinstance(values, dict)
        summary = values["all_cases"]
        assert isinstance(summary, dict)

        lines.append(
            f"| {name} "
            f"| {float(summary['hd95_mean_mm']):.2f} "
            f"| {float(summary['hd95_median_mm']):.2f} "
            f"| {values['remaining_outliers_at_original_threshold']} "
            f"| {values['changed_count']} "
            f"| {values['target_outliers_resolved']}/12 "
            f"| {float(values['target_hd95_reduction_mean_mm']):+.2f} "
            f"| {values['non_target_hd95_worsened_gt_1mm']} "
            f"| {values['non_target_dice_worsened_gt_0_01']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(payload["interpretation_boundary"]),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_feature_frequency(
    payload: dict[str, object],
    path: Path,
) -> None:
    frequencies = payload["selected_feature_frequency"]
    assert isinstance(frequencies, dict)

    labels = list(frequencies.keys())
    values = [int(frequencies[key]) for key in labels]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("LOOCV folds selected")
    ax.set_title("Selected prediction-only feature by held-out fold")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    root = discover_project_root()

    trigger_csv = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_prediction_geometry_trigger"
        / "prediction_geometry_cases.csv"
    )
    filter_csv = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_component_filter_sensitivity"
        / "case_comparison.csv"
    )
    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_selective_trigger_loocv"
    )

    trigger_rows = read_trigger_cases(trigger_csv)
    filter_rows = read_filter_cases(filter_csv)

    payload = analyze_selective_filtering(
        trigger_rows=trigger_rows,
        filter_rows=filter_rows,
    )

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "selective_trigger_loocv.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    decisions_path = output_root / "fold_decisions.csv"
    _write_decisions(payload, decisions_path)

    md_path = output_root / "selective_trigger_loocv.md"
    _write_markdown(payload, md_path)

    _render_feature_frequency(
        payload,
        output_root / "figures" / "feature_selection_frequency.png",
    )

    trigger = payload["trigger"]
    variants = payload["variants"]

    assert isinstance(trigger, dict)
    assert isinstance(variants, dict)

    print("NV-Segment-CT LOOCV selective-trigger validation")
    print(
        "Trigger confusion TP/FP/TN/FN: "
        f"{trigger['tp']}/{trigger['fp']}/"
        f"{trigger['tn']}/{trigger['fn']}"
    )
    print(
        "Sensitivity / specificity / precision: "
        f"{100.0 * float(trigger['sensitivity']):.1f}% / "
        f"{100.0 * float(trigger['specificity']):.1f}% / "
        f"{100.0 * float(trigger['precision']):.1f}%"
    )
    print(
        "Balanced accuracy: "
        f"{float(trigger['balanced_accuracy']):.3f}"
    )
    print()

    for name, values in variants.items():
        assert isinstance(values, dict)
        summary = values["all_cases"]
        assert isinstance(summary, dict)

        print(name)
        print(
            f"  selective mean HD95: "
            f"{float(summary['hd95_mean_mm']):.2f} mm"
        )
        print(
            f"  remaining outliers: "
            f"{values['remaining_outliers_at_original_threshold']}/100"
        )
        print(
            f"  target outliers resolved: "
            f"{values['target_outliers_resolved']}/12"
        )
        print(
            f"  target mean HD95 reduction: "
            f"{float(values['target_hd95_reduction_mean_mm']):+.2f} mm"
        )
        print(
            "  non-target harm: "
            f"HD95>1mm="
            f"{values['non_target_hd95_worsened_gt_1mm']}, "
            f"Dice>0.01="
            f"{values['non_target_dice_worsened_gt_0_01']}"
        )
        print()

    print(f"JSON: {json_path}")
    print(f"CSV:  {decisions_path}")
    print(f"MD:   {md_path}")
    print("LOOCV selective-trigger validation PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
