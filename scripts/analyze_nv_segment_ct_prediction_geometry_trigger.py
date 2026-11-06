from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.prediction_geometry_trigger import (
    analyze_prediction_geometry_trigger,
    build_prediction_geometry_rows,
)


def _write_cases(
    payload: dict[str, object],
    path: Path,
) -> None:
    rows = payload["case_rows"]
    assert isinstance(rows, list)

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_feature_table(
    payload: dict[str, object],
    path: Path,
) -> None:
    rankings = payload["feature_rankings"]
    assert isinstance(rankings, list)

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rankings[0].keys())

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rankings)


def _render_auc(
    payload: dict[str, object],
    path: Path,
) -> None:
    rankings = payload["feature_rankings"]
    assert isinstance(rankings, list)

    labels = [str(item["feature"]) for item in rankings]
    aucs = [float(item["auc"]) for item in rankings]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    y = list(range(len(labels)))
    ax.barh(y, aucs)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Univariate AUROC for target separation")
    ax.set_title("Prediction-only geometry trigger candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(
    payload: dict[str, object],
    path: Path,
) -> None:
    rankings = payload["feature_rankings"]
    best = payload["best_feature"]

    assert isinstance(rankings, list)
    assert isinstance(best, dict)

    lines = [
        "# Expanded100 prediction-geometry trigger audit",
        "",
        "## Objective",
        "",
        (
            "Identify prediction-only geometry features that separate the "
            "12 Block 35 prediction→GT-dominated HD95 outliers from the "
            "remaining 88 cases."
        ),
        "",
        "No trigger is registered in this block.",
        "",
        "## Feature ranking",
        "",
        "| Rank | Feature | Direction | AUROC | Target median | Non-target median | Top-12 target count | Top-20 recall |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    for index, item in enumerate(rankings, start=1):
        assert isinstance(item, dict)
        lines.append(
            f"| {index} "
            f"| {item['feature']} "
            f"| {item['direction']} "
            f"| {float(item['auc']):.3f} "
            f"| {float(item['target_median']):.3f} "
            f"| {float(item['non_target_median']):.3f} "
            f"| {item['top12_target_count']}/12 "
            f"| {100.0 * float(item['top20_recall']):.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Best univariate feature",
            "",
            (
                f"`{best['feature']}` with direction `{best['direction']}` "
                f"achieved AUROC={float(best['auc']):.3f}."
            ),
            "",
            "## Interpretation boundary",
            "",
            str(payload["interpretation_boundary"]),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = discover_project_root()

    taxonomy_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_hd95_tail"
        / "case_taxonomy.csv"
    )
    geometry_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_residual_geometry"
        / "outlier_geometry.csv"
    )
    prediction_root = (
        root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
    )
    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_prediction_geometry_trigger"
    )

    rows = build_prediction_geometry_rows(
        taxonomy_path=taxonomy_path,
        geometry_path=geometry_path,
        prediction_root=prediction_root,
    )
    payload = analyze_prediction_geometry_trigger(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "prediction_geometry_trigger.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    cases_path = output_root / "prediction_geometry_cases.csv"
    _write_cases(payload, cases_path)

    features_path = output_root / "feature_rankings.csv"
    _write_feature_table(payload, features_path)

    md_path = output_root / "prediction_geometry_trigger.md"
    _write_markdown(payload, md_path)

    _render_auc(
        payload,
        output_root / "figures" / "feature_auc.png",
    )

    rankings = payload["feature_rankings"]
    assert isinstance(rankings, list)

    print("NV-Segment-CT prediction-geometry trigger audit")
    print("Prediction->GT targets: 12/100")
    print()
    print("Top prediction-only features:")

    for index, item in enumerate(rankings[:6], start=1):
        assert isinstance(item, dict)
        print(
            f"  {index:02d}. {item['feature']}: "
            f"AUROC={float(item['auc']):.3f}, "
            f"direction={item['direction']}, "
            f"top12 targets={item['top12_target_count']}/12, "
            f"top20 recall={100.0 * float(item['top20_recall']):.1f}%"
        )

    print()
    print(f"JSON: {json_path}")
    print(f"CSV:  {features_path}")
    print(f"MD:   {md_path}")
    print("Prediction-geometry trigger audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
