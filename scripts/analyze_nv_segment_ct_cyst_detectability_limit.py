from __future__ import annotations

from vista3d_tfg.analysis.cyst_detectability_limit import (
    load_block50_cases,
    summarize_cyst_detectability,
    write_detectability_outputs,
    write_detectability_report,
)
from vista3d_tfg.config import discover_project_root


def _print_predictor(
    name: str,
    payload: dict[str, object],
) -> None:
    ci = payload["bootstrap_95_ci"]
    threshold = payload[
        "best_descriptive_threshold"
    ]
    transition = payload[
        "empirical_transition"
    ]

    assert isinstance(ci, dict)
    assert isinstance(threshold, dict)
    assert isinstance(transition, dict)

    print(name)
    print(
        f"  volume AUROC: "
        f"{payload['auroc']:.3f}"
    )
    print(
        f"  bootstrap 95% CI: "
        f"[{ci['lower_95']:.3f}, "
        f"{ci['upper_95']:.3f}]"
    )
    print(
        f"  best descriptive threshold: "
        f"{threshold['threshold_ml']:.2f} mL"
    )
    print(
        f"  balanced accuracy: "
        f"{threshold['balanced_accuracy']:.3f}"
    )
    print(
        f"  TP/FP/TN/FN: "
        f"{threshold['tp']}/"
        f"{threshold['fp']}/"
        f"{threshold['tn']}/"
        f"{threshold['fn']}"
    )
    print(
        f"  largest negative / smallest positive: "
        f"{transition['largest_negative_ml']} / "
        f"{transition['smallest_positive_ml']} mL"
    )
    print(
        f"  perfectly separable by volume: "
        f"{transition['separable']}"
    )
    print()


def main() -> int:
    root = discover_project_root()

    block50_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_cyst_failures"
        / "summary.json"
    )

    cases = load_block50_cases(
        block50_path
    )
    payload = summarize_cyst_detectability(
        cases
    )

    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_cyst_detectability_limit"
    )
    write_detectability_outputs(
        payload=payload,
        output_root=output_root,
    )
    write_detectability_report(
        payload=payload,
        path=output_root
        / "cyst_detectability_report.md",
    )

    events = payload["event_counts"]
    predictors = payload[
        "volume_as_predictor"
    ]

    print("NV-Segment-CT cyst detectability limit audit")
    print("Cyst-positive cases: 46")
    print("Source: frozen Block 50 case metrics")
    print("GPU inference: none")
    print("Predictions modified: no")
    print(
        "Thresholds: descriptive in-sample only; "
        "not clinical cutoffs"
    )
    print()

    print("Events")
    print(
        f"  any overlap: "
        f"{events['any_overlap']}/46"
    )
    print(
        f"  zero overlap: "
        f"{events['zero_overlap']}/46"
    )
    print(
        f"  Dice >=0.50: "
        f"{events['dice_ge_050']}/46"
    )
    print()

    _print_predictor(
        "Any-overlap detectability",
        predictors["any_overlap"],
    )
    _print_predictor(
        "Dice >=0.50",
        predictors["dice_ge_050"],
    )

    print("Volume-response bins")
    for row in payload[
        "volume_response_bins"
    ]:
        print(
            f"  bin {row['bin']}: "
            f"N={row['count']}, "
            f"median={row['volume_median_ml']:.2f} mL, "
            f"overlap={100.0 * row['any_overlap_fraction']:.0f}%, "
            f"Dice>=.50={100.0 * row['dice_ge_050_fraction']:.0f}%, "
            f"Dice={row['dice_mean']:.3f}, "
            f"recall={row['recall_mean']:.3f}"
        )

    print()
    print(
        f"JSON: {output_root / 'summary.json'}"
    )
    print(
        f"CSV:  "
        f"{output_root / 'volume_response_bins.csv'}"
    )
    print(
        f"MD:   "
        f"{output_root / 'cyst_detectability_report.md'}"
    )
    print(
        "Cyst detectability limit audit PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
