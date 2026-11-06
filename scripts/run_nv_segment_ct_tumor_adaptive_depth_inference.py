from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    load_adaptive_protocol,
    run_adaptive_depth_comparison,
    summarize_adaptive_depth_results,
    write_adaptive_depth_outputs,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen 70%-relative-depth tumor point "
            "against the deepest-interior point on 12 stratified cases."
        )
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    return parser.parse_args()


def _print_group(
    name: str,
    group: dict[str, object],
) -> None:
    adaptive = group["adaptive"]
    deep = group["deep"]
    paired = group["paired_adaptive_minus_deep"]

    assert isinstance(adaptive, dict)
    assert isinstance(deep, dict)
    assert isinstance(paired, dict)

    adaptive_dice = adaptive["tumor_dice"]
    deep_dice = deep["tumor_dice"]
    dice_delta = paired["tumor_dice"]
    adaptive_hd95 = adaptive["hec_hd95_reduction_mm"]
    deep_hd95 = deep["hec_hd95_reduction_mm"]
    hd95_delta = paired["hec_hd95_reduction_mm"]

    assert isinstance(adaptive_dice, dict)
    assert isinstance(deep_dice, dict)
    assert isinstance(dice_delta, dict)
    assert isinstance(adaptive_hd95, dict)
    assert isinstance(deep_hd95, dict)
    assert isinstance(hd95_delta, dict)

    print(name)
    print(
        f"  adaptive tumor Dice mean/median: "
        f"{adaptive_dice['mean']:.4f} / "
        f"{adaptive_dice['median']:.4f}"
    )
    print(
        f"  deep tumor Dice mean/median: "
        f"{deep_dice['mean']:.4f} / "
        f"{deep_dice['median']:.4f}"
    )
    print(
        f"  adaptive-deep Dice delta mean: "
        f"{dice_delta['mean']:+.4f}"
    )
    print(
        f"  within ±0.05 / ±0.10 Dice: "
        f"{paired['cases_within_0_05_tumor_dice']}/"
        f"{group['count']} / "
        f"{paired['cases_within_0_10_tumor_dice']}/"
        f"{group['count']}"
    )
    print(
        f"  adaptive HD95 reduction mean: "
        f"{adaptive_hd95['mean']:+.2f} mm"
    )
    print(
        f"  deep HD95 reduction mean: "
        f"{deep_hd95['mean']:+.2f} mm"
    )
    print(
        f"  adaptive-deep HD95 reduction delta: "
        f"{hd95_delta['mean']:+.2f} mm"
    )
    print()


def main() -> int:
    args = parse_args()
    root = discover_project_root()

    protocol_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_adaptive_relative_depth"
        / "protocol.json"
    )
    output_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_adaptive_relative_depth"
        / "inference"
    )

    protocol = load_adaptive_protocol(
        protocol_path
    )
    nv_paths = runtime_paths(root)

    rows = []

    print(
        "NV-Segment-CT adaptive relative-depth "
        "tumor inference"
    )
    print("Cases: 12")
    print("Adaptive point: 70% maximum interior depth")
    print("Reference point: deepest interior")
    print("point_labels=[3], label_prompt=None")
    print("Oracle-simulated clicks; deployment-valid: no")
    print("Maximum new inferences: 24")
    print()

    for index, row in enumerate(
        protocol,
        start=1,
    ):
        print(
            f"[{index:02d}/12] {row.case_id} "
            f"({row.quartile})"
        )
        print(
            f"  tumor={row.largest_component_volume_ml:.2f} mL, "
            f"max_depth={row.max_interior_depth_mm:.2f} mm, "
            f"adaptive_depth={row.adaptive_actual_depth_mm:.2f} mm"
        )

        result = run_adaptive_depth_comparison(
            row=row,
            project_root=root,
            nv_paths=nv_paths,
            environment_name=args.environment,
            force=args.force,
        )
        rows.append(result)

        print(
            f"  adaptive: Dice="
            f"{result.adaptive.interactive_target_dice:.4f}, "
            f"coverage="
            f"{100.0 * result.adaptive.tumor_coverage_gain:+.1f} pp, "
            f"HD95 reduction="
            f"{result.adaptive.hec_hd95_reduction_mm:+.2f} mm"
        )
        print(
            f"  deep:     Dice="
            f"{result.deep.interactive_target_dice:.4f}, "
            f"coverage="
            f"{100.0 * result.deep.tumor_coverage_gain:+.1f} pp, "
            f"HD95 reduction="
            f"{result.deep.hec_hd95_reduction_mm:+.2f} mm"
        )
        print(
            f"  paired Dice delta="
            f"{result.adaptive.interactive_target_dice - result.deep.interactive_target_dice:+.4f}"
        )
        print()

    payload = summarize_adaptive_depth_results(
        rows
    )
    write_adaptive_depth_outputs(
        payload=payload,
        output_root=output_root,
    )

    overall = payload["overall"]
    by_quartile = payload["by_quartile"]

    assert isinstance(overall, dict)
    assert isinstance(by_quartile, dict)

    print("Adaptive relative-depth summary")
    _print_group("overall", overall)

    for quartile in QUARTILE_NAMES:
        group = by_quartile[quartile]
        assert isinstance(group, dict)
        _print_group(quartile, group)

    paired = overall["paired_adaptive_minus_deep"]
    assert isinstance(paired, dict)
    dice_delta = paired["tumor_dice"]
    hd95_delta = paired["hec_hd95_reduction_mm"]
    assert isinstance(dice_delta, dict)
    assert isinstance(hd95_delta, dict)

    dice_ci = dice_delta["bootstrap_mean_95_ci"]
    hd95_ci = hd95_delta["bootstrap_mean_95_ci"]
    assert isinstance(dice_ci, dict)
    assert isinstance(hd95_ci, dict)

    print("Paired bootstrap 95% CI")
    print(
        f"  adaptive-deep tumor Dice mean delta: "
        f"[{dice_ci['lower_95']:+.4f}, "
        f"{dice_ci['upper_95']:+.4f}]"
    )
    print(
        f"  adaptive-deep HD95-reduction mean delta: "
        f"[{hd95_ci['lower_95']:+.2f}, "
        f"{hd95_ci['upper_95']:+.2f}] mm"
    )
    print()

    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(
        "Adaptive relative-depth tumor inference PASSED "
        "(oracle-simulated interaction; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
