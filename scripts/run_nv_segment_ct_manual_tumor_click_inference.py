from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.manual_tumor_click_inference import (
    load_frozen_manual_clicks,
    run_manual_vs_deep,
    summarize_manual_click_results,
    write_manual_click_outputs,
    write_manual_click_report,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    load_adaptive_protocol,
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
            "Run the frozen CT-only manual tumor clicks and compare "
            "them with the existing deepest-interior oracle reference."
        )
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun manual-click inference; deep references remain cached.",
    )
    return parser.parse_args()


def _print_group(
    name: str,
    group: dict[str, object],
) -> None:
    manual = group["manual"]
    deep = group["deep"]
    paired = group["paired_manual_minus_deep"]

    assert isinstance(manual, dict)
    assert isinstance(deep, dict)
    assert isinstance(paired, dict)

    manual_dice = manual["tumor_dice"]
    deep_dice = deep["tumor_dice"]
    delta = paired["tumor_dice"]

    assert isinstance(manual_dice, dict)
    assert isinstance(deep_dice, dict)
    assert isinstance(delta, dict)

    print(name)
    print(
        f"  manual tumor Dice mean/median: "
        f"{manual_dice['mean']:.4f} / "
        f"{manual_dice['median']:.4f}"
    )
    print(
        f"  deep tumor Dice mean/median: "
        f"{deep_dice['mean']:.4f} / "
        f"{deep_dice['median']:.4f}"
    )
    print(
        f"  manual-deep Dice delta mean: "
        f"{delta['mean']:+.4f}"
    )
    print(
        f"  within ±0.05 / ±0.10 Dice: "
        f"{paired['cases_within_0_05_tumor_dice']}/"
        f"{group['count']} / "
        f"{paired['cases_within_0_10_tumor_dice']}/"
        f"{group['count']}"
    )
    print(
        f"  manual HD95 reduction mean: "
        f"{manual['hec_hd95_reduction_mm']['mean']:+.2f} mm"
    )
    print(
        f"  deep HD95 reduction mean: "
        f"{deep['hec_hd95_reduction_mm']['mean']:+.2f} mm"
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
    clicks_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_manual_tumor_clicks"
        / "clicks"
    )
    output_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_manual_tumor_clicks"
        / "inference"
    )

    protocol = load_adaptive_protocol(
        protocol_path
    )
    pairs = load_frozen_manual_clicks(
        clicks_root=clicks_root,
        protocol=protocol,
    )

    nv_paths = runtime_paths(root)
    rows = []

    print("NV-Segment-CT manual CT-only tumor-click inference")
    print("Frozen cases: 12")
    print("Manual clicks: canonical-RAS CT display; GT hidden")
    print("Inference point_labels=[3], label_prompt=None")
    print("Deep references: reused from Block 46B")
    print("New GPU inferences: 12")
    print("Localization audit: post-hoc only")
    print("Study type: simulated-user localization, not clinician validation")
    print()

    for index, (case, click) in enumerate(
        pairs,
        start=1,
    ):
        print(
            f"[{index:02d}/12] {case.case_id} "
            f"({case.quartile})"
        )
        print(
            f"  manual point original={click.point_xyz}, "
            f"display_RAS={click.display_point_ras_xyz}"
        )

        result = run_manual_vs_deep(
            case=case,
            click=click,
            project_root=root,
            nv_paths=nv_paths,
            environment_name=args.environment,
            force=args.force,
        )
        rows.append(result)

        localization = result.localization

        if localization.inside_target_component:
            status = "TARGET HIT"
        elif localization.inside_any_tumor:
            status = "OTHER TUMOR HIT"
        else:
            status = (
                "MISS "
                f"(distance={localization.distance_to_any_tumor_mm:.2f} mm)"
            )

        print(
            f"  post-hoc localization: {status}, "
            f"GT label={localization.gt_label_at_point}"
        )
        print(
            f"  manual: Dice="
            f"{result.manual.interactive_target_dice:.4f}, "
            f"coverage="
            f"{100.0 * result.manual.tumor_coverage_gain:+.1f} pp, "
            f"HD95 reduction="
            f"{result.manual.hec_hd95_reduction_mm:+.2f} mm"
        )
        print(
            f"  deep:   Dice="
            f"{result.deep.interactive_target_dice:.4f}, "
            f"coverage="
            f"{100.0 * result.deep.tumor_coverage_gain:+.1f} pp, "
            f"HD95 reduction="
            f"{result.deep.hec_hd95_reduction_mm:+.2f} mm"
        )
        print(
            f"  paired Dice delta="
            f"{result.manual.interactive_target_dice - result.deep.interactive_target_dice:+.4f}"
        )
        print()

    payload = summarize_manual_click_results(
        rows
    )
    write_manual_click_outputs(
        payload=payload,
        output_root=output_root,
    )
    write_manual_click_report(
        payload=payload,
        path=output_root / "manual_click_report.md",
    )

    localization = payload["localization_audit"]
    overall = payload["overall"]
    by_quartile = payload["by_quartile"]

    assert isinstance(localization, dict)
    assert isinstance(overall, dict)
    assert isinstance(by_quartile, dict)

    print("Manual CT-only localization summary")
    print(
        f"  target-component hits: "
        f"{localization['target_component_hit_count']}/12"
    )
    print(
        f"  any-tumor hits: "
        f"{localization['any_tumor_hit_count']}/12"
    )
    print(
        f"  misses: "
        f"{localization['miss_count']}/12"
    )
    print(
        "  distance-to-target vs manual Dice Pearson: "
        f"{localization['distance_to_target_vs_manual_dice_pearson']:+.3f}"
    )
    print()

    print("Manual CT-only segmentation summary")
    _print_group("overall", overall)

    for quartile in QUARTILE_NAMES:
        group = by_quartile[quartile]
        assert isinstance(group, dict)
        _print_group(quartile, group)

    paired = overall["paired_manual_minus_deep"]
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
        f"  manual-deep tumor Dice mean delta: "
        f"[{dice_ci['lower_95']:+.4f}, "
        f"{dice_ci['upper_95']:+.4f}]"
    )
    print(
        f"  manual-deep HD95-reduction mean delta: "
        f"[{hd95_ci['lower_95']:+.2f}, "
        f"{hd95_ci['upper_95']:+.2f}] mm"
    )
    print()

    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(f"MD:   {output_root / 'manual_click_report.md'}")
    print(
        "Manual CT-only tumor-click inference PASSED "
        "(simulated-user localization; not clinician validation)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
