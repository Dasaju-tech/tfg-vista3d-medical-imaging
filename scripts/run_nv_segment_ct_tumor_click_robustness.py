from __future__ import annotations

import argparse
import json

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickRobustnessCase,
    derive_click_variants,
    load_all_frozen_prompts,
    run_click_variant,
    summarize_click_robustness,
    write_click_robustness_outputs,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate tumor special-point robustness to less ideal "
            "GT-simulated clicks."
        )
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = discover_project_root()

    protocol_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_oracle_point"
        / "protocol.json"
    )
    output_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_click_robustness"
    )
    dataset_root = root / "data" / "raw" / "kits23"

    prompts = load_all_frozen_prompts(protocol_path)
    nv_paths = runtime_paths(root)

    rows = []

    print("NV-Segment-CT tumor click robustness")
    print("Cases: 9")
    print("Variants: deep / centroid / shallow_5mm")
    print("Deep reuses Block 41C.")
    print("Centroid and shallow_5mm are GT-simulated clicks.")
    print("Deployment-valid: no")
    print()

    for case_index, prompt in enumerate(prompts, start=1):
        gt_path = (
            dataset_root
            / prompt.case_id
            / "segmentation.nii.gz"
        )
        variants = derive_click_variants(
            prompt=prompt,
            ground_truth_path=gt_path,
        )

        print(
            f"[{case_index:02d}/09] {prompt.case_id}"
        )

        results = {}
        for variant in variants:
            print(
                f"  {variant.variant}: "
                f"point={variant.point_xyz}, "
                f"depth={variant.interior_radius_mm:.2f} mm"
            )
            result = run_click_variant(
                base_prompt=prompt,
                variant=variant,
                project_root=root,
                nv_paths=nv_paths,
                environment_name=args.environment,
                force=args.force,
            )
            results[variant.variant] = result
            print(
                f"    tumor Dice={result.interactive_target_dice:.4f}, "
                f"coverage gain="
                f"{100.0 * result.tumor_coverage_gain:+.1f} pp, "
                f"HD95 reduction="
                f"{result.hec_hd95_reduction_mm:+.2f} mm"
            )

        rows.append(
            ClickRobustnessCase(
                case_id=prompt.case_id,
                deep=results["deep"],
                centroid=results["centroid"],
                shallow_5mm=results["shallow_5mm"],
            )
        )

    payload = summarize_click_robustness(rows)
    write_click_robustness_outputs(
        payload=payload,
        output_root=output_root,
    )

    variants = payload["variants"]
    robustness = payload["robustness_vs_deep"]
    assert isinstance(variants, dict)
    assert isinstance(robustness, dict)

    print()
    print("Tumor click robustness summary")
    for name in ("deep", "centroid", "shallow_5mm"):
        item = variants[name]
        assert isinstance(item, dict)
        print(name)
        print(
            f"  tumor Dice mean/median: "
            f"{item['tumor_dice_mean']:.4f} / "
            f"{item['tumor_dice_median']:.4f}"
        )
        print(
            f"  Dice >=0.50: {item['tumor_dice_ge_050']}/9"
        )
        print(
            f"  tumor coverage gain mean: "
            f"{100.0 * item['tumor_coverage_gain_mean']:+.1f} pp"
        )
        print(
            f"  HEC HD95 reduction mean: "
            f"{item['hec_hd95_reduction_mean_mm']:+.2f} mm"
        )
        print(
            f"  HD95 improved: "
            f"{item['hd95_improved_count']}/9"
        )

    print()
    print("Robustness relative to deep oracle")
    print(
        "  centroid tumor-Dice delta mean: "
        f"{robustness['centroid_tumor_dice_delta_mean']:+.4f}"
    )
    print(
        "  centroid within ±0.05 Dice: "
        f"{robustness['centroid_cases_within_0_05_dice']}/9"
    )
    print(
        "  shallow_5mm tumor-Dice delta mean: "
        f"{robustness['shallow_5mm_tumor_dice_delta_mean']:+.4f}"
    )
    print(
        "  shallow_5mm within ±0.05 Dice: "
        f"{robustness['shallow_5mm_cases_within_0_05_dice']}/9"
    )
    print()
    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(
        "Tumor click robustness PASSED "
        "(GT-simulated interaction; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
