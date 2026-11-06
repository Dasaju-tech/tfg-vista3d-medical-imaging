from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.cyst_oracle_prompt import (
    load_prompt_manifest,
)
from vista3d_tfg.experiments.cyst_semantic_point import (
    run_semantic_point_case,
    summarize_semantic_point_results,
    write_semantic_point_summary,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare point-only cyst prompting against side-specific "
            "semantic class plus the same oracle point."
        )
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help="Optional case-number subset for smoke/debug execution.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    block16_root = (
        project_root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_cyst_oracle_prompt"
    )
    prompts = load_prompt_manifest(
        block16_root / "prompts.json"
    )

    if args.cases is not None:
        requested = {
            f"case_{case_number:05d}"
            for case_number in args.cases
        }
        prompts = tuple(
            prompt
            for prompt in prompts
            if prompt.case_id in requested
        )

        if not prompts:
            raise ValueError(
                "No prepared cyst prompts match requested cases."
            )

    nv_paths = runtime_paths(project_root)
    rows = []

    print("NV-Segment-CT cyst semantic + oracle-point experiment")
    print(f"Cases: {len(prompts)}")
    print(
        "Protocol: same frozen Block 16 positive point plus "
        "GT-derived side-specific cyst class (116 left / 117 right)."
    )
    print(
        "IMPORTANT: point and semantic laterality are GT-derived; "
        "this remains an oracle, non-deployment-valid experiment."
    )
    print()

    for index, prompt in enumerate(prompts, start=1):
        print(
            f"[{index:02d}/{len(prompts):02d}] "
            f"{prompt.case_id}"
        )

        result = run_semantic_point_case(
            prompt=prompt,
            project_root=project_root,
            nv_paths=nv_paths,
            environment_name=args.environment,
            force=args.force,
        )
        rows.append(result)

        print(
            f"  class={result.class_label} "
            f"({result.class_name}), side={result.laterality}"
        )
        print(
            f"  automatic side-class Dice="
            f"{result.automatic_side_class_dice:.4f}"
        )
        print(
            f"  point-only Dice={result.point_only_dice:.4f}"
        )
        print(
            f"  semantic+point Dice="
            f"{result.semantic_plus_point_dice:.4f} "
            f"(vs point-only "
            f"{result.semantic_plus_point_minus_point_only:+.4f})"
        )
        print(
            f"  labels={result.labels_present}, "
            f"contains_255={result.contains_255_unprocessed}, "
            f"time={result.wall_time_seconds:.2f} s"
        )

    summary = summarize_semantic_point_results(rows)

    print()
    print("Semantic + spatial conditioning result")
    print(
        "Automatic side-class Dice: "
        f"mean={summary.automatic_side_class_dice_mean:.4f}, "
        f"median={summary.automatic_side_class_dice_median:.4f}"
    )
    print(
        "Point-only Dice: "
        f"mean={summary.point_only_dice_mean:.4f}, "
        f"median={summary.point_only_dice_median:.4f}"
    )
    print(
        "Semantic+point Dice: "
        f"mean={summary.semantic_plus_point_dice_mean:.4f}, "
        f"median={summary.semantic_plus_point_dice_median:.4f}"
    )
    print(
        "Semantic+point minus point-only: "
        f"mean={summary.semantic_plus_point_minus_point_only_mean:+.4f}, "
        f"median="
        f"{summary.semantic_plus_point_minus_point_only_median:+.4f}, "
        "bootstrap 95% CI="
        f"["
        f"{summary.semantic_plus_point_minus_point_only_bootstrap_95ci.lower:+.4f}, "
        f"{summary.semantic_plus_point_minus_point_only_bootstrap_95ci.upper:+.4f}"
        f"]"
    )
    print(
        f"Improved over point-only: "
        f"{summary.improved_over_point_only_count}/{summary.case_count}"
    )
    print(
        f"Regressed vs point-only: "
        f"{summary.regressed_vs_point_only_count}/{summary.case_count}"
    )
    print(
        f"Semantic+point Dice >=0.50: "
        f"{summary.semantic_plus_point_dice_ge_050_count}/"
        f"{summary.case_count}"
    )
    print(
        f"Semantic+point Dice >=0.80: "
        f"{summary.semantic_plus_point_dice_ge_080_count}/"
        f"{summary.case_count}"
    )
    print(
        "Point-only zero-overlap rescued by semantic conditioning: "
        f"{summary.point_only_zero_overlap_rescued_count}/"
        f"{summary.point_only_zero_overlap_count}"
    )
    print(
        f"Laterality: left={summary.left_target_count}, "
        f"right={summary.right_target_count}"
    )

    if args.cases is None:
        output_root = (
            project_root
            / "outputs"
            / "experiments"
            / "nv_segment_ct_cyst_semantic_point"
        )
        write_semantic_point_summary(
            summary,
            output_root,
        )
        print(f"Summary: {output_root / 'summary.json'}")
        print(f"CSV:     {output_root / 'cases.csv'}")

    print(
        "Semantic+oracle-point experiment PASSED "
        "(experimental upper-bound; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
