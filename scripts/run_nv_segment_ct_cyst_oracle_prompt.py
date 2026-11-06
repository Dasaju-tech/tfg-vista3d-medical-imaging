from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.cyst_oracle_prompt import (
    load_prompt_manifest,
    prepare_oracle_prompts,
    run_interactive_case,
    summarize_interactive_results,
    write_interactive_summary,
    write_prompt_manifest,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled single-positive-click oracle localization "
            "experiment on the largest cyst component of each cyst-positive "
            "pilot30 case."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "all"),
        default="all",
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
        help=(
            "Optional case-number subset for smoke/debug execution. "
            "The frozen prompt manifest remains the source of prompts."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    output_root = (
        project_root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_cyst_oracle_prompt"
    )
    prompt_manifest = output_root / "prompts.json"

    if args.stage in {"prepare", "all"}:
        prompts = prepare_oracle_prompts(project_root)

        write_prompt_manifest(
            prompts,
            prompt_manifest,
        )

        print("Cyst oracle prompt manifest prepared")
        print(f"Cyst-positive cases: {len(prompts)}")
        print(f"Manifest: {prompt_manifest}")
        print(
            "IMPORTANT: prompts are GT-derived and define an oracle "
            "localization experiment, not a deployable protocol."
        )

        for prompt in prompts:
            print(
                f"  {prompt.case_id} "
                f"point={prompt.point_xyz} "
                f"target={prompt.target_volume_ml:.2f} mL "
                f"interior_radius={prompt.interior_radius_mm:.2f} mm"
            )

        if args.stage == "prepare":
            print("Oracle prompt preparation PASSED.")
            return 0

    prompts = load_prompt_manifest(prompt_manifest)

    if args.cases is not None:
        wanted = {
            f"case_{case_number:05d}"
            for case_number in args.cases
        }
        prompts = tuple(
            prompt for prompt in prompts if prompt.case_id in wanted
        )

        if not prompts:
            raise ValueError(
                "Requested cases contain no prepared cyst-positive prompts."
            )

    nv_paths = runtime_paths(project_root)
    rows = []

    print()
    print("NV-Segment-CT cyst oracle point-prompt experiment")
    print(f"Cases to run: {len(prompts)}")

    for index, prompt in enumerate(prompts, start=1):
        print(
            f"[{index:02d}/{len(prompts):02d}] "
            f"{prompt.case_id} point={prompt.point_xyz}"
        )

        result = run_interactive_case(
            prompt=prompt,
            project_root=project_root,
            nv_paths=nv_paths,
            environment_name=args.environment,
            force=args.force,
        )
        rows.append(result)

        print(
            f"  automatic target Dice={result.automatic_target_dice:.4f}"
        )
        print(
            f"  interactive target Dice={result.interactive_target_dice:.4f} "
            f"(delta={result.dice_delta:+.4f})"
        )
        print(
            f"  labels={result.interactive_labels_present}, "
            f"contains_255={result.contains_255_unprocessed}, "
            f"time={result.wall_time_seconds:.2f} s"
        )

    summary = summarize_interactive_results(rows)

    # Only write the registered full-cohort summary when no debug subset was
    # requested. Subset runs retain per-case results and can be resumed later.
    if args.cases is None:
        write_interactive_summary(summary, output_root)

    print()
    print("Oracle point-prompt result")
    print(
        f"Automatic target Dice: "
        f"mean={summary.automatic_target_dice_mean:.4f}, "
        f"median={summary.automatic_target_dice_median:.4f}"
    )
    print(
        f"Interactive target Dice: "
        f"mean={summary.interactive_target_dice_mean:.4f}, "
        f"median={summary.interactive_target_dice_median:.4f}"
    )
    print(
        f"Dice delta: mean={summary.dice_delta_mean:+.4f}, "
        f"median={summary.dice_delta_median:+.4f}"
    )
    print(
        f"Improved cases: {summary.improved_case_count}/"
        f"{summary.case_count}"
    )
    print(
        f"Interactive Dice >=0.50: "
        f"{summary.interactive_dice_ge_050_count}/{summary.case_count}"
    )
    print(
        f"Interactive Dice >=0.80: "
        f"{summary.interactive_dice_ge_080_count}/{summary.case_count}"
    )
    print(
        "Automatic zero-overlap rescued: "
        f"{summary.zero_overlap_rescued_case_count}/"
        f"{summary.automatic_zero_overlap_case_count}"
    )

    if args.cases is None:
        print(f"Summary: {output_root / 'summary.json'}")
        print(f"CSV:     {output_root / 'cases.csv'}")

    print(
        "Oracle cyst point-prompt experiment PASSED "
        "(experimental upper-bound; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
