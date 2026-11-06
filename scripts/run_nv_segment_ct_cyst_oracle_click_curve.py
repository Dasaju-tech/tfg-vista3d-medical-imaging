from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.cyst_oracle_click_curve import (
    render_click_curve,
    run_oracle_click_trajectory,
    summarize_click_curve,
    write_click_curve_summary,
)
from vista3d_tfg.experiments.cyst_oracle_prompt import load_prompt_manifest
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic iterative oracle correction experiment "
            "with cumulative positive/negative point clicks."
        )
    )
    parser.add_argument(
        "--maximum-clicks",
        type=int,
        default=3,
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
    prompts = load_prompt_manifest(block16_root / "prompts.json")

    if args.cases is not None:
        requested = {
            f"case_{case_number:05d}"
            for case_number in args.cases
        }
        prompts = tuple(
            prompt for prompt in prompts if prompt.case_id in requested
        )

        if not prompts:
            raise ValueError("No prepared prompts match requested cases.")

    nv_paths = runtime_paths(project_root)
    trajectories = []

    print("NV-Segment-CT iterative cyst oracle click curve")
    print(f"Cases: {len(prompts)}")
    print(f"Maximum cumulative clicks: {args.maximum_clicks}")
    print(
        "Protocol: click 1 = deepest GT interior point; later clicks = "
        "largest remaining FN (positive) or FP (negative) component."
    )
    print(
        "IMPORTANT: every correction uses GT and is an oracle upper-bound "
        "experiment, not a deployment-valid protocol."
    )
    print()

    for index, prompt in enumerate(prompts, start=1):
        print(f"[{index:02d}/{len(prompts):02d}] {prompt.case_id}")

        trajectory = run_oracle_click_trajectory(
            prompt=prompt,
            project_root=project_root,
            nv_paths=nv_paths,
            maximum_clicks=args.maximum_clicks,
            environment_name=args.environment,
            force=args.force,
        )
        trajectories.append(trajectory)

        for round_result in trajectory.rounds:
            print(
                f"  clicks={round_result.click_count}: "
                f"Dice={round_result.dice:.4f}, "
                f"time={round_result.wall_time_seconds:.2f} s"
            )

        for click in trajectory.clicks[1:]:
            print(
                f"    correction {click.index}: "
                f"label={click.point_label}, "
                f"point={click.point_xyz}, "
                f"source={click.source}"
            )

    summary = summarize_click_curve(
        trajectories,
        maximum_clicks=args.maximum_clicks,
    )

    print()
    print("Oracle click-response curve")

    for round_summary in summary.round_summaries:
        print(
            f"  {round_summary.click_count} click(s): "
            f"mean={round_summary.dice_mean:.4f}, "
            f"median={round_summary.dice_median:.4f}, "
            f">=0.50={round_summary.dice_ge_050_count}/"
            f"{round_summary.case_count}, "
            f">=0.80={round_summary.dice_ge_080_count}/"
            f"{round_summary.case_count}"
        )

    print()
    print(
        "Final vs click 1: "
        f"mean delta={summary.mean_delta_click1_to_final:+.4f}, "
        f"median delta={summary.median_delta_click1_to_final:+.4f}"
    )
    print(
        f"Improved from click 1: "
        f"{summary.improved_from_click1_count}/{summary.case_count}"
    )
    print(
        f"Regressed from click 1: "
        f"{summary.regressed_from_click1_count}/{summary.case_count}"
    )
    print(
        "Automatic zero-overlap rescued by final round: "
        f"{summary.automatic_zero_overlap_rescued_by_final_count}/"
        f"{summary.automatic_zero_overlap_case_count}"
    )

    if args.cases is None:
        output_root = (
            project_root
            / "outputs"
            / "experiments"
            / "nv_segment_ct_cyst_oracle_click_curve"
        )
        write_click_curve_summary(summary, output_root)
        render_click_curve(
            summary,
            output_root / "dice_vs_click_count.png",
        )
        print(f"Summary: {output_root / 'summary.json'}")
        print(f"CSV:     {output_root / 'trajectories.csv'}")
        print(f"Figure:  {output_root / 'dice_vs_click_count.png'}")

    print(
        "Iterative oracle click-curve experiment PASSED "
        "(experimental upper-bound; not deployment-valid)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
