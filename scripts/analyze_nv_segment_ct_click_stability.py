from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.interaction_stability import (
    load_click_curve_summary,
    render_stability_figures,
    summarize_interaction_stability,
    write_stability_outputs,
)


def format_group(name: str, group) -> None:
    if group is None:
        print(f"{name}: no observations")
        return

    print(
        f"{name}: n={group.count}, "
        f"mean delta={group.mean_delta:+.4f}, "
        f"median delta={group.median_delta:+.4f}, "
        f"improved={group.improved_count}/{group.count}, "
        f"regressed={group.regressed_count}/{group.count}"
    )


def main() -> int:
    project_root = discover_project_root()

    click_curve_root = (
        project_root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_cyst_oracle_click_curve"
    )

    trajectories = load_click_curve_summary(
        click_curve_root / "summary.json"
    )
    summary = summarize_interaction_stability(trajectories)

    output_root = (
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_cyst_interaction_stability"
    )

    write_stability_outputs(summary, output_root)
    figures = render_stability_figures(
        summary,
        output_root / "figures",
    )

    print("NV-Segment-CT cyst interaction stability audit")
    print(f"Cases: {summary.case_count}")
    print()
    print(
        f"1 click: mean={summary.click1_mean:.4f}, "
        f"median={summary.click1_median:.4f}"
    )
    print(
        f"2 clicks: mean={summary.click2_mean:.4f}, "
        f"median={summary.click2_median:.4f}"
    )
    print(
        f"3 clicks: mean={summary.click3_mean:.4f}, "
        f"median={summary.click3_median:.4f}"
    )
    print()
    print(
        f"Oracle best observed round: "
        f"mean={summary.best_observed_mean:.4f}, "
        f"median={summary.best_observed_median:.4f}"
    )
    print(
        f"Oracle-best gain over click 1: "
        f"mean={summary.best_minus_click1_mean:+.4f}, "
        f"median={summary.best_minus_click1_median:+.4f}, "
        "bootstrap 95% CI="
        f"[{summary.best_minus_click1_bootstrap_95ci.lower:+.4f}, "
        f"{summary.best_minus_click1_bootstrap_95ci.upper:+.4f}]"
    )
    print(
        "Best round distribution: "
        f"click1={summary.best_round_click1_count}, "
        f"click2={summary.best_round_click2_count}, "
        f"click3={summary.best_round_click3_count}"
    )
    print(
        f"Oracle-best Dice >=0.50: "
        f"{summary.best_observed_dice_ge_050_count}/{summary.case_count}"
    )
    print(
        f"Oracle-best Dice >=0.80: "
        f"{summary.best_observed_dice_ge_080_count}/{summary.case_count}"
    )

    print()
    print(
        f"Final-round regret vs best: "
        f"mean={summary.final_regret_vs_best_mean:.4f}, "
        f"median={summary.final_regret_vs_best_median:.4f}"
    )
    print(
        f"Regret >=0.05: "
        f"{summary.final_regret_ge_005_count}/{summary.case_count}"
    )
    print(
        f"Regret >=0.10: "
        f"{summary.final_regret_ge_010_count}/{summary.case_count}"
    )

    print()
    format_group("Positive corrections", summary.positive_correction)
    format_group("Negative corrections", summary.negative_correction)
    format_group("Click 1->2", summary.click2_transition)
    format_group("Click 2->3", summary.click3_transition)

    print()
    print(
        "Largest regressions: "
        + ", ".join(summary.largest_regression_case_ids)
    )
    print(
        "Largest improvements: "
        + ", ".join(summary.largest_improvement_case_ids)
    )

    print()
    print(f"JSON: {output_root / 'stability_summary.json'}")
    print(f"CSV:  {output_root / 'transitions.csv'}")
    for figure in figures:
        print(f"Figure: {figure}")

    print("Cyst interaction stability audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
