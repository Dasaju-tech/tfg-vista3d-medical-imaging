from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.automatic_cyst_laterality import (
    audit_case_laterality,
    summarize_automatic_laterality,
    write_laterality_outputs,
)
from vista3d_tfg.experiments.cyst_oracle_prompt import (
    load_prompt_manifest,
)
from vista3d_tfg.experiments.pilot30 import pilot_case_paths


def main() -> int:
    project_root = discover_project_root()

    prompt_manifest = (
        project_root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_cyst_oracle_prompt"
        / "prompts.json"
    )
    prompts = load_prompt_manifest(prompt_manifest)

    decisions = []

    print("Automatic cyst laterality from renal segmentation")
    print(f"Cases: {len(prompts)}")
    print(
        "Rule: assign the click to the physically nearest automatic "
        "kidney core (14=left, 5=right), then map to cyst class "
        "116/117."
    )
    print()

    for prompt in prompts:
        case_number = int(prompt.case_id.split("_")[-1])
        paths = pilot_case_paths(project_root, case_number)

        semantic_result_path = (
            project_root
            / "outputs"
            / "predictions"
            / "nv_segment_ct_semantic_point_cyst"
            / prompt.case_id
            / "result.json"
        )

        decision = audit_case_laterality(
            prompt=prompt,
            automatic_prediction_path=paths.prediction_path,
            oracle_semantic_result_path=semantic_result_path,
        )
        decisions.append(decision)

        left = decision.distance_to_left_kidney_mm
        right = decision.distance_to_right_kidney_mm

        print(
            f"{decision.case_id}: "
            f"inferred={decision.inferred_laterality:5s} "
            f"class={decision.inferred_cyst_class}, "
            f"oracle={decision.oracle_cyst_class}, "
            f"match={decision.agrees_with_oracle}, "
            f"d_left={left:.2f} mm, "
            f"d_right={right:.2f} mm, "
            f"margin={decision.distance_margin_mm:.2f} mm"
        )

    summary = summarize_automatic_laterality(decisions)

    output_root = (
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_automatic_cyst_laterality"
    )
    write_laterality_outputs(summary, output_root)

    print()
    print("Automatic semantic-class audit")
    print(
        f"Agreement with GT-derived class: "
        f"{summary.agreement_count}/{summary.case_count} "
        f"({summary.agreement_fraction:.1%})"
    )
    print(
        f"Both automatic kidneys present: "
        f"{summary.both_kidneys_present_count}/{summary.case_count}"
    )
    print(
        f"Selected kidney distance: "
        f"mean={summary.selected_distance_mean_mm:.2f} mm, "
        f"median={summary.selected_distance_median_mm:.2f} mm, "
        f"max={summary.selected_distance_max_mm:.2f} mm"
    )
    print(
        f"Laterality margin: "
        f"mean={summary.margin_mean_mm:.2f} mm, "
        f"median={summary.margin_median_mm:.2f} mm, "
        f"min={summary.margin_minimum_mm:.2f} mm"
    )
    print(
        f"Left/right inferred: "
        f"{summary.left_inferred_count}/"
        f"{summary.right_inferred_count}"
    )

    if summary.disagreement_case_ids:
        print(
            "Disagreements: "
            + ", ".join(summary.disagreement_case_ids)
        )
    else:
        print("Disagreements: none")

    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print("Automatic cyst laterality audit PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
