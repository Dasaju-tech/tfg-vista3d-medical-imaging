from __future__ import annotations

from vista3d_tfg.analysis.interaction_study_synthesis import (
    load_interaction_sources,
    synthesize_interaction_study,
    write_markdown_report,
    write_synthesis_outputs,
)
from vista3d_tfg.config import discover_project_root


def main() -> int:
    root = discover_project_root()

    sources = load_interaction_sources(
        project_root=root,
    )
    payload = synthesize_interaction_study(
        sources
    )

    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_tumor_interaction_study"
    )

    write_synthesis_outputs(
        payload=payload,
        output_root=output_root,
    )
    write_markdown_report(
        payload=payload,
        path=output_root / "interaction_study.md",
    )

    key = payload["key_results"]
    final_decision = payload["final_decision"]

    print("NV-Segment-CT tumor interaction study synthesis")
    print("Blocks: 41C, 42, 43, 44, 45, 46B, 47B")
    print("GPU inference: none")
    print()

    oracle = key["oracle_cohort"]
    click = key["click_robustness"]
    safe = key["safe_region_generalization"]
    relative = key["relative_depth"]
    dual = key["dual_point"]

    print("Capability")
    print(
        f"  oracle tumor Dice mean: "
        f"{oracle['tumor_dice_mean']:.4f}"
    )
    print(
        f"  oracle HEC HD95 reduction mean: "
        f"{oracle['hec_hd95_reduction_mean_mm']:+.2f} mm"
    )
    print(
        f"  HD95 improved: "
        f"{oracle['hd95_improved_count']}/9"
    )
    print()

    print("Click-location sensitivity")
    print(
        f"  deep / centroid / 5mm Dice: "
        f"{click['deep_tumor_dice_mean']:.4f} / "
        f"{click['centroid_tumor_dice_mean']:.4f} / "
        f"{click['shallow_5mm_tumor_dice_mean']:.4f}"
    )
    print()

    print("Generalization")
    print(
        f"  15mm safe region: "
        f"{safe['15mm_nonempty_count']}/"
        f"{safe['tumor_positive_count']}"
    )
    print(
        f"  smallest quartile 15mm: "
        f"{safe['smallest_quartile_15mm_nonempty_count']}/"
        f"{safe['smallest_quartile_count']}"
    )
    print()

    print("Alternative interaction rules")
    print(
        f"  70% relative-depth paired Dice delta: "
        f"{relative['paired_tumor_dice_delta_mean']:+.4f}"
    )
    print(
        f"  70% upper-middle delta: "
        f"{relative['upper_middle_delta_mean']:+.4f}"
    )
    print(
        f"  dual-point paired Dice delta: "
        f"{dual['paired_tumor_dice_delta_mean']:+.4f}"
    )
    print(
        f"  dual-point upper-middle delta: "
        f"{dual['upper_middle_delta_mean']:+.4f}"
    )
    print()

    print("Decision")
    print(
        "  oracle prompt engineering: "
        f"{final_decision['oracle_prompt_engineering_status']}"
    )
    print(
        "  preferred concept: "
        f"{final_decision['preferred_interaction_concept']}"
    )
    print(
        "  next validation: "
        f"{final_decision['next_validation_priority']}"
    )
    print()

    print(
        f"JSON: "
        f"{output_root / 'interaction_study_summary.json'}"
    )
    print(
        f"CSV:  "
        f"{output_root / 'decision_matrix.csv'}"
    )
    print(
        f"MD:   "
        f"{output_root / 'interaction_study.md'}"
    )
    print(
        "Tumor interaction study synthesis PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
