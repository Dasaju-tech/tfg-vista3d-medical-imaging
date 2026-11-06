from __future__ import annotations

from vista3d_tfg.analysis.final_experimental_synthesis import (
    load_and_synthesize,
    write_outputs,
    write_report,
)
from vista3d_tfg.config import discover_project_root


def main() -> int:
    root = discover_project_root()

    payload = load_and_synthesize(
        project_root=root,
    )

    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_final_experimental_synthesis"
    )

    write_outputs(
        payload=payload,
        output_root=output_root,
    )
    write_report(
        payload=payload,
        path=output_root
        / "final_experimental_report.md",
    )

    baseline = payload[
        "automatic_baseline"
    ]
    tail = payload["hd95_tail"]
    interaction = payload[
        "tumor_interaction"
    ]
    manual = payload[
        "manual_ct_only_localization"
    ]
    cyst = payload["cyst_branch"]

    print("NV-Segment-CT final experimental synthesis")
    print("GPU inference: none")
    print()

    print("Automatic expanded100")
    print(
        f"  Dice: "
        f"{baseline['kidney_and_mass_dice_mean']:.4f} "
        f"± {baseline['kidney_and_mass_dice_sd']:.4f}"
    )
    print(
        f"  Surface Dice: "
        f"{baseline['surface_dice_mean']:.4f}"
    )
    print(
        f"  HD95 median/mean: "
        f"{baseline['hd95_median_mm']:.2f} / "
        f"{baseline['hd95_mean_mm']:.2f} mm"
    )
    print()

    print("HD95 tail")
    print(
        f"  outliers: "
        f"{tail['outlier_count']}/100"
    )
    print(
        f"  mechanism split P->GT / GT->P / balanced: "
        f"{tail['prediction_to_gt_dominant_count']} / "
        f"{tail['gt_to_prediction_dominant_count']} / "
        f"{tail['balanced_count']}"
    )
    print()

    print("Tumor interaction")
    print(
        f"  oracle Dice mean: "
        f"{interaction['oracle_tumor_dice_mean']:.4f}"
    )
    print(
        f"  oracle HD95 reduction: "
        f"{interaction['oracle_hec_hd95_reduction_mean_mm']:+.2f} mm"
    )
    print(
        f"  oracle prompt engineering: "
        f"{interaction['oracle_prompt_engineering_status']}"
    )
    print()

    print("Manual CT-only localization")
    print(
        f"  tumor hits: "
        f"{manual['any_tumor_hit_count']}/"
        f"{manual['case_count']}"
    )
    print(
        f"  manual/deep Dice: "
        f"{manual['manual_tumor_dice_mean']:.4f} / "
        f"{manual['deep_tumor_dice_mean']:.4f}"
    )
    print(
        f"  geometry contract valid: "
        f"{manual['geometry_contract_valid']}"
    )
    print()

    print("Cyst branch")
    print(
        f"  Dice mean/median: "
        f"{cyst['dice_mean']:.4f} / "
        f"{cyst['dice_median']:.4f}"
    )
    print(
        f"  FN-dominant: "
        f"{cyst['fn_dominant_count']}/"
        f"{cyst['case_count']}"
    )
    print(
        f"  volume AUROC any-overlap / Dice>=.50: "
        f"{cyst['any_overlap_volume_auroc']:.3f} / "
        f"{cyst['dice_ge_050_volume_auroc']:.3f}"
    )
    print()

    print("Final conclusion")
    print(
        f"  {payload['final_experimental_conclusion']}"
    )
    print()

    print(
        f"JSON: {output_root / 'summary.json'}"
    )
    print(
        f"CSV:  {output_root / 'claim_matrix.csv'}"
    )
    print(
        f"MD:   {output_root / 'final_experimental_report.md'}"
    )
    print(
        "Final experimental synthesis PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
