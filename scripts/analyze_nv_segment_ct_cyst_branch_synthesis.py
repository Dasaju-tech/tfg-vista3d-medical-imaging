from __future__ import annotations

from vista3d_tfg.analysis.cyst_branch_synthesis import (
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
        / "nv_segment_ct_cyst_branch_synthesis"
    )

    write_outputs(
        payload=payload,
        output_root=output_root,
    )
    write_report(
        payload=payload,
        path=output_root
        / "cyst_branch_report.md",
    )

    key = payload["key_results"]
    final = payload["final_decision"]

    print("NV-Segment-CT cyst branch synthesis")
    print("Source blocks: 50, 51")
    print("GPU inference: none")
    print()

    print("Overall")
    print(
        f"  cyst-positive N: "
        f"{payload['case_count']}"
    )
    print(
        f"  Dice mean/median: "
        f"{key['dice_mean']:.4f} / "
        f"{key['dice_median']:.4f}"
    )
    print(
        f"  FN/FP dominant: "
        f"{key['fn_dominant_count']} / "
        f"{key['fp_dominant_count']}"
    )
    print(
        f"  zero Dice: "
        f"{key['zero_dice_count']}/"
        f"{payload['case_count']}"
    )
    print()

    print("Size dependence")
    print(
        f"  volume vs Dice Pearson: "
        f"{key['volume_vs_dice_pearson']:+.3f}"
    )
    print(
        f"  volume vs recall Pearson: "
        f"{key['volume_vs_recall_pearson']:+.3f}"
    )
    print(
        f"  volume AUROC -> any overlap: "
        f"{key['any_overlap_volume_auroc']:.3f}"
    )
    print(
        f"  volume AUROC -> Dice>=0.50: "
        f"{key['dice_ge_050_volume_auroc']:.3f}"
    )
    print()

    print("Decision")
    print(
        f"  cyst branch: "
        f"{final['cyst_branch_status']}"
    )
    print(
        f"  primary failure mode: "
        f"{final['primary_failure_mode']}"
    )
    print(
        "  validated clinical cutoff: no"
    )
    print(
        "  global postprocessing: not recommended"
    )
    print()

    print(
        f"JSON: {output_root / 'summary.json'}"
    )
    print(
        f"CSV:  {output_root / 'decision_matrix.csv'}"
    )
    print(
        f"MD:   {output_root / 'cyst_branch_report.md'}"
    )
    print(
        "Cyst branch synthesis PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
