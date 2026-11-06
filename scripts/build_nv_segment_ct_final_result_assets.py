from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reporting.final_result_assets import (
    build_all_assets,
    load_final_summary,
)


def main() -> int:
    root = discover_project_root()
    summary_path = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_final_experimental_synthesis"
        / "summary.json"
    )
    output_root = root / "outputs" / "reporting" / "final_results"

    payload = load_final_summary(summary_path)
    build_all_assets(payload=payload, output_root=output_root)

    print("NV-Segment-CT final result assets")
    print("Source: Block 53 final synthesis")
    print("GPU inference: none")
    print()

    print("LaTeX")
    for name in (
        "results_macros.tex",
        "table_final_results.tex",
        "table_claim_matrix.tex",
    ):
        print(f"  {output_root / 'latex' / name}")

    print()
    print("Figures")
    for stem in (
        "fig01_automatic_baseline",
        "fig02_hd95_failure_mechanisms",
        "fig03_tumor_interaction_comparison",
        "fig04_cyst_volume_response",
    ):
        print(f"  {output_root / 'figures' / (stem + '.pdf')}")
        print(f"  {output_root / 'figures' / (stem + '.png')}")

    print()
    print("Final result assets PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
