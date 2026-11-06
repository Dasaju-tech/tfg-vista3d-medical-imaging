from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
    prepare_adaptive_protocol,
    write_adaptive_protocol,
)


def main() -> int:
    root = discover_project_root()

    safe_region_cases_csv = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_expanded100_tumor_safe_regions"
        / "cases.csv"
    )
    dataset_root = (
        root
        / "data"
        / "raw"
        / "kits23"
    )
    output_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_adaptive_relative_depth"
        / "protocol.json"
    )

    rows = prepare_adaptive_protocol(
        safe_region_cases_csv=safe_region_cases_csv,
        dataset_root=dataset_root,
    )
    write_adaptive_protocol(
        rows=rows,
        path=output_path,
    )

    print(
        "NV-Segment-CT adaptive relative-depth "
        "tumor protocol freeze"
    )
    print("Source cohort: expanded100")
    print("Tumor-positive source cases: 100")
    print("Selected cases: 12")
    print("Cases per tumor-size quartile: 3")
    print("Adaptive depth: 70% of per-case maximum interior depth")
    print("Reference: deepest-interior point")
    print("point_labels=[3], label_prompt=None")
    print("GPU inference: none")
    print("Oracle-simulated clicks; deployment-valid: no")
    print()

    for quartile in QUARTILE_NAMES:
        print(quartile)
        for row in rows:
            if row.quartile != quartile:
                continue
            print(
                f"  {row.case_id}: "
                f"tumor={row.largest_component_volume_ml:.2f} mL, "
                f"max_depth={row.max_interior_depth_mm:.2f} mm, "
                f"adaptive={row.adaptive_actual_depth_mm:.2f} mm, "
                f"point={row.adaptive_point_xyz}"
            )
        print()

    print(f"Manifest: {output_path}")
    print(
        "Adaptive relative-depth protocol freeze PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
