from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_dual_point_protocol import (
    prepare_dual_point_protocol,
    write_dual_point_protocol,
)


def main() -> int:
    root = discover_project_root()

    block46a_protocol_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_adaptive_relative_depth"
        / "protocol.json"
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
        / "nv_segment_ct_tumor_dual_interior_point"
        / "protocol.json"
    )

    rows = prepare_dual_point_protocol(
        block46a_protocol_path=block46a_protocol_path,
        dataset_root=dataset_root,
    )
    write_dual_point_protocol(
        rows=rows,
        path=output_path,
    )

    print(
        "NV-Segment-CT dual interior-point tumor protocol freeze"
    )
    print("Source: frozen Block 46A 12-case stratified cohort")
    print("Cases: 12")
    print("First point: deepest interior voxel")
    print(
        "Second point: farthest voxel from first among "
        "depth >= 50% max-depth candidates"
    )
    print("point_labels=[3, 3], label_prompt=None")
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
                f"p2_depth={row.second_point_depth_mm:.2f} mm "
                f"({100.0 * row.second_point_relative_depth:.1f}% max), "
                f"separation={row.point_separation_mm:.2f} mm"
            )
            print(
                f"    p1={row.first_point_xyz}, "
                f"p2={row.second_point_xyz}"
            )
        print()

    print(f"Manifest: {output_path}")
    print(
        "Dual interior-point tumor protocol freeze PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
