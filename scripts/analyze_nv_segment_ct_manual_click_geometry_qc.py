from __future__ import annotations

from vista3d_tfg.analysis.manual_click_geometry_qc import (
    audit_manual_click_geometry,
    render_manual_click_qc,
    summarize_geometry_qc,
    write_geometry_qc_outputs,
)
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    load_adaptive_protocol,
)
from vista3d_tfg.interaction.manual_tumor_clicks import (
    load_manual_click,
)


def main() -> int:
    root = discover_project_root()

    protocol_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_adaptive_relative_depth"
        / "protocol.json"
    )
    clicks_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_manual_tumor_clicks"
        / "clicks"
    )
    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_manual_click_geometry_qc"
    )

    protocol = load_adaptive_protocol(
        protocol_path
    )

    rows = []

    print("NV-Segment-CT frozen manual-click geometry QC")
    print("Cases: 12")
    print("GPU inference: none")
    print("Clicks modified: no")
    print("Ground truth use: post-hoc QC only")
    print()

    for index, case in enumerate(
        protocol,
        start=1,
    ):
        click_path = (
            clicks_root
            / f"{case.case_id}.json"
        )
        click = load_manual_click(
            click_path
        )

        image_path = (
            root
            / "data"
            / "raw"
            / "kits23"
            / case.case_id
            / "imaging.nii.gz"
        )
        segmentation_path = (
            root
            / "data"
            / "raw"
            / "kits23"
            / case.case_id
            / "segmentation.nii.gz"
        )

        qc = audit_manual_click_geometry(
            click=click,
            image_path=image_path,
            segmentation_path=segmentation_path,
        )
        rows.append(qc)

        figure_path = (
            output_root
            / "figures"
            / f"{case.case_id}.png"
        )
        render_manual_click_qc(
            qc=qc,
            image_path=image_path,
            segmentation_path=segmentation_path,
            output_path=figure_path,
        )

        status = (
            "TUMOR HIT"
            if qc.inside_any_tumor
            else "MISS"
        )

        print(
            f"[{index:02d}/12] {case.case_id}: {status}"
        )
        print(
            f"  grids equal: {qc.grids_equal}, "
            f"affine max diff={qc.affine_max_abs_diff:.3e}"
        )
        print(
            f"  saved RAS={qc.saved_display_ras_xyz}, "
            f"recomputed RAS={qc.recomputed_display_ras_xyz}, "
            f"error={qc.display_roundtrip_error_vox:.3e} vox"
        )
        print(
            f"  image voxel={qc.image_point_xyz}, "
            f"GT voxel={qc.segmentation_point_xyz}, "
            f"world error={qc.world_roundtrip_error_mm:.3e} mm"
        )
        print(
            f"  GT label={qc.gt_label_at_mapped_point}, "
            f"distance to tumor={qc.distance_to_any_tumor_mm:.2f} mm"
        )
        print(
            f"  figure={figure_path}"
        )
        print()

    payload = summarize_geometry_qc(
        rows
    )
    write_geometry_qc_outputs(
        payload=payload,
        output_root=output_root,
    )

    print("Geometry QC summary")
    print(
        f"  identical image/GT grids: "
        f"{payload['grid_equal_count']}/12"
    )
    print(
        f"  tumor hits after world-space mapping: "
        f"{payload['tumor_hit_count']}/12"
    )
    print(
        f"  max RAS display round-trip error: "
        f"{payload['display_roundtrip_error_vox_max']:.3e} vox"
    )
    print(
        f"  max image->GT mapping rounding error: "
        f"{payload['segmentation_mapping_rounding_error_vox_max']:.3e} vox"
    )
    print(
        f"  max image->GT world round-trip error: "
        f"{payload['world_roundtrip_error_mm_max']:.3e} mm"
    )
    print(
        f"  geometry contract valid: "
        f"{payload['geometry_contract_valid']}"
    )
    print()

    print(
        f"JSON: {output_root / 'summary.json'}"
    )
    print(
        f"CSV:  {output_root / 'cases.csv'}"
    )
    print(
        f"Figures: {output_root / 'figures'}"
    )
    print(
        "Frozen manual-click geometry QC PASSED."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
