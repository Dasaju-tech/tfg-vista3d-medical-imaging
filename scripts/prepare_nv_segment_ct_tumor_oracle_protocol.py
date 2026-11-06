from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    audit_tumor_runtime_contract,
    load_label_dict,
    prepare_tumor_oracle_prompts,
    write_protocol_manifest,
)
from vista3d_tfg.integration.nv_segment import runtime_paths


def main() -> int:
    root = discover_project_root()
    nv_paths = runtime_paths(root)

    label_dict_path = nv_paths.ct_bundle_root / "configs" / "label_dict.json"
    geometry_csv = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_residual_geometry"
        / "outlier_geometry.csv"
    )
    dataset_root = root / "data" / "raw" / "kits23"
    output_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_oracle_point"
        / "protocol.json"
    )

    label_dict = load_label_dict(label_dict_path)
    contract = audit_tumor_runtime_contract(label_dict)
    prompts, no_tumor = prepare_tumor_oracle_prompts(
        geometry_csv=geometry_csv,
        dataset_root=dataset_root,
    )

    if contract["automatic_kidney_tumor_available"]:
        raise RuntimeError(
            "A kidney tumor/mass automatic class is present in the frozen runtime. "
            "Review the upstream class before using the special-point protocol."
        )

    write_protocol_manifest(
        runtime_contract=contract,
        prompts=prompts,
        no_tumor_case_ids=no_tumor,
        output_path=output_path,
    )

    print("NV-Segment-CT tumor oracle protocol freeze")
    print("GT->prediction source cases: 9")
    print(f"Tumor-positive prompts: {len(prompts)}")
    print(f"No-tumor cases: {len(no_tumor)}")
    print("Automatic kidney-tumor class: unavailable in frozen label_dict")
    print("Interactive protocol: point_labels=[3], label_prompt=None")
    print("Oracle status: GT-derived localization; not deployment-valid")
    print()

    for prompt in prompts:
        print(
            f"  {prompt.case_id}: point={prompt.point_xyz}, "
            f"largest_tumor={prompt.target_volume_ml:.2f} mL, "
            f"total_tumor={prompt.total_tumor_volume_ml:.2f} mL, "
            f"interior_radius={prompt.interior_radius_mm:.2f} mm"
        )

    print()
    print(f"Manifest: {output_path}")
    print("Tumor oracle protocol freeze PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
