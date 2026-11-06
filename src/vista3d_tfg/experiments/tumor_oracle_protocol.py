from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


SPECIAL_TUMOR_POSITIVE_POINT_LABEL = 3
EXPECTED_AUTOMATIC_RENAL_LABELS = {
    "right kidney": 5,
    "left kidney": 14,
    "left kidney cyst": 116,
    "right kidney cyst": 117,
}


@dataclass(frozen=True, slots=True)
class TumorOraclePrompt:
    case_id: str
    component_id: int
    point_xyz: tuple[int, int, int]
    target_voxels: int
    target_volume_ml: float
    total_tumor_volume_ml: float
    interior_radius_mm: float


def load_label_dict(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("NV-Segment label_dict.json must contain an object.")
    return {str(name): int(value) for name, value in payload.items()}


def audit_tumor_runtime_contract(label_dict: dict[str, int]) -> dict[str, object]:
    for name, expected in EXPECTED_AUTOMATIC_RENAL_LABELS.items():
        actual = label_dict.get(name)
        if actual != expected:
            raise RuntimeError(
                f"Upstream label contract changed for {name!r}: "
                f"expected {expected}, got {actual}."
            )

    normalized = {name.strip().lower(): value for name, value in label_dict.items()}
    kidney_tumor_names = tuple(
        sorted(
            name
            for name in normalized
            if "kidney" in name and ("tumor" in name or "mass" in name)
        )
    )

    return {
        "automatic_renal_labels_verified": True,
        "automatic_kidney_tumor_names": list(kidney_tumor_names),
        "automatic_kidney_tumor_available": bool(kidney_tumor_names),
        "interactive_special_positive_point_label": SPECIAL_TUMOR_POSITIVE_POINT_LABEL,
        "recommended_protocol": (
            "oracle_special_positive_point_without_label_prompt"
            if not kidney_tumor_names
            else "review_upstream_tumor_class_before_experiment"
        ),
    }


def _load_gt(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(path))
    data = np.asanyarray(nii.dataobj)
    if len(nii.shape) != 3:
        raise ValueError(f"Expected 3D NIfTI: {path}")
    if not np.allclose(data, np.rint(data), atol=1e-5, rtol=0.0):
        raise TypeError(f"Expected integer labels: {path}")
    spacing = tuple(float(v) for v in nii.header.get_zooms()[:3])
    return np.rint(data).astype(np.int16, copy=False), spacing


def largest_tumor_deepest_point(
    tumor_mask: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
) -> tuple[int, tuple[int, int, int], int, float]:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(tumor_mask, structure=structure)
    if count == 0:
        raise ValueError("Tumor mask is empty.")

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component_id = int(np.argmax(sizes))
    component = labels == component_id
    target_voxels = int(sizes[component_id])

    distance = ndimage.distance_transform_edt(component, sampling=spacing_mm)
    flat = int(np.argmax(distance))
    point = tuple(int(v) for v in np.unravel_index(flat, component.shape))
    radius = float(distance[point])

    if not component[point]:
        raise RuntimeError("Deepest point is outside the selected tumor component.")

    return component_id, point, target_voxels, radius


def read_gt_to_prediction_case_ids(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    case_ids = tuple(
        row["case_id"]
        for row in rows
        if row["dominant_direction"] == "gt_to_prediction"
    )
    if len(case_ids) != 9:
        raise ValueError(f"Expected 9 GT->prediction cases, got {len(case_ids)}.")
    return case_ids


def prepare_tumor_oracle_prompts(
    *,
    geometry_csv: Path,
    dataset_root: Path,
) -> tuple[tuple[TumorOraclePrompt, ...], tuple[str, ...]]:
    prompts: list[TumorOraclePrompt] = []
    no_tumor: list[str] = []

    for case_id in read_gt_to_prediction_case_ids(geometry_csv):
        gt, spacing = _load_gt(dataset_root / case_id / "segmentation.nii.gz")
        tumor = gt == 2
        total_voxels = int(np.count_nonzero(tumor))

        if total_voxels == 0:
            no_tumor.append(case_id)
            continue

        component_id, point, target_voxels, radius = largest_tumor_deepest_point(
            tumor,
            spacing_mm=spacing,
        )
        voxel_volume_ml = float(np.prod(spacing) / 1000.0)

        prompts.append(
            TumorOraclePrompt(
                case_id=case_id,
                component_id=component_id,
                point_xyz=point,
                target_voxels=target_voxels,
                target_volume_ml=target_voxels * voxel_volume_ml,
                total_tumor_volume_ml=total_voxels * voxel_volume_ml,
                interior_radius_mm=radius,
            )
        )

    return tuple(prompts), tuple(no_tumor)


def write_protocol_manifest(
    *,
    runtime_contract: dict[str, object],
    prompts: tuple[TumorOraclePrompt, ...],
    no_tumor_case_ids: tuple[str, ...],
    output_path: Path,
) -> None:
    payload = {
        "schema_version": "1.0",
        "experiment": "nv_segment_ct_gt_to_prediction_tumor_oracle_point",
        "phase": "protocol_freeze",
        "source_case_count": 9,
        "runtime_contract": runtime_contract,
        "oracle": True,
        "deployment_valid": False,
        "input_protocol": {
            "label_prompt": None,
            "point_labels": [SPECIAL_TUMOR_POSITIVE_POINT_LABEL],
            "point_source": "largest_gt_tumor_component_deepest_interior",
        },
        "prompted_tumor_positive_case_count": len(prompts),
        "no_tumor_case_ids": list(no_tumor_case_ids),
        "prompts": [asdict(prompt) for prompt in prompts],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
