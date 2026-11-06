from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickVariant,
    run_click_variant,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)
from vista3d_tfg.integration.nv_segment import NVSegmentPaths


DEPTHS_MM = (5.0, 10.0, 15.0)


@dataclass(frozen=True, slots=True)
class DepthAuditCase:
    case_id: str
    shallow_5mm: TumorOracleSmokeResult
    depth_10mm: TumorOracleSmokeResult
    depth_15mm: TumorOracleSmokeResult
    centroid: TumorOracleSmokeResult
    deep: TumorOracleSmokeResult


def _integer_labels(array: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise ValueError("Label map contains non-finite values.")
    if not np.allclose(
        array,
        np.rint(array),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError("Label map must be integer-valued.")
    return np.rint(array).astype(np.int16, copy=False)


def _target_component(
    *,
    ground_truth_path: Path,
    prompt: TumorOraclePrompt,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    nii = nib.load(str(ground_truth_path))
    gt = _integer_labels(np.asanyarray(nii.dataobj))
    spacing = tuple(
        float(v)
        for v in nii.header.get_zooms()[:3]
    )

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(
        gt == 2,
        structure=structure,
    )

    if prompt.component_id < 1 or prompt.component_id > count:
        raise ValueError(
            f"{prompt.case_id}: frozen tumor component is invalid."
        )

    component = labels == prompt.component_id

    if int(np.count_nonzero(component)) != prompt.target_voxels:
        raise ValueError(
            f"{prompt.case_id}: frozen tumor component size changed."
        )

    return component, spacing


def derive_depth_variant(
    *,
    prompt: TumorOraclePrompt,
    ground_truth_path: Path,
    depth_mm: float,
) -> ClickVariant:
    if depth_mm <= 0.0:
        raise ValueError("depth_mm must be positive.")

    component, spacing = _target_component(
        ground_truth_path=ground_truth_path,
        prompt=prompt,
    )

    distance = ndimage.distance_transform_edt(
        component,
        sampling=spacing,
    )
    coords = np.argwhere(component)
    component_distances = distance[component]

    index = int(
        np.argmin(
            np.abs(component_distances - depth_mm)
        )
    )
    point = tuple(
        int(v)
        for v in coords[index]
    )
    actual_depth = float(distance[point])

    return ClickVariant(
        case_id=prompt.case_id,
        variant=f"depth_{depth_mm:g}mm",
        point_xyz=point,
        interior_radius_mm=actual_depth,
    )


def load_existing_click_result(
    *,
    summary_csv: Path,
    case_id: str,
    variant: str,
) -> dict[str, str]:
    import csv

    with summary_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = tuple(csv.DictReader(handle))

    matches = [
        row
        for row in rows
        if row["case_id"] == case_id
        and row["variant"] == variant
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one {variant} row for {case_id}, "
            f"got {len(matches)}."
        )

    return matches[0]


def summarize_depth_audit(
    rows: Iterable[DepthAuditCase],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 9:
        raise ValueError(
            f"Expected 9 depth-audit cases, got {len(rows)}."
        )

    variant_names = (
        "shallow_5mm",
        "depth_10mm",
        "depth_15mm",
        "centroid",
        "deep",
    )

    payload: dict[str, object] = {
        "schema_version": "1.0",
        "experiment": "nv_segment_ct_tumor_click_depth_audit",
        "case_count": 9,
        "deployment_valid": False,
        "gt_simulated_clicks": True,
        "variants": {},
    }

    variants_payload = payload["variants"]
    assert isinstance(variants_payload, dict)

    for name in variant_names:
        results = tuple(
            getattr(row, name)
            for row in rows
        )
        dice = np.asarray(
            [row.interactive_target_dice for row in results],
            dtype=float,
        )
        coverage = np.asarray(
            [row.tumor_coverage_gain for row in results],
            dtype=float,
        )
        hd95 = np.asarray(
            [row.hec_hd95_reduction_mm for row in results],
            dtype=float,
        )
        hec_dice = np.asarray(
            [row.hec_dice_delta for row in results],
            dtype=float,
        )
        surface = np.asarray(
            [row.hec_surface_dice_delta for row in results],
            dtype=float,
        )

        variants_payload[name] = {
            "tumor_dice_mean": float(dice.mean()),
            "tumor_dice_median": float(np.median(dice)),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(dice >= 0.50)
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(dice >= 0.70)
            ),
            "tumor_coverage_gain_mean": float(
                coverage.mean()
            ),
            "hec_hd95_reduction_mean_mm": float(
                hd95.mean()
            ),
            "hec_hd95_reduction_median_mm": float(
                np.median(hd95)
            ),
            "hd95_improved_count": int(
                np.count_nonzero(hd95 > 0.0)
            ),
            "hec_dice_delta_mean": float(
                hec_dice.mean()
            ),
            "hec_surface_dice_delta_mean": float(
                surface.mean()
            ),
        }

    deep = np.asarray(
        [
            row.deep.interactive_target_dice
            for row in rows
        ],
        dtype=float,
    )

    robustness = {}
    for name in (
        "shallow_5mm",
        "depth_10mm",
        "depth_15mm",
        "centroid",
    ):
        values = np.asarray(
            [
                getattr(row, name).interactive_target_dice
                for row in rows
            ],
            dtype=float,
        )
        difference = np.abs(values - deep)
        robustness[name] = {
            "dice_delta_vs_deep_mean": float(
                (values - deep).mean()
            ),
            "cases_within_0_05_dice_of_deep": int(
                np.count_nonzero(
                    difference <= (0.05 + 1e-12)
                )
            ),
            "cases_within_0_10_dice_of_deep": int(
                np.count_nonzero(
                    difference <= (0.10 + 1e-12)
                )
            ),
        }

    payload["robustness_vs_deep"] = robustness
    payload["case_rows"] = [
        {
            "case_id": row.case_id,
            "shallow_5mm": row.shallow_5mm.to_dict(),
            "depth_10mm": row.depth_10mm.to_dict(),
            "depth_15mm": row.depth_15mm.to_dict(),
            "centroid": row.centroid.to_dict(),
            "deep": row.deep.to_dict(),
        }
        for row in rows
    ]

    return payload


def write_depth_audit_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "summary.json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
