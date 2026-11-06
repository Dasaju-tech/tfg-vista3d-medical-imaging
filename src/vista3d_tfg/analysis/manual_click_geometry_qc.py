from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage

from vista3d_tfg.interaction.manual_tumor_clicks import (
    ManualTumorClick,
    load_manual_click,
    validate_point,
    window_ct,
)


@dataclass(frozen=True, slots=True)
class ManualClickGeometryQC:
    case_id: str
    quartile: str
    image_shape: tuple[int, int, int]
    segmentation_shape: tuple[int, int, int]
    shapes_equal: bool
    affine_max_abs_diff: float
    grids_equal: bool
    image_point_xyz: tuple[int, int, int]
    saved_display_ras_xyz: tuple[int, int, int]
    recomputed_display_ras_xyz: tuple[int, int, int]
    display_roundtrip_error_vox: float
    segmentation_point_xyz: tuple[int, int, int]
    segmentation_point_float_xyz: tuple[float, float, float]
    segmentation_mapping_rounding_error_vox: float
    world_roundtrip_error_mm: float
    gt_label_at_mapped_point: int
    inside_any_tumor: bool
    distance_to_any_tumor_mm: float
    nearest_tumor_seg_xyz: tuple[int, int, int] | None
    nearest_tumor_display_ras_xyz: tuple[int, int, int] | None


def _shape3(image: nib.spatialimages.SpatialImage) -> tuple[int, int, int]:
    if len(image.shape) != 3:
        raise ValueError(
            f"Expected a 3D NIfTI image, got shape {image.shape}."
        )
    return tuple(int(value) for value in image.shape)


def _voxel_to_world(
    point_xyz: tuple[int, int, int],
    affine: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        nib.affines.apply_affine(
            affine,
            np.asarray(point_xyz, dtype=float),
        ),
        dtype=float,
    )


def _world_to_voxel_float(
    world_xyz: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        nib.affines.apply_affine(
            np.linalg.inv(affine),
            world_xyz,
        ),
        dtype=float,
    )


def _rounded_voxel(
    point_float: np.ndarray,
) -> tuple[int, int, int]:
    return tuple(
        int(value)
        for value in np.rint(point_float).astype(int)
    )


def audit_manual_click_geometry(
    *,
    click: ManualTumorClick,
    image_path: Path,
    segmentation_path: Path,
) -> ManualClickGeometryQC:
    if click.display_point_ras_xyz is None:
        raise ValueError(
            f"{click.case_id}: schema-1.1 RAS display point is required."
        )

    image_nii = nib.load(str(image_path))
    seg_nii = nib.load(str(segmentation_path))

    image_shape = _shape3(image_nii)
    seg_shape = _shape3(seg_nii)

    validate_point(
        point_xyz=click.point_xyz,
        shape=image_shape,
    )

    image_affine = np.asarray(
        image_nii.affine,
        dtype=float,
    )
    seg_affine = np.asarray(
        seg_nii.affine,
        dtype=float,
    )

    affine_diff = float(
        np.max(
            np.abs(
                image_affine - seg_affine
            )
        )
    )
    grids_equal = bool(
        image_shape == seg_shape
        and np.allclose(
            image_affine,
            seg_affine,
            rtol=0.0,
            atol=1e-5,
        )
    )

    click_world = _voxel_to_world(
        click.point_xyz,
        image_affine,
    )

    seg_point_float = _world_to_voxel_float(
        click_world,
        seg_affine,
    )
    seg_point = _rounded_voxel(
        seg_point_float
    )
    validate_point(
        point_xyz=seg_point,
        shape=seg_shape,
    )

    mapping_rounding_error = float(
        np.linalg.norm(
            seg_point_float
            - np.asarray(
                seg_point,
                dtype=float,
            )
        )
    )

    mapped_world = _voxel_to_world(
        seg_point,
        seg_affine,
    )
    world_roundtrip_error = float(
        np.linalg.norm(
            click_world - mapped_world
        )
    )

    canonical_image = nib.as_closest_canonical(
        image_nii
    )
    canonical_affine = np.asarray(
        canonical_image.affine,
        dtype=float,
    )
    display_float = _world_to_voxel_float(
        click_world,
        canonical_affine,
    )
    display_point = _rounded_voxel(
        display_float
    )
    display_error = float(
        np.linalg.norm(
            np.asarray(
                display_point,
                dtype=float,
            )
            - np.asarray(
                click.display_point_ras_xyz,
                dtype=float,
            )
        )
    )

    gt_raw = np.asanyarray(
        seg_nii.dataobj
    )
    if not np.allclose(
        gt_raw,
        np.rint(gt_raw),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError(
            f"{click.case_id}: segmentation is not integer-valued."
        )

    gt = np.rint(
        gt_raw
    ).astype(
        np.int16,
        copy=False,
    )
    tumor = gt == 2

    if not np.any(tumor):
        raise ValueError(
            f"{click.case_id}: no tumor voxels found."
        )

    spacing = tuple(
        float(value)
        for value in seg_nii.header.get_zooms()[:3]
    )

    inside_tumor = bool(
        tumor[seg_point]
    )

    distance_map, nearest_indices = ndimage.distance_transform_edt(
        ~tumor,
        sampling=spacing,
        return_indices=True,
    )
    distance = float(
        distance_map[seg_point]
    )

    nearest_seg = tuple(
        int(nearest_indices[axis][seg_point])
        for axis in range(3)
    )

    nearest_world = _voxel_to_world(
        nearest_seg,
        seg_affine,
    )
    nearest_display = _rounded_voxel(
        _world_to_voxel_float(
            nearest_world,
            canonical_affine,
        )
    )

    return ManualClickGeometryQC(
        case_id=click.case_id,
        quartile=click.quartile,
        image_shape=image_shape,
        segmentation_shape=seg_shape,
        shapes_equal=(image_shape == seg_shape),
        affine_max_abs_diff=affine_diff,
        grids_equal=grids_equal,
        image_point_xyz=click.point_xyz,
        saved_display_ras_xyz=click.display_point_ras_xyz,
        recomputed_display_ras_xyz=display_point,
        display_roundtrip_error_vox=display_error,
        segmentation_point_xyz=seg_point,
        segmentation_point_float_xyz=tuple(
            float(value)
            for value in seg_point_float
        ),
        segmentation_mapping_rounding_error_vox=(
            mapping_rounding_error
        ),
        world_roundtrip_error_mm=world_roundtrip_error,
        gt_label_at_mapped_point=int(
            gt[seg_point]
        ),
        inside_any_tumor=inside_tumor,
        distance_to_any_tumor_mm=distance,
        nearest_tumor_seg_xyz=nearest_seg,
        nearest_tumor_display_ras_xyz=nearest_display,
    )


def render_manual_click_qc(
    *,
    qc: ManualClickGeometryQC,
    image_path: Path,
    segmentation_path: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    image_nii = nib.load(str(image_path))
    seg_nii = nib.load(str(segmentation_path))

    canonical_image = nib.as_closest_canonical(
        image_nii
    )
    canonical_seg = resample_from_to(
        seg_nii,
        canonical_image,
        order=0,
    )

    image = window_ct(
        np.asanyarray(
            canonical_image.dataobj,
            dtype=np.float32,
        )
    )
    seg = np.rint(
        np.asanyarray(
            canonical_seg.dataobj
        )
    ).astype(
        np.int16,
        copy=False,
    )
    tumor = seg == 2

    click = qc.recomputed_display_ras_xyz
    nearest = qc.nearest_tumor_display_ras_xyz
    assert nearest is not None

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
    )
    fig.suptitle(
        (
            f"{qc.case_id} | {qc.quartile} | "
            f"post-hoc GT QC | "
            f"distance to tumor={qc.distance_to_any_tumor_mm:.2f} mm"
        )
    )

    def draw_row(
        *,
        center: tuple[int, int, int],
        row_index: int,
        prefix: str,
    ) -> None:
        x, y, z = center
        views = (
            (
                image[:, :, z].T,
                tumor[:, :, z].T,
                (click[0], click[1]),
                (nearest[0], nearest[1]),
                "Axial",
            ),
            (
                image[:, y, :].T,
                tumor[:, y, :].T,
                (click[0], click[2]),
                (nearest[0], nearest[2]),
                "Coronal",
            ),
            (
                image[x, :, :].T,
                tumor[x, :, :].T,
                (click[1], click[2]),
                (nearest[1], nearest[2]),
                "Sagittal",
            ),
        )

        for column, (
            view,
            tumor_view,
            click_marker,
            nearest_marker,
            title,
        ) in enumerate(views):
            axis = axes[row_index, column]
            axis.imshow(
                view,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
            )

            if np.any(tumor_view):
                axis.contour(
                    tumor_view.astype(float),
                    levels=[0.5],
                    linewidths=1.0,
                    origin="lower",
                )

            axis.plot(
                click_marker[0],
                click_marker[1],
                marker="x",
                markersize=10,
                markeredgewidth=2,
                label="frozen click",
            )
            axis.plot(
                nearest_marker[0],
                nearest_marker[1],
                marker="+",
                markersize=10,
                markeredgewidth=2,
                label="nearest tumor",
            )
            axis.set_title(
                f"{prefix}: {title}"
            )
            axis.set_axis_off()

    draw_row(
        center=click,
        row_index=0,
        prefix="click-centered",
    )
    draw_row(
        center=nearest,
        row_index=1,
        prefix="tumor-centered",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
        )

    fig.tight_layout(
        rect=(0.0, 0.05, 1.0, 0.95)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def summarize_geometry_qc(
    rows: Iterable[ManualClickGeometryQC],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 frozen clicks, got {len(rows)}."
        )

    grid_equal_count = sum(
        row.grids_equal
        for row in rows
    )
    tumor_hit_count = sum(
        row.inside_any_tumor
        for row in rows
    )

    display_errors = np.asarray(
        [
            row.display_roundtrip_error_vox
            for row in rows
        ],
        dtype=float,
    )
    world_errors = np.asarray(
        [
            row.world_roundtrip_error_mm
            for row in rows
        ],
        dtype=float,
    )
    mapping_errors = np.asarray(
        [
            row.segmentation_mapping_rounding_error_vox
            for row in rows
        ],
        dtype=float,
    )
    distances = np.asarray(
        [
            row.distance_to_any_tumor_mm
            for row in rows
        ],
        dtype=float,
    )

    geometry_valid = bool(
        np.max(display_errors) <= 1e-6
        and np.max(world_errors) <= 1e-4
        and np.max(mapping_errors) <= 1e-4
    )

    return {
        "schema_version": "1.0",
        "analysis": "manual_click_frozen_geometry_qc",
        "case_count": len(rows),
        "grid_equal_count": grid_equal_count,
        "tumor_hit_count": tumor_hit_count,
        "tumor_miss_count": len(rows) - tumor_hit_count,
        "display_roundtrip_error_vox_max": float(
            np.max(display_errors)
        ),
        "segmentation_mapping_rounding_error_vox_max": float(
            np.max(mapping_errors)
        ),
        "world_roundtrip_error_mm_max": float(
            np.max(world_errors)
        ),
        "distance_to_tumor_mm": {
            "mean": float(
                np.mean(distances)
            ),
            "median": float(
                np.median(distances)
            ),
            "min": float(
                np.min(distances)
            ),
            "max": float(
                np.max(distances)
            ),
        },
        "geometry_contract_valid": geometry_valid,
        "rows": [
            asdict(row)
            for row in rows
        ],
        "interpretation": (
            "This audit is post-hoc. Frozen manual clicks are not modified. "
            "If the geometry contract is valid and tumor hits remain zero, "
            "the manual CT-only localization failure is genuine for this "
            "simulated-user experiment rather than a coordinate-transform "
            "artifact."
        ),
    }


def write_geometry_qc_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_root / "summary.json").write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    rows = payload["rows"]
    assert isinstance(rows, list)

    with (output_root / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        if not rows:
            raise ValueError(
                "No geometry-QC rows."
            )
        fieldnames = list(
            rows[0].keys()
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)
