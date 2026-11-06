from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


PLANE_TO_AXIS = {
    "sagittal": 0,
    "coronal": 1,
    "axial": 2,
}

NV_RENAL_LABELS = (5, 14, 116, 117)


@dataclass(frozen=True, slots=True)
class ComparisonSlice:
    plane: str
    axis: int
    index: int
    source: str
    disagreement_voxels: int


@dataclass(frozen=True, slots=True)
class CaseComparisonQC:
    case_id: str
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    orientation: tuple[str, str, str]
    selected_slices: tuple[ComparisonSlice, ...]
    figure_files: tuple[str, ...]
    total_false_positive_voxels: int
    total_false_negative_voxels: int
    total_disagreement_voxels: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_comparison_case(
    *,
    image_path: Path,
    ground_truth_path: Path,
    prediction_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float, float],
    tuple[str, str, str],
]:
    image_nii = nib.load(str(image_path))
    ground_truth_nii = nib.load(str(ground_truth_path))
    prediction_nii = nib.load(str(prediction_path))

    if (
        len(image_nii.shape) != 3
        or len(ground_truth_nii.shape) != 3
        or len(prediction_nii.shape) != 3
    ):
        raise ValueError("Image, ground truth, and prediction must be 3D.")

    if not (
        image_nii.shape
        == ground_truth_nii.shape
        == prediction_nii.shape
    ):
        raise ValueError("Image, ground truth, and prediction shapes differ.")

    if not np.allclose(
        image_nii.affine,
        ground_truth_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("Image and ground-truth affines differ.")

    if not np.allclose(
        image_nii.affine,
        prediction_nii.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError("Image and prediction affines differ.")

    image = np.asanyarray(image_nii.dataobj)
    ground_truth = np.asanyarray(ground_truth_nii.dataobj)
    prediction = np.asanyarray(prediction_nii.dataobj)

    if not np.issubdtype(ground_truth.dtype, np.integer):
        raise TypeError("Ground truth must contain integer labels.")

    if not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction contains non-finite values.")

    if not np.allclose(
        prediction,
        np.rint(prediction),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError("Prediction must be integer-valued.")

    prediction = np.rint(prediction).astype(np.int16, copy=False)

    unexpected_gt = set(int(v) for v in np.unique(ground_truth)) - {
        0, 1, 2, 3
    }
    if unexpected_gt:
        raise ValueError(
            f"Unexpected KiTS23 labels: {sorted(unexpected_gt)}."
        )

    unexpected_pred = set(int(v) for v in np.unique(prediction)) - {
        0, *NV_RENAL_LABELS
    }
    if unexpected_pred:
        raise ValueError(
            f"Unexpected NV-Segment labels: {sorted(unexpected_pred)}."
        )

    spacing = tuple(
        float(value)
        for value in image_nii.header.get_zooms()[:3]
    )
    orientation = tuple(
        str(value) for value in nib.aff2axcodes(image_nii.affine)
    )

    return image, ground_truth, prediction, spacing, orientation


def renal_envelopes(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.isin(ground_truth, (1, 2, 3)),
        np.isin(prediction, NV_RENAL_LABELS),
    )


def disagreement_map(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    reference, predicted = renal_envelopes(
        ground_truth,
        prediction,
    )

    result = np.zeros(reference.shape, dtype=np.uint8)
    result[reference & predicted] = 1
    result[~reference & predicted] = 2
    result[reference & ~predicted] = 3
    return result


def select_comparison_slice(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    plane: str,
) -> ComparisonSlice:
    if plane not in PLANE_TO_AXIS:
        raise ValueError(
            f"Unknown plane {plane!r}. "
            f"Expected one of {sorted(PLANE_TO_AXIS)}."
        )

    axis = PLANE_TO_AXIS[plane]
    errors = disagreement_map(ground_truth, prediction)
    disagreement = errors >= 2

    reduction_axes = tuple(
        candidate
        for candidate in range(3)
        if candidate != axis
    )

    disagreement_counts = np.count_nonzero(
        disagreement,
        axis=reduction_axes,
    )

    if np.any(disagreement_counts):
        index = int(np.argmax(disagreement_counts))
        source = "maximum_disagreement"
        count = int(disagreement_counts[index])
    else:
        lesion = np.isin(ground_truth, (2, 3))
        lesion_counts = np.count_nonzero(
            lesion,
            axis=reduction_axes,
        )

        if np.any(lesion_counts):
            index = int(np.argmax(lesion_counts))
            source = "maximum_ground_truth_lesion"
        else:
            renal = np.isin(ground_truth, (1, 2, 3))
            renal_counts = np.count_nonzero(
                renal,
                axis=reduction_axes,
            )

            if np.any(renal_counts):
                index = int(np.argmax(renal_counts))
                source = "maximum_ground_truth_renal_area"
            else:
                index = ground_truth.shape[axis] // 2
                source = "volume_center"

        count = 0

    return ComparisonSlice(
        plane=plane,
        axis=axis,
        index=index,
        source=source,
        disagreement_voxels=count,
    )


def _slice_2d(
    volume: np.ndarray,
    *,
    axis: int,
    index: int,
) -> np.ndarray:
    return np.rot90(
        np.take(volume, indices=index, axis=axis)
    )


def _window_ct(
    image: np.ndarray,
    window_hu: tuple[float, float],
) -> np.ndarray:
    lower, upper = window_hu

    if not (
        np.isfinite(lower)
        and np.isfinite(upper)
        and lower < upper
    ):
        raise ValueError("Invalid CT display window.")

    return np.clip(image, lower, upper)


def render_comparison_figure(
    *,
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    selection: ComparisonSlice,
    output_path: Path,
    window_hu: tuple[float, float] = (-200.0, 300.0),
) -> None:
    image_slice = _slice_2d(
        image,
        axis=selection.axis,
        index=selection.index,
    )
    gt_slice = _slice_2d(
        ground_truth,
        axis=selection.axis,
        index=selection.index,
    )
    pred_slice = _slice_2d(
        prediction,
        axis=selection.axis,
        index=selection.index,
    )
    error_slice = _slice_2d(
        disagreement_map(ground_truth, prediction),
        axis=selection.axis,
        index=selection.index,
    )

    ct = _window_ct(image_slice, window_hu)

    figure, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(
        ct,
        cmap="gray",
        vmin=window_hu[0],
        vmax=window_hu[1],
        interpolation="nearest",
    )
    axes[0].set_title("CT")

    axes[1].imshow(
        ct,
        cmap="gray",
        vmin=window_hu[0],
        vmax=window_hu[1],
        interpolation="nearest",
    )
    gt_overlay = np.ma.masked_where(gt_slice == 0, gt_slice)
    axes[1].imshow(
        gt_overlay,
        cmap=plt.get_cmap("tab10", 4),
        vmin=0,
        vmax=3,
        alpha=0.45,
        interpolation="nearest",
    )
    axes[1].set_title("KiTS23 ground truth")

    axes[2].imshow(
        ct,
        cmap="gray",
        vmin=window_hu[0],
        vmax=window_hu[1],
        interpolation="nearest",
    )
    pred_overlay = np.ma.masked_where(
        pred_slice == 0,
        pred_slice,
    )
    axes[2].imshow(
        pred_overlay,
        cmap="tab20",
        alpha=0.45,
        interpolation="nearest",
    )
    axes[2].set_title("NV-Segment prediction")

    axes[3].imshow(
        ct,
        cmap="gray",
        vmin=window_hu[0],
        vmax=window_hu[1],
        interpolation="nearest",
    )
    error_overlay = np.ma.masked_where(
        error_slice == 0,
        error_slice,
    )
    axes[3].imshow(
        error_overlay,
        cmap=plt.get_cmap("tab10", 4),
        vmin=0,
        vmax=3,
        alpha=0.55,
        interpolation="nearest",
    )
    axes[3].set_title("Envelope agreement / error")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"{selection.plane.capitalize()} slice {selection.index} "
        f"({selection.source}, "
        f"{selection.disagreement_voxels} disagreement voxels)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_case_comparison_qc(
    *,
    case_id: str,
    image_path: Path,
    ground_truth_path: Path,
    prediction_path: Path,
    output_directory: Path,
    window_hu: tuple[float, float] = (-200.0, 300.0),
) -> CaseComparisonQC:
    (
        image,
        ground_truth,
        prediction,
        spacing,
        orientation,
    ) = load_comparison_case(
        image_path=image_path,
        ground_truth_path=ground_truth_path,
        prediction_path=prediction_path,
    )

    selections = tuple(
        select_comparison_slice(
            ground_truth,
            prediction,
            plane=plane,
        )
        for plane in ("axial", "coronal", "sagittal")
    )

    figure_files: list[str] = []

    for selection in selections:
        output_path = (
            output_directory
            / case_id
            / f"{selection.plane}_prediction_vs_gt.png"
        )

        render_comparison_figure(
            image=image,
            ground_truth=ground_truth,
            prediction=prediction,
            selection=selection,
            output_path=output_path,
            window_hu=window_hu,
        )

        figure_files.append(str(output_path))

    reference, predicted = renal_envelopes(
        ground_truth,
        prediction,
    )

    false_positive = int(
        np.count_nonzero(~reference & predicted)
    )
    false_negative = int(
        np.count_nonzero(reference & ~predicted)
    )

    return CaseComparisonQC(
        case_id=case_id,
        shape=tuple(int(value) for value in image.shape),
        spacing_mm=spacing,
        orientation=orientation,
        selected_slices=selections,
        figure_files=tuple(figure_files),
        total_false_positive_voxels=false_positive,
        total_false_negative_voxels=false_negative,
        total_disagreement_voxels=false_positive + false_negative,
    )


def write_comparison_qc_index(
    result: CaseComparisonQC,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "KiTS23",
                "model": "NV-Segment-CT",
                "analysis": "prediction_vs_ground_truth_visual_qc",
                "error_map": {
                    "0": "background",
                    "1": "true_positive",
                    "2": "false_positive",
                    "3": "false_negative",
                },
                "result": result.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
