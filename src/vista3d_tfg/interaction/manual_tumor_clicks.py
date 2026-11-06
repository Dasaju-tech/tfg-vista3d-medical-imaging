from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np

from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    load_adaptive_protocol,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    QUARTILE_NAMES,
)


@dataclass(frozen=True, slots=True)
class ManualClickCase:
    case_id: str
    quartile: str
    image_path: str


@dataclass(frozen=True, slots=True)
class ManualTumorClick:
    case_id: str
    quartile: str
    point_xyz: tuple[int, int, int]
    display_point_ras_xyz: tuple[int, int, int] | None = None
    source: str = "manual_ct_only_ras_display"
    gt_visible_during_selection: bool = False


def load_manual_click_cases(
    *,
    block46a_protocol_path: Path,
    dataset_root: Path,
) -> tuple[ManualClickCase, ...]:
    protocol = load_adaptive_protocol(
        block46a_protocol_path
    )

    rows = tuple(
        ManualClickCase(
            case_id=row.case_id,
            quartile=row.quartile,
            image_path=str(
                dataset_root
                / row.case_id
                / "imaging.nii.gz"
            ),
        )
        for row in protocol
    )

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 frozen cases, got {len(rows)}."
        )

    counts = {
        quartile: sum(
            row.quartile == quartile
            for row in rows
        )
        for quartile in QUARTILE_NAMES
    }
    if any(count != 3 for count in counts.values()):
        raise ValueError(
            f"Manual-click cohort is unbalanced: {counts}."
        )

    for row in rows:
        if not Path(row.image_path).is_file():
            raise FileNotFoundError(row.image_path)

    return rows


def save_manual_click(
    *,
    click: ManualTumorClick,
    output_path: Path,
) -> None:
    payload = {
        "schema_version": "1.1",
        "experiment": (
            "nv_segment_ct_manual_tumor_click"
        ),
        "deployment_plausible_localization": True,
        "gt_visible_during_selection": False,
        "display_orientation": "RAS",
        "coordinate_contract": {
            "display_point_ras_xyz": (
                "canonical RAS display voxel coordinate"
            ),
            "point_xyz": (
                "original NIfTI voxel coordinate used by VISTA3D"
            ),
        },
        "click": asdict(click),
        "interpretation_boundary": (
            "The click is selected from the CT image without displaying "
            "the ground-truth segmentation. Ground truth may be used only "
            "after click capture for retrospective evaluation."
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_manual_click(
    path: Path,
) -> ManualTumorClick:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("gt_visible_during_selection") is not False:
        raise ValueError(
            "Manual-click artifact must certify GT was hidden."
        )

    if payload.get("schema_version") != "1.1":
        raise ValueError(
            "Legacy manual-click artifact predates the RAS coordinate "
            "contract and must be recollected."
        )
    if payload.get("display_orientation") != "RAS":
        raise ValueError(
            "Manual-click artifact must use canonical RAS display."
        )

    click = payload.get("click")
    if not isinstance(click, dict):
        raise ValueError("Manual-click artifact has no click object.")

    point = tuple(
        int(value)
        for value in click["point_xyz"]
    )
    if len(point) != 3:
        raise ValueError(
            f"Expected xyz point, got {point}."
        )

    display_raw = click.get("display_point_ras_xyz")
    display_point = (
        None
        if display_raw is None
        else tuple(int(value) for value in display_raw)
    )

    return ManualTumorClick(
        case_id=str(click["case_id"]),
        quartile=str(click["quartile"]),
        point_xyz=point,
        display_point_ras_xyz=display_point,
        source=str(click["source"]),
        gt_visible_during_selection=bool(
            click["gt_visible_during_selection"]
        ),
    )


def window_ct(
    array: np.ndarray,
    *,
    lower_hu: float = -200.0,
    upper_hu: float = 300.0,
) -> np.ndarray:
    if upper_hu <= lower_hu:
        raise ValueError(
            "upper_hu must be greater than lower_hu."
        )

    clipped = np.clip(
        np.asarray(array, dtype=np.float32),
        lower_hu,
        upper_hu,
    )
    return (
        (clipped - lower_hu)
        / (upper_hu - lower_hu)
    )


def validate_point(
    *,
    point_xyz: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> None:
    if len(point_xyz) != 3 or len(shape) != 3:
        raise ValueError("Expected three-dimensional point and shape.")

    for value, upper in zip(point_xyz, shape):
        if value < 0 or value >= upper:
            raise ValueError(
                f"Point {point_xyz} lies outside image shape {shape}."
            )


def map_voxel_point(
    *,
    point_xyz: tuple[int, int, int],
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    world = nib.affines.apply_affine(
        source_affine,
        np.asarray(point_xyz, dtype=float),
    )
    target_float = nib.affines.apply_affine(
        np.linalg.inv(target_affine),
        world,
    )
    target_rounded = np.rint(target_float).astype(int)

    if not np.allclose(
        target_float,
        target_rounded,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Voxel mapping is not an exact permutation/flip transform."
        )

    target = tuple(int(value) for value in target_rounded)
    validate_point(
        point_xyz=target,
        shape=target_shape,
    )
    return target


class OrthogonalClickViewer:
    """Minimal CT-only orthogonal point-selection viewer.

    Ground-truth segmentation is deliberately not accepted by this class.
    """

    def __init__(
        self,
        *,
        image_path: Path,
        case_id: str,
        quartile: str,
        lower_hu: float = -200.0,
        upper_hu: float = 300.0,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._plt = plt
        self.case_id = case_id
        self.quartile = quartile

        original_nii = nib.load(str(image_path))
        original_volume = np.asanyarray(
            original_nii.dataobj,
            dtype=np.float32,
        )

        if original_volume.ndim != 3:
            raise ValueError(
                f"Expected 3D CT, got shape {original_volume.shape}."
            )

        canonical_nii = nib.as_closest_canonical(original_nii)
        canonical_volume = np.asanyarray(
            canonical_nii.dataobj,
            dtype=np.float32,
        )

        self.original_affine = np.asarray(
            original_nii.affine,
            dtype=float,
        )
        self.original_shape = tuple(
            int(value) for value in original_volume.shape
        )
        self.display_affine = np.asarray(
            canonical_nii.affine,
            dtype=float,
        )

        self.volume = window_ct(
            canonical_volume,
            lower_hu=lower_hu,
            upper_hu=upper_hu,
        )
        self.shape = tuple(
            int(value)
            for value in self.volume.shape
        )
        self.cursor = [
            dimension // 2
            for dimension in self.shape
        ]
        self.selected: tuple[int, int, int] | None = None
        self.accepted = False
        self.aborted = False

        self.fig, self.axes = plt.subplots(
            1,
            3,
            figsize=(15, 5),
        )
        self.fig.subplots_adjust(
            bottom=0.18,
        )
        self.fig.suptitle(
            (
                f"{case_id} | {quartile}\n"
                "CT only — GT hidden — canonical RAS display. "
                "Scroll to navigate, left-click to place point."
            )
        )

        accept_axis = self.fig.add_axes(
            [0.36, 0.04, 0.10, 0.07]
        )
        reset_axis = self.fig.add_axes(
            [0.47, 0.04, 0.10, 0.07]
        )
        abort_axis = self.fig.add_axes(
            [0.58, 0.04, 0.10, 0.07]
        )

        self.accept_button = Button(
            accept_axis,
            "Accept",
        )
        self.reset_button = Button(
            reset_axis,
            "Reset",
        )
        self.abort_button = Button(
            abort_axis,
            "Abort",
        )

        self.accept_button.on_clicked(
            self._accept,
        )
        self.reset_button.on_clicked(
            self._reset,
        )
        self.abort_button.on_clicked(
            self._abort,
        )

        self.fig.canvas.mpl_connect(
            "scroll_event",
            self._on_scroll,
        )
        self.fig.canvas.mpl_connect(
            "button_press_event",
            self._on_click,
        )
        self.fig.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )

        self._draw()

    def _axis_index(self, axis: object) -> int | None:
        for index, candidate in enumerate(self.axes):
            if axis is candidate:
                return index
        return None

    def _draw(self) -> None:
        x, y, z = self.cursor

        views = (
            (
                self.volume[:, :, z].T,
                "Axial (RAS)",
                (x, y),
            ),
            (
                self.volume[:, y, :].T,
                "Coronal (RAS)",
                (x, z),
            ),
            (
                self.volume[x, :, :].T,
                "Sagittal (RAS)",
                (y, z),
            ),
        )

        for axis, (image, title, crosshair) in zip(
            self.axes,
            views,
        ):
            axis.clear()
            axis.imshow(
                image,
                cmap="gray",
                origin="lower",
                interpolation="nearest",
            )
            axis.set_title(title)
            axis.axvline(
                crosshair[0],
                linewidth=0.8,
            )
            axis.axhline(
                crosshair[1],
                linewidth=0.8,
            )
            axis.set_axis_off()

        if self.selected is not None:
            sx, sy, sz = self.selected
            markers = (
                (sx, sy),
                (sx, sz),
                (sy, sz),
            )
            for axis, marker in zip(
                self.axes,
                markers,
            ):
                axis.plot(
                    marker[0],
                    marker[1],
                    marker="x",
                    markersize=10,
                    markeredgewidth=2,
                )

        self.fig.canvas.draw_idle()

    def _on_scroll(self, event: object) -> None:
        index = self._axis_index(
            getattr(event, "inaxes", None)
        )
        if index is None:
            return

        step = 1 if getattr(event, "button", "") == "up" else -1

        # Axial scrolls z, coronal scrolls y, sagittal scrolls x.
        dimension = (2, 1, 0)[index]
        self.cursor[dimension] = int(
            np.clip(
                self.cursor[dimension] + step,
                0,
                self.shape[dimension] - 1,
            )
        )
        self._draw()

    def _on_click(self, event: object) -> None:
        if getattr(event, "button", None) != 1:
            return

        index = self._axis_index(
            getattr(event, "inaxes", None)
        )
        if index is None:
            return

        xdata = getattr(event, "xdata", None)
        ydata = getattr(event, "ydata", None)
        if xdata is None or ydata is None:
            return

        a = int(round(float(xdata)))
        b = int(round(float(ydata)))

        x, y, z = self.cursor

        if index == 0:
            point = (a, b, z)
        elif index == 1:
            point = (a, y, b)
        else:
            point = (x, a, b)

        try:
            validate_point(
                point_xyz=point,
                shape=self.shape,
            )
        except ValueError:
            return

        self.selected = point
        self.cursor[:] = list(point)
        self._draw()

    def _accept(self, _event: object | None = None) -> None:
        if self.selected is None:
            print(
                "No point selected. Click inside the suspected lesion first."
            )
            return

        self.accepted = True
        self._plt.close(self.fig)

    def _reset(self, _event: object | None = None) -> None:
        self.selected = None
        self._draw()

    def _abort(self, _event: object | None = None) -> None:
        self.aborted = True
        self._plt.close(self.fig)

    def _on_key(self, event: object) -> None:
        key = str(
            getattr(event, "key", "")
        ).lower()

        if key in {"enter", "return"}:
            self._accept()
        elif key == "r":
            self._reset()
        elif key == "q":
            self._abort()

    def collect(self) -> ManualTumorClick | None:
        self._plt.show()

        if self.aborted:
            return None
        if not self.accepted or self.selected is None:
            return None

        original_point = map_voxel_point(
            point_xyz=self.selected,
            source_affine=self.display_affine,
            target_affine=self.original_affine,
            target_shape=self.original_shape,
        )

        return ManualTumorClick(
            case_id=self.case_id,
            quartile=self.quartile,
            point_xyz=original_point,
            display_point_ras_xyz=self.selected,
        )
