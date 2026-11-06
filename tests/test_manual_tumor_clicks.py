from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vista3d_tfg.interaction.manual_tumor_clicks import (
    ManualTumorClick,
    load_manual_click,
    map_voxel_point,
    save_manual_click,
    validate_point,
    window_ct,
)


def test_window_ct_maps_requested_range() -> None:
    array = np.asarray(
        [-500.0, -200.0, 50.0, 300.0, 700.0],
        dtype=np.float32,
    )

    result = window_ct(
        array,
        lower_hu=-200.0,
        upper_hu=300.0,
    )

    assert result.tolist() == pytest.approx(
        [0.0, 0.0, 0.5, 1.0, 1.0]
    )


def test_manual_click_round_trip_certifies_gt_hidden(
    tmp_path: Path,
) -> None:
    path = tmp_path / "click.json"
    click = ManualTumorClick(
        case_id="case_00141",
        quartile="smallest_quartile",
        point_xyz=(10, 20, 30),
        display_point_ras_xyz=(30, 20, 10),
    )

    save_manual_click(
        click=click,
        output_path=path,
    )
    loaded = load_manual_click(path)

    assert loaded == click

    text = path.read_text(encoding="utf-8")
    assert '"gt_visible_during_selection": false' in text
    assert '"deployment_plausible_localization": true' in text
    assert '"display_orientation": "RAS"' in text
    assert '"schema_version": "1.1"' in text


def test_validate_point_rejects_out_of_bounds() -> None:
    validate_point(
        point_xyz=(0, 9, 4),
        shape=(10, 10, 5),
    )

    with pytest.raises(ValueError, match="outside"):
        validate_point(
            point_xyz=(10, 9, 4),
            shape=(10, 10, 5),
        )


def test_manual_click_viewer_api_has_no_ground_truth_input() -> None:
    import inspect
    import vista3d_tfg.interaction.manual_tumor_clicks as module

    parameters = inspect.signature(
        module.OrthogonalClickViewer.__init__
    ).parameters

    assert "segmentation_path" not in parameters
    assert "ground_truth_path" not in parameters
    assert "mask_path" not in parameters
    assert "image_path" in parameters


def test_voxel_mapping_handles_axis_permutation_and_flip() -> None:
    source_affine = np.asarray(
        [
            [0.0, 0.0, -1.0, 9.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    mapped = map_voxel_point(
        point_xyz=(2, 3, 4),
        source_affine=source_affine,
        target_affine=np.eye(4),
        target_shape=(10, 10, 10),
    )

    assert mapped == (5, 2, 3)


def test_legacy_click_artifact_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        """{
  "schema_version": "1.0",
  "gt_visible_during_selection": false,
  "click": {
    "case_id": "case_00141",
    "quartile": "smallest_quartile",
    "point_xyz": [1, 2, 3],
    "source": "manual_ct_only",
    "gt_visible_during_selection": false
  }
}""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy"):
        load_manual_click(path)
