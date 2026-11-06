from __future__ import annotations

import numpy as np
import pytest

from vista3d_tfg.experiments.cyst_semantic_point import (
    LEFT_CYST_LABEL,
    RIGHT_CYST_LABEL,
)
from vista3d_tfg.workflows.user_guided_cyst import (
    LEFT_KIDNEY_LABEL,
    RIGHT_KIDNEY_LABEL,
    assign_cyst_class_from_automatic_kidneys,
)


def make_prediction() -> np.ndarray:
    prediction = np.zeros((30, 20, 20), dtype=np.uint8)
    prediction[2:8, 5:15, 5:15] = LEFT_KIDNEY_LABEL
    prediction[22:28, 5:15, 5:15] = RIGHT_KIDNEY_LABEL
    return prediction


def test_left_click_routes_to_left_cyst_class() -> None:
    assignment = assign_cyst_class_from_automatic_kidneys(
        automatic_prediction=make_prediction(),
        spacing_mm=(1.0, 1.0, 1.0),
        point_xyz=(9, 10, 10),
    )

    assert assignment.laterality == "left"
    assert assignment.class_label == LEFT_CYST_LABEL
    assert assignment.distance_to_left_kidney_mm < (
        assignment.distance_to_right_kidney_mm
    )


def test_right_click_routes_to_right_cyst_class() -> None:
    assignment = assign_cyst_class_from_automatic_kidneys(
        automatic_prediction=make_prediction(),
        spacing_mm=(2.0, 1.0, 1.0),
        point_xyz=(20, 10, 10),
    )

    assert assignment.laterality == "right"
    assert assignment.class_label == RIGHT_CYST_LABEL


def test_out_of_bounds_click_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        assign_cyst_class_from_automatic_kidneys(
            automatic_prediction=make_prediction(),
            spacing_mm=(1.0, 1.0, 1.0),
            point_xyz=(100, 10, 10),
        )


def test_no_automatic_kidneys_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="neither"):
        assign_cyst_class_from_automatic_kidneys(
            automatic_prediction=np.zeros(
                (10, 10, 10),
                dtype=np.uint8,
            ),
            spacing_mm=(1.0, 1.0, 1.0),
            point_xyz=(5, 5, 5),
        )


def test_ambiguous_click_is_rejected() -> None:
    prediction = np.zeros((11, 5, 5), dtype=np.uint8)
    prediction[1, 2, 2] = LEFT_KIDNEY_LABEL
    prediction[9, 2, 2] = RIGHT_KIDNEY_LABEL

    with pytest.raises(RuntimeError, match="ambiguous"):
        assign_cyst_class_from_automatic_kidneys(
            automatic_prediction=prediction,
            spacing_mm=(1.0, 1.0, 1.0),
            point_xyz=(5, 2, 2),
        )


def test_single_available_kidney_routes_deterministically() -> None:
    prediction = make_prediction()
    prediction[prediction == RIGHT_KIDNEY_LABEL] = 0

    assignment = assign_cyst_class_from_automatic_kidneys(
        automatic_prediction=prediction,
        spacing_mm=(1.0, 1.0, 1.0),
        point_xyz=(20, 10, 10),
    )

    assert assignment.class_label == LEFT_CYST_LABEL
    assert np.isinf(assignment.distance_to_right_kidney_mm)
