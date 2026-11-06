from __future__ import annotations

import numpy as np
import pytest

from vista3d_tfg.evaluation.automatic_cyst_laterality import (
    LEFT_KIDNEY_LABEL,
    RIGHT_KIDNEY_LABEL,
    infer_cyst_class_from_automatic_kidneys,
    summarize_automatic_laterality,
)
from vista3d_tfg.experiments.cyst_semantic_point import (
    LEFT_CYST_LABEL,
    RIGHT_CYST_LABEL,
)


def make_prediction() -> np.ndarray:
    prediction = np.zeros((30, 20, 20), dtype=np.uint8)
    prediction[2:8, 5:15, 5:15] = LEFT_KIDNEY_LABEL
    prediction[22:28, 5:15, 5:15] = RIGHT_KIDNEY_LABEL
    return prediction


def test_point_near_left_kidney_maps_to_left_cyst() -> None:
    decision = infer_cyst_class_from_automatic_kidneys(
        case_id="fixture",
        automatic_prediction=make_prediction(),
        spacing_mm=(1.0, 1.0, 1.0),
        point_xyz=(9, 10, 10),
        oracle_cyst_class=LEFT_CYST_LABEL,
    )

    assert decision.inferred_laterality == "left"
    assert decision.inferred_cyst_class == LEFT_CYST_LABEL
    assert decision.agrees_with_oracle is True
    assert decision.distance_to_left_kidney_mm < (
        decision.distance_to_right_kidney_mm
    )


def test_point_near_right_kidney_maps_to_right_cyst() -> None:
    decision = infer_cyst_class_from_automatic_kidneys(
        case_id="fixture",
        automatic_prediction=make_prediction(),
        spacing_mm=(2.0, 1.0, 1.0),
        point_xyz=(20, 10, 10),
        oracle_cyst_class=RIGHT_CYST_LABEL,
    )

    assert decision.inferred_laterality == "right"
    assert decision.inferred_cyst_class == RIGHT_CYST_LABEL
    assert decision.agrees_with_oracle is True


def test_missing_one_kidney_uses_available_side() -> None:
    prediction = make_prediction()
    prediction[prediction == RIGHT_KIDNEY_LABEL] = 0

    decision = infer_cyst_class_from_automatic_kidneys(
        case_id="fixture",
        automatic_prediction=prediction,
        spacing_mm=(1.0, 1.0, 1.0),
        point_xyz=(20, 10, 10),
        oracle_cyst_class=LEFT_CYST_LABEL,
    )

    assert decision.inferred_cyst_class == LEFT_CYST_LABEL
    assert decision.right_kidney_present is False
    assert np.isinf(decision.distance_to_right_kidney_mm)


def test_no_kidneys_is_rejected() -> None:
    prediction = np.zeros((10, 10, 10), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="neither kidney"):
        infer_cyst_class_from_automatic_kidneys(
            case_id="fixture",
            automatic_prediction=prediction,
            spacing_mm=(1.0, 1.0, 1.0),
            point_xyz=(5, 5, 5),
            oracle_cyst_class=LEFT_CYST_LABEL,
        )


def test_equal_distance_is_rejected_as_ambiguous() -> None:
    prediction = np.zeros((11, 5, 5), dtype=np.uint8)
    prediction[1, 2, 2] = LEFT_KIDNEY_LABEL
    prediction[9, 2, 2] = RIGHT_KIDNEY_LABEL

    with pytest.raises(RuntimeError, match="ambiguous"):
        infer_cyst_class_from_automatic_kidneys(
            case_id="fixture",
            automatic_prediction=prediction,
            spacing_mm=(1.0, 1.0, 1.0),
            point_xyz=(5, 2, 2),
            oracle_cyst_class=LEFT_CYST_LABEL,
        )


def test_summary_counts_agreement() -> None:
    prediction = make_prediction()

    left = infer_cyst_class_from_automatic_kidneys(
        case_id="left",
        automatic_prediction=prediction,
        spacing_mm=(1.0, 1.0, 1.0),
        point_xyz=(9, 10, 10),
        oracle_cyst_class=LEFT_CYST_LABEL,
    )
    right = infer_cyst_class_from_automatic_kidneys(
        case_id="right",
        automatic_prediction=prediction,
        spacing_mm=(1.0, 1.0, 1.0),
        point_xyz=(20, 10, 10),
        oracle_cyst_class=RIGHT_CYST_LABEL,
    )

    summary = summarize_automatic_laterality([left, right])

    assert summary.case_count == 2
    assert summary.agreement_count == 2
    assert summary.agreement_fraction == 1.0
    assert summary.both_kidneys_present_count == 2
