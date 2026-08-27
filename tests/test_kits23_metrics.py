from math import isinf

import numpy as np
import pytest

from vista3d_tfg.metrics.kits23 import (
    KITS23_HECS,
    construct_hec_mask,
    dice_score,
    evaluate_kits23_case,
    validate_kits23_segmentation,
)


def make_reference() -> np.ndarray:
    segmentation = np.zeros((12, 12, 12), dtype=np.uint8)

    # Kidney parenchyma.
    segmentation[2:10, 2:10, 2:10] = 1

    # Tumor.
    segmentation[4:6, 4:6, 4:6] = 2

    # Cyst.
    segmentation[7:9, 7:9, 7:9] = 3

    return segmentation


def test_official_hec_definitions() -> None:
    assert [item.name for item in KITS23_HECS] == [
        "kidney_and_mass",
        "mass",
        "tumor",
    ]

    assert [item.labels for item in KITS23_HECS] == [
        (1, 2, 3),
        (2, 3),
        (2,),
    ]

    assert KITS23_HECS[0].surface_tolerance_mm == pytest.approx(
        1.0330772532390826
    )
    assert KITS23_HECS[1].surface_tolerance_mm == pytest.approx(
        1.1328796488598762
    )
    assert KITS23_HECS[2].surface_tolerance_mm == pytest.approx(
        1.1498198361434828
    )


def test_construct_hec_mask_merges_labels() -> None:
    segmentation = make_reference()

    mass = construct_hec_mask(segmentation, (2, 3))

    assert np.count_nonzero(mass) == 16
    assert np.all(mass[4:6, 4:6, 4:6])
    assert np.all(mass[7:9, 7:9, 7:9])


def test_dice_perfect_overlap() -> None:
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True

    assert dice_score(mask, mask) == pytest.approx(1.0)


def test_dice_both_empty_is_one() -> None:
    empty = np.zeros((5, 5, 5), dtype=bool)

    assert dice_score(empty, empty) == pytest.approx(1.0)


def test_dice_one_empty_is_zero() -> None:
    prediction = np.zeros((5, 5, 5), dtype=bool)
    reference = np.zeros((5, 5, 5), dtype=bool)
    reference[2, 2, 2] = True

    assert dice_score(prediction, reference) == pytest.approx(0.0)


def test_perfect_case_scores_one() -> None:
    reference = make_reference()

    result = evaluate_kits23_case(
        reference.copy(),
        reference,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    assert result.mean_dice == pytest.approx(1.0)
    assert result.mean_surface_dice == pytest.approx(1.0)

    for hec in result.hecs:
        assert hec.dice == pytest.approx(1.0)
        assert hec.surface_dice == pytest.approx(1.0)
        assert hec.hd95_mm == pytest.approx(0.0)


def test_absent_tumor_in_both_is_perfect_for_tumor_hec() -> None:
    segmentation = make_reference()
    segmentation[segmentation == 2] = 1

    result = evaluate_kits23_case(
        segmentation.copy(),
        segmentation,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    tumor = result.hecs[2]

    assert tumor.name == "tumor"
    assert tumor.dice == pytest.approx(1.0)
    assert tumor.surface_dice == pytest.approx(1.0)
    assert tumor.hd95_mm == pytest.approx(0.0)


def test_missing_prediction_tumor_scores_zero_and_infinite_hd95() -> None:
    reference = make_reference()
    prediction = reference.copy()
    prediction[prediction == 2] = 1

    result = evaluate_kits23_case(
        prediction,
        reference,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    tumor = result.hecs[2]

    assert tumor.dice == pytest.approx(0.0)
    assert tumor.surface_dice == pytest.approx(0.0)
    assert isinf(tumor.hd95_mm)


def test_shifted_tumor_degrades_metrics() -> None:
    reference = make_reference()
    prediction = reference.copy()

    prediction[prediction == 2] = 1
    prediction[4:6, 4:6, 7:9] = 2

    result = evaluate_kits23_case(
        prediction,
        reference,
        spacing_mm=(1.0, 1.0, 1.0),
    )

    tumor = result.hecs[2]

    assert 0.0 <= tumor.dice < 1.0
    assert 0.0 <= tumor.surface_dice < 1.0
    assert tumor.hd95_mm > 0.0


def test_case_result_serializes_protocol_metadata() -> None:
    reference = make_reference()

    payload = evaluate_kits23_case(
        reference,
        reference,
        spacing_mm=(1.0, 1.0, 1.0),
    ).to_dict()

    assert payload["protocol"] == "KiTS23"
    assert payload["official_metrics"] == ["dice", "surface_dice"]
    assert payload["diagnostic_metrics"] == ["hd95_mm"]
    assert len(payload["hecs"]) == 3


@pytest.mark.parametrize(
    "spacing",
    [
        (1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, np.nan, 1.0),
    ],
)
def test_invalid_spacing_is_rejected(spacing) -> None:
    reference = make_reference()

    with pytest.raises(ValueError):
        evaluate_kits23_case(
            reference,
            reference,
            spacing_mm=spacing,
        )


def test_shape_mismatch_is_rejected() -> None:
    prediction = np.zeros((4, 4, 4), dtype=np.uint8)
    reference = np.zeros((5, 5, 5), dtype=np.uint8)

    with pytest.raises(ValueError):
        evaluate_kits23_case(
            prediction,
            reference,
            spacing_mm=(1.0, 1.0, 1.0),
        )


def test_unexpected_label_is_rejected() -> None:
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1, 1, 1] = 4

    with pytest.raises(ValueError, match="unexpected"):
        validate_kits23_segmentation(segmentation)


def test_floating_segmentation_is_rejected() -> None:
    segmentation = np.zeros((4, 4, 4), dtype=np.float32)

    with pytest.raises(TypeError, match="integer dtype"):
        validate_kits23_segmentation(segmentation)
