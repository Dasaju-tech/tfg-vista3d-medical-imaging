from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class CystDetectabilityCase:
    case_id: str
    reference_volume_ml: float
    dice: float
    recall: float

    @property
    def any_overlap(self) -> bool:
        return self.dice > 0.0

    @property
    def dice_ge_050(self) -> bool:
        return self.dice >= 0.50


def load_block50_cases(
    path: Path,
) -> tuple[CystDetectabilityCase, ...]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    rows = payload.get("rows")

    if not isinstance(rows, list):
        raise ValueError(
            f"Block 50 summary has no rows list: {path}"
        )

    cases = tuple(
        CystDetectabilityCase(
            case_id=str(row["case_id"]),
            reference_volume_ml=float(
                row["reference_volume_ml"]
            ),
            dice=float(row["dice"]),
            recall=float(row["recall"]),
        )
        for row in rows
    )

    if len(cases) != 46:
        raise ValueError(
            f"Expected 46 cyst-positive cases, got {len(cases)}."
        )

    return cases


def binary_auroc(
    scores: Iterable[float],
    labels: Iterable[bool],
) -> float:
    scores = np.asarray(
        tuple(scores),
        dtype=float,
    )
    labels = np.asarray(
        tuple(labels),
        dtype=bool,
    )

    positive = scores[labels]
    negative = scores[~labels]

    if positive.size == 0 or negative.size == 0:
        raise ValueError(
            "AUROC requires both positive and negative examples."
        )

    comparisons = (
        positive[:, None]
        - negative[None, :]
    )
    wins = np.count_nonzero(comparisons > 0.0)
    ties = np.count_nonzero(comparisons == 0.0)

    return float(
        (wins + 0.5 * ties)
        / comparisons.size
    )


def _balanced_accuracy(
    labels: np.ndarray,
    predicted: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)

    positive = labels
    negative = ~labels

    if not np.any(positive) or not np.any(negative):
        raise ValueError(
            "Balanced accuracy requires both classes."
        )

    sensitivity = float(
        np.mean(predicted[positive])
    )
    specificity = float(
        np.mean(~predicted[negative])
    )
    return 0.5 * (
        sensitivity + specificity
    )


def best_volume_threshold(
    *,
    volumes_ml: Iterable[float],
    labels: Iterable[bool],
) -> dict[str, float | int]:
    volumes = np.asarray(
        tuple(volumes_ml),
        dtype=float,
    )
    labels = np.asarray(
        tuple(labels),
        dtype=bool,
    )

    if volumes.size != labels.size:
        raise ValueError(
            "Volume and label lengths differ."
        )
    if volumes.size < 2:
        raise ValueError(
            "At least two cases are required."
        )
    if not np.any(labels) or np.all(labels):
        raise ValueError(
            "Threshold selection requires both classes."
        )

    unique = np.unique(volumes)
    candidates = np.concatenate(
        (
            np.asarray(
                [np.nextafter(unique[0], -np.inf)]
            ),
            (unique[:-1] + unique[1:]) / 2.0,
            np.asarray(
                [np.nextafter(unique[-1], np.inf)]
            ),
        )
    )

    best = None

    for threshold in candidates:
        predicted = volumes >= threshold
        ba = _balanced_accuracy(
            labels,
            predicted,
        )

        tp = int(
            np.count_nonzero(
                labels & predicted
            )
        )
        fn = int(
            np.count_nonzero(
                labels & ~predicted
            )
        )
        tn = int(
            np.count_nonzero(
                ~labels & ~predicted
            )
        )
        fp = int(
            np.count_nonzero(
                ~labels & predicted
            )
        )

        record = {
            "threshold_ml": float(threshold),
            "balanced_accuracy": float(ba),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

        if best is None:
            best = record
            continue

        if ba > float(best["balanced_accuracy"]) + 1e-12:
            best = record
            continue

        if (
            abs(
                ba - float(
                    best["balanced_accuracy"]
                )
            )
            <= 1e-12
            and threshold
            < float(best["threshold_ml"])
        ):
            best = record

    assert best is not None
    return best


def empirical_transition(
    *,
    volumes_ml: Iterable[float],
    labels: Iterable[bool],
) -> dict[str, float | None]:
    volumes = np.asarray(
        tuple(volumes_ml),
        dtype=float,
    )
    labels = np.asarray(
        tuple(labels),
        dtype=bool,
    )

    if volumes.size != labels.size:
        raise ValueError(
            "Volume and label lengths differ."
        )

    positive = volumes[labels]
    negative = volumes[~labels]

    if positive.size == 0 or negative.size == 0:
        return {
            "largest_negative_ml": (
                None
                if negative.size == 0
                else float(negative.max())
            ),
            "smallest_positive_ml": (
                None
                if positive.size == 0
                else float(positive.min())
            ),
            "separable": False,
            "gap_ml": None,
        }

    largest_negative = float(
        negative.max()
    )
    smallest_positive = float(
        positive.min()
    )

    return {
        "largest_negative_ml": largest_negative,
        "smallest_positive_ml": smallest_positive,
        "separable": bool(
            largest_negative < smallest_positive
        ),
        "gap_ml": float(
            smallest_positive
            - largest_negative
        ),
    }


def _bootstrap_auroc(
    cases: tuple[CystDetectabilityCase, ...],
    *,
    label_name: str,
    seed: int,
    samples: int = 5000,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []

    for _ in range(samples):
        indices = rng.integers(
            0,
            len(cases),
            size=len(cases),
        )
        sample = tuple(
            cases[int(index)]
            for index in indices
        )
        labels = np.asarray(
            [
                getattr(case, label_name)
                for case in sample
            ],
            dtype=bool,
        )

        if not np.any(labels) or np.all(labels):
            continue

        values.append(
            binary_auroc(
                (
                    case.reference_volume_ml
                    for case in sample
                ),
                labels,
            )
        )

    if not values:
        raise RuntimeError(
            "No valid bootstrap AUROC samples."
        )

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "lower_95": float(
            np.quantile(array, 0.025)
        ),
        "upper_95": float(
            np.quantile(array, 0.975)
        ),
        "valid_bootstrap_samples": int(
            array.size
        ),
    }


def _volume_bins(
    cases: tuple[CystDetectabilityCase, ...],
) -> list[dict[str, float | int]]:
    ordered = sorted(
        cases,
        key=lambda case: (
            case.reference_volume_ml,
            case.case_id,
        ),
    )

    groups = np.array_split(
        np.arange(len(ordered)),
        8,
    )
    rows = []

    for index, indices in enumerate(
        groups,
        start=1,
    ):
        group = tuple(
            ordered[int(item)]
            for item in indices
        )

        volumes = np.asarray(
            [
                case.reference_volume_ml
                for case in group
            ],
            dtype=float,
        )
        dice = np.asarray(
            [
                case.dice
                for case in group
            ],
            dtype=float,
        )
        recall = np.asarray(
            [
                case.recall
                for case in group
            ],
            dtype=float,
        )

        rows.append(
            {
                "bin": index,
                "count": len(group),
                "volume_min_ml": float(
                    volumes.min()
                ),
                "volume_median_ml": float(
                    np.median(volumes)
                ),
                "volume_max_ml": float(
                    volumes.max()
                ),
                "any_overlap_fraction": float(
                    np.mean(
                        [
                            case.any_overlap
                            for case in group
                        ]
                    )
                ),
                "dice_ge_050_fraction": float(
                    np.mean(
                        [
                            case.dice_ge_050
                            for case in group
                        ]
                    )
                ),
                "dice_mean": float(
                    dice.mean()
                ),
                "recall_mean": float(
                    recall.mean()
                ),
            }
        )

    return rows


def summarize_cyst_detectability(
    cases: Iterable[CystDetectabilityCase],
) -> dict[str, object]:
    cases = tuple(cases)

    if len(cases) != 46:
        raise ValueError(
            f"Expected 46 cyst-positive cases, got {len(cases)}."
        )

    volumes = np.asarray(
        [
            case.reference_volume_ml
            for case in cases
        ],
        dtype=float,
    )
    any_overlap = np.asarray(
        [
            case.any_overlap
            for case in cases
        ],
        dtype=bool,
    )
    dice_050 = np.asarray(
        [
            case.dice_ge_050
            for case in cases
        ],
        dtype=bool,
    )

    any_overlap_auroc = binary_auroc(
        volumes,
        any_overlap,
    )
    dice_050_auroc = binary_auroc(
        volumes,
        dice_050,
    )

    return {
        "schema_version": "1.0",
        "analysis": (
            "expanded100_cyst_detectability_limit"
        ),
        "case_count": len(cases),
        "event_counts": {
            "any_overlap": int(
                np.count_nonzero(
                    any_overlap
                )
            ),
            "zero_overlap": int(
                np.count_nonzero(
                    ~any_overlap
                )
            ),
            "dice_ge_050": int(
                np.count_nonzero(
                    dice_050
                )
            ),
            "dice_lt_050": int(
                np.count_nonzero(
                    ~dice_050
                )
            ),
        },
        "volume_as_predictor": {
            "any_overlap": {
                "auroc": any_overlap_auroc,
                "bootstrap_95_ci": _bootstrap_auroc(
                    cases,
                    label_name="any_overlap",
                    seed=20260911,
                ),
                "best_descriptive_threshold": (
                    best_volume_threshold(
                        volumes_ml=volumes,
                        labels=any_overlap,
                    )
                ),
                "empirical_transition": (
                    empirical_transition(
                        volumes_ml=volumes,
                        labels=any_overlap,
                    )
                ),
            },
            "dice_ge_050": {
                "auroc": dice_050_auroc,
                "bootstrap_95_ci": _bootstrap_auroc(
                    cases,
                    label_name="dice_ge_050",
                    seed=20260912,
                ),
                "best_descriptive_threshold": (
                    best_volume_threshold(
                        volumes_ml=volumes,
                        labels=dice_050,
                    )
                ),
                "empirical_transition": (
                    empirical_transition(
                        volumes_ml=volumes,
                        labels=dice_050,
                    )
                ),
            },
        },
        "volume_response_bins": _volume_bins(
            cases
        ),
        "interpretation_boundary": (
            "All thresholds are descriptive and selected in-sample on the "
            "same frozen expanded100 cyst-positive cohort. They quantify "
            "a size-dependent detectability transition and must not be "
            "interpreted as validated clinical cutoffs."
        ),
    }


def write_detectability_outputs(
    *,
    payload: dict[str, object],
    output_root: Path,
) -> None:
    import csv

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

    bins = payload["volume_response_bins"]
    assert isinstance(bins, list)

    with (
        output_root
        / "volume_response_bins.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        if not bins:
            raise ValueError(
                "No volume-response bins."
            )

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                bins[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(bins)


def write_detectability_report(
    *,
    payload: dict[str, object],
    path: Path,
) -> None:
    events = payload["event_counts"]
    predictors = payload[
        "volume_as_predictor"
    ]
    bins = payload[
        "volume_response_bins"
    ]

    assert isinstance(events, dict)
    assert isinstance(predictors, dict)
    assert isinstance(bins, list)

    overlap = predictors[
        "any_overlap"
    ]
    useful = predictors[
        "dice_ge_050"
    ]

    lines = [
        "# Expanded100 cyst detectability limit",
        "",
        "## Events",
        "",
        (
            f"- Any cyst overlap: "
            f"{events['any_overlap']}/{payload['case_count']}."
        ),
        (
            f"- Zero overlap: "
            f"{events['zero_overlap']}/{payload['case_count']}."
        ),
        (
            f"- Dice >= 0.50: "
            f"{events['dice_ge_050']}/{payload['case_count']}."
        ),
        "",
        "## Reference volume as a predictor",
        "",
        (
            f"- Any-overlap AUROC: "
            f"{overlap['auroc']:.3f}, bootstrap 95% CI "
            f"[{overlap['bootstrap_95_ci']['lower_95']:.3f}, "
            f"{overlap['bootstrap_95_ci']['upper_95']:.3f}]."
        ),
        (
            f"- Dice>=0.50 AUROC: "
            f"{useful['auroc']:.3f}, bootstrap 95% CI "
            f"[{useful['bootstrap_95_ci']['lower_95']:.3f}, "
            f"{useful['bootstrap_95_ci']['upper_95']:.3f}]."
        ),
        "",
        "## Descriptive thresholds",
        "",
        (
            f"- Any overlap: "
            f"{overlap['best_descriptive_threshold']['threshold_ml']:.2f} mL "
            f"(balanced accuracy "
            f"{overlap['best_descriptive_threshold']['balanced_accuracy']:.3f})."
        ),
        (
            f"- Dice>=0.50: "
            f"{useful['best_descriptive_threshold']['threshold_ml']:.2f} mL "
            f"(balanced accuracy "
            f"{useful['best_descriptive_threshold']['balanced_accuracy']:.3f})."
        ),
        "",
        "## Volume-response bins",
        "",
        "| Bin | N | Volume median mL | Any overlap | Dice>=0.50 | Dice mean | Recall mean |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in bins:
        lines.append(
            f"| {row['bin']} "
            f"| {row['count']} "
            f"| {row['volume_median_ml']:.2f} "
            f"| {row['any_overlap_fraction']:.2f} "
            f"| {row['dice_ge_050_fraction']:.2f} "
            f"| {row['dice_mean']:.3f} "
            f"| {row['recall_mean']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(
                payload[
                    "interpretation_boundary"
                ]
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
