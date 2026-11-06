from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


FEATURES = (
    "largest_to_second_ratio",
    "largest_component_volume_ml",
    "third_plus_volume_ml",
    "total_renal_components",
    "third_plus_fraction",
    "right_components",
    "left_components",
    "total_renal_volume_ml",
    "second_component_volume_ml",
    "smallest_component_volume_ml",
    "max_component_centroid_distance_mm",
)


@dataclass(frozen=True)
class TriggerCase:
    case_id: str
    target: bool
    hd95_outlier: bool
    features: dict[str, float]


@dataclass(frozen=True)
class MetricTriplet:
    dice: float
    surface_dice: float
    hd95_mm: float


@dataclass(frozen=True)
class FilterCase:
    case_id: str
    original: MetricTriplet
    global_top2: MetricTriplet
    side_largest: MetricTriplet


@dataclass(frozen=True)
class FoldDecision:
    case_id: str
    target: bool
    triggered: bool
    selected_feature: str
    direction: str
    threshold: float
    training_auc: float
    training_balanced_accuracy: float


def _float_or_zero(value: str | None) -> float:
    if value is None:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    return float(value)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def read_trigger_cases(path: Path) -> tuple[TriggerCase, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    if len(rows) != 100:
        raise ValueError(
            f"Expected 100 prediction-geometry rows, got {len(rows)}."
        )

    output = tuple(
        TriggerCase(
            case_id=row["case_id"],
            target=_parse_bool(row["prediction_to_gt_target"]),
            hd95_outlier=_parse_bool(row["hd95_outlier"]),
            features={
                feature: _float_or_zero(row.get(feature))
                for feature in FEATURES
            },
        )
        for row in rows
    )

    if sum(row.target for row in output) != 12:
        raise ValueError("Expected exactly 12 target cases.")

    if len({row.case_id for row in output}) != 100:
        raise ValueError("Duplicate case IDs in trigger table.")

    return output


def read_filter_cases(path: Path) -> tuple[FilterCase, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(csv.DictReader(handle))

    if len(rows) != 100:
        raise ValueError(
            f"Expected 100 component-filter rows, got {len(rows)}."
        )

    output = tuple(
        FilterCase(
            case_id=row["case_id"],
            original=MetricTriplet(
                dice=float(row["original_dice"]),
                surface_dice=float(row["original_surface_dice"]),
                hd95_mm=float(row["original_hd95_mm"]),
            ),
            global_top2=MetricTriplet(
                dice=float(row["global_top2_dice"]),
                surface_dice=float(
                    row["global_top2_surface_dice"]
                ),
                hd95_mm=float(row["global_top2_hd95_mm"]),
            ),
            side_largest=MetricTriplet(
                dice=float(row["side_largest_dice"]),
                surface_dice=float(
                    row["side_largest_surface_dice"]
                ),
                hd95_mm=float(row["side_largest_hd95_mm"]),
            ),
        )
        for row in rows
    )

    if len({row.case_id for row in output}) != 100:
        raise ValueError("Duplicate case IDs in filter table.")

    return output


def rank_auc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    positives = int(np.count_nonzero(labels))
    negatives = int(labels.size - positives)

    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)

    start = 0
    while start < scores.size:
        end = start + 1
        while (
            end < scores.size
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[labels].sum())
    u = (
        positive_rank_sum
        - positives * (positives + 1) / 2.0
    )
    return u / (positives * negatives)


def _balanced_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    predictions = np.asarray(predictions, dtype=bool)
    labels = np.asarray(labels, dtype=bool)

    tp = int(np.count_nonzero(predictions & labels))
    fn = int(np.count_nonzero(~predictions & labels))
    tn = int(np.count_nonzero(~predictions & ~labels))
    fp = int(np.count_nonzero(predictions & ~labels))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return 0.5 * (sensitivity + specificity)


def _candidate_thresholds(values: np.ndarray) -> tuple[float, ...]:
    unique = np.unique(np.asarray(values, dtype=float))

    if unique.size == 1:
        value = float(unique[0])
        return (value,)

    mids = [
        float((left + right) / 2.0)
        for left, right in zip(unique[:-1], unique[1:])
    ]

    epsilon = max(
        1e-12,
        float(np.ptp(unique)) * 1e-9,
    )

    return tuple(
        [float(unique[0] - epsilon)]
        + mids
        + [float(unique[-1] + epsilon)]
    )


def fit_univariate_trigger(
    rows: Iterable[TriggerCase],
) -> tuple[str, str, float, float, float]:
    rows = tuple(rows)

    if sum(row.target for row in rows) == 0:
        raise ValueError("Training fold has no targets.")
    if sum(not row.target for row in rows) == 0:
        raise ValueError("Training fold has no non-targets.")

    labels = np.asarray(
        [row.target for row in rows],
        dtype=bool,
    )

    best: tuple[
        float,
        float,
        float,
        str,
        str,
        float,
    ] | None = None

    for feature in FEATURES:
        values = np.asarray(
            [row.features[feature] for row in rows],
            dtype=float,
        )

        auc_high = rank_auc(values, labels)
        auc_low = rank_auc(-values, labels)

        if auc_high >= auc_low:
            direction = "high"
            auc = auc_high
        else:
            direction = "low"
            auc = auc_low

        for threshold in _candidate_thresholds(values):
            if direction == "high":
                predictions = values >= threshold
            else:
                predictions = values <= threshold

            balanced = _balanced_accuracy(
                predictions,
                labels,
            )
            predicted_positive = int(
                np.count_nonzero(predictions)
            )

            # Deterministic preference:
            # 1) highest balanced accuracy,
            # 2) highest AUROC,
            # 3) fewer triggered training cases,
            # 4) lexical feature/direction/threshold stability.
            candidate = (
                balanced,
                auc,
                -float(predicted_positive),
                feature,
                direction,
                float(threshold),
            )

            if best is None or candidate > best:
                best = candidate

    assert best is not None
    balanced, auc, _, feature, direction, threshold = best

    return (
        feature,
        direction,
        threshold,
        auc,
        balanced,
    )


def leave_one_out_decisions(
    rows: Iterable[TriggerCase],
) -> tuple[FoldDecision, ...]:
    rows = tuple(rows)

    if len(rows) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(rows)}."
        )

    decisions = []

    for index, held_out in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]

        (
            feature,
            direction,
            threshold,
            auc,
            balanced,
        ) = fit_univariate_trigger(training)

        value = held_out.features[feature]
        triggered = (
            value >= threshold
            if direction == "high"
            else value <= threshold
        )

        decisions.append(
            FoldDecision(
                case_id=held_out.case_id,
                target=held_out.target,
                triggered=bool(triggered),
                selected_feature=feature,
                direction=direction,
                threshold=threshold,
                training_auc=auc,
                training_balanced_accuracy=balanced,
            )
        )

    return tuple(decisions)


def confusion_summary(
    decisions: Iterable[FoldDecision],
) -> dict[str, float | int]:
    decisions = tuple(decisions)

    tp = sum(d.triggered and d.target for d in decisions)
    fp = sum(d.triggered and not d.target for d in decisions)
    tn = sum(
        not d.triggered and not d.target
        for d in decisions
    )
    fn = sum(
        not d.triggered and d.target
        for d in decisions
    )

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if tp + fp else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": (
            0.5 * (sensitivity + specificity)
        ),
        "triggered_count": tp + fp,
    }


def _metric_summary(
    metrics: Iterable[MetricTriplet],
) -> dict[str, float | int]:
    metrics = tuple(metrics)

    dice = np.asarray(
        [metric.dice for metric in metrics],
        dtype=float,
    )
    surface = np.asarray(
        [metric.surface_dice for metric in metrics],
        dtype=float,
    )
    hd95 = np.asarray(
        [metric.hd95_mm for metric in metrics],
        dtype=float,
    )

    return {
        "count": len(metrics),
        "dice_mean": float(dice.mean()),
        "surface_dice_mean": float(surface.mean()),
        "hd95_mean_mm": float(hd95.mean()),
        "hd95_median_mm": float(np.median(hd95)),
    }


def analyze_selective_filtering(
    *,
    trigger_rows: Iterable[TriggerCase],
    filter_rows: Iterable[FilterCase],
) -> dict[str, object]:
    trigger_rows = tuple(trigger_rows)
    filter_rows = tuple(filter_rows)

    decisions = leave_one_out_decisions(trigger_rows)
    trigger_summary = confusion_summary(decisions)

    filters_by_id = {
        row.case_id: row
        for row in filter_rows
    }
    decisions_by_id = {
        row.case_id: row
        for row in decisions
    }

    if set(filters_by_id) != set(decisions_by_id):
        raise ValueError(
            "Trigger and filter case IDs do not match."
        )

    original_metrics = tuple(
        filters_by_id[case_id].original
        for case_id in sorted(filters_by_id)
    )
    original_hd95 = np.asarray(
        [metric.hd95_mm for metric in original_metrics],
        dtype=float,
    )
    q1, q3 = np.percentile(original_hd95, [25.0, 75.0])
    original_threshold = float(q3 + 1.5 * (q3 - q1))

    variants = {}

    for variant in ("global_top2", "side_largest"):
        selected_metrics = []
        changed_case_ids = []
        target_changed = []
        non_target_changed = []

        for case_id in sorted(filters_by_id):
            filter_case = filters_by_id[case_id]
            decision = decisions_by_id[case_id]

            if decision.triggered:
                metric = getattr(filter_case, variant)
                changed_case_ids.append(case_id)
                if decision.target:
                    target_changed.append(case_id)
                else:
                    non_target_changed.append(case_id)
            else:
                metric = filter_case.original

            selected_metrics.append(metric)

        selected_metrics = tuple(selected_metrics)

        original_by_id = {
            row.case_id: row.original
            for row in filter_rows
        }
        selected_by_id = {
            case_id: metric
            for case_id, metric in zip(
                sorted(filters_by_id),
                selected_metrics,
            )
        }

        hd95_reductions = {
            case_id: (
                original_by_id[case_id].hd95_mm
                - selected_by_id[case_id].hd95_mm
            )
            for case_id in selected_by_id
        }
        dice_deltas = {
            case_id: (
                selected_by_id[case_id].dice
                - original_by_id[case_id].dice
            )
            for case_id in selected_by_id
        }

        target_ids = {
            row.case_id
            for row in trigger_rows
            if row.target
        }
        non_target_ids = set(selected_by_id) - target_ids

        variants[variant] = {
            "all_cases": _metric_summary(selected_metrics),
            "remaining_outliers_at_original_threshold": sum(
                metric.hd95_mm > original_threshold
                for metric in selected_metrics
            ),
            "changed_count": len(changed_case_ids),
            "changed_case_ids": changed_case_ids,
            "target_changed_count": len(target_changed),
            "non_target_changed_count": len(non_target_changed),
            "target_hd95_reduction_mean_mm": float(
                np.mean(
                    [
                        hd95_reductions[case_id]
                        for case_id in target_ids
                    ]
                )
            ),
            "target_outliers_resolved": sum(
                original_by_id[case_id].hd95_mm
                > original_threshold
                and selected_by_id[case_id].hd95_mm
                <= original_threshold
                for case_id in target_ids
            ),
            "non_target_hd95_worsened_gt_1mm": sum(
                hd95_reductions[case_id] < -1.0
                for case_id in non_target_ids
            ),
            "non_target_dice_worsened_gt_0_01": sum(
                dice_deltas[case_id] < -0.01
                for case_id in non_target_ids
            ),
        }

    feature_counts: dict[str, int] = {}
    for decision in decisions:
        feature_counts[decision.selected_feature] = (
            feature_counts.get(
                decision.selected_feature,
                0,
            )
            + 1
        )

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_expanded100_loocv_selective_filter_validation"
        ),
        "case_count": 100,
        "target_count": 12,
        "validation": (
            "leave-one-out feature+threshold fitting; each held-out "
            "decision is generated without its own target label"
        ),
        "trigger": trigger_summary,
        "selected_feature_frequency": dict(
            sorted(
                feature_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "original_hd95_outlier_threshold_mm": original_threshold,
        "original": _metric_summary(original_metrics),
        "variants": variants,
        "fold_decisions": [
            {
                "case_id": d.case_id,
                "target": d.target,
                "triggered": d.triggered,
                "selected_feature": d.selected_feature,
                "direction": d.direction,
                "threshold": d.threshold,
                "training_auc": d.training_auc,
                "training_balanced_accuracy": (
                    d.training_balanced_accuracy
                ),
            }
            for d in decisions
        ],
        "interpretation_boundary": (
            "LOOCV reduces direct threshold fitting bias within expanded100 "
            "but does not constitute external validation. Feature discovery "
            "originated in the same cohort, so a final trigger still requires "
            "validation on additional cases before deployment claims."
        ),
    }
