from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vista3d_tfg.experiments.cyst_oracle_click_curve import (
    OracleClickTrajectory,
)


BOOTSTRAP_SEED = 20260828
BOOTSTRAP_SAMPLES = 5000


@dataclass(frozen=True, slots=True)
class TransitionObservation:
    case_id: str
    from_click: int
    to_click: int
    point_label: int
    source: str
    dice_before: float
    dice_after: float
    dice_delta: float


@dataclass(frozen=True, slots=True)
class TransitionGroupSummary:
    count: int
    mean_delta: float
    median_delta: float
    improved_count: int
    regressed_count: int
    unchanged_count: int
    improved_fraction: float
    regressed_fraction: float


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class InteractionStabilitySummary:
    case_count: int

    click1_mean: float
    click1_median: float
    click2_mean: float
    click2_median: float
    click3_mean: float
    click3_median: float

    best_observed_mean: float
    best_observed_median: float
    best_minus_click1_mean: float
    best_minus_click1_median: float
    best_minus_click1_bootstrap_95ci: BootstrapInterval

    best_round_click1_count: int
    best_round_click2_count: int
    best_round_click3_count: int

    best_observed_dice_ge_050_count: int
    best_observed_dice_ge_080_count: int

    final_regret_vs_best_mean: float
    final_regret_vs_best_median: float
    final_regret_ge_005_count: int
    final_regret_ge_010_count: int

    positive_correction: TransitionGroupSummary | None
    negative_correction: TransitionGroupSummary | None
    click2_transition: TransitionGroupSummary
    click3_transition: TransitionGroupSummary

    largest_regression_case_ids: tuple[str, ...]
    largest_improvement_case_ids: tuple[str, ...]

    transitions: tuple[TransitionObservation, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _dice_at_click(
    trajectory: OracleClickTrajectory,
    click_count: int,
) -> float:
    eligible = [
        result
        for result in trajectory.rounds
        if result.click_count <= click_count
    ]

    if not eligible:
        raise ValueError(
            f"{trajectory.case_id} has no result at/before click {click_count}."
        )

    return float(eligible[-1].dice)


def build_transition_observations(
    trajectories: Iterable[OracleClickTrajectory],
) -> tuple[TransitionObservation, ...]:
    trajectories = tuple(trajectories)
    observations: list[TransitionObservation] = []

    for trajectory in trajectories:
        round_by_count = {
            result.click_count: result
            for result in trajectory.rounds
        }
        click_by_index = {
            click.index: click
            for click in trajectory.clicks
        }

        for to_click in (2, 3):
            if to_click not in round_by_count:
                continue
            if (to_click - 1) not in round_by_count:
                continue
            if to_click not in click_by_index:
                raise ValueError(
                    f"{trajectory.case_id} is missing click metadata "
                    f"for round {to_click}."
                )

            before = float(round_by_count[to_click - 1].dice)
            after = float(round_by_count[to_click].dice)
            click = click_by_index[to_click]

            observations.append(
                TransitionObservation(
                    case_id=trajectory.case_id,
                    from_click=to_click - 1,
                    to_click=to_click,
                    point_label=int(click.point_label),
                    source=str(click.source),
                    dice_before=before,
                    dice_after=after,
                    dice_delta=after - before,
                )
            )

    return tuple(observations)


def summarize_transition_group(
    observations: Iterable[TransitionObservation],
) -> TransitionGroupSummary | None:
    observations = tuple(observations)

    if not observations:
        return None

    deltas = np.asarray(
        [observation.dice_delta for observation in observations],
        dtype=float,
    )

    improved = int(np.count_nonzero(deltas > 1e-12))
    regressed = int(np.count_nonzero(deltas < -1e-12))
    unchanged = len(observations) - improved - regressed

    return TransitionGroupSummary(
        count=len(observations),
        mean_delta=float(np.mean(deltas)),
        median_delta=float(np.median(deltas)),
        improved_count=improved,
        regressed_count=regressed,
        unchanged_count=unchanged,
        improved_fraction=float(improved / len(observations)),
        regressed_fraction=float(regressed / len(observations)),
    )


def bootstrap_mean_difference_ci(
    paired_differences: Iterable[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> BootstrapInterval:
    values = np.asarray(
        tuple(float(value) for value in paired_differences),
        dtype=float,
    )

    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sequence.")
    if samples <= 0:
        raise ValueError("samples must be positive.")

    rng = np.random.default_rng(seed)
    indexes = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    boot_means = values[indexes].mean(axis=1)

    return BootstrapInterval(
        lower=float(np.quantile(boot_means, 0.025)),
        upper=float(np.quantile(boot_means, 0.975)),
    )


def summarize_interaction_stability(
    trajectories: Iterable[OracleClickTrajectory],
) -> InteractionStabilitySummary:
    trajectories = tuple(trajectories)

    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    click1 = np.asarray(
        [_dice_at_click(trajectory, 1) for trajectory in trajectories],
        dtype=float,
    )
    click2 = np.asarray(
        [_dice_at_click(trajectory, 2) for trajectory in trajectories],
        dtype=float,
    )
    click3 = np.asarray(
        [_dice_at_click(trajectory, 3) for trajectory in trajectories],
        dtype=float,
    )

    all_rounds = np.stack([click1, click2, click3], axis=1)
    best = np.max(all_rounds, axis=1)

    # np.argmax is deterministic and chooses the earliest round in a tie,
    # matching the minimal-interaction interpretation.
    best_rounds = np.argmax(all_rounds, axis=1) + 1

    final_regret = best - click3
    best_gain = best - click1

    transitions = build_transition_observations(trajectories)

    positive = tuple(
        observation
        for observation in transitions
        if observation.point_label == 1
    )
    negative = tuple(
        observation
        for observation in transitions
        if observation.point_label == 0
    )
    click2_transitions = tuple(
        observation
        for observation in transitions
        if observation.to_click == 2
    )
    click3_transitions = tuple(
        observation
        for observation in transitions
        if observation.to_click == 3
    )

    click2_summary = summarize_transition_group(click2_transitions)
    click3_summary = summarize_transition_group(click3_transitions)

    if click2_summary is None or click3_summary is None:
        raise ValueError(
            "Full three-round trajectories are required for stability audit."
        )

    per_case_change = [
        (
            trajectory.case_id,
            float(_dice_at_click(trajectory, 3) - _dice_at_click(trajectory, 1)),
        )
        for trajectory in trajectories
    ]

    largest_regressions = tuple(
        case_id
        for case_id, _ in sorted(
            per_case_change,
            key=lambda item: (item[1], item[0]),
        )[:5]
    )
    largest_improvements = tuple(
        case_id
        for case_id, _ in sorted(
            per_case_change,
            key=lambda item: (-item[1], item[0]),
        )[:5]
    )

    return InteractionStabilitySummary(
        case_count=len(trajectories),

        click1_mean=float(np.mean(click1)),
        click1_median=float(np.median(click1)),
        click2_mean=float(np.mean(click2)),
        click2_median=float(np.median(click2)),
        click3_mean=float(np.mean(click3)),
        click3_median=float(np.median(click3)),

        best_observed_mean=float(np.mean(best)),
        best_observed_median=float(np.median(best)),
        best_minus_click1_mean=float(np.mean(best_gain)),
        best_minus_click1_median=float(np.median(best_gain)),
        best_minus_click1_bootstrap_95ci=bootstrap_mean_difference_ci(
            best_gain
        ),

        best_round_click1_count=int(np.count_nonzero(best_rounds == 1)),
        best_round_click2_count=int(np.count_nonzero(best_rounds == 2)),
        best_round_click3_count=int(np.count_nonzero(best_rounds == 3)),

        best_observed_dice_ge_050_count=int(np.count_nonzero(best >= 0.5)),
        best_observed_dice_ge_080_count=int(np.count_nonzero(best >= 0.8)),

        final_regret_vs_best_mean=float(np.mean(final_regret)),
        final_regret_vs_best_median=float(np.median(final_regret)),
        final_regret_ge_005_count=int(
            np.count_nonzero(final_regret >= 0.05)
        ),
        final_regret_ge_010_count=int(
            np.count_nonzero(final_regret >= 0.10)
        ),

        positive_correction=summarize_transition_group(positive),
        negative_correction=summarize_transition_group(negative),
        click2_transition=click2_summary,
        click3_transition=click3_summary,

        largest_regression_case_ids=largest_regressions,
        largest_improvement_case_ids=largest_improvements,

        transitions=transitions,
    )


def load_click_curve_summary(
    path: Path,
) -> tuple[OracleClickTrajectory, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Click-curve summary does not contain result object.")

    raw_trajectories = result.get("trajectories")
    if not isinstance(raw_trajectories, list):
        raise ValueError("Click-curve summary has no trajectory list.")

    from vista3d_tfg.experiments.cyst_oracle_click_curve import (
        OracleClickRound,
        OracleCorrectionClick,
    )

    trajectories = []

    for raw in raw_trajectories:
        clicks = tuple(
            OracleCorrectionClick(
                index=int(click["index"]),
                point_xyz=tuple(int(v) for v in click["point_xyz"]),
                point_label=int(click["point_label"]),
                source=str(click["source"]),
                error_component_voxels=int(click["error_component_voxels"]),
                interior_radius_mm=float(click["interior_radius_mm"]),
            )
            for click in raw["clicks"]
        )
        rounds = tuple(
            OracleClickRound(
                click_count=int(round_result["click_count"]),
                dice=float(round_result["dice"]),
                prediction_volume_ml=float(
                    round_result["prediction_volume_ml"]
                ),
                signed_volume_error_ml=float(
                    round_result["signed_volume_error_ml"]
                ),
                labels_present=tuple(
                    int(v) for v in round_result["labels_present"]
                ),
                contains_255_unprocessed=bool(
                    round_result["contains_255_unprocessed"]
                ),
                wall_time_seconds=float(
                    round_result["wall_time_seconds"]
                ),
                prediction_path=str(round_result["prediction_path"]),
            )
            for round_result in raw["rounds"]
        )

        trajectories.append(
            OracleClickTrajectory(
                case_id=str(raw["case_id"]),
                target_volume_ml=float(raw["target_volume_ml"]),
                automatic_target_dice=float(raw["automatic_target_dice"]),
                clicks=clicks,
                rounds=rounds,
            )
        )

    return tuple(trajectories)


def write_stability_outputs(
    summary: InteractionStabilitySummary,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    (output_root / "stability_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiment": "oracle_click_interaction_stability_audit",
                "deployment_valid": False,
                "oracle_early_stopping": True,
                "result": summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with (output_root / "transitions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        fieldnames = list(asdict(summary.transitions[0]).keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for observation in summary.transitions:
            writer.writerow(asdict(observation))


def render_stability_figures(
    summary: InteractionStabilitySummary,
    output_root: Path,
) -> tuple[Path, ...]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = []

    positive = [
        observation.dice_delta
        for observation in summary.transitions
        if observation.point_label == 1
    ]
    negative = [
        observation.dice_delta
        for observation in summary.transitions
        if observation.point_label == 0
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    positions = []
    data = []
    labels = []

    if positive:
        positions.append(len(positions) + 1)
        data.append(positive)
        labels.append("Positive correction")
    if negative:
        positions.append(len(positions) + 1)
        data.append(negative)
        labels.append("Negative correction")

    axis.boxplot(data, positions=positions)
    axis.axhline(0.0)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Dice change after correction")
    axis.set_title("Oracle correction stability by click label")
    figure.tight_layout()

    path = output_root / "correction_delta_by_label.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    fixed_means = [
        summary.click1_mean,
        summary.click2_mean,
        summary.click3_mean,
        summary.best_observed_mean,
    ]
    fixed_medians = [
        summary.click1_median,
        summary.click2_median,
        summary.click3_median,
        summary.best_observed_median,
    ]
    x = np.arange(4)

    figure, axis = plt.subplots(figsize=(8, 6))
    width = 0.35
    axis.bar(x - width / 2, fixed_means, width, label="Mean")
    axis.bar(x + width / 2, fixed_medians, width, label="Median")
    axis.set_xticks(x, ["1 click", "2 clicks", "3 clicks", "Oracle best"])
    axis.set_ylabel("Target cyst Dice")
    axis.set_title("Fixed click count versus oracle early stopping")
    axis.legend()
    figure.tight_layout()

    path = output_root / "fixed_round_vs_oracle_best.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    outputs.append(path)

    return tuple(outputs)
