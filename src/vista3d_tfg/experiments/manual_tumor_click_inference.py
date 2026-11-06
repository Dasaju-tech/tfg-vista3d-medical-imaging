from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.experiments.tumor_adaptive_depth_inference import (
    load_adaptive_protocol,
)
from vista3d_tfg.experiments.tumor_adaptive_depth_protocol import (
    AdaptiveDepthCase,
    QUARTILE_NAMES,
)
from vista3d_tfg.experiments.tumor_click_robustness import (
    ClickVariant,
    run_click_variant,
)
from vista3d_tfg.experiments.tumor_oracle_protocol import (
    TumorOraclePrompt,
)
from vista3d_tfg.experiments.tumor_oracle_smoke import (
    TumorOracleSmokeResult,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NVSegmentPaths,
)
from vista3d_tfg.interaction.manual_tumor_clicks import (
    ManualTumorClick,
    load_manual_click,
)


@dataclass(frozen=True, slots=True)
class ManualClickLocalization:
    case_id: str
    point_xyz: tuple[int, int, int]
    gt_label_at_point: int
    inside_any_tumor: bool
    inside_target_component: bool
    distance_to_any_tumor_mm: float
    distance_to_target_component_mm: float


@dataclass(frozen=True, slots=True)
class ManualClickComparison:
    case_id: str
    quartile: str
    largest_component_volume_ml: float
    localization: ManualClickLocalization
    manual: TumorOracleSmokeResult
    deep: TumorOracleSmokeResult


def load_frozen_manual_clicks(
    *,
    clicks_root: Path,
    protocol: Iterable[AdaptiveDepthCase],
) -> tuple[tuple[AdaptiveDepthCase, ManualTumorClick], ...]:
    protocol = tuple(protocol)

    if len(protocol) != 12:
        raise ValueError(
            f"Expected frozen Block 46A N=12, got {len(protocol)}."
        )

    pairs = []

    for case in protocol:
        path = clicks_root / f"{case.case_id}.json"

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen manual click: {path}"
            )

        click = load_manual_click(path)

        if click.case_id != case.case_id:
            raise ValueError(
                f"Click case mismatch: {click.case_id} != {case.case_id}."
            )
        if click.quartile != case.quartile:
            raise ValueError(
                f"{case.case_id}: click quartile mismatch."
            )
        if click.gt_visible_during_selection:
            raise ValueError(
                f"{case.case_id}: GT-visible click is not admissible."
            )
        if click.source != "manual_ct_only_ras_display":
            raise ValueError(
                f"{case.case_id}: unexpected click source {click.source!r}."
            )
        if click.display_point_ras_xyz is None:
            raise ValueError(
                f"{case.case_id}: missing canonical-RAS display point."
            )

        pairs.append((case, click))

    if len(pairs) != 12:
        raise RuntimeError(
            f"Expected 12 frozen manual clicks, got {len(pairs)}."
        )

    return tuple(pairs)


def assess_manual_click_localization(
    *,
    case: AdaptiveDepthCase,
    click: ManualTumorClick,
    segmentation_path: Path,
) -> ManualClickLocalization:
    nii = nib.load(str(segmentation_path))
    gt_raw = np.asanyarray(nii.dataobj)

    if not np.all(np.isfinite(gt_raw)):
        raise ValueError(
            f"{case.case_id}: GT contains non-finite values."
        )
    if not np.allclose(
        gt_raw,
        np.rint(gt_raw),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError(
            f"{case.case_id}: GT is not integer-valued."
        )

    gt = np.rint(gt_raw).astype(
        np.int16,
        copy=False,
    )
    point = tuple(
        int(value)
        for value in click.point_xyz
    )

    if any(
        value < 0 or value >= upper
        for value, upper in zip(point, gt.shape)
    ):
        raise ValueError(
            f"{case.case_id}: manual point {point} outside GT shape "
            f"{gt.shape}."
        )

    tumor = gt == 2
    structure = ndimage.generate_binary_structure(3, 1)
    components, component_count = ndimage.label(
        tumor,
        structure=structure,
    )

    if not (1 <= case.component_id <= component_count):
        raise ValueError(
            f"{case.case_id}: frozen target component unavailable."
        )

    target = components == case.component_id
    target_voxels = int(np.count_nonzero(target))

    if target_voxels != case.target_voxels:
        raise ValueError(
            f"{case.case_id}: frozen target size changed "
            f"({target_voxels} != {case.target_voxels})."
        )

    spacing = tuple(
        float(value)
        for value in nii.header.get_zooms()[:3]
    )

    inside_any = bool(tumor[point])
    inside_target = bool(target[point])

    if inside_any:
        distance_any = 0.0
    else:
        distance_any_map = ndimage.distance_transform_edt(
            ~tumor,
            sampling=spacing,
        )
        distance_any = float(
            distance_any_map[point]
        )

    if inside_target:
        distance_target = 0.0
    else:
        distance_target_map = ndimage.distance_transform_edt(
            ~target,
            sampling=spacing,
        )
        distance_target = float(
            distance_target_map[point]
        )

    return ManualClickLocalization(
        case_id=case.case_id,
        point_xyz=point,
        gt_label_at_point=int(gt[point]),
        inside_any_tumor=inside_any,
        inside_target_component=inside_target,
        distance_to_any_tumor_mm=distance_any,
        distance_to_target_component_mm=distance_target,
    )


def _prompt(
    *,
    case: AdaptiveDepthCase,
    point_xyz: tuple[int, int, int],
    interior_radius_mm: float,
) -> TumorOraclePrompt:
    return TumorOraclePrompt(
        case_id=case.case_id,
        component_id=case.component_id,
        point_xyz=point_xyz,
        target_voxels=case.target_voxels,
        target_volume_ml=case.largest_component_volume_ml,
        total_tumor_volume_ml=case.largest_component_volume_ml,
        interior_radius_mm=interior_radius_mm,
    )


def run_manual_vs_deep(
    *,
    case: AdaptiveDepthCase,
    click: ManualTumorClick,
    project_root: Path,
    nv_paths: NVSegmentPaths,
    environment_name: str = DEFAULT_RUNTIME_ENV,
    force: bool = False,
) -> ManualClickComparison:
    segmentation_path = (
        project_root
        / "data"
        / "raw"
        / "kits23"
        / case.case_id
        / "segmentation.nii.gz"
    )

    localization = assess_manual_click_localization(
        case=case,
        click=click,
        segmentation_path=segmentation_path,
    )

    manual_prompt = _prompt(
        case=case,
        point_xyz=click.point_xyz,
        interior_radius_mm=0.0,
    )
    deep_prompt = _prompt(
        case=case,
        point_xyz=case.deep_point_xyz,
        interior_radius_mm=case.max_interior_depth_mm,
    )

    manual = run_click_variant(
        base_prompt=manual_prompt,
        variant=ClickVariant(
            case_id=case.case_id,
            variant="manual_ct_only",
            point_xyz=click.point_xyz,
            interior_radius_mm=0.0,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=force,
        require_prompt_inside_target=False,
    )

    deep = run_click_variant(
        base_prompt=deep_prompt,
        variant=ClickVariant(
            case_id=case.case_id,
            variant="deep",
            point_xyz=case.deep_point_xyz,
            interior_radius_mm=case.max_interior_depth_mm,
        ),
        project_root=project_root,
        nv_paths=nv_paths,
        environment_name=environment_name,
        force=False,
    )

    return ManualClickComparison(
        case_id=case.case_id,
        quartile=case.quartile,
        largest_component_volume_ml=(
            case.largest_component_volume_ml
        ),
        localization=localization,
        manual=manual,
        deep=deep,
    )


def _summary(
    values: Iterable[float],
) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)

    if array.size == 0:
        raise ValueError("Cannot summarize empty values.")

    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _paired_bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 5000,
) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Bootstrap requires a non-empty 1D array."
        )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        values.size,
        size=(samples, values.size),
    )
    means = values[indices].mean(axis=1)

    return {
        "lower_95": float(
            np.quantile(means, 0.025)
        ),
        "upper_95": float(
            np.quantile(means, 0.975)
        ),
    }


def _group_payload(
    rows: tuple[ManualClickComparison, ...],
) -> dict[str, object]:
    manual_dice = np.asarray(
        [
            row.manual.interactive_target_dice
            for row in rows
        ],
        dtype=float,
    )
    deep_dice = np.asarray(
        [
            row.deep.interactive_target_dice
            for row in rows
        ],
        dtype=float,
    )
    dice_delta = manual_dice - deep_dice

    manual_coverage = np.asarray(
        [
            row.manual.tumor_coverage_gain
            for row in rows
        ],
        dtype=float,
    )
    deep_coverage = np.asarray(
        [
            row.deep.tumor_coverage_gain
            for row in rows
        ],
        dtype=float,
    )
    coverage_delta = manual_coverage - deep_coverage

    manual_hd95 = np.asarray(
        [
            row.manual.hec_hd95_reduction_mm
            for row in rows
        ],
        dtype=float,
    )
    deep_hd95 = np.asarray(
        [
            row.deep.hec_hd95_reduction_mm
            for row in rows
        ],
        dtype=float,
    )
    hd95_delta = manual_hd95 - deep_hd95

    return {
        "count": len(rows),
        "manual": {
            "tumor_dice": _summary(manual_dice),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(
                    manual_dice >= 0.50
                )
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(
                    manual_dice >= 0.70
                )
            ),
            "tumor_coverage_gain": _summary(
                manual_coverage
            ),
            "hec_hd95_reduction_mm": _summary(
                manual_hd95
            ),
            "hd95_improved_count": int(
                np.count_nonzero(
                    manual_hd95 > 0.0
                )
            ),
        },
        "deep": {
            "tumor_dice": _summary(deep_dice),
            "tumor_dice_ge_050_count": int(
                np.count_nonzero(
                    deep_dice >= 0.50
                )
            ),
            "tumor_dice_ge_070_count": int(
                np.count_nonzero(
                    deep_dice >= 0.70
                )
            ),
            "tumor_coverage_gain": _summary(
                deep_coverage
            ),
            "hec_hd95_reduction_mm": _summary(
                deep_hd95
            ),
            "hd95_improved_count": int(
                np.count_nonzero(
                    deep_hd95 > 0.0
                )
            ),
        },
        "paired_manual_minus_deep": {
            "tumor_dice": {
                **_summary(dice_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        dice_delta,
                        seed=20260908,
                    )
                ),
            },
            "tumor_coverage_gain": {
                **_summary(coverage_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        coverage_delta,
                        seed=20260909,
                    )
                ),
            },
            "hec_hd95_reduction_mm": {
                **_summary(hd95_delta),
                "bootstrap_mean_95_ci": (
                    _paired_bootstrap_ci(
                        hd95_delta,
                        seed=20260910,
                    )
                ),
            },
            "cases_within_0_05_tumor_dice": int(
                np.count_nonzero(
                    np.abs(dice_delta)
                    <= (0.05 + 1e-12)
                )
            ),
            "cases_within_0_10_tumor_dice": int(
                np.count_nonzero(
                    np.abs(dice_delta)
                    <= (0.10 + 1e-12)
                )
            ),
            "manual_tumor_dice_better_count": int(
                np.count_nonzero(
                    dice_delta > 0.0
                )
            ),
        },
    }


def summarize_manual_click_results(
    rows: Iterable[ManualClickComparison],
) -> dict[str, object]:
    rows = tuple(rows)

    if len(rows) != 12:
        raise ValueError(
            f"Expected 12 manual-click comparisons, got {len(rows)}."
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
            f"Manual-click quartiles are unbalanced: {counts}."
        )

    target_hits = tuple(
        row
        for row in rows
        if row.localization.inside_target_component
    )
    any_tumor_hits = tuple(
        row
        for row in rows
        if row.localization.inside_any_tumor
    )
    misses = tuple(
        row
        for row in rows
        if not row.localization.inside_any_tumor
    )

    miss_distances = np.asarray(
        [
            row.localization.distance_to_any_tumor_mm
            for row in misses
        ],
        dtype=float,
    )

    manual_dice = np.asarray(
        [
            row.manual.interactive_target_dice
            for row in rows
        ],
        dtype=float,
    )
    target_distances = np.asarray(
        [
            row.localization.distance_to_target_component_mm
            for row in rows
        ],
        dtype=float,
    )

    if (
        np.std(target_distances) > 0.0
        and np.std(manual_dice) > 0.0
    ):
        distance_vs_dice = float(
            np.corrcoef(
                target_distances,
                manual_dice,
            )[0, 1]
        )
    else:
        distance_vs_dice = 0.0

    localization_payload: dict[str, object] = {
        "target_component_hit_count": len(target_hits),
        "any_tumor_hit_count": len(any_tumor_hits),
        "miss_count": len(misses),
        "target_component_hit_fraction": (
            len(target_hits) / len(rows)
        ),
        "any_tumor_hit_fraction": (
            len(any_tumor_hits) / len(rows)
        ),
        "distance_to_target_vs_manual_dice_pearson": (
            distance_vs_dice
        ),
    }

    if miss_distances.size:
        localization_payload[
            "miss_distance_to_any_tumor_mm"
        ] = _summary(miss_distances)
    else:
        localization_payload[
            "miss_distance_to_any_tumor_mm"
        ] = None

    if target_hits:
        localization_payload[
            "manual_dice_on_target_hits"
        ] = _summary(
            row.manual.interactive_target_dice
            for row in target_hits
        )
    else:
        localization_payload[
            "manual_dice_on_target_hits"
        ] = None

    if misses:
        localization_payload[
            "manual_dice_on_misses"
        ] = _summary(
            row.manual.interactive_target_dice
            for row in misses
        )
    else:
        localization_payload[
            "manual_dice_on_misses"
        ] = None

    return {
        "schema_version": "1.0",
        "experiment": (
            "nv_segment_ct_manual_ct_only_tumor_click_inference"
        ),
        "case_count": 12,
        "localization_source": "manual_ct_only_ras_display",
        "gt_visible_during_selection": False,
        "deployment_valid": False,
        "simulated_user_localization": True,
        "localization_audit": localization_payload,
        "overall": _group_payload(rows),
        "by_quartile": {
            quartile: _group_payload(
                tuple(
                    row
                    for row in rows
                    if row.quartile == quartile
                )
            )
            for quartile in QUARTILE_NAMES
        },
        "case_rows": [
            {
                "case_id": row.case_id,
                "quartile": row.quartile,
                "largest_component_volume_ml": (
                    row.largest_component_volume_ml
                ),
                "localization": {
                    "point_xyz": list(
                        row.localization.point_xyz
                    ),
                    "gt_label_at_point": (
                        row.localization.gt_label_at_point
                    ),
                    "inside_any_tumor": (
                        row.localization.inside_any_tumor
                    ),
                    "inside_target_component": (
                        row.localization.inside_target_component
                    ),
                    "distance_to_any_tumor_mm": (
                        row.localization.distance_to_any_tumor_mm
                    ),
                    "distance_to_target_component_mm": (
                        row.localization.distance_to_target_component_mm
                    ),
                },
                "manual": row.manual.to_dict(),
                "deep": row.deep.to_dict(),
            }
            for row in rows
        ],
        "interpretation_boundary": (
            "Manual click locations were frozen from CT-only canonical-RAS "
            "views before ground truth was consulted. Ground truth is used "
            "only for retrospective localization and segmentation evaluation. "
            "Because the project author supplied the clicks, this is a "
            "simulated-user localization experiment, not a clinician study."
        ),
    }


def write_manual_click_outputs(
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

    rows = payload["case_rows"]
    assert isinstance(rows, list)

    fields = [
        "case_id",
        "quartile",
        "largest_component_volume_ml",
        "point_xyz",
        "gt_label_at_point",
        "inside_any_tumor",
        "inside_target_component",
        "distance_to_any_tumor_mm",
        "distance_to_target_component_mm",
        "variant",
        "tumor_dice",
        "tumor_coverage_gain",
        "hec_dice_delta",
        "hec_surface_dice_delta",
        "hec_hd95_reduction_mm",
        "wall_time_seconds",
    ]

    with (output_root / "cases.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            assert isinstance(row, dict)
            localization = row["localization"]
            assert isinstance(localization, dict)

            for variant in ("manual", "deep"):
                result = row[variant]
                assert isinstance(result, dict)

                writer.writerow(
                    {
                        "case_id": row["case_id"],
                        "quartile": row["quartile"],
                        "largest_component_volume_ml": row[
                            "largest_component_volume_ml"
                        ],
                        "point_xyz": ";".join(
                            str(value)
                            for value in localization["point_xyz"]
                        ),
                        "gt_label_at_point": localization[
                            "gt_label_at_point"
                        ],
                        "inside_any_tumor": localization[
                            "inside_any_tumor"
                        ],
                        "inside_target_component": localization[
                            "inside_target_component"
                        ],
                        "distance_to_any_tumor_mm": localization[
                            "distance_to_any_tumor_mm"
                        ],
                        "distance_to_target_component_mm": localization[
                            "distance_to_target_component_mm"
                        ],
                        "variant": variant,
                        "tumor_dice": result[
                            "interactive_target_dice"
                        ],
                        "tumor_coverage_gain": result[
                            "tumor_coverage_gain"
                        ],
                        "hec_dice_delta": result[
                            "hec_dice_delta"
                        ],
                        "hec_surface_dice_delta": result[
                            "hec_surface_dice_delta"
                        ],
                        "hec_hd95_reduction_mm": result[
                            "hec_hd95_reduction_mm"
                        ],
                        "wall_time_seconds": result[
                            "wall_time_seconds"
                        ],
                    }
                )


def write_manual_click_report(
    *,
    payload: dict[str, object],
    path: Path,
) -> None:
    localization = payload["localization_audit"]
    overall = payload["overall"]
    by_quartile = payload["by_quartile"]

    assert isinstance(localization, dict)
    assert isinstance(overall, dict)
    assert isinstance(by_quartile, dict)

    manual = overall["manual"]
    deep = overall["deep"]
    paired = overall["paired_manual_minus_deep"]

    assert isinstance(manual, dict)
    assert isinstance(deep, dict)
    assert isinstance(paired, dict)

    manual_dice = manual["tumor_dice"]
    deep_dice = deep["tumor_dice"]
    paired_dice = paired["tumor_dice"]

    assert isinstance(manual_dice, dict)
    assert isinstance(deep_dice, dict)
    assert isinstance(paired_dice, dict)

    lines = [
        "# Manual CT-only tumor click inference",
        "",
        "## Localization",
        "",
        (
            f"- Target-component hits: "
            f"{localization['target_component_hit_count']}/12."
        ),
        (
            f"- Any-tumor hits: "
            f"{localization['any_tumor_hit_count']}/12."
        ),
        (
            f"- Tumor misses: "
            f"{localization['miss_count']}/12."
        ),
        "",
        "## Segmentation",
        "",
        (
            f"- Manual tumor Dice: "
            f"{float(manual_dice['mean']):.4f} mean, "
            f"{float(manual_dice['median']):.4f} median."
        ),
        (
            f"- Deep-oracle tumor Dice: "
            f"{float(deep_dice['mean']):.4f} mean, "
            f"{float(deep_dice['median']):.4f} median."
        ),
        (
            f"- Paired manual-minus-deep Dice: "
            f"{float(paired_dice['mean']):+.4f}."
        ),
        "",
        "## By tumor-size quartile",
        "",
        "| Quartile | Manual Dice mean | Deep Dice mean | Paired delta |",
        "|---|---:|---:|---:|",
    ]

    for quartile in QUARTILE_NAMES:
        group = by_quartile[quartile]
        assert isinstance(group, dict)
        group_manual = group["manual"]["tumor_dice"]
        group_deep = group["deep"]["tumor_dice"]
        group_delta = group[
            "paired_manual_minus_deep"
        ]["tumor_dice"]

        lines.append(
            f"| {quartile} "
            f"| {float(group_manual['mean']):.4f} "
            f"| {float(group_deep['mean']):.4f} "
            f"| {float(group_delta['mean']):+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            str(payload["interpretation_boundary"]),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
