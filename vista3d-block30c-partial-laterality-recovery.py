from __future__ import annotations

from pathlib import Path


ROOT = Path.cwd()

FILES = {
    "nv": ROOT / "src" / "vista3d_tfg" / "evaluation" / "nv_kits23.py",
    "cohort": ROOT / "src" / "vista3d_tfg" / "evaluation" / "cohort.py",
    "analysis": ROOT / "src" / "vista3d_tfg" / "evaluation" / "pilot_analysis.py",
    "pilot": ROOT / "src" / "vista3d_tfg" / "experiments" / "pilot30.py",
    "runner": ROOT / "scripts" / "run_nv_segment_ct_expanded100.py",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}. "
            "No files were written."
        )
    return text.replace(old, new, 1)


for label, path in FILES.items():
    if not path.is_file():
        raise SystemExit(f"Missing expected file [{label}]: {path}")

original = {
    label: path.read_text(encoding="utf-8")
    for label, path in FILES.items()
}

if (
    "ground_truth_laterality_resolved: bool" in original["nv"]
    and "laterality_unresolved_case_count" in original["cohort"]
):
    print("Block 30C already appears to be applied.")
    raise SystemExit(0)

updated = dict(original)

updated["nv"] = replace_once(
    updated["nv"],
    '''    kidney_core_diagnostic: BinaryComparison
    cyst: BinaryComparison
    left_renal_envelope: BinaryComparison
    right_renal_envelope: BinaryComparison
    left_cyst: BinaryComparison
    right_cyst: BinaryComparison
''',
    '''    kidney_core_diagnostic: BinaryComparison
    cyst: BinaryComparison
    ground_truth_laterality_resolved: bool
    ground_truth_laterality_reason: str | None
    left_renal_envelope: BinaryComparison | None
    right_renal_envelope: BinaryComparison | None
    left_cyst: BinaryComparison | None
    right_cyst: BinaryComparison | None
''',
    "SupportedRenalComparison fields",
)

updated["nv"] = replace_once(
    updated["nv"],
    '''def _ground_truth_side_masks(
    ground_truth: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> dict[str, np.ndarray]:
    anatomy = analyze_kits23_anatomy(
        case_id="evaluation",
        segmentation=ground_truth,
        spacing_mm=spacing_mm,
        affine=affine,
    )

    if not anatomy.is_laterality_resolved:
        raise RuntimeError(
            "Ground-truth renal laterality is unresolved; "
            "side-aware evaluation is unavailable."
        )

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )
    component_labels, _ = ndimage.label(
        np.isin(ground_truth, (1, 2, 3)),
        structure=structure,
    )

    side_masks: dict[str, np.ndarray] = {}

    for unit in anatomy.renal_units:
        if unit.laterality not in {"left", "right"}:
            continue

        side_masks[unit.laterality] = (
            component_labels == unit.component_id
        )

    if set(side_masks) != {"left", "right"}:
        raise RuntimeError(
            "Could not recover both dominant ground-truth renal masks."
        )

    return side_masks
''',
    '''def _ground_truth_side_masks(
    ground_truth: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float],
    affine: np.ndarray,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    # Global kidney-and-mass, kidney-core, and cyst metrics do not require
    # left/right assignment. Unresolved laterality disables only side-aware
    # diagnostics rather than invalidating the entire case.
    anatomy = analyze_kits23_anatomy(
        case_id="evaluation",
        segmentation=ground_truth,
        spacing_mm=spacing_mm,
        affine=affine,
    )

    if not anatomy.is_laterality_resolved:
        reason = (
            ";".join(anatomy.qc_flags)
            if anatomy.qc_flags
            else "ground_truth_laterality_unresolved"
        )
        return None, reason

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=1,
    )
    component_labels, _ = ndimage.label(
        np.isin(ground_truth, (1, 2, 3)),
        structure=structure,
    )

    side_masks: dict[str, np.ndarray] = {}

    for unit in anatomy.renal_units:
        if unit.laterality not in {"left", "right"}:
            continue

        side_masks[unit.laterality] = (
            component_labels == unit.component_id
        )

    if set(side_masks) != {"left", "right"}:
        return None, "dominant_ground_truth_renal_masks_unavailable"

    return side_masks, None
''',
    "_ground_truth_side_masks",
)

updated["nv"] = replace_once(
    updated["nv"],
    '''    gt_side_masks = _ground_truth_side_masks(
        ground_truth,
        spacing_mm=spacing_mm,
        affine=affine,
    )

    left_gt = gt_side_masks["left"]
    right_gt = gt_side_masks["right"]
''',
    '''    gt_side_masks, laterality_reason = _ground_truth_side_masks(
        ground_truth,
        spacing_mm=spacing_mm,
        affine=affine,
    )

    left_gt = (
        gt_side_masks["left"]
        if gt_side_masks is not None
        else None
    )
    right_gt = (
        gt_side_masks["right"]
        if gt_side_masks is not None
        else None
    )
''',
    "GT side-mask resolution use",
)

updated["nv"] = replace_once(
    updated["nv"],
    '''    left_gt_cyst = gt_cyst & left_gt
    right_gt_cyst = gt_cyst & right_gt
''',
    '''    left_gt_cyst = (
        gt_cyst & left_gt
        if left_gt is not None
        else None
    )
    right_gt_cyst = (
        gt_cyst & right_gt
        if right_gt is not None
        else None
    )
''',
    "side-specific GT cyst masks",
)

updated["nv"] = replace_once(
    updated["nv"],
    '''        cyst=compare_binary_masks(
            gt_cyst,
            predicted_cyst,
            spacing_mm=spacing_mm,
        ),
        left_renal_envelope=compare_binary_masks(
            left_gt,
            left_prediction,
            spacing_mm=spacing_mm,
        ),
        right_renal_envelope=compare_binary_masks(
            right_gt,
            right_prediction,
            spacing_mm=spacing_mm,
        ),
        left_cyst=compare_binary_masks(
            left_gt_cyst,
            left_pred_cyst,
            spacing_mm=spacing_mm,
        ),
        right_cyst=compare_binary_masks(
            right_gt_cyst,
            right_pred_cyst,
            spacing_mm=spacing_mm,
        ),
''',
    '''        cyst=compare_binary_masks(
            gt_cyst,
            predicted_cyst,
            spacing_mm=spacing_mm,
        ),
        ground_truth_laterality_resolved=(gt_side_masks is not None),
        ground_truth_laterality_reason=laterality_reason,
        left_renal_envelope=(
            compare_binary_masks(
                left_gt,
                left_prediction,
                spacing_mm=spacing_mm,
            )
            if left_gt is not None
            else None
        ),
        right_renal_envelope=(
            compare_binary_masks(
                right_gt,
                right_prediction,
                spacing_mm=spacing_mm,
            )
            if right_gt is not None
            else None
        ),
        left_cyst=(
            compare_binary_masks(
                left_gt_cyst,
                left_pred_cyst,
                spacing_mm=spacing_mm,
            )
            if left_gt_cyst is not None
            else None
        ),
        right_cyst=(
            compare_binary_masks(
                right_gt_cyst,
                right_pred_cyst,
                spacing_mm=spacing_mm,
            )
            if right_gt_cyst is not None
            else None
        ),
''',
    "SupportedRenalComparison construction",
)

updated["nv"] = replace_once(
    updated["nv"],
    '''                "schema_version": "1.0",
                "dataset": "KiTS23",
                "model": "NV-Segment-CT",
''',
    '''                "schema_version": "1.1",
                "dataset": "KiTS23",
                "model": "NV-Segment-CT",
''',
    "evaluation schema version",
)

updated["cohort"] = replace_once(
    updated["cohort"],
    '''    left_renal_dice: float
    right_renal_dice: float
    left_renal_volume_error_ml: float
    right_renal_volume_error_ml: float
    gt_tumor_volume_ml: float
    inference_wall_time_seconds: float | None
    qc_false_positive_voxels: int | None
    qc_false_negative_voxels: int | None
    qc_total_disagreement_voxels: int | None
''',
    '''    left_renal_dice: float | None
    right_renal_dice: float | None
    left_renal_volume_error_ml: float | None
    right_renal_volume_error_ml: float | None
    gt_tumor_volume_ml: float
    inference_wall_time_seconds: float | None
    qc_false_positive_voxels: int | None
    qc_false_negative_voxels: int | None
    qc_total_disagreement_voxels: int | None
    ground_truth_laterality_resolved: bool = True
''',
    "CaseCohortRow optional laterality",
)

updated["cohort"] = replace_once(
    updated["cohort"],
    '''    kidney_and_mass_hd95_mm: MetricSummary
    kidney_core_dice: MetricSummary
    left_renal_dice: MetricSummary
    right_renal_dice: MetricSummary
    cyst_dice_all_cases: MetricSummary
''',
    '''    kidney_and_mass_hd95_mm: MetricSummary
    kidney_core_dice: MetricSummary
    laterality_resolved_case_count: int
    laterality_unresolved_case_count: int
    left_renal_dice: MetricSummary | None
    right_renal_dice: MetricSummary | None
    cyst_dice_all_cases: MetricSummary
''',
    "CohortSummary laterality fields",
)

updated["cohort"] = replace_once(
    updated["cohort"],
    '''    kidney_hec = result["kidney_and_mass_hec"]
    kidney_core = result["kidney_core_diagnostic"]
    cyst = result["cyst"]
    left = result["left_renal_envelope"]
    right = result["right_renal_envelope"]

    inference_wall_time: float | None = None
''',
    '''    kidney_hec = result["kidney_and_mass_hec"]
    kidney_core = result["kidney_core_diagnostic"]
    cyst = result["cyst"]
    left = result.get("left_renal_envelope")
    right = result.get("right_renal_envelope")

    if left is not None and not isinstance(left, dict):
        raise ValueError("left_renal_envelope must be an object or null.")
    if right is not None and not isinstance(right, dict):
        raise ValueError("right_renal_envelope must be an object or null.")

    laterality_value = result.get("ground_truth_laterality_resolved")
    if laterality_value is None:
        laterality_resolved = left is not None and right is not None
    elif isinstance(laterality_value, bool):
        laterality_resolved = laterality_value
    else:
        raise ValueError(
            "ground_truth_laterality_resolved must be boolean when present."
        )

    inference_wall_time: float | None = None
''',
    "cohort evaluation parsing",
)

updated["cohort"] = replace_once(
    updated["cohort"],
    '''        left_renal_dice=float(left["dice"]),
        right_renal_dice=float(right["dice"]),
        left_renal_volume_error_ml=float(
            left["signed_volume_error_ml"]
        ),
        right_renal_volume_error_ml=float(
            right["signed_volume_error_ml"]
        ),
        gt_tumor_volume_ml=float(result["gt_tumor_volume_ml"]),
        inference_wall_time_seconds=inference_wall_time,
        qc_false_positive_voxels=qc_fp,
        qc_false_negative_voxels=qc_fn,
        qc_total_disagreement_voxels=qc_total,
''',
    '''        left_renal_dice=(
            float(left["dice"])
            if left is not None
            else None
        ),
        right_renal_dice=(
            float(right["dice"])
            if right is not None
            else None
        ),
        left_renal_volume_error_ml=(
            float(left["signed_volume_error_ml"])
            if left is not None
            else None
        ),
        right_renal_volume_error_ml=(
            float(right["signed_volume_error_ml"])
            if right is not None
            else None
        ),
        gt_tumor_volume_ml=float(result["gt_tumor_volume_ml"]),
        inference_wall_time_seconds=inference_wall_time,
        qc_false_positive_voxels=qc_fp,
        qc_false_negative_voxels=qc_fn,
        qc_total_disagreement_voxels=qc_total,
        ground_truth_laterality_resolved=laterality_resolved,
''',
    "CaseCohortRow parsing construction",
)

updated["cohort"] = replace_once(
    updated["cohort"],
    '''    runtime_values = tuple(
        row.inference_wall_time_seconds
        for row in rows
        if row.inference_wall_time_seconds is not None
    )

    return CohortSummary(
        schema_version="1.0",
''',
    '''    runtime_values = tuple(
        row.inference_wall_time_seconds
        for row in rows
        if row.inference_wall_time_seconds is not None
    )

    laterality_resolved_count = sum(
        1
        for row in rows
        if row.ground_truth_laterality_resolved
    )
    left_dice_values = tuple(
        row.left_renal_dice
        for row in rows
        if row.left_renal_dice is not None
    )
    right_dice_values = tuple(
        row.right_renal_dice
        for row in rows
        if row.right_renal_dice is not None
    )

    return CohortSummary(
        schema_version="1.1",
''',
    "cohort laterality availability calculation",
)

updated["cohort"] = replace_once(
    updated["cohort"],
    '''        kidney_core_dice=summarize_values(
            row.kidney_core_dice for row in rows
        ),
        left_renal_dice=summarize_values(
            row.left_renal_dice for row in rows
        ),
        right_renal_dice=summarize_values(
            row.right_renal_dice for row in rows
        ),
        cyst_dice_all_cases=summarize_values(
''',
    '''        kidney_core_dice=summarize_values(
            row.kidney_core_dice for row in rows
        ),
        laterality_resolved_case_count=laterality_resolved_count,
        laterality_unresolved_case_count=(
            len(rows) - laterality_resolved_count
        ),
        left_renal_dice=(
            summarize_values(left_dice_values)
            if left_dice_values
            else None
        ),
        right_renal_dice=(
            summarize_values(right_dice_values)
            if right_dice_values
            else None
        ),
        cyst_dice_all_cases=summarize_values(
''',
    "cohort side-aware summaries",
)

updated["analysis"] = replace_once(
    updated["analysis"],
    '''    largest_undersegmentation = tuple(
        row.case_id
        for row in sorted(
            rows,
            key=lambda row: (
                row.left_renal_volume_error_ml
                + row.right_renal_volume_error_ml,
                row.case_id,
            ),
        )[:5]
    )
''',
    '''    laterality_rows = tuple(
        row
        for row in rows
        if row.left_renal_volume_error_ml is not None
        and row.right_renal_volume_error_ml is not None
    )

    largest_undersegmentation = tuple(
        row.case_id
        for row in sorted(
            laterality_rows,
            key=lambda row: (
                row.left_renal_volume_error_ml
                + row.right_renal_volume_error_ml,
                row.case_id,
            ),
        )[:5]
    )
''',
    "pilot analysis laterality-eligible ranking",
)

updated["pilot"] = replace_once(
    updated["pilot"],
    '''def write_run_manifest(
    *,
    output_path: Path,
    requested_case_ids: Iterable[str],
    successful_case_ids: Iterable[str],
    failed_cases: dict[str, str],
    worst_qc_case_ids: Iterable[str],
) -> None:
''',
    '''def write_run_manifest(
    *,
    output_path: Path,
    requested_case_ids: Iterable[str],
    successful_case_ids: Iterable[str],
    failed_cases: dict[str, str],
    worst_qc_case_ids: Iterable[str],
    experiment: str = "nv_segment_ct_pilot30",
) -> None:
''',
    "write_run_manifest signature",
)

updated["pilot"] = replace_once(
    updated["pilot"],
    '''                "schema_version": "1.0",
                "experiment": "nv_segment_ct_pilot30",
                "requested_case_ids": list(requested_case_ids),
''',
    '''                "schema_version": "1.0",
                "experiment": experiment,
                "requested_case_ids": list(requested_case_ids),
''',
    "write_run_manifest experiment field",
)

updated["runner"] = replace_once(
    updated["runner"],
    '''        successful_case_ids=successful,
        failed_cases=failures,
        worst_qc_case_ids=worst_case_ids,
    )
''',
    '''        successful_case_ids=successful,
        failed_cases=failures,
        worst_qc_case_ids=worst_case_ids,
        experiment=EXPANDED100_EXPERIMENT,
    )
''',
    "expanded100 manifest experiment",
)

for label in FILES:
    if updated[label] == original[label]:
        raise RuntimeError(f"No change prepared for {label}; aborting.")

for label, path in FILES.items():
    path.write_text(updated[label], encoding="utf-8")

test_path = ROOT / "tests" / "test_partial_laterality_evaluation.py"
test_path.write_text(
    r'''from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from vista3d_tfg.evaluation.cohort import (
    load_case_cohort_row,
    summarize_cohort,
)
from vista3d_tfg.evaluation.nv_kits23 import (
    evaluate_supported_renal_prediction,
    write_supported_renal_comparison,
)
from vista3d_tfg.experiments.pilot30 import write_run_manifest


def test_unresolved_gt_laterality_preserves_global_metrics(
    tmp_path: Path,
) -> None:
    ground_truth = np.zeros((12, 12, 12), dtype=np.uint8)
    ground_truth[2:10, 2:10, 2:10] = 1

    prediction = np.zeros_like(ground_truth)
    prediction[2:10, 2:10, 2:10] = 14

    gt_path = tmp_path / "segmentation.nii.gz"
    pred_path = tmp_path / "prediction.nii.gz"

    nib.save(nib.Nifti1Image(ground_truth, np.eye(4)), gt_path)
    nib.save(nib.Nifti1Image(prediction, np.eye(4)), pred_path)

    result = evaluate_supported_renal_prediction(
        case_id="unilateral_fixture",
        ground_truth_path=gt_path,
        prediction_path=pred_path,
    )

    assert result.ground_truth_laterality_resolved is False
    assert result.ground_truth_laterality_reason
    assert result.kidney_and_mass_hec.dice == 1.0
    assert result.kidney_core_diagnostic.dice == 1.0
    assert result.left_renal_envelope is None
    assert result.right_renal_envelope is None
    assert result.left_cyst is None
    assert result.right_cyst is None

    evaluation_path = tmp_path / "supported_renal_metrics.json"
    write_supported_renal_comparison(result, evaluation_path)

    payload = json.loads(
        evaluation_path.read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "1.1"

    row = load_case_cohort_row(
        case_id="unilateral_fixture",
        evaluation_path=evaluation_path,
    )

    assert row.ground_truth_laterality_resolved is False
    assert row.kidney_and_mass_dice == 1.0
    assert row.left_renal_dice is None
    assert row.right_renal_dice is None

    summary = summarize_cohort([row])

    assert summary.case_count == 1
    assert summary.kidney_and_mass_dice.count == 1
    assert summary.kidney_and_mass_dice.mean == 1.0
    assert summary.laterality_resolved_case_count == 0
    assert summary.laterality_unresolved_case_count == 1
    assert summary.left_renal_dice is None
    assert summary.right_renal_dice is None


def test_schema10_artifacts_remain_backward_compatible(
    tmp_path: Path,
) -> None:
    evaluation_path = tmp_path / "legacy.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": {
                    "kidney_and_mass_hec": {
                        "dice": 0.9,
                        "surface_dice": 0.8,
                        "diagnostic_hd95_mm": 3.0,
                    },
                    "kidney_core_diagnostic": {"dice": 0.89},
                    "cyst": {
                        "dice": 1.0,
                        "reference_volume_ml": 0.0,
                        "prediction_volume_ml": 0.0,
                    },
                    "left_renal_envelope": {
                        "dice": 0.91,
                        "signed_volume_error_ml": -1.0,
                    },
                    "right_renal_envelope": {
                        "dice": 0.92,
                        "signed_volume_error_ml": -2.0,
                    },
                    "gt_tumor_volume_ml": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )

    row = load_case_cohort_row(
        case_id="legacy",
        evaluation_path=evaluation_path,
    )

    assert row.ground_truth_laterality_resolved is True
    assert row.left_renal_dice == 0.91
    assert row.right_renal_dice == 0.92


def test_run_manifest_accepts_explicit_experiment_name(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run_manifest.json"

    write_run_manifest(
        output_path=output,
        requested_case_ids=("case_a",),
        successful_case_ids=("case_a",),
        failed_cases={},
        worst_qc_case_ids=(),
        experiment="nv_segment_ct_expanded100",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment"] == "nv_segment_ct_expanded100"
''',
    encoding="utf-8",
)

doc_path = ROOT / "docs" / "block30c_partial_laterality_recovery.md"
doc_path.write_text(
    r'''# Block 30C — Partial-Laterality Recovery

## Trigger

Expanded100 completed model inference for all 100 registered cases. Evaluation
succeeded for 99 cases and failed only for `case_00563`.

The failure was not a model failure. Ground-truth morphology analysis found one
connected kidney-and-mass component, so a defensible left/right assignment was
unavailable:

```text
kidney_and_mass_components = 1
qc_flags = ["fewer_than_two_kidney_and_mass_components"]
```

## Protocol correction

The primary supported quantities do not require ground-truth left/right
assignment:

- kidney-and-mass Dice;
- kidney-and-mass Surface Dice;
- diagnostic HD95;
- kidney-core Dice;
- global cyst Dice and physical cyst volumes.

Therefore unresolved laterality must not invalidate those quantities.

Side-aware renal-envelope and side-specific cyst diagnostics are nullable.
No anatomical side is guessed from the model prediction or from an arbitrary
coordinate split.

## Cohort semantics

Expanded100 keeps all 100 frozen cases.

Expected recovered cohort:

```text
primary/global evaluation N = 100
laterality-resolved N       = 99
laterality-unresolved N     = 1
```

Left/right cohort summaries are computed only over cases with resolved
ground-truth laterality.

Schema-1.0 evaluation artifacts remain readable and are treated as
laterality-resolved when both historical side objects are present.

## Reproducibility

No prediction is regenerated. Re-running the resumable expanded100 runner
should reuse all 100 inference artifacts and the 99 existing evaluation
artifacts, computing only the missing evaluation for `case_00563`.

The expanded100 run manifest also records the correct experiment identifier
`nv_segment_ct_expanded100`.
''',
    encoding="utf-8",
)

print("Block 30C applied successfully.")
print("Modified:")
for path in FILES.values():
    print(f"  {path.relative_to(ROOT)}")
print("Added:")
print(f"  {test_path.relative_to(ROOT)}")
print(f"  {doc_path.relative_to(ROOT)}")
print()
print("Run focused tests before re-running expanded100.")
