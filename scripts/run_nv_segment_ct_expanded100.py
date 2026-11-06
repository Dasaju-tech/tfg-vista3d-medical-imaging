from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.acquisition import download_kits23_case
from vista3d_tfg.data.audit import audit_cases, write_audit
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.data.provenance import write_acquisition_provenance
from vista3d_tfg.evaluation.cohort import (
    load_case_cohort_row,
    summarize_cohort,
    write_cohort_csv,
    write_cohort_json,
)
from vista3d_tfg.evaluation.nv_kits23 import (
    evaluate_supported_renal_prediction,
    write_supported_renal_comparison,
)
from vista3d_tfg.experiments.expanded100 import (
    EXPANDED100_EXPERIMENT,
    EXPANDED100_SIZE,
    load_frozen_selection,
)
from vista3d_tfg.experiments.pilot30 import (
    PILOT30_CASES,
    pilot_case_paths,
    select_worst_surface_dice_cases,
    write_run_manifest,
)
from vista3d_tfg.integration.nv_inference import run_renal_inference
from vista3d_tfg.integration.nv_segment import DEFAULT_RUNTIME_ENV, runtime_paths
from vista3d_tfg.visualization.nv_compare import (
    build_case_comparison_qc,
    write_comparison_qc_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire, run, evaluate, and summarize the frozen 100-case "
            "KiTS23 NV-Segment-CT validation cohort."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("acquire", "run", "all"),
        default="all",
    )
    parser.add_argument("--environment", default=DEFAULT_RUNTIME_ENV)
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help=(
            "Frozen selection manifest. Defaults to "
            "configs/experiments/nv_segment_ct_expanded100_selection.json."
        ),
    )
    parser.add_argument(
        "--worst-qc",
        type=int,
        default=10,
        help="Render prediction-vs-GT QC for the N worst Surface Dice cases.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--force-qc", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue after a case-level failure, preserve completed cases, "
            "and return a non-zero status at the end."
        ),
    )
    return parser.parse_args()


def default_selection_path(project_root: Path) -> Path:
    return (
        project_root
        / "configs"
        / "experiments"
        / "nv_segment_ct_expanded100_selection.json"
    )


def experiment_root(project_root: Path) -> Path:
    return project_root / "outputs" / "cohorts" / EXPANDED100_EXPERIMENT


def read_runtime_commit(project_root: Path) -> str:
    manifest_path = project_root / "outputs" / "runtime" / "nv_segment_ct_runtime.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    probe = payload.get("probe", {})

    if probe.get("cuda_available") is not True:
        raise RuntimeError("Recorded NV-Segment runtime is not CUDA-enabled.")

    commit = payload.get("repository_commit")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("Runtime manifest does not contain repository_commit.")

    return commit


def acquire_cases(
    *,
    project_root: Path,
    case_numbers: tuple[int, ...],
    retries: int,
) -> None:
    dataset_root = project_root / "data" / "raw" / "kits23"
    records = []

    print("=== Expanded100 acquisition and integrity audit ===")

    for index, case_number in enumerate(case_numbers, start=1):
        case_id = f"case_{case_number:05d}"
        print(f"[{index:03d}/{len(case_numbers):03d}] Acquiring {case_id} ...")

        case_records = download_kits23_case(
            case_number,
            dataset_root,
            force=False,
            retries=retries,
        )
        records.extend(case_records)
        print(
            "  "
            + ", ".join(
                f"{record.kind}={record.status}" for record in case_records
            )
        )

    cases = tuple(
        discover_case(dataset_root / f"case_{number:05d}")
        for number in case_numbers
    )
    audit = audit_cases(cases)

    provenance_path = (
        project_root
        / "outputs"
        / "manifests"
        / "kits23_expanded100_provenance.json"
    )
    audit_path = (
        project_root
        / "outputs"
        / "audits"
        / "kits23_expanded100_audit.json"
    )

    write_acquisition_provenance(records, provenance_path)
    write_audit(audit, audit_path)

    print(f"Audit: {audit.valid_cases}/{len(audit.cases)} valid")

    if not audit.is_valid:
        raise RuntimeError(
            "Expanded100 acquisition contains invalid KiTS23 cases. "
            f"See {audit_path}."
        )


def load_rows(
    *,
    project_root: Path,
    case_numbers: tuple[int, ...],
) -> list:
    rows = []

    for case_number in case_numbers:
        paths = pilot_case_paths(project_root, case_number)
        if not paths.evaluation_json.is_file():
            continue

        rows.append(
            load_case_cohort_row(
                case_id=paths.case_id,
                evaluation_path=paths.evaluation_json,
                inference_path=paths.inference_json,
                qc_path=paths.comparison_qc_json,
            )
        )

    return rows


def write_summary(*, project_root: Path, rows: list) -> None:
    if not rows:
        raise RuntimeError("No completed cases are available to summarize.")

    summary = summarize_cohort(rows)
    output_root = experiment_root(project_root)

    write_cohort_json(summary, output_root / "summary.json")
    write_cohort_csv(summary, output_root / "cases.csv")

    print()
    print("=== Expanded100 cohort summary ===")
    print(f"Completed cases: {summary.case_count}/{EXPANDED100_SIZE}")

    kidney = summary.kidney_and_mass_dice
    if kidney.std is not None:
        print(f"Kidney-and-mass Dice: {kidney.mean:.4f} +/- {kidney.std:.4f}")
    else:
        print(f"Kidney-and-mass Dice: {kidney.mean:.4f}")

    print(
        "Surface Dice: "
        f"{summary.kidney_and_mass_surface_dice.mean:.4f}"
    )
    print(
        "HD95 median: "
        f"{summary.kidney_and_mass_hd95_mm.median:.2f} mm"
    )
    print(f"Cyst-positive cases: {summary.cyst_positive_case_count}")

    if summary.cyst_dice_cyst_positive_cases is not None:
        print(
            "Cyst-positive-only Dice: "
            f"{summary.cyst_dice_cyst_positive_cases.mean:.4f}"
        )

    if summary.inference_wall_time_seconds is not None:
        print(
            "Inference wall time mean: "
            f"{summary.inference_wall_time_seconds.mean:.2f} s"
        )

    print("Full KiTS23 score: unavailable by protocol")


def render_worst_case_qc(
    *,
    project_root: Path,
    case_numbers: tuple[int, ...],
    count: int,
    force_qc: bool,
) -> tuple[str, ...]:
    if count < 0:
        raise ValueError("--worst-qc must be non-negative.")
    if count == 0:
        return ()

    rows = load_rows(project_root=project_root, case_numbers=case_numbers)
    worst_case_ids = select_worst_surface_dice_cases(
        rows,
        count=min(count, len(rows)),
    )
    number_by_id = {
        f"case_{number:05d}": number for number in case_numbers
    }

    print()
    print(f"=== Expanded100 worst-case visual QC (N={len(worst_case_ids)}) ===")

    for case_id in worst_case_ids:
        case_number = number_by_id[case_id]
        paths = pilot_case_paths(project_root, case_number)

        if paths.comparison_qc_json.is_file() and not force_qc:
            print(f"{case_id}: QC reused")
            continue

        case = discover_case(paths.data_directory)
        result = build_case_comparison_qc(
            case_id=case_id,
            image_path=case.imaging_path,
            ground_truth_path=case.segmentation_path,
            prediction_path=paths.prediction_path,
            output_directory=(
                project_root / "outputs" / "qc" / "nv_segment_ct_comparison"
            ),
            window_hu=(-200.0, 300.0),
        )
        write_comparison_qc_index(result, paths.comparison_qc_json)
        print(
            f"{case_id}: QC rendered, "
            f"disagreement={result.total_disagreement_voxels}"
        )

    return worst_case_ids


def run_cases(
    *,
    project_root: Path,
    case_numbers: tuple[int, ...],
    environment_name: str,
    force_inference: bool,
    force_evaluation: bool,
    continue_on_error: bool,
) -> tuple[tuple[str, ...], dict[str, str], int, int, int, int]:
    nv_paths = runtime_paths(project_root)
    repository_commit = read_runtime_commit(project_root)

    successful: list[str] = []
    failures: dict[str, str] = {}
    reused_inference = 0
    reused_evaluation = 0
    new_inference = 0
    new_evaluation = 0

    print("=== Expanded100 inference and evaluation ===")
    print(f"Runtime commit: {repository_commit}")

    for index, case_number in enumerate(case_numbers, start=1):
        paths = pilot_case_paths(project_root, case_number)
        case_id = paths.case_id
        print()
        print(f"[{index:03d}/{len(case_numbers):03d}] {case_id}")

        try:
            case = discover_case(paths.data_directory)

            if (
                paths.prediction_path.is_file()
                and paths.inference_json.is_file()
                and not force_inference
            ):
                reused_inference += 1
                print("  inference: reused")
            else:
                result = run_renal_inference(
                    case_id=case_id,
                    image_path=case.imaging_path,
                    paths=nv_paths,
                    repository_commit=repository_commit,
                    project_output_root=(
                        project_root
                        / "outputs"
                        / "predictions"
                        / "nv_segment_ct"
                    ),
                    environment_name=environment_name,
                    force=force_inference,
                )

                from vista3d_tfg.integration.nv_inference import write_inference_result

                write_inference_result(result, paths.inference_json)
                new_inference += 1
                print(f"  inference: {result.wall_time_seconds:.2f} s")

            if paths.evaluation_json.is_file() and not force_evaluation:
                reused_evaluation += 1
                print("  evaluation: reused")
            else:
                evaluation = evaluate_supported_renal_prediction(
                    case_id=case_id,
                    ground_truth_path=case.segmentation_path,
                    prediction_path=paths.prediction_path,
                )
                write_supported_renal_comparison(
                    evaluation,
                    paths.evaluation_json,
                )
                new_evaluation += 1
                print(
                    "  evaluation: "
                    f"Dice={evaluation.kidney_and_mass_hec.dice:.4f}, "
                    "SurfaceDice="
                    f"{evaluation.kidney_and_mass_hec.surface_dice:.4f}, "
                    "HD95="
                    f"{evaluation.kidney_and_mass_hec.diagnostic_hd95_mm:.2f} mm"
                )

            successful.append(case_id)

        except Exception as exc:
            failures[case_id] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failures[case_id]}")

            if not continue_on_error:
                raise

            traceback.print_exc()

    return (
        tuple(successful),
        failures,
        reused_inference,
        reused_evaluation,
        new_inference,
        new_evaluation,
    )


def write_expanded_run_metadata(
    *,
    output_path: Path,
    selection_path: Path,
    case_numbers: tuple[int, ...],
    successful: tuple[str, ...],
    failures: dict[str, str],
    reused_inference: int,
    reused_evaluation: int,
    new_inference: int,
    new_evaluation: int,
) -> None:
    payload = {
        "schema_version": "1.0",
        "experiment": EXPANDED100_EXPERIMENT,
        "selection_manifest": str(selection_path),
        "requested_case_count": len(case_numbers),
        "retained_pilot30_count": len(PILOT30_CASES),
        "additional_case_count": len(case_numbers) - len(PILOT30_CASES),
        "successful_case_count": len(successful),
        "failed_case_count": len(failures),
        "reused_inference_count": reused_inference,
        "reused_evaluation_count": reused_evaluation,
        "new_inference_count": new_inference,
        "new_evaluation_count": new_evaluation,
        "protocol_changed_from_pilot30": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    selection_path = (
        args.selection.resolve()
        if args.selection is not None
        else default_selection_path(project_root)
    )

    if not selection_path.is_file():
        raise FileNotFoundError(
            "Frozen expanded100 selection manifest not found: "
            f"{selection_path}. Run Block 30A first."
        )

    case_numbers = load_frozen_selection(selection_path)
    if len(case_numbers) != EXPANDED100_SIZE:
        raise RuntimeError("Frozen expanded100 selection is not exactly 100 cases.")

    root = experiment_root(project_root)

    print("NV-Segment-CT expanded100 KiTS23 validation")
    print(f"Selection: {selection_path}")
    print(f"Cases: {len(case_numbers)}")
    print(f"Retained pilot30: {len(PILOT30_CASES)}")
    print(f"Additional cases: {len(case_numbers) - len(PILOT30_CASES)}")
    print()

    if args.stage in {"acquire", "all"}:
        acquire_cases(
            project_root=project_root,
            case_numbers=case_numbers,
            retries=args.retries,
        )

    if args.stage == "acquire":
        print("Expanded100 acquisition PASSED.")
        return 0

    (
        successful,
        failures,
        reused_inference,
        reused_evaluation,
        new_inference,
        new_evaluation,
    ) = run_cases(
        project_root=project_root,
        case_numbers=case_numbers,
        environment_name=args.environment,
        force_inference=args.force_inference,
        force_evaluation=args.force_evaluation,
        continue_on_error=args.continue_on_error,
    )

    initial_rows = load_rows(project_root=project_root, case_numbers=case_numbers)
    write_summary(project_root=project_root, rows=initial_rows)

    worst_case_ids = render_worst_case_qc(
        project_root=project_root,
        case_numbers=case_numbers,
        count=args.worst_qc,
        force_qc=args.force_qc,
    )

    final_rows = load_rows(project_root=project_root, case_numbers=case_numbers)
    write_summary(project_root=project_root, rows=final_rows)

    write_run_manifest(
        output_path=root / "run_manifest.json",
        requested_case_ids=(f"case_{number:05d}" for number in case_numbers),
        successful_case_ids=successful,
        failed_cases=failures,
        worst_qc_case_ids=worst_case_ids,
        experiment=EXPANDED100_EXPERIMENT,
    )
    write_expanded_run_metadata(
        output_path=root / "expanded_run_metadata.json",
        selection_path=selection_path,
        case_numbers=case_numbers,
        successful=successful,
        failures=failures,
        reused_inference=reused_inference,
        reused_evaluation=reused_evaluation,
        new_inference=new_inference,
        new_evaluation=new_evaluation,
    )

    print()
    print(f"Run manifest: {root / 'run_manifest.json'}")
    print(f"Run metadata: {root / 'expanded_run_metadata.json'}")
    print(f"Reused inference artifacts: {reused_inference}")
    print(f"Reused evaluation artifacts: {reused_evaluation}")
    print(f"New inference artifacts: {new_inference}")
    print(f"New evaluation artifacts: {new_evaluation}")

    if failures:
        print(f"Expanded100 completed with {len(failures)} failed case(s).")
        return 3

    print("NV-Segment-CT expanded100 PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
