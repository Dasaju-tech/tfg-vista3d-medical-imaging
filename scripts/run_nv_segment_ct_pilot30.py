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
from vista3d_tfg.experiments.pilot30 import (
    PILOT30_CASES,
    pilot_case_paths,
    select_worst_surface_dice_cases,
    write_run_manifest,
    write_selection_manifest,
)
from vista3d_tfg.integration.nv_inference import run_renal_inference
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)
from vista3d_tfg.visualization.nv_compare import (
    build_case_comparison_qc,
    write_comparison_qc_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire, run, evaluate, and summarize the deterministic "
            "30-case KiTS23 NV-Segment-CT pilot cohort."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("acquire", "run", "all"),
        default="all",
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Override the frozen pilot case list for debugging only. "
            "Omit this option for the registered pilot30 experiment."
        ),
    )
    parser.add_argument(
        "--worst-qc",
        type=int,
        default=5,
        help=(
            "Render prediction-vs-GT QC for the N worst Surface Dice cases."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
    )
    parser.add_argument(
        "--force-evaluation",
        action="store_true",
    )
    parser.add_argument(
        "--force-qc",
        action="store_true",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue with later cases after a case-level failure and "
            "record failures in the run manifest."
        ),
    )
    return parser.parse_args()


def read_runtime_commit(project_root: Path) -> str:
    manifest_path = (
        project_root
        / "outputs"
        / "runtime"
        / "nv_segment_ct_runtime.json"
    )

    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    probe = payload.get("probe", {})

    if probe.get("cuda_available") is not True:
        raise RuntimeError(
            "Recorded NV-Segment runtime is not CUDA-enabled."
        )

    commit = payload.get("repository_commit")

    if not isinstance(commit, str) or not commit:
        raise RuntimeError(
            "Runtime manifest does not contain repository_commit."
        )

    return commit


def acquire_cases(
    *,
    project_root: Path,
    case_numbers: tuple[int, ...],
    retries: int,
) -> None:
    dataset_root = project_root / "data" / "raw" / "kits23"
    records = []

    print("=== Acquisition and integrity audit ===")

    for index, case_number in enumerate(case_numbers, start=1):
        case_id = f"case_{case_number:05d}"
        print(
            f"[{index:02d}/{len(case_numbers):02d}] "
            f"Acquiring {case_id} ..."
        )

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
                f"{record.kind}={record.status}"
                for record in case_records
            )
        )

    cases = tuple(
        discover_case(
            dataset_root / f"case_{number:05d}"
        )
        for number in case_numbers
    )

    audit = audit_cases(cases)

    provenance_path = (
        project_root
        / "outputs"
        / "manifests"
        / "kits23_pilot30_provenance.json"
    )
    audit_path = (
        project_root
        / "outputs"
        / "audits"
        / "kits23_pilot30_audit.json"
    )

    write_acquisition_provenance(records, provenance_path)
    write_audit(audit, audit_path)

    print(
        f"Audit: {audit.valid_cases}/{len(audit.cases)} valid"
    )

    if not audit.is_valid:
        raise RuntimeError(
            "Pilot acquisition contains invalid KiTS23 cases. "
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


def write_summary(
    *,
    project_root: Path,
    rows: list,
) -> None:
    if not rows:
        raise RuntimeError("No completed cases are available to summarize.")

    summary = summarize_cohort(rows)

    output_root = (
        project_root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_pilot30"
    )

    write_cohort_json(
        summary,
        output_root / "summary.json",
    )
    write_cohort_csv(
        summary,
        output_root / "cases.csv",
    )

    print()
    print("=== Pilot cohort summary ===")
    print(
        f"Completed cases: {summary.case_count}"
    )
    print(
        "Kidney-and-mass Dice: "
        f"{summary.kidney_and_mass_dice.mean:.4f} "
        f"± {summary.kidney_and_mass_dice.std:.4f}"
        if summary.kidney_and_mass_dice.std is not None
        else (
            "Kidney-and-mass Dice: "
            f"{summary.kidney_and_mass_dice.mean:.4f}"
        )
    )
    print(
        "Surface Dice: "
        f"{summary.kidney_and_mass_surface_dice.mean:.4f}"
    )
    print(
        "HD95 median: "
        f"{summary.kidney_and_mass_hd95_mm.median:.2f} mm"
    )
    print(
        f"Cyst-positive cases: "
        f"{summary.cyst_positive_case_count}"
    )

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

    print(
        "Full KiTS23 score: unavailable by protocol"
    )


def render_worst_case_qc(
    *,
    project_root: Path,
    case_numbers: tuple[int, ...],
    count: int,
    force_qc: bool,
) -> tuple[str, ...]:
    if count == 0:
        return ()

    rows = load_rows(
        project_root=project_root,
        case_numbers=case_numbers,
    )

    worst_case_ids = select_worst_surface_dice_cases(
        rows,
        count=min(count, len(rows)),
    )

    number_by_id = {
        f"case_{number:05d}": number
        for number in case_numbers
    }

    print()
    print(
        "=== Worst-case visual QC "
        f"(N={len(worst_case_ids)}) ==="
    )

    for case_id in worst_case_ids:
        case_number = number_by_id[case_id]
        paths = pilot_case_paths(
            project_root,
            case_number,
        )

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
                project_root
                / "outputs"
                / "qc"
                / "nv_segment_ct_comparison"
            ),
            window_hu=(-200.0, 300.0),
        )

        write_comparison_qc_index(
            result,
            paths.comparison_qc_json,
        )

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
) -> tuple[tuple[str, ...], dict[str, str]]:
    nv_paths = runtime_paths(project_root)
    repository_commit = read_runtime_commit(project_root)

    successful: list[str] = []
    failures: dict[str, str] = {}

    print("=== Inference and evaluation ===")
    print(f"Runtime commit: {repository_commit}")

    for index, case_number in enumerate(case_numbers, start=1):
        paths = pilot_case_paths(
            project_root,
            case_number,
        )
        case_id = paths.case_id

        print()
        print(
            f"[{index:02d}/{len(case_numbers):02d}] "
            f"{case_id}"
        )

        try:
            case = discover_case(paths.data_directory)

            if (
                paths.prediction_path.is_file()
                and paths.inference_json.is_file()
                and not force_inference
            ):
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

                from vista3d_tfg.integration.nv_inference import (
                    write_inference_result,
                )

                write_inference_result(
                    result,
                    paths.inference_json,
                )

                print(
                    f"  inference: {result.wall_time_seconds:.2f} s"
                )

            if (
                paths.evaluation_json.is_file()
                and not force_evaluation
            ):
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

                print(
                    "  evaluation: "
                    f"Dice="
                    f"{evaluation.kidney_and_mass_hec.dice:.4f}, "
                    f"SurfaceDice="
                    f"{evaluation.kidney_and_mass_hec.surface_dice:.4f}, "
                    f"HD95="
                    f"{evaluation.kidney_and_mass_hec.diagnostic_hd95_mm:.2f} mm"
                )

            successful.append(case_id)

        except Exception as exc:
            failures[case_id] = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f"  FAILED: {failures[case_id]}"
            )

            if not continue_on_error:
                raise

            traceback.print_exc()

    return tuple(successful), failures


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    case_numbers = tuple(
        args.cases
        if args.cases is not None
        else PILOT30_CASES
    )

    experiment_root = (
        project_root
        / "outputs"
        / "cohorts"
        / "nv_segment_ct_pilot30"
    )

    write_selection_manifest(
        experiment_root / "selection.json",
        case_numbers,
    )

    print("NV-Segment-CT deterministic KiTS23 pilot")
    print(f"Cases: {len(case_numbers)}")
    print(
        "IDs: "
        + " ".join(
            f"{number:05d}"
            for number in case_numbers
        )
    )
    print(
        "Selection manifest: "
        f"{experiment_root / 'selection.json'}"
    )
    print()

    if args.stage in {"acquire", "all"}:
        acquire_cases(
            project_root=project_root,
            case_numbers=case_numbers,
            retries=args.retries,
        )

    if args.stage == "acquire":
        print("Pilot acquisition PASSED.")
        return 0

    successful, failures = run_cases(
        project_root=project_root,
        case_numbers=case_numbers,
        environment_name=args.environment,
        force_inference=args.force_inference,
        force_evaluation=args.force_evaluation,
        continue_on_error=args.continue_on_error,
    )

    initial_rows = load_rows(
        project_root=project_root,
        case_numbers=case_numbers,
    )
    write_summary(
        project_root=project_root,
        rows=initial_rows,
    )

    worst_case_ids = render_worst_case_qc(
        project_root=project_root,
        case_numbers=case_numbers,
        count=args.worst_qc,
        force_qc=args.force_qc,
    )

    # Re-load so QC counts are included for the selected worst cases.
    final_rows = load_rows(
        project_root=project_root,
        case_numbers=case_numbers,
    )
    write_summary(
        project_root=project_root,
        rows=final_rows,
    )

    write_run_manifest(
        output_path=experiment_root / "run_manifest.json",
        requested_case_ids=(
            f"case_{number:05d}"
            for number in case_numbers
        ),
        successful_case_ids=successful,
        failed_cases=failures,
        worst_qc_case_ids=worst_case_ids,
    )

    print()
    print(
        f"Run manifest: "
        f"{experiment_root / 'run_manifest.json'}"
    )

    if failures:
        print(
            f"Pilot completed with {len(failures)} failed case(s)."
        )
        return 3

    print("NV-Segment-CT pilot30 PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
