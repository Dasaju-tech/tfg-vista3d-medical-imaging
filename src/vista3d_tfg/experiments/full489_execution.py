"""Block 58: measured, resumable execution of the frozen internal partitions."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import time
from uuid import uuid4

from vista3d_tfg.data.acquisition import download_kits23_case
from vista3d_tfg.data.audit import audit_case
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.data.nifti import inspect_nifti
from vista3d_tfg.evaluation.nv_kits23 import (
    KITS23_KIDNEY_AND_MASS_TOLERANCE_MM,
    evaluate_supported_renal_prediction,
    write_supported_renal_comparison,
)
from vista3d_tfg.experiments.full489 import (
    build_protocol, canonical_sha256, load_protocol,
)
from vista3d_tfg.experiments.pilot30 import pilot_case_paths
from vista3d_tfg.integration.nv_inference import (
    RENAL_LABEL_PROMPT, audit_renal_prediction, discover_single_prediction,
    load_upstream_label_dict, sha256_file, validate_renal_label_contract,
)
from vista3d_tfg.integration.nv_segment import resolve_conda_executable, runtime_paths


PROTOCOL_FILE = "configs/experiments/nv_segment_ct_full489_protocol.json"
WORKER_FILE = "scripts/nv_segment_ct_full489_worker.py"
PROFILE_SIZE = 20


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex[:8] + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def frozen_json(path: Path, payload: dict) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        atomic_json(path, payload)


@contextmanager
def execution_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "execution.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(
            f"An execution lock already exists: {lock}. If a process was killed, "
            "confirm that it has stopped before removing this stale lock."
        ) from exc
    try:
        with handle:
            handle.write(json.dumps({"pid": os.getpid(), "host": platform.node()}))
        yield
    finally:
        lock.unlink(missing_ok=True)


def verified_protocol(root: Path) -> dict:
    payload = load_protocol(root / PROTOCOL_FILE)
    snapshot = root / "outputs/protocols/nv_segment_ct_full489/kits23_metadata.json"
    rebuilt = build_protocol(
        metadata_raw=snapshot.read_bytes(),
        expanded100_raw=(root / "configs/experiments/nv_segment_ct_expanded100_selection.json").read_bytes(),
    )
    if payload != rebuilt:
        raise ValueError("The current inputs differ from the Block 57 freeze.")
    result = subprocess.run(
        ["git", "-C", str(root), "show", "HEAD:" + PROTOCOL_FILE],
        capture_output=True, text=True, encoding="utf-8", check=False, timeout=30,
    )
    if result.returncode or json.loads(result.stdout.lstrip("\ufeff")) != payload:
        raise ValueError(
            "Commit the validated Block 57 protocol before Block 58. "
            "HEAD must contain the same full489 JSON as the working tree."
        )
    return payload


def selected_cases(protocol: dict, partition: str, *, acquire: bool = False) -> list[str]:
    if partition == "all" and acquire:
        return protocol["cohorts"]["all489"]["case_ids"]
    if partition == "test" and acquire:
        return protocol["partition"]["test"]["case_ids"]
    if partition not in ("development", "validation"):
        raise ValueError("Test inference/evaluation is sealed in Block 58; all is acquisition-only.")
    return protocol["partition"][partition]["case_ids"]


def worker_command(root: Path, environment: str, job: Path, report: Path) -> list[str]:
    return [
        resolve_conda_executable(), "run", "--no-capture-output", "-n", environment,
        "python", str((root / WORKER_FILE).resolve()),
        "--job", str(job.resolve()), "--report", str(report.resolve()),
    ]


def invoke_worker(command: list[str], root: Path, timeout: float) -> int:
    """Keep the Conda child tree together so an interrupted case releases its GPU."""
    options = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt" else {"start_new_session": True}
    )
    with subprocess.Popen(command, cwd=str(root), shell=False, **options) as process:
        try:
            return process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False, capture_output=True, timeout=15,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=15)
            raise


def run_worker(root: Path, environment: str, job: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    job_path, report_path = folder / "job.json", folder / "worker.json"
    atomic_json(job_path, job)
    # An attempt always has a new directory, so an old success cannot mask failure.
    if report_path.exists():
        raise FileExistsError(report_path)
    started = time.perf_counter()
    returncode = invoke_worker(
        worker_command(root, environment, job_path, report_path), root,
        timeout=180 if job["mode"] == "probe" else 3600,
    )
    if not report_path.is_file():
        raise RuntimeError(f"The runtime returned no report (exit {returncode}).")
    report = read_json(report_path)
    if returncode or report.get("status") != "complete":
        raise RuntimeError(report.get("error", f"Worker exit code {returncode}"))
    report["process_wall_seconds"] = time.perf_counter() - started
    return report


def _tree_hashes(base: Path, folders: tuple[str, ...]) -> dict[str, str]:
    files = [
        path for folder in folders for path in (base / folder).rglob("*")
        if path.is_file() and path.suffix in (".py", ".json", ".yaml", ".yml", ".txt")
        and "__pycache__" not in path.parts
    ]
    return {p.relative_to(base).as_posix(): sha256_file(p) for p in sorted(files)}


def prepare_run(root: Path, protocol: dict, environment: str) -> tuple[Path, dict, dict]:
    paths = runtime_paths(root)
    checkpoint = paths.ct_bundle_root / "models/model.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(
            f"Existing model checkpoint not found: {checkpoint}. "
            "Restore the working NV-Segment runtime before profiling."
        )
    config = read_json(paths.ct_bundle_root / "configs/inference.json")
    if config.get("checkpointloader", {}).get("load_path") != "@checkpoint_path":
        raise ValueError("The bundle no longer loads the declared checkpoint_path.")
    labels = load_upstream_label_dict(paths.ct_bundle_root / "configs/label_dict.json")
    validate_renal_label_contract(labels)
    commit = subprocess.run(
        ["git", "-C", str(paths.checkout_root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True, timeout=30,
    ).stdout.strip()
    folder = root / "outputs/full489/probes" / uuid4().hex[:12]
    probe = run_worker(root, environment, {"mode": "probe"}, folder)
    if probe.get("fp32_fp16_conv3d_smoke") is not True:
        raise ValueError("The runtime did not complete the CUDA FP32/FP16 kernel check.")
    identity = probe["identity"]
    packages = {}
    for package in ("numpy", "scipy", "nibabel", "surface-distance"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "missing"
    evaluation_files = _tree_hashes(
        root / "src/vista3d_tfg", ("data", "metrics"),
    )
    for relative in (
        "analysis/anatomy.py", "evaluation/nv_kits23.py",
        "integration/nv_inference.py", "integration/nv_segment.py",
        "experiments/full489.py", "experiments/full489_execution.py",
    ):
        evaluation_files[relative] = sha256_file(root / "src/vista3d_tfg" / relative)
    contract = {
        "schema_version": "1.0",
        "protocol_sha256": protocol["protocol_sha256"],
        "upstream_commit": commit,
        "bundle_files": _tree_hashes(paths.ct_bundle_root, ("configs", "scripts")),
        "checkpoint_sha256": sha256_file(checkpoint),
        "runtime_identity": identity,
        "worker_sha256": sha256_file(root / WORKER_FILE),
        "evaluation_files": evaluation_files,
        "evaluation_python": platform.python_version(),
        "evaluation_packages": packages,
        "label_prompt": list(RENAL_LABEL_PROMPT),
        "kidney_mass_surface_tolerance_mm": KITS23_KIDNEY_AND_MASS_TOLERANCE_MM,
        "execution": "one fresh MONAI bundle process per case; frozen upstream settings",
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    run_root = root / "outputs/full489" / contract["contract_sha256"][:16]
    frozen_json(run_root / "contract.json", contract)
    atomic_json(run_root / "latest_probe.json", probe)
    return run_root, contract, probe


def input_identity(root: Path, cid: str) -> tuple[object, dict]:
    paths = pilot_case_paths(root, int(cid[5:]))
    case = discover_case(paths.data_directory)
    return case, {
        "case_id": cid,
        "image_sha256": sha256_file(case.imaging_path),
        "ground_truth_sha256": sha256_file(case.segmentation_path),
    }


def checked_receipt(case_root: Path, expected: dict) -> dict | None:
    receipt_path = case_root / "complete.json"
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != canonical_sha256(unsigned):
        raise ValueError(f"Completion receipt changed: {receipt_path}")
    if receipt.get("identity") != expected:
        raise ValueError(f"Input/model/evaluation identity changed for {case_root.name}.")
    artifacts = receipt.get("artifacts")
    if (
        receipt.get("status") != "complete"
        or receipt.get("case_id") != expected["case_id"]
        or not isinstance(artifacts, dict)
        or len(artifacts) != 4
        or {Path(name).name for name in artifacts} != {
            "prediction.nii.gz", "supported_renal_metrics.json", "worker.json", "job.json",
        }
    ):
        raise ValueError("Incomplete completion receipt.")
    for relative, digest in receipt["artifacts"].items():
        path = case_root / relative
        if not path.is_relative_to(case_root) or ".." in Path(relative).parts:
            raise ValueError("Invalid artifact path in receipt.")
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Completed artifact is missing or changed: {path}")
    return receipt


def _finite(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def execute_case(
    root: Path, run_root: Path, contract: dict, cid: str, environment: str,
) -> tuple[dict, bool]:
    started = time.perf_counter()
    if sha256_file(root / WORKER_FILE) != contract["worker_sha256"]:
        raise ValueError("The worker changed after the run contract was prepared.")
    case, inputs = input_identity(root, cid)
    identity = {**inputs, "contract_sha256": contract["contract_sha256"]}
    case_root = run_root / "cases" / cid
    receipt = checked_receipt(case_root, identity)
    if receipt is not None:
        return receipt, True
    frozen_json(case_root / "inputs.json", inputs)
    audit = audit_case(case)
    if not audit.is_valid:
        raise ValueError(f"Invalid input geometry/labels for {cid}: {audit.issues}")
    attempt = case_root / ("attempt_" + uuid4().hex[:8])
    attempt.mkdir(parents=True)
    job = {
        "mode": "infer",
        "bundle_root": str(runtime_paths(root).ct_bundle_root),
        "bundle_files": contract["bundle_files"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "runtime_identity": contract["runtime_identity"],
        "image_path": str(case.imaging_path),
        "image_sha256": inputs["image_sha256"],
        "output_dir": str(attempt / "raw"),
        "label_prompt": contract["label_prompt"],
    }
    report = run_worker(root, environment, job, attempt)
    for key in ("runtime_identity", "checkpoint_sha256", "image_sha256"):
        if report.get(key) != job[key]:
            raise ValueError(f"The worker reported a different {key}.")
    for key in ("bundle_wall_seconds", "process_wall_seconds",
                "peak_allocated_bytes", "peak_reserved_bytes"):
        value = report.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid runtime measurement: {key}.")
    labels = load_upstream_label_dict(runtime_paths(root).ct_bundle_root / "configs/label_dict.json")
    raw = discover_single_prediction(attempt / "raw")
    prediction = attempt / "prediction.nii.gz"
    shutil.copy2(raw, prediction)
    prediction_audit = audit_renal_prediction(
        input_path=case.imaging_path, prediction_path=prediction, labels=labels,
    )
    if not prediction_audit.is_valid:
        raise ValueError(f"Prediction geometry/labels failed for {cid}.")
    evaluation = evaluate_supported_renal_prediction(
        case_id=cid, ground_truth_path=case.segmentation_path, prediction_path=prediction,
    )
    evaluation_path = attempt / "supported_renal_metrics.json"
    write_supported_renal_comparison(evaluation, evaluation_path)
    if input_identity(root, cid)[1] != inputs:
        raise ValueError(f"Input data changed during the case: {cid}")
    metrics = evaluation.kidney_and_mass_hec
    receipt = {
        "status": "complete",
        "case_id": cid,
        "identity": identity,
        "artifacts": {
            p.relative_to(case_root).as_posix(): sha256_file(p)
            for p in (prediction, evaluation_path, attempt / "worker.json", attempt / "job.json")
        },
        "timing": {
            "case_total_seconds": time.perf_counter() - started,
            "worker_process_seconds": report["process_wall_seconds"],
            "bundle_seconds": report["bundle_wall_seconds"],
        },
        "memory": {k: report[k] for k in ("peak_allocated_bytes", "peak_reserved_bytes")},
        "kidney_mass": {
            "dice": _finite(metrics.dice),
            "surface_dice": _finite(metrics.surface_dice),
            "hd95_mm": _finite(metrics.diagnostic_hd95_mm),
            "hd95_nonfinite": not math.isfinite(metrics.diagnostic_hd95_mm),
        },
        "prediction_audit": asdict(prediction_audit),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    atomic_json(case_root / "complete.json", receipt)
    return receipt, False


def profile_plan(root: Path, protocol: dict) -> dict:
    """Five largest CT grids plus fifteen hash-selected historical cases."""
    grids = {}
    for cid in protocol["cohorts"]["historical100"]["case_ids"]:
        metadata = inspect_nifti(pilot_case_paths(root, int(cid[5:])).imaging_path)
        grids[cid] = list(metadata.shape)
    largest = sorted(grids, key=lambda cid: (-math.prod(grids[cid]), cid))[:5]
    remaining = sorted(
        set(grids) - set(largest),
        key=lambda cid: sha256(("full489-profile-v1|" + cid).encode()).hexdigest(),
    )
    return {
        "protocol_sha256": protocol["protocol_sha256"],
        "method": "five largest historical CT grids plus fifteen fixed hash-ranked cases",
        "case_ids": largest + remaining[:PROFILE_SIZE - len(largest)],
        "historical_grid_shapes": grids,
        "selection_uses_model_outcomes": False,
    }


def _progress(requested: list[str], completed: list, failed: dict, deferred: list,
              newly_run: list[str], contract: dict) -> dict:
    return {
        "contract_sha256": contract["contract_sha256"],
        "requested_case_ids": requested,
        "completed_case_ids": [r["case_id"] for r in completed],
        "new_inference_case_ids": newly_run,
        "failed_cases": failed,
        "deferred_case_ids": deferred,
        "requested_count": len(requested),
        "completed_count": len(completed),
        "status": "failed" if failed else "partial" if len(completed) < len(requested) else "complete",
    }


def execute_batch(
    root: Path, run_root: Path, contract: dict, protocol: dict, environment: str,
    *, phase: str, partition: str = "development", max_cases: int | None = None,
    continue_on_error: bool = False,
) -> dict:
    if phase not in ("profile", "run"):
        raise ValueError("Unknown inference phase.")
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError("max_cases must be a positive integer.")
    if phase == "profile":
        if partition != "development":
            raise ValueError("Profiling is restricted to historical development cases.")
        plan = profile_plan(root, protocol)
        requested = plan["case_ids"]
        if len(requested) != PROFILE_SIZE or not set(requested).issubset(
            protocol["cohorts"]["historical100"]["case_ids"]
        ):
            raise ValueError("The profiling cohort must contain 20 historical cases.")
        frozen_json(run_root / "profile_plan.json", plan)
        label = "profile"
    else:
        requested = selected_cases(protocol, partition)
        require_profile(root, run_root, contract, protocol)
        label = partition
    progress_path = run_root / f"{label}_progress.json"
    completed, newly_run, failed, deferred = [], [], {}, []
    attempted = 0
    atomic_json(progress_path, _progress(requested, [], {}, requested, [], contract))
    for index, cid in enumerate(requested):
        has_receipt = (run_root / "cases" / cid / "complete.json").is_file()
        if not has_receipt and max_cases is not None and attempted >= max_cases:
            deferred.append(cid)
            continue
        print(f"[{index + 1:03d}/{len(requested):03d}] {cid}", flush=True)
        try:
            if not has_receipt:
                attempted += 1
            receipt, reused = execute_case(root, run_root, contract, cid, environment)
            completed.append(receipt)
            if not reused:
                newly_run.append(cid)
            print(
                f"  {'verified cache' if reused else 'new inference'}; "
                f"Dice={receipt['kidney_mass']['dice']}; "
                f"case={receipt['timing']['case_total_seconds']:.2f} s; "
                f"peak reserved={receipt['memory']['peak_reserved_bytes'] / 2**30:.2f} GiB",
                flush=True,
            )
        except Exception as exc:
            failed[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[cid]}", flush=True)
            if not continue_on_error:
                deferred.extend(requested[index + 1:])
        pending = list(dict.fromkeys(deferred + requested[index + 1:]))
        state = _progress(requested, completed, failed, pending, newly_run, contract)
        atomic_json(progress_path, state)
        if failed and not continue_on_error:
            break
    state = _progress(requested, completed, failed, deferred, newly_run, contract)
    atomic_json(progress_path, state)
    if phase == "profile":
        report = summarize_profile(root, contract, state, completed)
        atomic_json(run_root / "profile_summary.json", report)
    return state


def summarize_profile(root: Path, contract: dict, state: dict, completed: list) -> dict:
    comparisons = []
    for receipt in completed:
        cid = receipt["case_id"]
        legacy_path = pilot_case_paths(root, int(cid[5:])).evaluation_json
        if legacy_path.is_file():
            try:
                old = read_json(legacy_path)["result"]["kidney_and_mass_hec"]
                comparisons.append({
                    "case_id": cid,
                    "legacy_dice": _finite(old["dice"]),
                    "current_dice": receipt["kidney_mass"]["dice"],
                    "dice_difference": _finite(receipt["kidney_mass"]["dice"] - old["dice"]),
                    "legacy_evaluation_sha256": sha256_file(legacy_path),
                })
            except (ValueError, KeyError, TypeError) as exc:
                comparisons.append({"case_id": cid, "legacy_read_error": str(exc)})
    seconds = [r["timing"]["case_total_seconds"] for r in completed]
    return {
        **state,
        "gpu": contract["runtime_identity"],
        "timing_scope": "separate process per case; includes initialization, audit and evaluation",
        "median_case_seconds": statistics.median(seconds) if seconds else None,
        "maximum_case_seconds": max(seconds) if seconds else None,
        "maximum_peak_allocated_gib": max(
            (r["memory"]["peak_allocated_bytes"] / 2**30 for r in completed), default=None,
        ),
        "maximum_peak_reserved_gib": max(
            (r["memory"]["peak_reserved_bytes"] / 2**30 for r in completed), default=None,
        ),
        "legacy_comparison_scope": (
            "descriptive paired check only; legacy receipts lack full input/checkpoint hashes"
        ),
        "legacy_comparisons": comparisons,
        "statistical_test_or_clinical_validation": False,
    }


def require_profile(root: Path, run_root: Path, contract: dict, protocol: dict) -> None:
    path = run_root / "profile_summary.json"
    if not path.is_file():
        raise ValueError("Complete the 20-case historical profile before a cohort run.")
    report = read_json(path)
    expected_ids = profile_plan(root, protocol)["case_ids"]
    if (
        report.get("status") != "complete"
        or report.get("contract_sha256") != contract["contract_sha256"]
        or report.get("completed_case_ids") != expected_ids
        or report.get("requested_case_ids") != expected_ids
    ):
        raise ValueError("The required 20-case profile is incomplete or belongs to another contract.")
    for cid in expected_ids:
        _, inputs = input_identity(root, cid)
        if checked_receipt(
            run_root / "cases" / cid,
            {**inputs, "contract_sha256": contract["contract_sha256"]},
        ) is None:
            raise ValueError(f"Missing verified profile receipt for {cid}.")


def acquire_partition(
    root: Path, protocol: dict, partition: str, *, max_cases: int | None = None,
    continue_on_error: bool = False, retries: int = 2,
) -> dict:
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError("max_cases must be a positive integer.")
    requested = selected_cases(protocol, partition, acquire=True)
    ledger_path = root / "outputs/full489/acquisition.json"
    ledger = read_json(ledger_path) if ledger_path.is_file() else {
        "protocol_sha256": protocol["protocol_sha256"], "cases": {},
    }
    if ledger.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Acquisition ledger belongs to a different protocol.")
    completed, failed, deferred, attempted = [], {}, [], 0
    for index, cid in enumerate(requested):
        try:
            previous = ledger["cases"].get(cid)
            if previous and previous.get("status") == "complete":
                _, identity = input_identity(root, cid)
                if identity != previous["identity"]:
                    raise ValueError(f"Previously acquired inputs changed for {cid}.")
                completed.append(cid)
                continue
            if max_cases is not None and attempted >= max_cases:
                deferred.append(cid)
                continue
            attempted += 1
            print(f"[{index + 1:03d}/{len(requested):03d}] acquire/audit {cid}", flush=True)
            downloads = download_kits23_case(
                int(cid[5:]), root / "data/raw/kits23", retries=retries,
            )
            case, identity = input_identity(root, cid)
            audit = audit_case(case)
            if not audit.is_valid:
                raise ValueError(f"Input audit failed for {cid}: {audit.issues}")
            # A duplicate byte-identical CT must not enter different split IDs silently.
            duplicate = next((
                other for other, entry in ledger["cases"].items()
                if other != cid and entry.get("identity", {}).get("image_sha256")
                == identity["image_sha256"]
            ), None)
            if duplicate:
                raise ValueError(f"Byte-identical CT for {cid} and {duplicate}; review patient identity.")
            ledger["cases"][cid] = {
                "status": "complete", "identity": identity,
                "downloads": [asdict(item) for item in downloads],
                "input_audit_valid": True,
            }
            completed.append(cid)
            atomic_json(ledger_path, ledger)
        except Exception as exc:
            failed[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[cid]}", flush=True)
            if not continue_on_error:
                deferred.extend(requested[index + 1:])
                break
    state = {
        "protocol_sha256": protocol["protocol_sha256"],
        "requested_case_ids": requested,
        "completed_case_ids": completed,
        "failed_cases": failed,
        "deferred_case_ids": deferred,
        "status": "failed" if failed else "partial" if deferred else "complete",
        "operation": "input acquisition and geometry/label audit only",
        "predictions_metrics_or_visual_qc_generated": False,
        "duplicate_check_scope": "compressed CT byte identity; not a complete patient-linkage audit",
    }
    atomic_json(root / "outputs/full489" / f"acquisition_{partition}_progress.json", state)
    return state
