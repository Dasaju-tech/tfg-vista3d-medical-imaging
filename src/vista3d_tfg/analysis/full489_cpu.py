"""Block 61: isolated boolean-mask fast path and paired CPU evaluation benchmark."""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
import time
from uuid import uuid4
import zipfile

import numpy as np

from vista3d_tfg.analysis.full489_development import describe
from vista3d_tfg.analysis.full489_review import (
    check_evaluation_environment, check_plan, differences, verified_review, write_csv,
)
from vista3d_tfg.evaluation.nv_kits23 import evaluate_supported_renal_prediction
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import (
    atomic_json, checked_receipt, frozen_json, input_identity, read_json,
)
from vista3d_tfg.integration.nv_inference import sha256_file
from vista3d_tfg.metrics import morphology


_ORIGINAL_VALIDATOR = morphology.validate_binary_mask
_MODE_LOCK = threading.Lock()


def validate_binary_mask_fast(mask: np.ndarray) -> None:
    # Only ordinary Boolean ndarrays take the shortcut. Numeric arrays, memmaps
    # and subclasses retain the exact original behavior (including errors).
    if type(mask) is np.ndarray and mask.dtype == np.dtype(bool) and mask.ndim == 3:
        return
    _ORIGINAL_VALIDATOR(mask)


@contextmanager
def evaluation_mode(optimized: bool):
    """Serial benchmark-only substitution; restore even after interruption.

    Morphology functions resolve their validator in the morphology module. No
    NumPy function or source file is replaced. Do not embed in a threaded service.
    """
    if not _MODE_LOCK.acquire(blocking=False):
        raise RuntimeError("CPU benchmark modes cannot run concurrently or nest.")
    try:
        if morphology.validate_binary_mask is not _ORIGINAL_VALIDATOR:
            raise RuntimeError("Morphology validator already differs from the baseline.")
        try:
            if optimized:
                morphology.validate_binary_mask = validate_binary_mask_fast
            yield
        finally:
            morphology.validate_binary_mask = _ORIGINAL_VALIDATOR
    finally:
        _MODE_LOCK.release()


def machine_identity() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "thread_environment": {k: os.environ.get(k) for k in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def check_baseline(root: Path, contract: dict) -> None:
    check_evaluation_environment(root, contract)
    # Also verify loaded package sources, so a different installed copy cannot
    # silently be measured while only the project-root copy passes its hashes.
    for relative, expected in contract["evaluation_files"].items():
        parts = list(Path(relative).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        module = sys.modules.get(".".join(("vista3d_tfg", *parts)))
        if module is not None and (not getattr(module, "__file__", None) or sha256_file(Path(module.__file__)) != expected):
            raise ValueError(f"Loaded evaluation source differs from the original run: {relative}")


def check_completed_review(review_root: Path, rows: list[dict], contract: dict) -> dict:
    plan = read_json(review_root / "plan.json")
    check_plan(plan, contract)
    if not plan["case_ids"] or plan["source_receipts"] != {r["case_id"]: r["receipt_sha256"] for r in rows}:
        raise ValueError("Block 60 sources differ from verified development results.")
    if plan["source_rows_sha256"] != canonical_sha256(rows):
        raise ValueError("Block 60 screening rows changed.")
    state = read_json(review_root / "review_progress.json")
    done = state.get("completed_case_ids", [])
    if (state.get("status") != "complete" or state.get("plan_sha256") != plan["plan_sha256"]
            or state.get("failed_cases") or state.get("deferred_case_ids")
            or len(done) != len(plan["case_ids"]) or set(done) != set(plan["case_ids"])):
        raise ValueError("Complete Block 60 review is required before benchmarking.")
    receipts = {}
    for cid in plan["case_ids"]:
        receipt = verified_review(review_root, cid, plan)
        if receipt is None:
            raise ValueError(f"Missing Block 60 review: {cid}")
        report_path = next(p for p in receipt["artifacts"] if Path(p).name == "review.json")
        report = read_json(review_root / "cases" / cid / report_path)
        if report["original_receipt_sha256"] != plan["source_receipts"][cid] or report.get("metric_parity") is not True:
            raise ValueError(f"Block 60 original source changed: {cid}")
        receipts[cid] = receipt["receipt_sha256"]
    return {"plan_sha256": plan["plan_sha256"], "case_ids": plan["case_ids"],
            "review_receipts": receipts, "source_receipts": plan["source_receipts"]}


def prepare_benchmark(root: Path, contract: dict, review: dict, *, repeats: int = 2) -> tuple[Path, dict]:
    if type(repeats) is not int or repeats < 2 or repeats % 2:
        raise ValueError("repeats must be an even integer of at least 2 for balanced order.")
    plan = {"block": 61, "contract_sha256": contract["contract_sha256"],
            "protocol_sha256": contract["protocol_sha256"], **review, "repeats": repeats,
            "benchmark_module_sha256": sha256_file(Path(__file__)),
            "review_helper_sha256": sha256_file(Path(__file__).with_name("full489_review.py")),
            "packages": {p: version(p) for p in ("numpy", "scipy", "nibabel", "surface-distance")},
            "machine": machine_identity(),
            "optimization": "skip value enumeration only for ordinary 3D boolean ndarrays",
            "numeric_tolerance": {"absolute": 1e-9, "relative": 1e-9},
            "measurement": "serial full evaluator calls including NIfTI reads; no cProfile; no excluded warmup",
            "order": "alternate baseline/optimized and optimized/baseline by case index plus repetition index",
            "scope": "same outcome-guided development subset as Block 60; repeated file caches; no cohort extrapolation"}
    # The Block 60 plan hash is a source identity, not the new benchmark identity.
    plan["review_plan_sha256"] = plan.pop("plan_sha256")
    plan["plan_sha256"] = canonical_sha256(plan)
    output = root / "outputs/reporting/full489_block61" / contract["contract_sha256"][:16] / plan["plan_sha256"][:16]
    frozen_json(output / "plan.json", plan)
    return output, plan


def check_benchmark_plan(plan: dict, contract: dict) -> None:
    check_plan(plan, contract)
    if not plan["case_ids"] or set(plan["case_ids"]) != set(plan["review_receipts"]):
        raise ValueError("Benchmark selection differs from completed Block 60 reviews.")
    if type(plan["repeats"]) is not int or plan["repeats"] < 2 or plan["repeats"] % 2:
        raise ValueError("Unbalanced benchmark repetitions.")
    if plan["benchmark_module_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Benchmark source changed; prepare a new plan.")
    if plan["review_helper_sha256"] != sha256_file(Path(__file__).with_name("full489_review.py")):
        raise ValueError("Review helper changed; prepare a new plan.")
    if plan["machine"] != machine_identity() or any(version(p) != v for p, v in plan["packages"].items()):
        raise ValueError("Benchmark environment changed; prepare a new plan.")


def source_case(root: Path, run_root: Path, cid: str, plan: dict, contract: dict):
    if cid not in plan["case_ids"]:
        raise ValueError("Case is outside the frozen development benchmark selection.")
    case, inputs = input_identity(root, cid)
    receipt = checked_receipt(run_root / "cases" / cid, {**inputs, "contract_sha256": contract["contract_sha256"]})
    if receipt is None or receipt["receipt_sha256"] != plan["source_receipts"][cid]:
        raise ValueError(f"Original source differs from the benchmark plan: {cid}")
    artifacts = {Path(p).name: run_root / "cases" / cid / p for p in receipt["artifacts"]}
    return case, inputs, artifacts


def json_metrics(value):
    """Standard JSON: encode equal positive infinities explicitly, never NaNs."""
    if isinstance(value, dict):
        return {k: json_metrics(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_metrics(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if value == math.inf:
            return "+Infinity"
        raise ValueError("Nonfinite metric other than positive infinity.")
    return value


def verified_benchmark(output: Path, cid: str, plan: dict) -> dict | None:
    path = output / "cases" / cid / "complete.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if (receipt.get("receipt_sha256") != canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get("case_id") != cid
            or receipt.get("status") != "complete" or receipt.get("metric_parity") is not True):
        raise ValueError(f"Benchmark completion changed: {cid}")
    artifacts = receipt.get("artifacts", {})
    if len(artifacts) != 1:
        raise ValueError("Incomplete benchmark artifact list.")
    for name, digest in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.name != "benchmark.json" or sha256_file(path.parent / relative) != digest:
            raise ValueError(f"Benchmark artifact changed: {cid}")
    return receipt


def benchmark_case(root: Path, run_root: Path, output: Path, cid: str, plan: dict, contract: dict) -> tuple[dict, bool]:
    check_benchmark_plan(plan, contract)
    check_baseline(root, contract)
    case, inputs, artifacts = source_case(root, run_root, cid, plan, contract)
    cached = verified_benchmark(output, cid, plan)
    if cached:
        return cached, True
    attempt = output / "cases" / cid / ("attempt_" + uuid4().hex[:8])
    attempt.mkdir(parents=True)
    measurements = []
    try:
        baseline_saved = read_json(artifacts["supported_renal_metrics.json"])["result"]
        case_index = plan["case_ids"].index(cid)
        for repetition in range(plan["repeats"]):
            order = ("baseline", "optimized") if (case_index + repetition) % 2 == 0 else ("optimized", "baseline")
            pair = {}
            for position, mode in enumerate(order):
                print(f"  pair {repetition + 1}/{plan['repeats']}, {mode}: evaluating on CPU...", flush=True)
                with evaluation_mode(mode == "optimized"):
                    started = time.perf_counter()
                    result = evaluate_supported_renal_prediction(case_id=cid, ground_truth_path=case.segmentation_path,
                                                                prediction_path=artifacts["prediction.nii.gz"])
                    elapsed = time.perf_counter() - started
                payload = result.to_dict()
                delta = differences(baseline_saved, payload)
                if delta:
                    atomic_json(attempt / "parity_failure.json", {"mode": mode, "pair": repetition + 1, "differences": delta})
                    raise ValueError(f"{mode} metrics differ from saved evaluation in {len(delta)} fields.")
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("Nonpositive or nonfinite benchmark duration.")
                pair[mode] = payload
                measurements.append({"case_id": cid, "pair": repetition + 1, "position": position + 1,
                                     "mode": mode, "seconds": elapsed, "metric_parity": True,
                                     "result": json_metrics(payload)})
                print(f"    {elapsed:.2f} s; saved metric parity passed", flush=True)
            delta = differences(pair["baseline"], pair["optimized"])
            if delta:
                atomic_json(attempt / "parity_failure.json", {"mode": "paired", "pair": repetition + 1, "differences": delta})
                raise ValueError("Paired baseline/optimized metrics differ.")
        _, after, _ = source_case(root, run_root, cid, plan, contract)
        if after != inputs:
            raise ValueError("Raw inputs changed during CPU comparison.")
        check_baseline(root, contract)
        base = statistics.median(m["seconds"] for m in measurements if m["mode"] == "baseline")
        fast = statistics.median(m["seconds"] for m in measurements if m["mode"] == "optimized")
        summary = {"case_id": cid, "baseline_median_seconds": base, "optimized_median_seconds": fast,
                   "speedup_ratio": base / fast, "seconds_saved": base - fast}
        atomic_json(attempt / "benchmark.json", {"plan_sha256": plan["plan_sha256"],
                    "original_receipt_sha256": plan["source_receipts"][cid], "summary": summary,
                    "measurements": measurements, "metric_parity": True,
                    "nonfinite_encoding": "positive infinite metric values are stored as the string +Infinity"})
        receipt = {"status": "complete", "case_id": cid, "plan_sha256": plan["plan_sha256"],
                   "metric_parity": True, "summary": summary,
                   "artifacts": {(attempt / "benchmark.json").relative_to(attempt.parent).as_posix(): sha256_file(attempt / "benchmark.json")}}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(attempt.parent / "complete.json", receipt)
        return receipt, False
    except Exception as exc:
        atomic_json(attempt / "failure.json", {"case_id": cid, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}", "completed_measurements": measurements})
        raise


def export_benchmark(output: Path, plan: dict, state: dict) -> Path:
    summaries, measurements, files = [], [], [output / "plan.json"]
    for cid in state["completed_case_ids"]:
        receipt = verified_benchmark(output, cid, plan)
        if receipt is None:
            raise ValueError(f"Missing benchmark completion: {cid}")
        folder = output / "cases" / cid
        files.append(folder / "complete.json")
        path = folder / next(iter(receipt["artifacts"]))
        files.append(path)
        report = read_json(path)
        if report["summary"] != receipt["summary"] or len(report["measurements"]) != 2 * plan["repeats"]:
            raise ValueError("Benchmark report and completion disagree.")
        summaries.append(report["summary"])
        measurements.extend({k: v for k, v in m.items() if k != "result"} for m in report["measurements"])
    state["timing_summary"] = {"n_cases": len(summaries),
        "case_speedup_ratios": describe([r["speedup_ratio"] for r in summaries]),
        "cases_faster": sum(r["speedup_ratio"] > 1 for r in summaries),
        "ratio_of_summed_case_medians": (sum(r["baseline_median_seconds"] for r in summaries) /
                                         sum(r["optimized_median_seconds"] for r in summaries)) if summaries else None}
    state["interpretation"] = plan["scope"]
    atomic_json(output / "progress.json", state)
    write_csv(output / "case_timings.csv", summaries)
    write_csv(output / "measurements.csv", measurements)
    lines = ["# Block 61 — Paired CPU evaluation benchmark", "",
             f"Status: **{state['status']}**. Verified cases: {len(summaries)}/{len(plan['case_ids'])}.", "",
             "Both modes and every repetition must match the saved full evaluation, and each pair must match directly "
             "(absolute/relative tolerance 1e-9). NaNs fail; equal positive infinities are retained.", "",
             "Only ordinary 3D Boolean masks skip value enumeration. Other inputs use the original validator. "
             "The original evaluator files and GPU results are preserved. Validation/test remain reserved.", "",
             "Timings cover full evaluator calls including volume reads, without cProfile or excluded warmup. "
             "Order alternates within each case; the starting mode alternates between cases. Current file caches "
             "and system load affect results. Case medians and their ratios are descriptive for this selected subset; "
             "do not extrapolate to the whole cohort or claim segmentation improvement.", "",
             "| Case | Baseline median (s) | Optimized median (s) | Ratio |",
             "|---|---:|---:|---:|"]
    lines.extend(f"| {r['case_id']} | {r['baseline_median_seconds']:.3f} | {r['optimized_median_seconds']:.3f} | {r['speedup_ratio']:.3f} |" for r in summaries)
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.extend(output / n for n in ("progress.json", "case_timings.csv", "measurements.csv", "report.md"))
    for cid in state["failed_cases"]:
        for attempt in sorted((output / "cases" / cid).glob("attempt_*")):
            files.extend(attempt / n for n in ("failure.json", "parity_failure.json") if (attempt / n).is_file())
    archive = output / "block61_cpu_benchmark_report.zip"
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def run_benchmark(root: Path, run_root: Path, output: Path, plan: dict, contract: dict,
                  *, max_cases: int | None = None, continue_on_error: bool = False) -> tuple[dict, Path]:
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError("max_cases must be positive.")
    check_benchmark_plan(plan, contract)
    check_baseline(root, contract)
    completed, new, failed = [], [], {}
    attempted = 0
    def progress():
        return {"plan_sha256": plan["plan_sha256"], "requested_case_ids": plan["case_ids"],
                "completed_case_ids": completed.copy(), "newly_benchmarked_case_ids": new.copy(),
                "failed_cases": failed.copy(), "deferred_case_ids": [c for c in plan["case_ids"] if c not in completed and c not in failed],
                "status": "failed" if failed else "complete" if len(completed) == len(plan["case_ids"]) else "partial",
                "gpu_inference": False, "validation_test_outputs_opened": False}
    atomic_json(output / "progress.json", progress())
    for index, cid in enumerate(plan["case_ids"]):
        cached = (output / "cases" / cid / "complete.json").exists()
        if not cached and max_cases is not None and attempted >= max_cases:
            continue
        print(f"[{index + 1:02d}/{len(plan['case_ids']):02d}] benchmark {cid}", flush=True)
        try:
            if not cached:
                attempted += 1
            receipt, reused = benchmark_case(root, run_root, output, cid, plan, contract)
            completed.append(cid)
            if not reused:
                new.append(cid)
            print(f"  {'verified benchmark cache' if reused else 'benchmark complete'}; "
                  f"metric parity passed; ratio={receipt['summary']['speedup_ratio']:.3f}x", flush=True)
        except Exception as exc:
            failed[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[cid]}", flush=True)
            if not continue_on_error:
                break
        atomic_json(output / "progress.json", progress())
    state = progress()
    return state, export_benchmark(output, plan, state)
