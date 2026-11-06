"""Block 62: versioned evaluator parity on the entire frozen development set."""
from __future__ import annotations

from importlib.metadata import version
import math
from pathlib import Path
import time
from uuid import uuid4
import zipfile

from vista3d_tfg.analysis import full489_cpu as cpu
from vista3d_tfg.analysis.full489_development import describe
from vista3d_tfg.analysis.full489_review import check_plan, differences, write_csv
from vista3d_tfg.evaluation import nv_kits23_cpu_v1 as evaluator
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import atomic_json, frozen_json, read_json
from vista3d_tfg.integration.nv_inference import sha256_file


def completed_benchmark(root: Path, run_root: Path, benchmark_root: Path, rows: list[dict], contract: dict) -> dict:
    plan = read_json(benchmark_root / "plan.json")
    cpu.check_benchmark_plan(plan, contract)
    cpu.check_baseline(root, contract)
    review_root = root / "outputs/reporting/full489_block60" / run_root.name / plan["review_plan_sha256"][:16]
    review = cpu.check_completed_review(review_root, rows, contract)
    if (review["plan_sha256"] != plan["review_plan_sha256"] or review["case_ids"] != plan["case_ids"]
            or review["source_receipts"] != plan["source_receipts"] or review["review_receipts"] != plan["review_receipts"]):
        raise ValueError("Block 61 and completed Block 60 sources disagree.")
    state = read_json(benchmark_root / "progress.json")
    done = state.get("completed_case_ids", [])
    if (state.get("status") != "complete" or state.get("plan_sha256") != plan["plan_sha256"]
            or state.get("failed_cases") or state.get("deferred_case_ids")
            or len(done) != len(plan["case_ids"]) or set(done) != set(plan["case_ids"])):
        raise ValueError("Complete Block 61 benchmark is required.")
    receipts = {}
    for cid in plan["case_ids"]:
        receipt = cpu.verified_benchmark(benchmark_root, cid, plan)
        if receipt is None:
            raise ValueError(f"Missing Block 61 completion: {cid}")
        report = read_json(benchmark_root / "cases" / cid / next(iter(receipt["artifacts"])))
        if (report.get("plan_sha256") != plan["plan_sha256"] or report.get("metric_parity") is not True
                or report.get("original_receipt_sha256") != plan["source_receipts"][cid]):
            raise ValueError(f"Block 61 result identity differs: {cid}")
        # Compare exported evidence to the still-verified saved baseline, without
        # reading original voxel arrays or running the old evaluator again.
        original = read_json(run_root / "cases" / cid / "complete.json")
        metric_path = next(p for p in original["artifacts"] if Path(p).name == "supported_renal_metrics.json")
        saved = cpu.json_metrics(read_json(run_root / "cases" / cid / metric_path)["result"])
        measurements = report["measurements"]
        if len(measurements) != 2 * plan["repeats"]:
            raise ValueError("Incomplete Block 61 measurements.")
        for i, measurement in enumerate(measurements):
            pair, position = divmod(i, 2)
            order = ("baseline", "optimized") if (plan["case_ids"].index(cid) + pair) % 2 == 0 else ("optimized", "baseline")
            if (measurement.get("case_id") != cid or measurement.get("pair") != pair + 1
                    or measurement.get("position") != position + 1 or measurement.get("mode") != order[position]
                    or measurement.get("metric_parity") is not True or not math.isfinite(measurement["seconds"])
                    or measurement["seconds"] <= 0 or differences(saved, measurement["result"])):
                raise ValueError(f"Invalid Block 61 measurement or metric parity: {cid}")
        for i in range(0, len(measurements), 2):
            if differences(measurements[i]["result"], measurements[i + 1]["result"]):
                raise ValueError(f"Block 61 paired metrics disagree: {cid}")
        receipts[cid] = receipt["receipt_sha256"]
    return {"benchmark_plan_sha256": plan["plan_sha256"], "benchmark_receipts": receipts}


def implementation_identity() -> dict:
    return {"evaluator_version": evaluator.EVALUATOR_VERSION,
            "evaluator_module_sha256": sha256_file(Path(evaluator.__file__)),
            "optimizer_module_sha256": sha256_file(Path(cpu.__file__)),
            "adoption_module_sha256": sha256_file(Path(__file__)),
            "review_helper_sha256": sha256_file(Path(__file__).with_name("full489_review.py")),
            "packages": {p: version(p) for p in ("numpy", "scipy", "nibabel", "surface-distance")},
            "machine": cpu.machine_identity()}


def prepare_adoption(root: Path, rows: list[dict], contract: dict, protocol: dict, evidence: dict) -> tuple[Path, dict]:
    expected = protocol["partition"]["development"]["case_ids"]
    reserved = set(protocol["partition"]["validation"]["case_ids"]) | set(protocol["partition"]["test"]["case_ids"])
    if (not expected or [r["case_id"] for r in rows] != expected or len(set(expected)) != len(expected)
            or set(expected) & reserved or not set(evidence["benchmark_receipts"]).issubset(expected)):
        raise ValueError("Adoption must cover exactly the frozen development partition.")
    plan = {"block": 62, "contract_sha256": contract["contract_sha256"], "protocol_sha256": protocol["protocol_sha256"],
            "case_ids": expected, "source_receipts": {r["case_id"]: r["receipt_sha256"] for r in rows},
            "source_rows_sha256": canonical_sha256(rows), **evidence, "implementation": implementation_identity(),
            "numeric_tolerance": {"absolute": 1e-9, "relative": 1e-9},
            "scope": "full frozen development partition only; validation/test remain reserved",
            "measurement": "one optimized evaluator call per case; no paired timing baseline or speedup claim"}
    plan["plan_sha256"] = canonical_sha256(plan)
    output = root / "outputs/reporting/full489_block62" / contract["contract_sha256"][:16] / plan["plan_sha256"][:16]
    frozen_json(output / "plan.json", plan)
    return output, plan


def check_adoption(plan: dict, contract: dict, protocol: dict) -> None:
    check_plan(plan, contract)
    expected = protocol["partition"]["development"]["case_ids"]
    reserved = set(protocol["partition"]["validation"]["case_ids"]) | set(protocol["partition"]["test"]["case_ids"])
    if plan["case_ids"] != expected or set(plan["source_receipts"]) != set(expected) or set(expected) & reserved:
        raise ValueError("Adoption selection differs from frozen development.")
    if plan["implementation"] != implementation_identity():
        raise ValueError("Versioned evaluator code or environment changed; prepare a new plan.")


def verified_adoption(output: Path, cid: str, plan: dict) -> dict | None:
    path = output / "cases" / cid / "complete.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if (receipt.get("receipt_sha256") != canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get("case_id") != cid
            or receipt.get("metric_parity") is not True or receipt.get("status") != "complete"):
        raise ValueError(f"Block 62 completion changed: {cid}")
    if len(receipt.get("artifacts", {})) != 1:
        raise ValueError("Incomplete Block 62 artifact list.")
    for name, digest in receipt["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.name != "evaluation.json" or sha256_file(path.parent / relative) != digest:
            raise ValueError(f"Block 62 artifact changed: {cid}")
    return receipt


def adoption_case(root: Path, run_root: Path, output: Path, cid: str, plan: dict, contract: dict, protocol: dict) -> tuple[dict, bool]:
    check_adoption(plan, contract, protocol)
    cpu.check_baseline(root, contract)
    case, inputs, artifacts = cpu.source_case(root, run_root, cid, plan, contract)
    cached = verified_adoption(output, cid, plan)
    if cached:
        return cached, True
    attempt = output / "cases" / cid / ("attempt_" + uuid4().hex[:8])
    attempt.mkdir(parents=True)
    try:
        saved = read_json(artifacts["supported_renal_metrics.json"])["result"]
        print(f"  {evaluator.EVALUATOR_VERSION}: evaluating on CPU...", flush=True)
        started = time.perf_counter()
        result = evaluator.evaluate_supported_renal_prediction_cpu_v1(case_id=cid, ground_truth_path=case.segmentation_path,
                                                                    prediction_path=artifacts["prediction.nii.gz"])
        seconds = time.perf_counter() - started
        actual = result.to_dict()
        delta = differences(saved, actual)
        if delta:
            atomic_json(attempt / "parity_failure.json", {"case_id": cid, "differences": delta})
            raise ValueError(f"Versioned evaluation differs in {len(delta)} fields.")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Invalid evaluation duration.")
        _, after, _ = cpu.source_case(root, run_root, cid, plan, contract)
        if after != inputs:
            raise ValueError("Inputs changed during versioned evaluation.")
        cpu.check_baseline(root, contract)
        check_adoption(plan, contract, protocol)
        exact = cpu.json_metrics(saved) == cpu.json_metrics(actual)
        report = {"case_id": cid, "plan_sha256": plan["plan_sha256"], "status": "complete",
                  "evaluator_version": evaluator.EVALUATOR_VERSION, "original_receipt_sha256": plan["source_receipts"][cid],
                  "metric_parity": True, "exact_metric_match": exact, "seconds": seconds,
                  "numeric_tolerance": plan["numeric_tolerance"], "result": cpu.json_metrics(actual),
                  "nonfinite_encoding": "positive infinite metric values are stored as the string +Infinity"}
        atomic_json(attempt / "evaluation.json", report)
        receipt = {k: report[k] for k in ("case_id", "plan_sha256", "status", "metric_parity", "exact_metric_match", "seconds")}
        receipt["artifacts"] = {(attempt / "evaluation.json").relative_to(attempt.parent).as_posix(): sha256_file(attempt / "evaluation.json")}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(attempt.parent / "complete.json", receipt)
        return receipt, False
    except Exception as exc:
        atomic_json(attempt / "failure.json", {"case_id": cid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def export_adoption(output: Path, plan: dict, state: dict) -> Path:
    rows, files, receipts = [], [output / "plan.json"], {}
    for cid in state["completed_case_ids"]:
        receipt = verified_adoption(output, cid, plan)
        if receipt is None:
            raise ValueError(f"Missing Block 62 completion: {cid}")
        path = output / "cases" / cid / next(iter(receipt["artifacts"]))
        report = read_json(path)
        if (any(report[k] != receipt[k] for k in ("case_id", "plan_sha256", "status", "metric_parity", "exact_metric_match", "seconds"))
                or report["original_receipt_sha256"] != plan["source_receipts"][cid]
                or report["evaluator_version"] != plan["implementation"]["evaluator_version"]):
            raise ValueError("Block 62 report and receipt disagree.")
        files.extend((path, path.parent.parent / "complete.json"))
        receipts[cid] = receipt["receipt_sha256"]
        rows.append({"case_id": cid, "metric_parity": True, "exact_metric_match": receipt["exact_metric_match"],
                     "optimized_evaluation_seconds": receipt["seconds"]})
    state["evaluation_seconds"] = describe([r["optimized_evaluation_seconds"] for r in rows])
    state["exact_metric_matches"] = sum(r["exact_metric_match"] for r in rows)
    state["metric_parity_passed"] = len(rows)
    atomic_json(output / "progress.json", state)
    write_csv(output / "case_parity.csv", rows)
    certificate = output / "development_validation.json"
    if (state["status"] == "complete" and not state["failed_cases"] and not state["deferred_case_ids"]
            and len(rows) == len(plan["case_ids"]) and set(receipts) == set(plan["case_ids"])):
        payload = {"status": "development_validated", "plan_sha256": plan["plan_sha256"],
                   "evaluator_version": plan["implementation"]["evaluator_version"],
                   "protocol_sha256": plan["protocol_sha256"], "contract_sha256": plan["contract_sha256"],
                   "n_cases": len(rows), "case_receipts": receipts, "exact_metric_matches": state["exact_metric_matches"],
                   "scope": plan["scope"], "numeric_tolerance": plan["numeric_tolerance"]}
        payload["certificate_sha256"] = canonical_sha256(payload)
        atomic_json(certificate, payload)
        files.append(certificate)
    else:
        certificate.unlink(missing_ok=True)
    lines = ["# Block 62 — Versioned CPU evaluator development validation", "",
             f"Status: **{state['status']}**. Metric parity passed: {len(rows)}/{len(plan['case_ids'])}.", "",
             f"Evaluator: `{plan['implementation']['evaluator_version']}`. Exact metric matches: {state['exact_metric_matches']}.", "",
             "Every field is compared to the saved original evaluation at 1e-9 absolute/relative tolerance. "
             "Only a complete development run receives development_validation.json. This is numerical parity "
             "validation, not an improvement in segmentation accuracy or clinical validation.", "",
             "Each case runs the optimized evaluator once. The original baseline is read from saved JSON; "
             "no original CPU reevaluation or GPU inference runs. Timings are descriptive, not paired speedup estimates. "
             "Validation/test remain reserved. All original sources and outputs are preserved.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    files.extend(output / n for n in ("progress.json", "case_parity.csv", "report.md"))
    for cid in state["failed_cases"]:
        for attempt in sorted((output / "cases" / cid).glob("attempt_*")):
            files.extend(attempt / n for n in ("failure.json", "parity_failure.json") if (attempt / n).is_file())
    archive = output / "block62_cpu_validation_report.zip"
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def run_adoption(root: Path, run_root: Path, output: Path, plan: dict, contract: dict, protocol: dict,
                 *, max_cases: int | None = None, continue_on_error: bool = False) -> tuple[dict, Path]:
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError("max_cases must be positive.")
    # Invalidate any prior exported readiness claim before a new verification.
    (output / "development_validation.json").unlink(missing_ok=True)
    check_adoption(plan, contract, protocol)
    cpu.check_baseline(root, contract)
    completed, new, failed = [], [], {}
    attempted = 0
    def progress():
        return {"plan_sha256": plan["plan_sha256"], "requested_case_ids": plan["case_ids"],
                "completed_case_ids": completed.copy(), "newly_evaluated_case_ids": new.copy(), "failed_cases": failed.copy(),
                "deferred_case_ids": [c for c in plan["case_ids"] if c not in completed and c not in failed],
                "status": "failed" if failed else "complete" if len(completed) == len(plan["case_ids"]) else "partial",
                "gpu_inference": False, "validation_test_outputs_opened": False}
    atomic_json(output / "progress.json", progress())
    for index, cid in enumerate(plan["case_ids"]):
        cached = (output / "cases" / cid / "complete.json").exists()
        if not cached and max_cases is not None and attempted >= max_cases:
            continue
        print(f"[{index + 1:03d}/{len(plan['case_ids']):03d}] validate {cid}", flush=True)
        try:
            if not cached:
                attempted += 1
            receipt, reused = adoption_case(root, run_root, output, cid, plan, contract, protocol)
            completed.append(cid)
            if not reused:
                new.append(cid)
            print(f"  {'verified validation cache' if reused else 'versioned evaluation complete'}; "
                  f"metric parity passed; {receipt['seconds']:.2f} s", flush=True)
        except Exception as exc:
            failed[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[cid]}", flush=True)
            if not continue_on_error:
                break
        atomic_json(output / "progress.json", progress())
    state = progress()
    return state, export_adoption(output, plan, state)
