"""Block 64: frozen development-only feasibility experiment, separate from NV."""
from __future__ import annotations

from hashlib import sha256
import math
from pathlib import Path
import re
from uuid import uuid4
import zipfile

from vista3d_tfg.analysis import full489_cyst_confusion as confusion
from vista3d_tfg.analysis.full489_development import check_contract
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import (
    atomic_json, frozen_json, input_identity, invoke_worker, read_json,
)
from vista3d_tfg.integration.nv_inference import sha256_file
from vista3d_tfg.integration.nv_segment import resolve_conda_executable
from vista3d_tfg.training.contract64 import (
    CONFIG, WORKER, SOURCES, settings, code_identity, check_digest, check_plan, verify_inputs,
)


def select_smoke(reports: list[dict], case_ids: list[str]) -> list[str]:
    if [r["case_id"] for r in reports] != case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Block 63 reports differ from development membership.")
    eligible = []
    for report in reports:
        counts = report["counts"]
        if (len(counts) != 4 or any(len(row) != 5 for row in counts)
                or any(type(n) is not int or n < 0 for row in counts for n in row)
                or sum(map(sum, counts)) != math.prod(report["shape"])):
            raise ValueError("Invalid Block 63 counts.")
        # Only reference class presence is used; prediction outcomes do not rank cases.
        if all(sum(row) > 0 for row in counts):
            eligible.append(report["case_id"])
    if len(eligible) < 3:
        raise ValueError("Need three development cases with all four reference labels.")
    return sorted(eligible, key=lambda c: (sha256(f"block64-smoke-6401:{c}".encode()).hexdigest(), c))[:3]


def prepare(root: Path, protocol: dict, run_id: str, confusion_id: str) -> tuple[Path, dict]:
    if any(not re.fullmatch(r"[0-9a-f]{16}", x) for x in (run_id, confusion_id)):
        raise ValueError("Run and confusion identifiers must be 16 lowercase hex characters.")
    run_root = root / "outputs/full489" / run_id
    contract = read_json(run_root / "contract.json")
    check_contract(contract, protocol)
    if contract["contract_sha256"][:16] != run_id:
        raise ValueError("Run identifier mismatch.")
    folder = root / "outputs/reporting/full489_block63" / run_id / confusion_id
    source_plan = read_json(folder / "plan.json")
    confusion.check_analysis(source_plan, contract, protocol)
    if source_plan["plan_sha256"][:16] != confusion_id:
        raise ValueError("Block 63 identifier mismatch.")
    state = read_json(folder / "progress.json")
    ids = protocol["partition"]["development"]["case_ids"]
    if (state.get("status") != "complete" or state.get("completed_case_ids") != ids
            or state.get("requested_case_ids") != ids or state.get("failed_cases")
            or state.get("deferred_case_ids") or state.get("plan_sha256") != source_plan["plan_sha256"]):
        raise ValueError("Block 63 must be complete for all development cases.")
    reports, receipts = [], {}
    for cid in ids:
        receipt = confusion.verified_case(folder, cid, source_plan)
        if receipt is None:
            raise ValueError(f"Missing Block 63 receipt: {cid}")
        report = read_json(folder / "cases" / cid / next(iter(receipt["artifacts"])))
        if (report["case_id"] != cid or report["plan_sha256"] != source_plan["plan_sha256"]
                or report["count_parity"] is not True
                or report["original_receipt_sha256"] != source_plan["source_receipts"][cid]
                or report["validation_receipt_sha256"] != source_plan["validation_receipts"][cid]):
            raise ValueError("Block 63 report provenance changed.")
        reports.append(report)
        receipts[cid] = receipt["receipt_sha256"]
    smoke_ids = select_smoke(reports, ids)
    inputs = {}
    for cid in smoke_ids:
        _, identity = input_identity(root, cid)
        if identity != read_json(run_root / "cases" / cid / "inputs.json"):
            raise ValueError(f"Original development inputs changed: {cid}")
        inputs[cid] = identity
    plan = {
        "block": 64, "scope": "supervised feasibility smoke; no quality or generalization estimate",
        "protocol_sha256": protocol["protocol_sha256"], "partition": protocol["partition"],
        "source_contract_sha256": contract["contract_sha256"],
        "block63_plan_sha256": source_plan["plan_sha256"], "block63_receipts": receipts,
        "smoke_case_ids": smoke_ids, "inputs": inputs, "settings": settings(),
        "implementation": code_identity(root), "validation_test_reads": False,
    }
    plan["experiment_sha256"] = canonical_sha256(plan)
    check_plan(root, plan, protocol)
    frozen_json(root / CONFIG, plan)
    output = root / "outputs/reporting/full489_block64" / plan["experiment_sha256"][:16]
    frozen_json(output / "protocol.json", plan)
    return output, plan


def call_worker(root: Path, environment: str, job: dict, attempt: Path, timeout: float = 1800) -> dict:
    atomic_json(attempt / "job.json", job)
    command = [resolve_conda_executable(), "run", "--no-capture-output", "-n", environment,
               "python", str(root / WORKER), "--job", str(attempt / "job.json"),
               "--report", str(attempt / "worker.json")]
    code = invoke_worker(command, root, timeout)
    report = read_json(attempt / "worker.json")
    if code or report.get("status") != "complete":
        raise RuntimeError(report.get("error", f"Worker exit code {code}"))
    return report


def validate_result(report: dict, plan: dict) -> None:
    if (report.get("status") != "complete" or report.get("experiment_sha256") != plan["experiment_sha256"]
            or report.get("case_ids") != plan["smoke_case_ids"] or report.get("optimizer_steps") != 24
            or report.get("parameters_changed") is not True or report.get("checkpoint_reload_exact") is not True
            or report.get("device") != "cuda" or report.get("quality_evaluation_performed") is not False
            or report.get("validation_test_reads") is not False or report.get("observed_target_labels") != [0, 1, 2, 3]
            or len(report.get("losses", [])) != 24
            or any(not math.isfinite(v) for v in report["losses"])):
        raise ValueError("Training smoke did not satisfy its acceptance checks.")


def export_report(output: Path, plan: dict, state: dict, attempt: Path) -> Path:
    atomic_json(output / "progress.json", state)
    lines = ["# Block 64 — supervised GPU training feasibility", "",
             f"Status: {state['status']}", "", "Four labels: background, kidney, tumor, cyst.",
             "Randomly initialized 3D U-Net; three development cases, 24 optimizer steps.",
             "Loss is measured on training patches. It is not validation Dice or evidence of generalization.",
             "No validation/test image, label, prediction, metric or QC reads. No NV output is replaced.",
             "Checkpoint stays local and must not initialize the future full research training run.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    archive = output / "block64_supervised_smoke_report.zip"
    temporary = archive.with_suffix(".tmp")
    files = [output / "protocol.json", output / "progress.json", output / "report.md"]
    files += [p for p in (output / "complete.json", attempt / "job.json", attempt / "worker.json",
                           attempt / "failure.json", attempt / "trace.json", attempt / "runtime.json") if p.is_file()]
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def run_smoke(root: Path, protocol: dict, environment: str) -> tuple[dict, Path]:
    plan = read_json(root / CONFIG)
    check_plan(root, plan, protocol)
    verify_inputs(root, plan)
    output = root / "outputs/reporting/full489_block64" / plan["experiment_sha256"][:16]
    frozen_json(output / "protocol.json", plan)
    completion = output / "complete.json"
    if completion.exists():
        receipt = read_json(completion)
        check_digest(receipt, "receipt_sha256")
        if receipt["experiment_sha256"] != plan["experiment_sha256"] or not re.fullmatch(r"attempt_[0-9a-f]{12}", receipt["attempt"]):
            raise ValueError("Completion identity changed.")
        attempt = output / receipt["attempt"]
        if set(receipt["artifacts"]) != {"job.json", "worker.json", "trace.json", "checkpoint.pt", "runtime.json"}:
            raise ValueError("Incomplete training artifact list.")
        for name, digest in receipt["artifacts"].items():
            if sha256_file(attempt / name) != digest:
                raise ValueError(f"Completed training artifact changed: {name}")
        report = read_json(attempt / "worker.json")
        validate_result(report, plan)
        state = {"status": "complete", "cached": True, "experiment_sha256": plan["experiment_sha256"],
                 "attempt": attempt.name, "optimizer_steps": report["optimizer_steps"]}
        return state, export_report(output, plan, state, attempt)
    attempt = output / ("attempt_" + uuid4().hex[:12])
    attempt.mkdir(parents=True)
    state = {"status": "running", "cached": False, "experiment_sha256": plan["experiment_sha256"], "attempt": attempt.name}
    atomic_json(output / "progress.json", state)
    try:
        job = {"project_root": str(root), "output_dir": str(attempt), "plan": plan}
        report = call_worker(root, environment, job, attempt)
        check_plan(root, plan, protocol)
        verify_inputs(root, plan)
        validate_result(report, plan)
        if report["checkpoint_sha256"] != sha256_file(attempt / "checkpoint.pt"):
            raise ValueError("Checkpoint digest mismatch.")
        receipt = {"experiment_sha256": plan["experiment_sha256"], "attempt": attempt.name,
                   "artifacts": {name: sha256_file(attempt / name) for name in
                                 ("job.json", "worker.json", "trace.json", "checkpoint.pt", "runtime.json")}}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(completion, receipt)
        state.update(status="complete", optimizer_steps=report["optimizer_steps"])
    except (Exception, KeyboardInterrupt) as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(attempt / "failure.json", state)
    return state, export_report(output, plan, state, attempt)
