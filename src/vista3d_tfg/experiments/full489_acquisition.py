"""Block 58a: acquisition repair independent of the frozen inference code.

The CLI uses this implementation. The original acquisition helper in
full489_execution remains untouched because that entire file is part of the
existing inference fingerprint; this repair must not invalidate GPU receipts.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from vista3d_tfg.data.acquisition import download_kits23_case
from vista3d_tfg.data.audit import audit_case
from vista3d_tfg.experiments.full489_execution import (
    atomic_json, input_identity, read_json, selected_cases,
)


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
            duplicate = next((
                other for other, entry in ledger["cases"].items()
                if other != cid and entry.get("identity", {}).get("image_sha256")
                == identity["image_sha256"]
            ), None)
            if duplicate:
                raise ValueError(f"Byte-identical CT for {cid} and {duplicate}; review patient identity.")
            entry = {
                "status": "complete", "identity": identity,
                "downloads": [{**asdict(item), "path": str(item.path)} for item in downloads],
                "input_audit_valid": True,
            }
            # Publish first. A failed write must affect neither in-memory reuse
            # nor the completed count, even when continuing to another case.
            candidate = {**ledger, "cases": {**ledger["cases"], cid: entry}}
            atomic_json(ledger_path, candidate)
            ledger = candidate
            completed.append(cid)
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
