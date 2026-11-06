"""Block 57: metadata-only cohort and split freeze, with no model execution."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from vista3d_tfg.experiments.expanded100 import (
    EXPANDED100_EXPERIMENT,
    OFFICIAL_KITS23_METADATA_URL,
    parse_official_case_numbers,
    validate_official_universe,
)
from vista3d_tfg.experiments.pilot30 import PILOT30_CASES, pilot_case_paths


EXPERIMENT = "nv_segment_ct_full489"
SPLIT_SEED = 20260905
VALIDATION_COUNT = 80
TEST_COUNT = 120
_CASE_ID = re.compile(r"case_[0-9]{5}")
_ARTIFACT_CASE = re.compile(r"(?<![A-Za-z0-9])case_[0-9]{5}(?![0-9])")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_sha256(payload: object) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _json(raw: bytes, name: str) -> object:
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON.") from exc


def _case_ids(value: object, count: int, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(x, str) or not _CASE_ID.fullmatch(x) for x in value)
        or len(set(value)) != count
    ):
        raise ValueError(f"{name} must contain {count} unique canonical case IDs.")
    return value


def _group(ids: list[str]) -> dict:
    return {"case_count": len(ids), "case_ids": ids}


def _assemble(
    universe: list[str], historical: list[str],
    metadata_sha256: str, expanded100_sha256: str,
) -> dict:
    _case_ids(universe, 489, "Official universe")
    _case_ids(historical, 100, "Historical cohort")
    if historical[:30] != [f"case_{n:05d}" for n in PILOT30_CASES]:
        raise ValueError("Historical cohort must retain pilot30 in its original order.")
    if not set(historical).issubset(universe):
        raise ValueError("Historical cohort contains IDs absent from official metadata.")
    for digest in (metadata_sha256, expanded100_sha256):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("Source SHA-256 must be 64 lowercase hexadecimal digits.")

    additional = sorted(set(universe) - set(historical))
    # Hash ranking is stable across Python versions and does not use global RNG state.
    ranked = sorted(
        additional,
        key=lambda cid: (
            sha256(f"{EXPERIMENT}|v1|{SPLIT_SEED}|{cid}".encode("ascii")).digest(),
            cid,
        ),
    )
    test = sorted(ranked[:TEST_COUNT])
    validation = sorted(ranked[TEST_COUNT:TEST_COUNT + VALIDATION_COUNT])
    development = historical + sorted(ranked[TEST_COUNT + VALIDATION_COUNT:])
    payload = {
        "schema_version": "1.0",
        "block": 57,
        "experiment": EXPERIMENT,
        "dataset": "KiTS23 public training collection",
        "scope": "cohort_and_split_freeze_only",
        "sources": {
            "official_metadata_url": OFFICIAL_KITS23_METADATA_URL,
            "official_metadata_sha256": metadata_sha256,
            "expanded100_manifest_canonical_sha256": expanded100_sha256,
        },
        "cohorts": {
            "historical100": _group(historical),
            "additional389": _group(additional),
            "all489": _group(historical + additional),
        },
        "partition": {
            "unit": "KiTS23 case ID",
            "method": "SHA-256 ranking of additional IDs; test, then validation",
            "seed": SPLIT_SEED,
            "development": _group(development),
            "validation": _group(validation),
            "test": _group(test),
        },
        "protocol": {
            "uses_model_results_for_assignment": False,
            "historical_cases_in_development_only": True,
            "test_metrics_and_visual_qc_deferred_until_methods_are_frozen": True,
            "failed_cases_must_be_reported_not_replaced": True,
            "cache_reuse_requires_input_model_runtime_and_metric_verification": True,
            "existing_expanded100_runner_accepts_this_manifest": False,
            "pretraining_overlap_status": "not_established",
            "volume_and_patient_duplicate_audit_status": "pending",
            "external_validation_claim_supported": False,
            "official_kits23_test_score": False,
        },
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    return payload


def build_protocol(*, metadata_raw: bytes, expanded100_raw: bytes) -> dict:
    """Validate original metadata bytes and retain the supplied historical IDs."""
    baseline = _json(expanded100_raw, "Expanded100 manifest")
    if not isinstance(baseline, dict) or baseline.get("experiment") != EXPANDED100_EXPERIMENT:
        raise ValueError("Expected the frozen expanded100 manifest.")
    if baseline.get("dataset") != "KiTS23":
        raise ValueError("Expanded100 dataset must be KiTS23.")
    source = baseline.get("official_universe")
    if not isinstance(source, dict):
        raise ValueError("Expanded100 has no official_universe object.")
    if (
        source.get("case_count") != 489
        or source.get("source_url") != OFFICIAL_KITS23_METADATA_URL
    ):
        raise ValueError("Expanded100 official metadata source/count changed.")
    digest = sha256(metadata_raw).hexdigest()
    if digest != source.get("metadata_sha256"):
        raise ValueError(
            "Official metadata SHA-256 differs from the expanded100 freeze. "
            "Use the original metadata bytes; do not regenerate the old selection."
        )
    numbers = validate_official_universe(
        parse_official_case_numbers(_json(metadata_raw, "Official metadata"))
    )
    selection = baseline.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Expanded100 has no selection object.")
    values = selection.get("case_numbers")
    if (
        not isinstance(values, list)
        or any(type(x) is not int or not 0 <= x <= 99999 for x in values)
    ):
        raise ValueError("Expanded100 case_numbers must be integer identifiers.")
    historical = _case_ids(selection.get("case_ids"), 100, "Expanded100 case_ids")
    if historical != [f"case_{n:05d}" for n in values]:
        raise ValueError("Expanded100 case_ids and case_numbers disagree.")
    if (
        selection.get("requested_case_count") != 100
        or selection.get("retained_pilot30_count") != 30
        or selection.get("additional_case_count") != 70
        or selection.get("additional_case_numbers") != values[30:]
    ):
        raise ValueError("Expanded100 retained/additional counts or identifiers disagree.")
    return _assemble(
        [f"case_{n:05d}" for n in numbers], historical,
        digest, canonical_sha256(baseline),
    )


def load_protocol(path: Path) -> dict:
    """Validate the certificate AND the complete deterministic allocation."""
    payload = _json(path.read_bytes(), "Full489 protocol")
    if not isinstance(payload, dict):
        raise ValueError("Full489 protocol must be an object.")
    unsigned = {key: value for key, value in payload.items() if key != "protocol_sha256"}
    if payload.get("protocol_sha256") != canonical_sha256(unsigned):
        raise ValueError("Full489 protocol certificate mismatch.")
    try:
        expected = _assemble(
            payload["cohorts"]["all489"]["case_ids"],
            payload["cohorts"]["historical100"]["case_ids"],
            payload["sources"]["official_metadata_sha256"],
            payload["sources"]["expanded100_manifest_canonical_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Full489 protocol structure is incomplete.") from exc
    if payload != expected:
        raise ValueError("Full489 allocation or protocol contract changed.")
    return payload


def assert_existing_protocol_matches(path: Path, payload: dict) -> bool:
    if not path.exists():
        return False
    if load_protocol(path) != payload:
        raise ValueError(
            "An incompatible full489 freeze already exists. It was not overwritten."
        )
    return True


def write_protocol(path: Path, payload: dict) -> bool:
    """Create once; a compatible rerun preserves bytes and modification time."""
    if assert_existing_protocol_matches(path, payload):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return True


def fetch_metadata(*, opener=None) -> bytes:
    """Only this explicit function accesses the network; no CT or model download."""
    request = Request(
        OFFICIAL_KITS23_METADATA_URL,
        headers={"User-Agent": "vista3d-tfg-block57/1.0"},
    )
    with (opener or urlopen)(request, timeout=30) as response:
        raw = response.read(10 * 1024 * 1024 + 1)
    if not isinstance(raw, bytes) or not raw or len(raw) > 10 * 1024 * 1024:
        raise ValueError("Official metadata download is empty or exceeds 10 MiB.")
    return raw


def audit_inventory(project_root: Path, payload: dict) -> dict:
    """Inspect file names and sizes only; never open volumes or result contents."""
    all_ids = payload["cohorts"]["all489"]["case_ids"]
    roles = {
        cid: role for role in ("development", "validation", "test")
        for cid in payload["partition"][role]["case_ids"]
    }
    evidence: dict[str, list[str]] = {}
    outputs = project_root / "outputs"
    if outputs.is_dir():
        for path in sorted(outputs.rglob("*")):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            relative = path.relative_to(project_root).as_posix()
            for cid in set(_ARTIFACT_CASE.findall(relative)).intersection(roles):
                evidence.setdefault(cid, [])
                if len(evidence[cid]) < 3:
                    evidence[cid].append(relative)

    totals = {name: 0 for name in (
        "imaging", "segmentation", "prediction", "inference_json", "evaluation_json",
    )}
    rows = []
    for cid in all_ids:
        paths = pilot_case_paths(project_root, int(cid[5:]))
        files = {
            "imaging": paths.imaging_path,
            "segmentation": paths.segmentation_path,
            "prediction": paths.prediction_path,
            "inference_json": paths.inference_json,
            "evaluation_json": paths.evaluation_json,
        }
        states = {}
        for name, path in files.items():
            size = path.stat().st_size if path.is_file() else None
            states[name] = "missing" if size is None else "empty" if size == 0 else "present"
            totals[name] += states[name] == "present"
        rows.append({
            "case_id": cid, "partition": roles[cid], "files": states,
            "existing_output_examples": evidence.get(cid, []),
        })
    return {
        "experiment": EXPERIMENT,
        "protocol_sha256": payload["protocol_sha256"],
        "inspection": "names and sizes only; presence does not validate cache or NIfTI",
        "exposure_scan_scope": (
            "nonempty files under outputs with a case ID in their path; "
            "cannot detect manual viewing, other roots, or aggregate-only histories"
        ),
        "nonempty_file_counts": totals,
        "reserved_cases_with_existing_outputs": sorted(
            cid for cid in evidence if roles[cid] in ("validation", "test")
        ),
        "cases": rows,
    }


def enforce_initial_reservation(audit: dict, *, already_frozen: bool) -> None:
    exposed = audit["reserved_cases_with_existing_outputs"]
    if exposed and not already_frozen:
        raise ValueError(
            "Reserved validation/test cases already have output artifacts: "
            + ", ".join(exposed)
            + ". No protocol was written. Review their exposure history; "
            "do not delete outputs to bypass this check."
        )
