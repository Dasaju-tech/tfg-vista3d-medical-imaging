"""Verify the frozen method and its Validation80 evidence before final evaluation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from statistics import mean, median
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vista3d_tfg.training.contract64 import (  # noqa: E402
    canonical_sha256, check_digest, frozen_json, read_json, sha256_file,
)


CONFIG = Path("configs/experiments/kits23_supervised_block75_freeze.json")
BLOCK57 = Path("configs/experiments/nv_segment_ct_full489_protocol.json")
BLOCK71 = Path("configs/experiments/kits23_supervised_block71_compare.json")
GUARD = {
    "kidney_label": 1,
    "max_removed_fraction": [1, 3],
    "fallback_operator": ">",
    "empty_predicted_kidney": "use_original",
}
LABELS = ("renal_envelope", "kidney", "tumor", "cyst")


def keep_original(kidney_voxels_before: int, kidney_voxels_after: int) -> bool:
    if not 0 <= kidney_voxels_after <= kidney_voxels_before:
        raise ValueError("Invalid predicted kidney counts for a removal-only filter.")
    numerator, denominator = GUARD["max_removed_fraction"]
    removed = kidney_voxels_before - kidney_voxels_after
    return kidney_voxels_before == 0 or denominator * removed > numerator * kidney_voxels_before


def apply_guarded_top2(prediction):
    import numpy as np
    from run_kits23_supervised74_component_filter import filter_top2

    label = GUARD["kidney_label"]
    before = int(np.count_nonzero(prediction == label))
    if before:
        filtered, _ = filter_top2(prediction)
        after = int(np.count_nonzero(filtered == label))
    else:
        filtered, after = prediction, 0
    fallback = keep_original(before, after)
    return (prediction if fallback else filtered), {
        "used_original": fallback,
        "predicted_kidney_voxels_before": before,
        "predicted_kidney_voxels_after_top2": after,
    }


def source_sha256(path: Path) -> str:
    """Hash source bytes with a common LF representation for Git checkouts."""
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_frozen_protocol(root: Path) -> tuple[dict, dict]:
    protocol = read_json(root / CONFIG)
    split = read_json(root / BLOCK57)
    model = read_json(root / BLOCK71)
    check_digest(protocol, "freeze_sha256")
    check_digest(split, "protocol_sha256")
    check_digest(model, "experiment_sha256")
    if (
        protocol["block"] != 75
        or protocol["test_reads"] is not False
        or protocol["source_block57_sha256"] != split["protocol_sha256"]
        or protocol["source_block71_sha256"] != model["experiment_sha256"]
        or protocol["source_settings_sha256"] != canonical_sha256(model["settings"])
        or split["partition"] != model["partition"]
        or protocol["postprocessing"]["guard"] != GUARD
    ):
        raise ValueError("The freeze disagrees with the source protocol, settings or guard.")
    all_ids = []
    for name, count in (("development", 289), ("validation", 80), ("test", 120)):
        case_ids = split["partition"][name]["case_ids"]
        expected = {"case_count": count, "case_ids_sha256": canonical_sha256(case_ids)}
        if len(case_ids) != count or protocol["partitions"][name] != expected:
            raise ValueError(f"Changed frozen partition: {name}.")
        all_ids.extend(case_ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("The frozen partitions overlap.")
    for filename, expected in protocol["implementation_lf_sha256"].items():
        if source_sha256(root / filename) != expected:
            raise ValueError(f"Changed implementation: {filename}.")
    return protocol, model


def summarize_validation(cases: dict[str, dict]) -> dict:
    fallback = []
    for case_id, by_label in cases.items():
        kidney = by_label["kidney"]
        if keep_original(kidney["prediction_voxels_before"], kidney["prediction_voxels_after"]):
            fallback.append(case_id)
    variants = {}
    for name, raw in (("ct_raw", set(cases)), ("ct_top2", set()), ("ct_guarded_top2", set(fallback))):
        hd95 = []
        dice = {label: [] for label in LABELS}
        for case_id, by_label in cases.items():
            suffix = "before" if case_id in raw else "after"
            hd95.append(by_label["renal_envelope"][f"hd95_mm_{suffix}"])
            for label in LABELS:
                row = by_label[label]
                if row["reference_positive"]:
                    dice[label].append(row[f"dice_{suffix}"])
        variants[name] = {
            "mean_dice_reference_positive": {label: round(mean(values), 9) for label, values in dice.items()},
            "renal_envelope_mean_hd95_mm": round(mean(hd95), 9),
            "renal_envelope_median_hd95_mm": round(median(hd95), 9),
            "renal_envelope_true_positive_ml_removed": round(fsum(
                row["renal_envelope"]["true_positive_ml_removed"]
                for case_id, row in cases.items() if case_id not in raw
            ), 9),
        }
    return {"case_count": len(cases), "guard_fallback_cases": fallback, "variants": variants}


def read_validation_evidence(report_path: Path, case_ids: list[str], protocol: dict) -> dict:
    evidence = protocol["validation_evidence"]
    checkpoint = protocol["checkpoint"]
    cases, receipts = {}, {}
    with ZipFile(report_path) as archive:
        source = json.loads(archive.read("protocol.json"))
        summary = json.loads(archive.read("summary.json"))
        check_digest(source, "filter_sha256")
        if (
            source["filter_sha256"] != evidence["block74_filter_sha256"]
            or source["source_analysis_sha256"] != evidence["block72_analysis_sha256"]
            or source["checkpoint_sha256"] != checkpoint["sha256"]
            or source["arm"] != checkpoint["arm"]
            or source["epoch"] != checkpoint["epoch"]
            or source["filter"] != protocol["postprocessing"]["base_filter"]
            or summary["scope"] != "validation80"
            or summary["case_count"] != len(case_ids)
            or source["test_reads"] is not False
            or summary["test_reads"] is not False
        ):
            raise ValueError("Expected the frozen Block 74 report for all 80 validation cases.")
        members = [name for name in archive.namelist() if name.startswith("cases/")]
        if sorted(members) != sorted(f"cases/{case_id}.json" for case_id in case_ids):
            raise ValueError("The report has missing, duplicate or unexpected case records.")
        for case_id in case_ids:
            record = json.loads(archive.read(f"cases/{case_id}.json"))
            check_digest(record, "result_sha256")
            if record["identity"]["case_id"] != case_id or record["identity"]["filter_sha256"] != source["filter_sha256"]:
                raise ValueError(f"Mismatched validation receipt: {case_id}.")
            cases[case_id] = {row["label"]: row for row in record["rows"]}
            receipts[case_id] = record["result_sha256"]
    return {"case_receipts_sha256": canonical_sha256(receipts), "results": summarize_validation(cases)}


def verify_checkpoint(root: Path, protocol: dict) -> None:
    checkpoint = protocol["checkpoint"]
    path = root / checkpoint["path"]
    receipt = read_json(path.parent / "complete.json")
    check_digest(receipt, "receipt_sha256")
    if (
        receipt["receipt_sha256"] != checkpoint["receipt_sha256"]
        or receipt["experiment_sha256"] != protocol["source_block71_sha256"]
        or receipt["arm"] != checkpoint["arm"]
        or receipt["epoch"] != checkpoint["epoch"]
        or receipt["file"] != path.name
        or receipt["sha256"] != checkpoint["sha256"]
        or sha256_file(path) != checkpoint["sha256"]
    ):
        raise ValueError("Changed frozen Block 71 ct checkpoint or receipt.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Path to the Validation80 report from Block 74.")
    parser.add_argument("--evidence-only", action="store_true", help="Verify the protocol and validation ZIP.")
    args = parser.parse_args()
    protocol, model = load_frozen_protocol(ROOT)
    evidence = protocol["validation_evidence"]
    report_path = args.report or (
        ROOT / "outputs/reporting/full489_block74_component_filter"
        / evidence["block74_filter_sha256"][:16] / "validation80/block74_component_filter_report.zip"
    )
    observed = read_validation_evidence(report_path, model["partition"]["validation"]["case_ids"], protocol)
    if any(observed[key] != evidence[key] for key in observed):
        raise ValueError("The replayed validation evidence disagrees with the frozen record.")
    validation = observed["results"]
    print(f"Validation80 verified ({validation['case_count']}/80). "
          f"Guard fallback in {len(validation['guard_fallback_cases'])} cases.")
    print(f"Frozen protocol SHA-256: {protocol['freeze_sha256']}")
    if args.evidence_only:
        return
    verify_checkpoint(ROOT, protocol)
    path = (ROOT / "outputs/reporting/full489_block75_freeze"
            / protocol["freeze_sha256"][:16] / "verification.json")
    frozen_json(path, {
        "freeze_sha256": protocol["freeze_sha256"],
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "validation_case_receipts_sha256": observed["case_receipts_sha256"],
        "validation_cases_verified": 80,
        "test_case_ids_sha256": protocol["partitions"]["test"]["case_ids_sha256"],
        "test_reads": False,
    })
    print("Block 75 freeze verified. Local ct checkpoint epoch 19 matches.")
    print(f"Verification: {path}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f"Block 75 FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
