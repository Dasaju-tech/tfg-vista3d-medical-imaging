"""Evaluate the frozen Block 75 variants on the 120 reserved KiTS23 cases."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import run_kits23_supervised75_freeze as freeze75
from run_kits23_supervised71_resume_io import atomic_json_write
from vista3d_tfg.training.contract64 import canonical_sha256, check_digest, input_identity, read_json


ROOT = Path(__file__).resolve().parents[1]
RAW, GUARDED = "ct_raw", "ct_guarded_top2"
VARIANTS = (RAW, GUARDED)
TEST_COUNT = 120


def check_freeze(root):
    freeze, plan = freeze75.load_frozen_protocol(root)
    evaluation = freeze["evaluation_plan"]
    if (evaluation["partition"], evaluation["comparator"], evaluation["candidate"]) != ("Test120", RAW, GUARDED):
        raise ValueError("Unexpected frozen evaluation plan.")
    receipt_path = (
        root / "outputs/reporting/full489_block75_freeze" / freeze["freeze_sha256"][:16]
        / "verification.json"
    )
    verification = read_json(receipt_path)
    if verification != {
        "freeze_sha256": freeze["freeze_sha256"],
        "checkpoint_sha256": freeze["checkpoint"]["sha256"],
        "validation_case_receipts_sha256": freeze["validation_evidence"]["case_receipts_sha256"],
        "validation_cases_verified": 80,
        "test_case_ids_sha256": freeze["partitions"]["test"]["case_ids_sha256"],
        "test_reads": False,
    }:
        raise ValueError("Block 75 verification.json does not match the frozen protocol.")
    freeze75.verify_checkpoint(root, freeze)
    return freeze, plan, verification


def file_stamps(paths):
    stats = [path.stat() for path in paths]
    return [(item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns) for item in stats]


def preflight_inputs(root, case_ids):
    import nibabel as nib
    from run_kits23_supervised72_metrics import native_geometry_mm

    data = root / "data/raw/kits23"
    paths = {cid: (data / cid / "imaging.nii.gz", data / cid / "segmentation.nii.gz") for cid in case_ids}
    missing = [cid for cid, pair in paths.items() if not all(p.is_file() and p.stat().st_size for p in pair)]
    print(f"Test120 input pairs present: {len(paths) - len(missing)}/{TEST_COUNT}.", flush=True)
    if missing:
        raise FileNotFoundError(f"Missing Test120 pairs: {', '.join(missing[:6])} (total {len(missing)}).")

    stamps = {}
    geometry = {}
    for cid, pair in paths.items():
        try:
            stamps[cid] = file_stamps(pair)
            geometry[cid] = native_geometry_mm(*(nib.load(path) for path in pair))
        except (EOFError, OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
            raise ValueError(f"Invalid Test120 headers for {cid}: {exc}") from exc
    print(f"Test120 NIfTI header pairs aligned: {len(geometry)}/{TEST_COUNT}.", flush=True)
    return paths, stamps, geometry


def fingerprint_inputs(root, plan, paths, stamps):
    seen = {entry["image_sha256"]: cid for cid, entry in plan["inputs"].items()}
    inputs = {}
    for number, (cid, pair) in enumerate(paths.items(), 1):
        if number == 1 or number % 20 == 0:
            print(f"Fingerprint Test120 inputs [{number:03d}/{TEST_COUNT}]", flush=True)
        identity = input_identity(root, cid)
        if file_stamps(pair) != stamps[cid]:
            raise ValueError(f"Test120 input changed while checking {cid}.")
        image_hash = identity["image_sha256"]
        if image_hash in seen:
            raise ValueError(f"Byte-identical CT for {cid} and {seen[image_hash]}.")
        seen[image_hash] = cid
        inputs[cid] = identity
    return inputs


def preprocess_ct(image, settings):
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
    from vista3d_tfg.training.cache65 import grid

    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all():
        raise ValueError("Non-finite native CT.")
    canonical = nib.as_closest_canonical(nib.Nifti1Image(data, image.affine))
    # Use each test CT's geometry to size its grid at the frozen spacing.
    target = grid(canonical, {**settings, "max_resampled_voxels": sys.maxsize})
    low, high = settings["ct_window_hu"]
    ct = np.asarray(resample_from_to(canonical, target, order=1, cval=low).dataobj, dtype=np.float32)
    np.clip(ct, low, high, out=ct)
    ct -= low
    ct *= 2. / (high - low)
    ct -= 1.
    if not np.isfinite(ct).all():
        raise ValueError("Non-finite preprocessed CT.")
    return ct, {
        "original_shape": list(image.shape), "original_affine": image.affine.tolist(),
        "shape": list(ct.shape), "affine": target[1].tolist(),
    }


def evaluate_case(paths, block71, model, settings):
    import nibabel as nib
    import numpy as np
    from run_kits23_supervised72_metrics import segmentation_metrics

    image_path, reference_path = paths
    image = nib.load(image_path)
    ct, meta = preprocess_ct(image, settings)
    raw = block71.predict_original(model, ct, meta, settings, "cuda")
    del ct
    guarded, guard = freeze75.apply_guarded_top2(raw)

    # The reference is decoded only once the two predictions are fixed.
    reference = np.asanyarray(nib.load(reference_path).dataobj)
    for start in range(0, reference.shape[2], 16):
        if not np.isin(reference[:, :, start:start + 16], (0, 1, 2, 3)).all():
            raise ValueError("Unexpected KiTS23 reference labels.")
    reference = reference.astype(np.uint8, copy=False)
    affine = image.affine
    before = segmentation_metrics(reference, raw, affine)
    after = before if guarded is raw else segmentation_metrics(reference, guarded, affine)
    raw_hash = block71.array_sha(raw)
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    return {
        "metrics": {RAW: before, GUARDED: after},
        "prediction_sha256": {RAW: raw_hash, GUARDED: raw_hash if guarded is raw else block71.array_sha(guarded)},
        "guard": guard,
        "true_positive_ml_removed": {
            label: (value["intersection_voxels"] - after[label]["intersection_voxels"]) * voxel_ml
            for label, value in before.items()
        },
    }


def pair_summary(values):
    common = [(raw, guarded) for raw, guarded in values if raw is not None and guarded is not None]
    if not common:
        return {"paired_n": 0, RAW: None, GUARDED: None, "guarded_minus_raw": None}
    raw, guarded = zip(*common)
    deltas = [candidate - baseline for baseline, candidate in common]
    return {
        "paired_n": len(common), RAW: {"mean": mean(raw), "median": median(raw)},
        GUARDED: {"mean": mean(guarded), "median": median(guarded)},
        "guarded_minus_raw": {"mean": mean(deltas), "median": median(deltas)},
    }


def summarize(records):
    from run_kits23_supervised72_metrics import summarize_cases

    variants = {
        arm: summarize_cases([{"metrics": record["metrics"][arm]} for record in records.values()])
        for arm in VARIANTS
    }
    envelope = {
        metric: pair_summary([
            tuple(record["metrics"][arm]["renal_envelope"][metric] for arm in VARIANTS)
            for record in records.values()
        ]) for metric in ("dice", "hd95_euclidean_mm", "hd95_manhattan_mm")
    }
    fallback = [cid for cid, result in records.items() if result["guard"]["used_original"]]
    return {
        "partition": "Test120", "test_reads": True, "case_count": len(records),
        "variants": variants, "renal_envelope_paired": envelope,
        "guard_fallback_n": len(fallback), "guard_fallback_case_ids": fallback,
        "total_true_positive_ml_removed": {
            label: sum(record["true_positive_ml_removed"][label] for record in records.values())
            for label in freeze75.LABELS
        },
    }


def write_report(output, freeze, verification, protocol, records):
    from run_kits23_supervised72_metrics import DISTANCES

    if len(records) != TEST_COUNT or list(records) != protocol["case_ids"]:
        raise ValueError("The final report requires all 120 frozen test cases.")
    summary = {"run_sha256": protocol["run_sha256"], **summarize(records)}
    csv_buffer = io.StringIO(newline="")
    fields = ("dice", "reference_voxels", "predicted_voxels", "intersection_voxels",
              "false_positive_ml", "false_negative_ml", "distance_status", *DISTANCES)
    header = ["case_id", "label", "guard_fallback", "true_positive_ml_removed"]
    header += [f"{arm}_{field}" for arm in VARIANTS for field in fields]
    writer = csv.writer(csv_buffer)
    writer.writerow(header)
    for cid, record in records.items():
        for label in freeze75.LABELS:
            writer.writerow([cid, label, record["guard"]["used_original"], record["true_positive_ml_removed"][label]] + [
                record["metrics"][arm][label][field] for arm in VARIANTS for field in fields
            ])

    target = output / "block76_test120_report.zip"
    temporary = output / f"{target.name}.{uuid4().hex}.tmp"
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as report:
            for name, payload in (("freeze.json", freeze), ("verification.json", verification),
                                  ("protocol.json", protocol), ("summary.json", summary)):
                report.writestr(name, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            report.writestr("per_case.csv", csv_buffer.getvalue())
            for cid, record in records.items():
                report.writestr(f"cases/{cid}.json", json.dumps(record, allow_nan=False) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_model(root, freeze, plan, block71):
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    checkpoint = freeze["checkpoint"]
    path = root / checkpoint["path"]
    receipt = read_json(path.parent / "complete.json")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"experiment_sha256": plan["experiment_sha256"], "arm": "ct", "epoch": checkpoint["epoch"],
                "runtime": plan["runtime"], "stats": receipt["stats"]}
    if any(payload[key] != value for key, value in expected.items()):
        raise ValueError("Frozen checkpoint payload mismatch.")
    settings = plan["settings"]
    model = block71.initialized_model(settings["model"], "ct", settings["seed"])
    model.load_state_dict(payload["model"], strict=True)
    if block71.b.state_digest(model) != payload["model_state_sha256"]:
        raise ValueError("Frozen model state digest mismatch.")
    return model.to("cuda").eval()


def evaluate(root, freeze, plan, verification, paths, stamps, geometry):
    import nibabel as nib
    import torch
    import run_kits23_supervised71_compare as block71
    from run_kits23_supervised71_resume import compatible_plan

    if not torch.cuda.is_available():
        raise RuntimeError("Use the original vista3d-nv CUDA environment.")
    runtime = block71.b.runtime()
    observed_plan = {**plan, "runtime": runtime}
    del observed_plan["experiment_sha256"]
    observed_plan["experiment_sha256"] = canonical_sha256(observed_plan)
    compatible_plan(observed_plan, plan)

    with block71.execution_lock(root):
        output = root / "outputs/reporting/full489_block76_test120" / freeze["freeze_sha256"][:16]
        protocol = {
            "block": 76, "partition": "Test120", "freeze_sha256": freeze["freeze_sha256"],
            "checkpoint": freeze["checkpoint"], "case_ids": list(paths),
            "case_ids_sha256": verification["test_case_ids_sha256"], "geometry": geometry,
            "inputs": fingerprint_inputs(root, plan, paths, stamps),
            "settings_sha256": freeze["source_settings_sha256"], "runtime": plan["runtime"],
            "nibabel": nib.__version__, "evaluator_lf_sha256": freeze75.source_sha256(Path(__file__)),
        }
        protocol["run_sha256"] = canonical_sha256(protocol)
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if read_json(protocol_path) != protocol:
                raise ValueError("The Test120 protocol, inputs or evaluator changed since the first run.")
        else:
            atomic_json_write(protocol_path, protocol)
        print(f"Block 76 Test120 folder: {output}", flush=True)
        atomic_json_write(output / "runtime_audit" / f"{uuid4().hex}.json", {
            "run_sha256": protocol["run_sha256"], "observed_runtime": runtime,
        })

        model = None
        records = {}
        for number, (cid, pair) in enumerate(paths.items(), 1):
            path = output / "cases" / f"{cid}.json"
            identity = {"case_id": cid, "run_sha256": protocol["run_sha256"]}
            if path.exists():
                record = read_json(path)
                check_digest(record, "result_sha256")
                if record["identity"] != identity:
                    raise ValueError(f"Test120 receipt identity changed: {cid}.")
                print(f"Test120 [{number:03d}/{TEST_COUNT}] {cid} verified", flush=True)
            else:
                print(f"Test120 [{number:03d}/{TEST_COUNT}] {cid}", flush=True)
                if model is None:
                    model = load_model(root, freeze, plan, block71)
                if file_stamps(pair) != stamps[cid]:
                    raise ValueError(f"Test120 input changed before prediction: {cid}.")
                record = {"identity": identity, **evaluate_case(pair, block71, model, plan["settings"])}
                if file_stamps(pair) != stamps[cid]:
                    raise ValueError(f"Test120 input changed during evaluation: {cid}.")
                record["result_sha256"] = canonical_sha256(record)
                atomic_json_write(path, record)
            records[cid] = record

        if any(file_stamps(pair) != stamps[cid] for cid, pair in paths.items()):
            raise ValueError("A Test120 input changed before report publication.")
        return write_report(output, freeze, verification, protocol, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("preflight", "evaluate"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    freeze, plan, verification = check_freeze(ROOT)
    print("Block 75 verification and frozen checkpoint match.", flush=True)
    paths, stamps, geometry = preflight_inputs(ROOT, plan["partition"]["test"]["case_ids"])
    if args.stage == "preflight":
        print("Block 76 preflight PASSED. Checked file availability and NIfTI headers.", flush=True)
        return
    report = evaluate(ROOT, freeze, plan, verification, paths, stamps, geometry)
    print("Block 76 complete. All 120 paired test cases evaluated.", flush=True)
    print(f"Share this ZIP: {report}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Rerun --stage evaluate to resume verified cases.", file=sys.stderr)
        raise SystemExit(130)
    except (ImportError, KeyError, OSError, ValueError, RuntimeError) as exc:
        print(f"Block 76 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
