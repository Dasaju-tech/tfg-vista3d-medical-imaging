"""Sensibilidad al filtrado de componentes del checkpoint ct en Validation80."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from statistics import mean, median
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.ndimage import generate_binary_structure, label

import run_kits23_supervised72_metrics as block72
import run_kits23_supervised73_hd95_qc as block73
from run_kits23_supervised71_resume_io import atomic_json_write


FILTER = {
    "name": "global_top2",
    "connectivity": 6,
    "envelope_labels": [1, 2, 3],
    "components_kept": 2,
    "ranking": "voxel count descending, component index ascending on ties",
    "replacement_label": 0,
}


def filter_top2(prediction):
    envelope = np.isin(prediction, FILTER["envelope_labels"])
    components, count = label(envelope, structure=generate_binary_structure(3, 1))
    if count <= FILTER["components_kept"]:
        return prediction, {"components": count, "kept": count, "removed_voxels": 0}

    counts = np.zeros(count + 1, dtype=np.int64)
    for start in range(0, prediction.shape[0], 8):
        counts += np.bincount(components[start:start + 8].ravel(), minlength=count + 1)
    largest = np.argsort(-counts[1:], kind="stable")[:FILTER["components_kept"]] + 1
    remove = np.ones(count + 1, dtype=bool)
    remove[0] = False
    remove[largest] = False
    filtered = prediction.copy()
    for start in range(0, prediction.shape[0], 8):
        section = filtered[start:start + 8]
        section[remove[components[start:start + 8]]] = 0
    return filtered, {
        "components": count, "kept": len(largest), "removed_voxels": int(counts[remove].sum()),
    }


def compare_case(replay, case_id, figure_path):
    image, reference, prediction, affine, _ = block73.replay_case(replay, case_id)
    source = replay.receipts[case_id]["metrics"]
    truth = np.asanyarray(reference.dataobj)
    filtered, components = filter_top2(prediction)
    del prediction
    filtered_metrics = replay.runner.voxel_metrics(truth, filtered, affine)

    envelope_labels = FILTER["envelope_labels"]
    reference_envelope = np.isin(truth, envelope_labels)
    filtered_envelope = np.isin(filtered, envelope_labels)
    directions = block73.measure_directions(reference_envelope, filtered_envelope, affine)
    hd95 = max(direction["p95_mm"] for direction in directions.values())
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3]))) / 1000
    rows = []
    for name in block72.LABELS:
        before, after = source[name], filtered_metrics[name]
        envelope = name == "renal_envelope"
        rows.append({
            "case_id": case_id,
            "label": name,
            "reference_positive": before["reference_voxels"] > 0,
            "dice_before": before["dice"],
            "dice_after": after["dice"],
            "dice_change": after["dice"] - before["dice"],
            "hd95_mm_before": before["hd95_euclidean_mm"] if envelope else None,
            "hd95_mm_after": hd95 if envelope else None,
            "false_positive_ml_before": before["false_positive_ml"],
            "false_positive_ml_after": after["false_positive_ml"],
            "false_negative_ml_before": before["false_negative_ml"],
            "false_negative_ml_after": after["false_negative_ml"],
            "true_positive_ml_removed": voxel_ml * (before["intersection_voxels"] - after["intersection_voxels"]),
            "prediction_voxels_before": before["predicted_voxels"],
            "prediction_voxels_after": after["predicted_voxels"],
        })
    if case_id in block73.CASE_PURPOSES:
        block73.render_overlay(
            image, reference_envelope, filtered_envelope, directions,
            {"case_id": case_id, "epoch": replay.checkpoint["epoch"], "hd95_mm": hd95,
             "dice": filtered_metrics["renal_envelope"]["dice"]},
            figure_path,
        )
    replay.runner.c.verify_input(replay.root, replay.parent, "validation", case_id)
    return {"rows": rows, "components": components, "filtered_directions": directions}


def case_result(replay, protocol, case_id, directory):
    evidence = replay.runner.b
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "result.json"
    figure = directory / "filtered.png"
    identity = {
        "case_id": case_id, "filter_sha256": protocol["filter_sha256"],
        "source_case_sha256": replay.receipts[case_id]["report_sha256"],
    }
    if cache.exists():
        result = evidence.read(cache)
        evidence.check(result, "result_sha256")
        if result["identity"] != identity or (
            case_id in block73.CASE_PURPOSES and evidence.sha(figure) != result["figure_sha256"]
        ):
            raise ValueError(f"El resultado guardado de {case_id} difiere del protocolo.")
        replay.runner.c.verify_input(replay.root, replay.parent, "validation", case_id)
        print("  Resultado verificado y reutilizado.", flush=True)
        return result

    with TemporaryDirectory(dir=directory) as work:
        pending_figure = Path(work) / figure.name
        result = compare_case(replay, case_id, pending_figure)
        if pending_figure.exists():
            os.replace(pending_figure, figure)
            result["figure_sha256"] = evidence.sha(figure)
    result["identity"] = identity
    result["result_sha256"] = evidence.digest(result)
    atomic_json_write(cache, result)
    return result


def summarize(rows, results, scope):
    envelope = [r for r in rows if r["label"] == "renal_envelope"]
    old_hd = [r["hd95_mm_before"] for r in envelope]
    new_hd = [r["hd95_mm_after"] for r in envelope]
    classes = {}
    for name in block72.LABELS:
        entries = [r for r in rows if r["label"] == name]
        positives = [r for r in entries if r["reference_positive"]]
        classes[name] = {
            "positive_cases": len(positives),
            "mean_positive_dice_before": mean(r["dice_before"] for r in positives) if positives else None,
            "mean_positive_dice_after": mean(r["dice_after"] for r in positives) if positives else None,
            "positive_cases_with_dice_loss": sum(r["dice_change"] < -1e-9 for r in positives),
            "positive_cases_with_true_voxels_removed": sum(r["true_positive_ml_removed"] > 0 for r in positives),
            "true_positive_ml_removed": sum(r["true_positive_ml_removed"] for r in positives),
            "positive_cases_emptied": sum(r["prediction_voxels_before"] > 0 and r["prediction_voxels_after"] == 0
                                         for r in positives),
        }
    return {
        "scope": scope, "case_count": len(results), "test_reads": False,
        "mean_envelope_hd95_mm_before": mean(old_hd), "mean_envelope_hd95_mm_after": mean(new_hd),
        "median_envelope_hd95_mm_before": median(old_hd), "median_envelope_hd95_mm_after": median(new_hd),
        "envelope_hd95_improved": sum(a - b > 1e-6 for a, b in zip(old_hd, new_hd)),
        "envelope_hd95_worsened": sum(b - a > 1e-6 for a, b in zip(old_hd, new_hd)),
        "cases_with_removed_components": sum(r["components"]["removed_voxels"] > 0 for r in results),
        "classes": classes,
    }


def report_zip(output, scope, protocol, results):
    rows = [row for result in results for row in result["rows"]]
    summary = summarize(rows, results, scope)
    scope_folder = output / scope
    scope_folder.mkdir(parents=True, exist_ok=True)
    destination = scope_folder / "block74_component_filter_report.zip"
    with TemporaryDirectory(dir=scope_folder) as work:
        folder = Path(work)
        csv_path = folder / "per_case.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        atomic_json_write(folder / "summary.json", summary)
        pending = folder / destination.name
        with ZipFile(pending, "w", ZIP_DEFLATED) as archive:
            archive.write(output / "protocol.json", "protocol.json")
            archive.write(csv_path, "per_case.csv")
            archive.write(folder / "summary.json", "summary.json")
            for result in results:
                case_id = result["identity"]["case_id"]
                case_folder = output / "cases" / case_id
                archive.write(case_folder / "result.json", f"cases/{case_id}.json")
                if case_id in block73.CASE_PURPOSES:
                    archive.write(case_folder / "filtered.png", f"figures/{case_id}_filtered.png")
        os.replace(pending, destination)
    print(f"HD95 medio de la unión renal · {summary['mean_envelope_hd95_mm_before']:.1f} mm → "
          f"{summary['mean_envelope_hd95_mm_after']:.1f} mm", flush=True)
    print(f"Share this ZIP: {destination}", flush=True)


def run(root, report, scope):
    with block73.open_replay(root, report, validation80=scope == "validation80") as replay:
        evidence = replay.runner.b
        protocol = {
            "source_analysis_sha256": replay.protocol["analysis_sha256"],
            "checkpoint_sha256": replay.checkpoint["sha256"], "arm": block73.ARM,
            "epoch": replay.checkpoint["epoch"], "partition": "Validation80",
            "test_reads": False, "filter": FILTER,
            "implementation": {Path(path).name: evidence.sha(Path(path)) for path in
                               (__file__, block73.__file__, block72.__file__)},
        }
        protocol["filter_sha256"] = evidence.digest(protocol)
        output = root / "outputs/reporting/full489_block74_component_filter" / protocol["filter_sha256"][:16]
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            if evidence.read(protocol_path) != protocol:
                raise ValueError("El protocolo guardado del Bloque 74 difiere del actual.")
        else:
            atomic_json_write(protocol_path, protocol)
        results = []
        for number, case_id in enumerate(replay.receipts, 1):
            print(f"[{number}/{len(replay.receipts)}] {case_id}", flush=True)
            result = case_result(replay, protocol, case_id, output / "cases" / case_id)
            envelope = next(r for r in result["rows"] if r["label"] == "renal_envelope")
            print(f"  HD95 {envelope['hd95_mm_before']:.1f} mm → {envelope['hd95_mm_after']:.1f} mm · "
                  f"Dice {envelope['dice_before']:.3f} → {envelope['dice_after']:.3f}", flush=True)
            results.append(result)
        report_zip(output, scope, protocol, results)


def main():
    root = Path(__file__).resolve().parents[1]
    default_report = (root / "outputs/reporting/full489_block72_metrics" / block73.ANALYSIS_SHA256[:16]
                      / "block72_validation_metrics_report.zip")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "validation80"), default="pilot")
    parser.add_argument("--report", type=Path, default=default_report, help="ZIP de métricas del Bloque 72B.")
    args = parser.parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(f"No encuentro {args.report}. Indica la ruta al ZIP del Bloque 72B con --report.")
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(root, args.report, args.scope)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Block 74 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
