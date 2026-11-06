"""Block 63: annotation-relative cyst confusion, without inference or distances."""
from __future__ import annotations

from importlib.metadata import version
import math
from pathlib import Path
import time
from uuid import uuid4
import zipfile

import numpy as np

from vista3d_tfg.analysis import full489_cpu as cpu, full489_cpu_adoption as adoption
from vista3d_tfg.analysis.full489_development import describe
from vista3d_tfg.analysis.full489_review import check_plan, differences, write_csv
from vista3d_tfg.evaluation.nv_kits23 import _load_pair
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import atomic_json, frozen_json, read_json
from vista3d_tfg.integration.nv_inference import sha256_file


GT_NAMES = ("background", "kidney", "tumor", "cyst")
PRED_VALUES = (0, 5, 14, 116, 117)
PRED_NAMES = ("background", "right_kidney", "left_kidney", "left_cyst", "right_cyst")


def confusion_counts(gt: np.ndarray, pred: np.ndarray, *, slab_depth: int = 16) -> np.ndarray:
    if gt.ndim != 3 or gt.shape != pred.shape or not np.issubdtype(gt.dtype, np.integer) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("Confusion requires matching 3D integer label arrays.")
    if type(slab_depth) is not int or slab_depth < 1 or gt.size == 0:
        raise ValueError("Nonempty arrays and positive slab depth are required.")
    counts = np.zeros((4, 5), dtype=np.int64)
    values = np.asarray(PRED_VALUES)
    for start in range(0, gt.shape[2], slab_depth):
        reference = gt[:, :, start:start + slab_depth]
        prediction = pred[:, :, start:start + slab_depth]
        if np.any(reference < 0) or np.any(reference > 3):
            raise ValueError("Unexpected KiTS23 label.")
        mapped = np.searchsorted(values, prediction)
        if np.any(mapped >= len(values)) or not np.array_equal(values[mapped], prediction):
            raise ValueError("Unexpected NV-Segment label.")
        encoded = reference.astype(np.int64) * 5 + mapped
        counts += np.bincount(encoded.ravel(), minlength=20).reshape(4, 5)
    if int(counts.sum()) != gt.size:
        raise ValueError("Confusion matrix does not conserve grid voxels.")
    return counts


def binary_fields(reference: int, predicted: int, intersection: int, voxel_ml: float) -> dict:
    reference_ml, predicted_ml = reference * voxel_ml, predicted * voxel_ml
    error = predicted_ml - reference_ml
    return {"reference_voxels": reference, "prediction_voxels": predicted, "intersection_voxels": intersection,
            "dice": 2. * intersection / (reference + predicted) if reference + predicted else 1.,
            "reference_volume_ml": reference_ml, "prediction_volume_ml": predicted_ml,
            "signed_volume_error_ml": error, "absolute_volume_error_ml": abs(error),
            "relative_volume_error_percent": 100. * error / reference_ml if reference_ml > 0 else None,
            "false_positive_volume_ml": (predicted - intersection) * voxel_ml,
            "false_negative_volume_ml": (reference - intersection) * voxel_ml}


def derived_metrics(counts: np.ndarray, spacing) -> dict:
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("Voxel spacing must have three finite positive values.")
    voxel_ml = float(np.prod(spacing) / 1000.)
    def basic(rows, cols):
        return binary_fields(int(counts[rows, :].sum()), int(counts[:, cols].sum()),
                             int(counts[np.ix_(rows, cols)].sum()), voxel_ml)
    return {"cyst": basic([3], [3, 4]), "kidney_core_diagnostic": basic([1], [1, 2]),
            "kidney_mass_dice": basic([1, 2, 3], [1, 2, 3, 4])["dice"],
            "gt_tumor_volume_ml": float(counts[2].sum() * voxel_ml)}


def check_count_parity(counts: np.ndarray, shape, spacing, baseline: dict) -> None:
    derived = derived_metrics(counts, spacing)
    expected = {"cyst": baseline["cyst"], "kidney_core_diagnostic": baseline["kidney_core_diagnostic"],
                "kidney_mass_dice": baseline["kidney_and_mass_hec"]["dice"],
                "gt_tumor_volume_ml": baseline["gt_tumor_volume_ml"]}
    delta = differences(expected, derived)
    delta += differences(baseline["shape"], list(shape), "shape")
    delta += differences(baseline["spacing_mm"], list(spacing), "spacing_mm")
    present = [PRED_VALUES[i] for i in range(5) if counts[:, i].sum() > 0]
    delta += differences(baseline["prediction_labels_present"], present, "prediction_labels_present")
    if delta:
        raise ValueError("Confusion/count parity failed: " + "; ".join(delta))


def breakdown(counts_ml: np.ndarray) -> dict:
    """Direction-specific volumes; undefined proportions are null, never zero."""
    fp = {GT_NAMES[i]: float(counts_ml[i, 3:].sum()) for i in range(3)}
    fn = {PRED_NAMES[i]: float(counts_ml[3, i]) for i in range(3)}
    total_predicted = float(counts_ml[:, 3:].sum())
    total_reference = float(counts_ml[3, :].sum())
    tp = float(counts_ml[3, 3:].sum())
    def fractions(values, denominator):
        return {k: v / denominator if denominator > 0 else None for k, v in values.items()}
    return {"reference_cyst_ml": total_reference, "predicted_cyst_ml": total_predicted, "true_positive_cyst_ml": tp,
            "cyst_fp_by_reference_ml": fp, "cyst_fn_by_prediction_ml": fn,
            "cyst_fp_total_ml": sum(fp.values()), "cyst_fn_total_ml": sum(fn.values()),
            "fp_composition": fractions(fp, sum(fp.values())), "fn_composition": fractions(fn, sum(fn.values())),
            "predicted_cyst_composition": fractions({**fp, "cyst": tp}, total_predicted),
            "reference_cyst_destinations": fractions({**fn, "cyst": tp}, total_reference)}


def completed_validation(root: Path, run_root: Path, validation_root: Path, rows: list[dict], contract: dict, protocol: dict) -> dict:
    plan = read_json(validation_root / "plan.json")
    adoption.check_adoption(plan, contract, protocol)
    cpu.check_baseline(root, contract)
    if plan["source_receipts"] != {r["case_id"]: r["receipt_sha256"] for r in rows} or plan["source_rows_sha256"] != canonical_sha256(rows):
        raise ValueError("Block 62 sources differ from verified development results.")
    cert = read_json(validation_root / "development_validation.json")
    if (cert.get("certificate_sha256") != canonical_sha256({k: v for k, v in cert.items() if k != "certificate_sha256"})
            or cert.get("status") != "development_validated" or cert.get("plan_sha256") != plan["plan_sha256"]
            or cert.get("protocol_sha256") != plan["protocol_sha256"] or cert.get("contract_sha256") != contract["contract_sha256"]
            or cert.get("n_cases") != len(rows) or set(cert.get("case_receipts", {})) != set(plan["case_ids"])
            or cert.get("evaluator_version") != plan["implementation"]["evaluator_version"]):
        raise ValueError("Invalid or incomplete Block 62 development certificate.")
    state = read_json(validation_root / "progress.json")
    done = state.get("completed_case_ids", [])
    if (state.get("status") != "complete" or state.get("plan_sha256") != plan["plan_sha256"] or state.get("failed_cases")
            or state.get("deferred_case_ids") or len(done) != len(rows) or set(done) != set(plan["case_ids"])):
        raise ValueError("Complete Block 62 validation is required.")
    for cid in plan["case_ids"]:
        receipt = adoption.verified_adoption(validation_root, cid, plan)
        if receipt is None or receipt["receipt_sha256"] != cert["case_receipts"][cid]:
            raise ValueError(f"Block 62 case differs from certificate: {cid}")
        report = read_json(validation_root / "cases" / cid / next(iter(receipt["artifacts"])))
        source = read_json(run_root / "cases" / cid / "complete.json")
        metrics_path = next(p for p in source["artifacts"] if Path(p).name == "supported_renal_metrics.json")
        saved = cpu.json_metrics(read_json(run_root / "cases" / cid / metrics_path)["result"])
        if (report.get("original_receipt_sha256") != plan["source_receipts"][cid] or report.get("metric_parity") is not True
                or report.get("case_id") != cid or report.get("plan_sha256") != plan["plan_sha256"]
                or report.get("evaluator_version") != cert["evaluator_version"] or differences(saved, report["result"])):
            raise ValueError(f"Block 62 result differs from original: {cid}")
    return {"validation_plan_sha256": plan["plan_sha256"], "validation_certificate_sha256": cert["certificate_sha256"],
            "validation_receipts": cert["case_receipts"]}


def implementation() -> dict:
    return {"module_sha256": sha256_file(Path(__file__)),
            "adoption_helper_sha256": sha256_file(Path(adoption.__file__)),
            "cpu_helper_sha256": sha256_file(Path(cpu.__file__)),
            "review_helper_sha256": sha256_file(Path(__file__).with_name("full489_review.py")),
            "packages": {p: version(p) for p in ("numpy", "nibabel")}}


def prepare_analysis(root: Path, rows: list[dict], contract: dict, protocol: dict, evidence: dict) -> tuple[Path, dict]:
    expected = protocol["partition"]["development"]["case_ids"]
    if not expected or [r["case_id"] for r in rows] != expected or set(evidence["validation_receipts"]) != set(expected):
        raise ValueError("Confusion analysis requires all frozen development cases.")
    plan = {"block": 63, "contract_sha256": contract["contract_sha256"], "protocol_sha256": protocol["protocol_sha256"],
            "case_ids": expected, "source_receipts": {r["case_id"]: r["receipt_sha256"] for r in rows},
            "groups": {r["case_id"]: r["group"] for r in rows}, **evidence, "implementation": implementation(),
            "ground_truth_labels": dict(enumerate(GT_NAMES)), "prediction_labels": dict(zip(PRED_VALUES, PRED_NAMES)),
            "numeric_tolerance": {"absolute": 1e-9, "relative": 1e-9},
            "scope": "annotation-relative development voxel confusion; no clinical adjudication, lesion detection or method change"}
    # Use string keys from the beginning so frozen JSON round trips exactly.
    for k in ("ground_truth_labels", "prediction_labels"):
        plan[k] = {str(label): name for label, name in plan[k].items()}
    plan["plan_sha256"] = canonical_sha256(plan)
    check_analysis(plan, contract, protocol)
    output = root / "outputs/reporting/full489_block63" / contract["contract_sha256"][:16] / plan["plan_sha256"][:16]
    frozen_json(output / "plan.json", plan)
    return output, plan


def check_analysis(plan: dict, contract: dict, protocol: dict) -> None:
    check_plan(plan, contract)
    expected = protocol["partition"]["development"]["case_ids"]
    reserved = set(protocol["partition"]["validation"]["case_ids"]) | set(protocol["partition"]["test"]["case_ids"])
    if (plan["case_ids"] != expected or set(plan["source_receipts"]) != set(expected)
            or set(plan["validation_receipts"]) != set(expected) or set(expected) & reserved):
        raise ValueError("Confusion selection differs from frozen development.")
    if plan["implementation"] != implementation():
        raise ValueError("Confusion code or environment changed; prepare a new plan.")


def verified_case(output: Path, cid: str, plan: dict) -> dict | None:
    path = output / "cases" / cid / "complete.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if (receipt.get("receipt_sha256") != canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
            or receipt.get("case_id") != cid or receipt.get("plan_sha256") != plan["plan_sha256"]
            or receipt.get("status") != "complete" or receipt.get("count_parity") is not True):
        raise ValueError(f"Block 63 completion changed: {cid}")
    if len(receipt.get("artifacts", {})) != 1:
        raise ValueError("Incomplete Block 63 artifact list.")
    for name, digest in receipt["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.name != "confusion.json" or sha256_file(path.parent / relative) != digest:
            raise ValueError(f"Block 63 artifact changed: {cid}")
    return receipt


def analysis_case(root: Path, run_root: Path, output: Path, cid: str, plan: dict, contract: dict, protocol: dict) -> tuple[dict, bool]:
    check_analysis(plan, contract, protocol)
    cpu.check_baseline(root, contract)
    case, inputs, artifacts = cpu.source_case(root, run_root, cid, plan, contract)
    validation_root = root / "outputs/reporting/full489_block62" / run_root.name / plan["validation_plan_sha256"][:16]
    validation_plan = read_json(validation_root / "plan.json")
    def validate_source():
        receipt = adoption.verified_adoption(validation_root, cid, validation_plan)
        if receipt is None or receipt["receipt_sha256"] != plan["validation_receipts"][cid]:
            raise ValueError("Block 62 case source changed.")
    validate_source()
    cached = verified_case(output, cid, plan)
    if cached:
        return cached, True
    attempt = output / "cases" / cid / ("attempt_" + uuid4().hex[:8])
    attempt.mkdir(parents=True)
    try:
        started = time.perf_counter()
        gt, pred, spacing, _ = _load_pair(case.segmentation_path, artifacts["prediction.nii.gz"])
        counts = confusion_counts(gt, pred)
        saved = read_json(artifacts["supported_renal_metrics.json"])["result"]
        check_count_parity(counts, gt.shape, spacing, saved)
        voxel_ml = float(np.prod(spacing) / 1000.)
        matrix_ml = counts.astype(float) * voxel_ml
        report = {"case_id": cid, "group": plan["groups"][cid], "plan_sha256": plan["plan_sha256"], "count_parity": True,
                  "original_receipt_sha256": plan["source_receipts"][cid], "validation_receipt_sha256": plan["validation_receipts"][cid],
                  "shape": list(gt.shape), "spacing_mm": list(spacing), "voxel_ml": voxel_ml,
                  "counts": counts.tolist(), "volumes_ml": matrix_ml.tolist(), "breakdown": breakdown(matrix_ml),
                  "analysis_seconds": time.perf_counter() - started}
        del gt, pred
        _, after, _ = cpu.source_case(root, run_root, cid, plan, contract)
        if after != inputs:
            raise ValueError("Inputs changed during confusion analysis.")
        validate_source()
        cpu.check_baseline(root, contract)
        check_analysis(plan, contract, protocol)
        atomic_json(attempt / "confusion.json", report)
        receipt = {"case_id": cid, "plan_sha256": plan["plan_sha256"], "status": "complete", "count_parity": True,
                   "artifacts": {(attempt / "confusion.json").relative_to(attempt.parent).as_posix(): sha256_file(attempt / "confusion.json")}}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(attempt.parent / "complete.json", receipt)
        return receipt, False
    except Exception as exc:
        atomic_json(attempt / "failure.json", {"case_id": cid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def summarize_reports(reports: list[dict]) -> dict:
    summaries = {}
    for group in ("historical100", "additional_development189", "development289"):
        selected = reports if group == "development289" else [r for r in reports if r["group"] == group]
        counts = sum((np.asarray(r["counts"], dtype=np.int64) for r in selected), start=np.zeros((4, 5), dtype=np.int64))
        volumes = sum((np.asarray(r["volumes_ml"], dtype=float) for r in selected), start=np.zeros((4, 5)))
        macro = {}
        for category in ("fp_composition", "fn_composition", "predicted_cyst_composition", "reference_cyst_destinations"):
            keys = breakdown(volumes)[category]
            macro[category] = {k: describe([r["breakdown"][category][k] for r in selected]) for k in keys}
        summaries[group] = {"n_cases": len(selected), "pooled_counts": counts.tolist(), "pooled_volumes_ml": volumes.tolist(),
                            "pooled_volume_breakdown": breakdown(volumes), "per_case_fraction_summaries": macro,
                            "reference_cyst_positive_cases": sum(r["breakdown"]["reference_cyst_ml"] > 0 for r in selected),
                            "cases_with_cyst_fp": sum(r["breakdown"]["cyst_fp_total_ml"] > 0 for r in selected),
                            "cases_with_cyst_fn": sum(r["breakdown"]["cyst_fn_total_ml"] > 0 for r in selected)}
    rankings = {}
    for direction, labels in (("cyst_fp_by_reference_ml", GT_NAMES[:3]), ("cyst_fn_by_prediction_ml", PRED_NAMES[:3])):
        for label in labels:
            selected = sorted((r for r in reports if r["breakdown"][direction][label] > 0),
                              key=lambda r: (-r["breakdown"][direction][label], r["case_id"]))[:10]
            rankings[f"{direction}/{label}"] = [{"case_id": r["case_id"], "volume_ml": r["breakdown"][direction][label]} for r in selected]
    return {"groups": summaries, "rankings": rankings,
            "interpretation": "FP/FN relative to KiTS23 labels only; background disagreement is not proof of a clinical false positive",
            "aggregation": "physical volumes summed after per-case spacing conversion; macro fractions omit undefined denominators and report counts"}


def export_analysis(output: Path, plan: dict, state: dict) -> Path:
    reports, files, cells, cases = [], [output / "plan.json"], [], []
    for cid in state["completed_case_ids"]:
        receipt = verified_case(output, cid, plan)
        if receipt is None:
            raise ValueError(f"Missing Block 63 completion: {cid}")
        folder = output / "cases" / cid
        path = folder / next(iter(receipt["artifacts"]))
        report = read_json(path)
        if (report["case_id"] != cid or report["plan_sha256"] != plan["plan_sha256"] or report["count_parity"] is not True
                or report["original_receipt_sha256"] != plan["source_receipts"][cid]
                or report["validation_receipt_sha256"] != plan["validation_receipts"][cid]):
            raise ValueError("Confusion report identity changed.")
        reports.append(report)
        files.extend((path, folder / "complete.json"))
        cases.append({"case_id": cid, "group": report["group"], **{k: v for k, v in report["breakdown"].items() if not isinstance(v, dict)},
                      **{f"fp_on_{k}_ml": v for k, v in report["breakdown"]["cyst_fp_by_reference_ml"].items()},
                      **{f"fn_as_{k}_ml": v for k, v in report["breakdown"]["cyst_fn_by_prediction_ml"].items()}})
        for i, gt_name in enumerate(GT_NAMES):
            for j, pred_name in enumerate(PRED_NAMES):
                cells.append({"case_id": cid, "reference_label": i, "reference_name": gt_name, "prediction_label": PRED_VALUES[j],
                              "prediction_name": pred_name, "voxels": report["counts"][i][j], "volume_ml": report["volumes_ml"][i][j]})
    summary = {"status": state["status"], "plan_sha256": plan["plan_sha256"], "n_requested": len(plan["case_ids"]),
               "n_analyzed": len(reports), **summarize_reports(reports)}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", state)
    write_csv(output / "case_confusion.csv", cases)
    write_csv(output / "confusion_cells.csv", cells)
    group = summary["groups"]["development289"]
    lines = ["# Block 63 — Annotation-relative cyst confusion", "",
             f"Status: **{state['status']}**. Analyzed: {len(reports)}/{len(plan['case_ids'])} development cases.", "",
             "Rows are reference labels; columns are prediction labels. Entries below are pooled physical volumes in mL. "
             "Background dominates a full-grid matrix, so overall voxel accuracy is deliberately not reported.", "",
             "| Reference / Prediction | Background | Right kidney | Left kidney | Left cyst | Right cyst |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, row in zip(GT_NAMES, group["pooled_volumes_ml"]):
        lines.append("| " + name + " | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    b = group["pooled_volume_breakdown"]
    lines += ["", "## Cyst error directions", "", "| Direction | Reference/prediction label | Volume (mL) | Share of that error volume |",
              "|---|---|---:|---:|"]
    for direction, key, fraction in (("Predicted cyst on reference", "cyst_fp_by_reference_ml", "fp_composition"),
                                     ("Missed reference cyst labeled", "cyst_fn_by_prediction_ml", "fn_composition")):
        for label, value in b[key].items():
            share = b[fraction][label]
            lines.append(f"| {direction} | {label} | {value:.3f} | {100 * share:.2f}% |" if share is not None else
                         f"| {direction} | {label} | {value:.3f} | undefined |")
    lines += ["", "See summary.json for historical100/additional189 strata, denominator-aware per-case fractions and volume rankings. "
              "Partial exports describe only completed cases and cannot be presented as full-cohort results.", "",
              "The 4x5 matrix has different reference/prediction taxonomies: NV-Segment has no validated automatic tumor label here. "
              "GT cyst labeled kidney is a class confusion within the renal envelope; GT cyst labeled background is envelope omission. "
              "Predicted cyst on GT tumor indicates annotation-class disagreement, not a clinical diagnosis. "
              "Background disagreement alone cannot establish annotation omission or a clinical false positive.", "",
              "Counts, cyst/kidney-core binary metrics, renal-envelope Dice, tumor volume, grid and prediction labels "
              "are checked against original saved results. Surface distances and laterality assignments are not recomputed. "
              "No inference, postprocessing, prompt selection or validation/test output reads occur."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    files.extend(output / n for n in ("summary.json", "progress.json", "case_confusion.csv", "confusion_cells.csv", "report.md"))
    for cid in state["failed_cases"]:
        files.extend(sorted((output / "cases" / cid).glob("attempt_*/failure.json")))
    archive = output / "block63_cyst_confusion_report.zip"
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def run_analysis(root: Path, run_root: Path, output: Path, plan: dict, contract: dict, protocol: dict,
                 *, max_cases: int | None = None, continue_on_error: bool = False) -> tuple[dict, Path]:
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError("max_cases must be positive.")
    check_analysis(plan, contract, protocol)
    cpu.check_baseline(root, contract)
    completed, new, failed = [], [], {}
    attempted = 0
    def progress():
        return {"plan_sha256": plan["plan_sha256"], "requested_case_ids": plan["case_ids"], "completed_case_ids": completed.copy(),
                "newly_analyzed_case_ids": new.copy(), "failed_cases": failed.copy(),
                "deferred_case_ids": [c for c in plan["case_ids"] if c not in completed and c not in failed],
                "status": "failed" if failed else "complete" if len(completed) == len(plan["case_ids"]) else "partial",
                "gpu_inference": False, "surface_metric_recomputation": False, "validation_test_outputs_opened": False}
    atomic_json(output / "progress.json", progress())
    for index, cid in enumerate(plan["case_ids"]):
        cached = (output / "cases" / cid / "complete.json").exists()
        if not cached and max_cases is not None and attempted >= max_cases:
            continue
        print(f"[{index + 1:03d}/{len(plan['case_ids']):03d}] cyst confusion {cid}", flush=True)
        try:
            if not cached:
                attempted += 1
            _, reused = analysis_case(root, run_root, output, cid, plan, contract, protocol)
            completed.append(cid)
            if not reused:
                new.append(cid)
            print(f"  {'verified confusion cache' if reused else 'confusion complete'}; count parity passed", flush=True)
        except Exception as exc:
            failed[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[cid]}", flush=True)
            if not continue_on_error:
                break
        atomic_json(output / "progress.json", progress())
    state = progress()
    return state, export_analysis(output, plan, state)
