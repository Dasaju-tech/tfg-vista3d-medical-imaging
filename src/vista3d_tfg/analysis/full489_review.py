"""Block 60: outcome-guided development review and instrumented CPU reruns."""
from __future__ import annotations

import cProfile
import csv
from importlib.metadata import version
import math
from pathlib import Path
import pstats
import statistics
import time
from uuid import uuid4
import zipfile

import nibabel as nib
import numpy as np
from scipy import ndimage

from vista3d_tfg.analysis.full489_development import describe
from vista3d_tfg.data.audit import audit_case
from vista3d_tfg.evaluation.nv_kits23 import evaluate_supported_renal_prediction
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import (
    atomic_json, checked_receipt, frozen_json, input_identity, read_json,
)
from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS, audit_renal_prediction, sha256_file


def screen_case(row: dict) -> dict:
    reference = row["cyst_reference_ml"]
    tp = max(0., reference - row["cyst_false_negative_ml"])
    predicted = tp + row["cyst_false_positive_ml"]
    if not row["cyst_positive"]:
        category = "reference_negative_prediction" if row["cyst_prediction_present"] else "reference_negative_no_prediction"
    elif not row["cyst_prediction_present"]:
        category = "no_cyst_prediction"
    elif row["cyst_dice"] == 0:
        category = "zero_overlap_with_prediction"
    else:
        fp, fn = row["cyst_false_positive_ml"], row["cyst_false_negative_ml"]
        category = "balanced_volume_error" if abs(fp - fn) <= max(1e-9, .05 * max(fp, fn)) else (
            "fn_volume_dominant" if fn > fp else "fp_volume_dominant")
    return {**row, "cyst_error_category": category,
            "cyst_precision": tp / predicted if predicted > 0 else None,
            "cyst_recall": tp / reference if reference > 0 else None,
            "outside_worker_seconds": row["case_total_seconds"] - row["worker_process_seconds"],
            "worker_outside_bundle_seconds": row["worker_process_seconds"] - row["bundle_seconds"]}


def cohort_screen(rows: list[dict]) -> dict:
    screened = [screen_case(r) for r in rows]
    groups = {}
    for group in ("historical100", "additional_development189", "development289"):
        selected = screened if group == "development289" else [r for r in screened if r["group"] == group]
        positive = [r for r in selected if r["cyst_positive"]]
        categories = {}
        for r in selected:
            key = r["cyst_error_category"]
            categories[key] = categories.get(key, 0) + 1
        bins = []
        for label, low, high in (("(0,1) mL", 0., 1.), ("[1,5) mL", 1., 5.),
                                 ("[5,20) mL", 5., 20.), ("[20,+inf) mL", 20., math.inf)):
            subset = [r for r in positive if low <= r["cyst_reference_ml"] < high]
            bins.append({"total_reference_cyst_volume_bin": label, "n": len(subset),
                         "zero_overlap_n": sum(r["cyst_dice"] == 0 for r in subset),
                         "dice": describe([r["cyst_dice"] for r in subset]),
                         "recall": describe([r["cyst_recall"] for r in subset])})
        total = sum(r["case_total_seconds"] for r in selected)
        outside = sum(r["outside_worker_seconds"] for r in selected)
        groups[group] = {"n": len(selected), "categories": categories, "cyst_positive_n": len(positive),
                         "cyst_positive_zero_overlap_n": sum(r["cyst_dice"] == 0 for r in positive),
                         "cyst_volume_bins": bins, "recorded_case_seconds": total,
                         "recorded_outside_worker_seconds": outside,
                         "outside_worker_fraction_of_recorded_sum": outside / total if total else None}
    return {"groups": groups, "volume_scope": "total annotated cyst volume per case, not individual lesion size",
            "error_scope": "voxel/volume diagnostics relative to annotations; not clinical detection performance",
            "timing_scope": "differences of recorded successful-case timers; stages inside the parent are not identified by these differences"}


def build_plan(rows: list[dict], contract: dict) -> dict:
    new = [r for r in rows if r["group"] == "additional_development189"]
    positive = [r for r in rows if r["cyst_positive"]]
    missed = [r for r in positive if r["cyst_dice"] == 0]
    false_positive = [r for r in rows if not r["cyst_positive"] and r["cyst_prediction_present"]]
    median = statistics.median(r["case_total_seconds"] for r in rows)
    def ranked(items, metric, descending=False):
        return sorted(items, key=lambda r: (
            -(math.inf if r[metric] is None else r[metric]) if descending else r[metric], r["case_id"]))
    rules = (
        ("new_lowest_dice", ranked(new, "kidney_mass_dice")[:3]),
        ("new_highest_hd95", ranked(new, "hd95_mm", True)[:3]),
        ("no_prediction_largest_reference_cysts", ranked([r for r in missed if not r["cyst_prediction_present"]], "cyst_reference_ml", True)[:2]),
        ("zero_overlap_with_prediction_largest_reference_cysts", ranked([r for r in missed if r["cyst_prediction_present"]], "cyst_reference_ml", True)[:2]),
        ("reference_negative_largest_predicted_cysts", ranked(false_positive, "cyst_false_positive_ml", True)[:3]),
        ("best_cyst_positive_controls", ranked(positive, "cyst_dice", True)[:2]),
        ("slowest_recorded_cases", ranked(rows, "case_total_seconds", True)[:3]),
        ("near_median_runtime_controls", sorted(rows, key=lambda r: (abs(r["case_total_seconds"] - median), r["case_id"]))[:2]),
    )
    reasons = {}
    for reason, selected in rules:
        for row in selected:
            reasons.setdefault(row["case_id"], []).append(reason)
    from vista3d_tfg.analysis import full489_development
    plan = {"block": 60, "protocol_sha256": contract["protocol_sha256"],
            "contract_sha256": contract["contract_sha256"], "case_ids": list(reasons), "selection_reasons": reasons,
            "source_rows_sha256": canonical_sha256(rows),
            "source_receipts": {r["case_id"]: r["receipt_sha256"] for r in rows},
            "review_module_sha256": sha256_file(Path(__file__)),
            "collector_module_sha256": sha256_file(Path(full489_development.__file__)),
            "review_packages": {p: version(p) for p in ("numpy", "scipy", "nibabel", "matplotlib", "surface-distance")},
            "selection_scope": "deterministic outcome-guided development review; not a representative sample or a deployment trigger",
            "parity_tolerance": {"absolute": 1e-9, "relative": 1e-9},
            "component_definition": "6-connected voxel components; any overlap is diagnostic, not lesion detection validation"}
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_review(root: Path, rows: list[dict], contract: dict) -> tuple[Path, dict]:
    plan = build_plan(rows, contract)
    output = root / "outputs/reporting/full489_block60" / contract["contract_sha256"][:16] / plan["plan_sha256"][:16]
    frozen_json(output / "plan.json", plan)
    frozen_json(output / "cohort_screen.json", cohort_screen(rows))
    write_csv(output / "case_screening.csv", [screen_case(r) for r in rows])
    by_id = {r["case_id"]: r for r in rows}
    write_csv(output / "review_queue.csv", [{"case_id": cid, "reasons": "; ".join(plan["selection_reasons"][cid]),
              "group": by_id[cid]["group"], "dice": by_id[cid]["kidney_mass_dice"],
              "cyst_dice": by_id[cid]["cyst_dice"], "original_case_seconds": by_id[cid]["case_total_seconds"]}
              for cid in plan["case_ids"]])
    return output, plan


def differences(expected, actual, path="result") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            return [f"{path}: keys differ"]
        return [d for k in expected for d in differences(expected[k], actual[k], f"{path}.{k}")]
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            return [f"{path}: lengths differ"]
        return [d for i, (a, b) in enumerate(zip(expected, actual)) for d in differences(a, b, f"{path}[{i}]")]
    if isinstance(expected, bool) or isinstance(actual, bool) or expected is None or actual is None:
        equal = type(expected) is type(actual) and expected == actual
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        equal = (expected == actual and not math.isnan(expected)) or math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
    else:
        equal = type(expected) is type(actual) and expected == actual
    return [] if equal else [f"{path}: expected {expected!r}, got {actual!r}"]


def component_summary(reference: np.ndarray, predicted: np.ndarray, voxel_ml: float) -> dict:
    info = {}
    for name, mask, other in (("reference", reference, predicted), ("prediction", predicted, reference)):
        labels, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
        hit_ids = np.unique(labels[other])
        hits = int(np.count_nonzero(hit_ids))
        info[name] = {"component_count": int(count), "components_with_any_overlap": hits,
                      "components_without_overlap": int(count) - hits,
                      "largest_component_ml": float(sizes.max() * voxel_ml) if count else 0.,
                      "largest_ten_volumes_ml": [float(n * voxel_ml) for n in sorted(sizes, reverse=True)[:10]]}
    return info


def load_review_arrays(image: Path, ground_truth: Path, prediction: Path):
    images = [nib.load(str(p)) for p in (image, ground_truth, prediction)]
    if any(n.shape != images[0].shape or not np.allclose(n.affine, images[0].affine, rtol=0, atol=1e-5) for n in images[1:]):
        raise ValueError("Review geometry mismatch.")
    canonical = [nib.as_closest_canonical(n) for n in images]
    arrays = [np.asanyarray(n.dataobj) for n in canonical]
    spacing = tuple(float(x) for x in canonical[0].header.get_zooms()[:3])
    return (*arrays, spacing)


def render_review(image, gt, pred, spacing, output: Path, cid: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    selections = {}
    for scope in ("renal_mass", "cyst"):
        reference = gt > 0 if scope == "renal_mass" else gt == 3
        predicted = pred > 0 if scope == "renal_mass" else np.isin(pred, (116, 117))
        errors = reference != predicted
        figure, axes = plt.subplots(3, 4, figsize=(12, 10), constrained_layout=True)
        figure.get_layout_engine().set(rect=(0, .035, 1, .965))
        selections[scope] = []
        for row, (plane, axis) in enumerate((("axial", 2), ("coronal", 1), ("sagittal", 0))):
            reduce_axes = tuple(a for a in range(3) if a != axis)
            counts = np.count_nonzero(errors, axis=reduce_axes)
            if np.any(counts):
                index, source = int(np.argmax(counts)), "maximum_disagreement"
            else:
                counts = np.count_nonzero(reference, axis=reduce_axes)
                index = int(np.argmax(counts)) if np.any(counts) else reference.shape[axis] // 2
                source = "maximum_reference_area" if np.any(counts) else "volume_center"
            selections[scope].append({"plane": plane, "canonical_axis": axis, "index": index, "source": source})
            ct = np.take(image, index, axis=axis).T
            ref = np.take(reference, index, axis=axis).T
            pr = np.take(predicted, index, axis=axis).T
            kept = [a for a in range(3) if a != axis]
            extent = (0, ct.shape[1] * spacing[kept[0]], 0, ct.shape[0] * spacing[kept[1]])
            error = np.zeros(ref.shape, dtype=np.uint8)
            error[ref & pr], error[ref & ~pr], error[~ref & pr] = 1, 2, 3
            for column in range(4):
                ax = axes[row, column]
                ax.imshow(ct, cmap="gray", vmin=-200, vmax=300, origin="lower", extent=extent)
                mask = ref if column == 1 else pr if column == 2 else error
                if column:
                    cmap = ListedColormap(["none", "#56b870", "#ff9a36", "#38b9e8"]) if column == 3 else ListedColormap(["none", "#56b870"])
                    ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap, vmin=0,
                              vmax=3 if column == 3 else 1, alpha=.7, origin="lower", extent=extent,
                              interpolation="nearest")
                ax.set_title(f"{plane}, slice {index}\n" + ("CT", "Reference", "Prediction", "Overlap / FN / FP")[column], fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        figure.suptitle(f"{cid}: {scope} — GT-assisted development review\nRAS voxel grid; physical aspect ratio; selected slices are not a 3D diagnosis", fontsize=11)
        figure.legend(handles=[Patch(color="#56b870", label="Overlap / mask"), Patch(color="#ff9a36", label="FN"),
                               Patch(color="#38b9e8", label="FP")], loc="lower center", ncol=3, fontsize=8)
        figure.savefig(output / f"{scope}_review.png", dpi=130)
        plt.close(figure)
    return selections


def check_evaluation_environment(root: Path, contract: dict) -> None:
    for relative, digest in contract["evaluation_files"].items():
        if sha256_file(root / "src/vista3d_tfg" / relative) != digest:
            raise ValueError(f"Evaluation source differs from the original run: {relative}")
    for package, expected in contract["evaluation_packages"].items():
        if version(package) != expected:
            raise ValueError(f"Evaluation package differs from the original run: {package}")


def verified_review(output: Path, cid: str, plan: dict) -> dict | None:
    path = output / "cases" / cid / "complete.json"
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt.get("receipt_sha256") != canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"}):
        raise ValueError(f"Block 60 completion changed: {cid}")
    if receipt.get("plan_sha256") != plan["plan_sha256"] or receipt.get("case_id") != cid or receipt.get("status") != "complete" or receipt.get("metric_parity") is not True:
        raise ValueError(f"Block 60 completion identity changed: {cid}")
    expected = {"review.json", "evaluation_profile.csv", "renal_mass_review.png", "cyst_review.png"}
    if len(receipt.get("artifacts", {})) != 4 or {Path(p).name for p in receipt["artifacts"]} != expected:
        raise ValueError(f"Incomplete Block 60 artifact list: {cid}")
    for name, digest in receipt["artifacts"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or sha256_file(path.parent / relative) != digest:
            raise ValueError(f"Block 60 artifact changed: {cid}/{name}")
    return receipt


def check_plan(plan: dict, contract: dict) -> None:
    if plan.get("plan_sha256") != canonical_sha256({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("Review plan checksum changed.")
    if plan["contract_sha256"] != contract["contract_sha256"] or plan["protocol_sha256"] != contract["protocol_sha256"]:
        raise ValueError("Review plan belongs to another run/protocol.")
    if len(set(plan["case_ids"])) != len(plan["case_ids"]) or not set(plan["case_ids"]).issubset(plan["source_receipts"]):
        raise ValueError("Review cases differ from the verified development sources.")


def review_case(root: Path, run_root: Path, output: Path, cid: str, plan: dict, contract: dict) -> tuple[dict, bool]:
    check_plan(plan, contract)
    if cid not in plan["case_ids"]:
        raise ValueError("Case is not in the frozen development review plan.")
    cached = verified_review(output, cid, plan)
    if cached:
        _, current = input_identity(root, cid)
        source = checked_receipt(run_root / "cases" / cid, {**current, "contract_sha256": contract["contract_sha256"]})
        if source is None or source["receipt_sha256"] != plan["source_receipts"][cid]:
            raise ValueError("Cached review source differs from the plan.")
        return cached, True
    attempt = output / "cases" / cid / ("attempt_" + uuid4().hex[:8])
    attempt.mkdir(parents=True)
    timings = {}
    def timed(name, function):
        print(f"  {name}...", flush=True)
        started = time.perf_counter()
        try:
            return function()
        finally:
            timings[name] = time.perf_counter() - started
            print(f"    {timings[name]:.2f} s", flush=True)
    try:
        case, inputs = timed("input_hashes_before", lambda: input_identity(root, cid))
        original = timed("saved_artifact_verification", lambda: checked_receipt(
            run_root / "cases" / cid, {**inputs, "contract_sha256": contract["contract_sha256"]}))
        if original is None or original["receipt_sha256"] != plan["source_receipts"][cid]:
            raise ValueError("Original completion differs from the review plan.")
        artifacts = {Path(p).name: run_root / "cases" / cid / p for p in original["artifacts"]}
        input_audit = timed("input_geometry_and_labels", lambda: audit_case(case))
        if not input_audit.is_valid:
            raise ValueError(f"Input audit failed: {input_audit.issues}")
        audit = timed("prediction_audit", lambda: audit_renal_prediction(
            input_path=case.imaging_path, prediction_path=artifacts["prediction.nii.gz"], labels=EXPECTED_RENAL_LABELS))
        if not audit.is_valid or audit.sha256 != original["prediction_audit"]["sha256"]:
            raise ValueError("Prediction audit or digest changed.")
        profiler = cProfile.Profile()
        evaluated = timed("evaluation_with_cprofile", lambda: profiler.runcall(
            evaluate_supported_renal_prediction, case_id=cid, ground_truth_path=case.segmentation_path,
            prediction_path=artifacts["prediction.nii.gz"]))
        profile = []
        for (filename, line, function), (primitive, calls, self_time, cumulative, _) in pstats.Stats(profiler).stats.items():
            profile.append({"function": function, "file": str(filename), "line": line,
                            "primitive_calls": primitive, "total_calls": calls,
                            "self_seconds": self_time, "cumulative_seconds": cumulative})
        profile.sort(key=lambda r: (-r["cumulative_seconds"], r["file"], r["line"]))
        write_csv(attempt / "evaluation_profile.csv", profile[:50])
        baseline = read_json(artifacts["supported_renal_metrics.json"])["result"]
        delta = differences(baseline, evaluated.to_dict())
        if delta:
            atomic_json(attempt / "parity_failure.json", {"differences": delta})
            raise ValueError(f"Recomputed metrics differ in {len(delta)} fields; see parity_failure.json.")
        image, gt, prediction, spacing = timed("review_array_loading", lambda: load_review_arrays(
            case.imaging_path, case.segmentation_path, artifacts["prediction.nii.gz"]))
        components = timed("cyst_components", lambda: component_summary(gt == 3, np.isin(prediction, (116, 117)), math.prod(spacing) / 1000))
        selections = timed("review_rendering", lambda: render_review(image, gt, prediction, spacing, attempt, cid))
        del image, gt, prediction
        after = timed("input_hashes_after", lambda: input_identity(root, cid)[1])
        final_receipt = timed("saved_artifact_verification_after", lambda: checked_receipt(
            run_root / "cases" / cid, {**after, "contract_sha256": contract["contract_sha256"]}))
        if after != inputs or final_receipt is None or final_receipt["receipt_sha256"] != original["receipt_sha256"]:
            raise ValueError("Source data or saved results changed during review.")
        report = {"status": "complete", "case_id": cid, "plan_sha256": plan["plan_sha256"],
                  "original_receipt_sha256": original["receipt_sha256"], "metric_parity": True,
                  "numeric_tolerance": plan["parity_tolerance"], "stage_seconds": timings,
                  "original_stage_seconds": original["timing"], "cyst_components": components,
                  "slice_selections": selections, "gpu_inference": False,
                  "timing_scope": "instrumented CPU rerun; cProfile and current file caches affect times; not a speedup measurement",
                  "qc_scope": "GT-assisted development slice selection; components are voxel-connected regions, not adjudicated lesions"}
        atomic_json(attempt / "review.json", report)
        receipt = {"status": "complete", "case_id": cid, "plan_sha256": plan["plan_sha256"],
                   "stage_seconds": timings, "metric_parity": True,
                   "artifacts": {p.relative_to(attempt.parent).as_posix(): sha256_file(p) for p in (
                       attempt / "review.json", attempt / "evaluation_profile.csv",
                       attempt / "renal_mass_review.png", attempt / "cyst_review.png")}}
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        atomic_json(attempt.parent / "complete.json", receipt)
        return receipt, False
    except Exception as exc:
        atomic_json(attempt / "failure.json", {"status": "failed", "case_id": cid,
                    "stage_seconds": timings, "error": f"{type(exc).__name__}: {exc}"})
        raise


def run_review(root: Path, run_root: Path, output: Path, plan: dict, contract: dict,
               *, max_cases: int | None = None, continue_on_error: bool = False) -> dict:
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError("max_cases must be positive.")
    check_plan(plan, contract)
    check_evaluation_environment(root, contract)
    completed, new, failed, receipts = [], [], {}, []
    attempted = 0
    def progress():
        return {"plan_sha256": plan["plan_sha256"], "requested_case_ids": plan["case_ids"],
                "completed_case_ids": completed.copy(), "newly_reviewed_case_ids": new.copy(),
                "failed_cases": failed.copy(), "deferred_case_ids": [c for c in plan["case_ids"] if c not in completed and c not in failed],
                "status": "failed" if failed else "complete" if len(completed) == len(plan["case_ids"]) else "partial",
                "gpu_inference": False}
    atomic_json(output / "review_progress.json", progress())
    for index, cid in enumerate(plan["case_ids"]):
        has_receipt = (output / "cases" / cid / "complete.json").exists()
        if not has_receipt and max_cases is not None and attempted >= max_cases:
            continue
        print(f"[{index + 1:02d}/{len(plan['case_ids']):02d}] review {cid}", flush=True)
        try:
            if not has_receipt:
                attempted += 1
            receipt, reused = review_case(root, run_root, output, cid, plan, contract)
            completed.append(cid)
            receipts.append(receipt)
            if not reused:
                new.append(cid)
            print(f"  {'verified review cache' if reused else 'review complete'}; metric parity passed", flush=True)
        except Exception as exc:
            failed[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {failed[cid]}", flush=True)
            if not continue_on_error:
                break
        atomic_json(output / "review_progress.json", progress())
    state = progress()
    stages = sorted({k for receipt in receipts for k in receipt["stage_seconds"]})
    state["instrumented_stage_summaries"] = {k: describe([r["stage_seconds"][k] for r in receipts]) for k in stages}
    state["interpretation"] = "timings from a deliberately selected review subset; no cohort-wide runtime extrapolation"
    atomic_json(output / "review_progress.json", state)
    write_csv(output / "stage_timings.csv", [{"case_id": r["case_id"], **r["stage_seconds"]} for r in receipts])
    return state


def package_review(output: Path, plan: dict, state: dict) -> Path:
    lines = ["# Block 60 — Development failure review", "",
             "This is outcome-guided, GT-assisted development review. Validation/test remain reserved.", "",
             f"Status: **{state['status']}**. Completed reviews: {len(state['completed_case_ids'])}/{len(plan['case_ids'])}.", "",
             "See cohort_screen.json for all-development cyst categories and case-volume bins. "
             "The review subset is deliberately selected and is not representative of all cases.", "",
             "Original inference files are preserved. Evaluation is recalculated with cProfile and compared "
             "field by field at 1e-9 absolute/relative tolerance. Rerun times include instrumentation and cache effects. "
             "Cumulative function times overlap and must not be summed.", ""]
    files = [output / n for n in ("plan.json", "cohort_screen.json", "case_screening.csv", "review_queue.csv", "review_progress.json", "stage_timings.csv") if (output / n).is_file()]
    for cid in state["completed_case_ids"]:
        receipt = verified_review(output, cid, plan)
        if receipt is None:
            raise ValueError(f"Missing completed review while packaging: {cid}")
        folder = output / "cases" / cid
        files.append(folder / "complete.json")
        files.extend(folder / name for name in receipt["artifacts"])
        lines += [f"## {cid}", "", "; ".join(plan["selection_reasons"][cid]), ""]
        for name in receipt["artifacts"]:
            if name.endswith(".png"):
                relative = (folder / name).relative_to(output).as_posix()
                lines += [f"![{cid} development review]({relative})", ""]
    for cid in state.get("failed_cases", {}):
        for attempt in sorted((output / "cases" / cid).glob("attempt_*")):
            files.extend(attempt / name for name in ("failure.json", "parity_failure.json", "evaluation_profile.csv")
                         if (attempt / name).is_file())
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    files.append(output / "report.md")
    archive = output / "block60_review_report.zip"
    temporary = output / "block60_review_report.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive
