"""Block 59: integrity-checked descriptive analysis of development only."""
from __future__ import annotations

import csv
from hashlib import sha256
import math
from pathlib import Path
import platform
import re
import zipfile

import numpy as np

from vista3d_tfg.evaluation.nv_kits23 import KITS23_KIDNEY_AND_MASS_TOLERANCE_MM
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import atomic_json, checked_receipt, read_json
from vista3d_tfg.integration.nv_inference import sha256_file


GROUPS = ("historical100", "additional_development189", "development289")
PRIMARY = ("kidney_mass_dice", "surface_dice", "hd95_mm")
SCOPE = (
    "Descriptive automatic baseline on development; no validation/test outcomes. "
    "Historical and additional development are disjoint; development289 contains both. "
    "This is not a full KiTS23 score, an independent test, or clinical validation. "
    "Case bootstrap intervals assume independent cases; patient linkage and "
    "foundation-model pretraining overlap remain unresolved."
)


def number(value, name: str, *, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid numeric field: {name}")
    if not math.isfinite(value) or value < 0 or (upper is not None and value > upper):
        raise ValueError(f"Out-of-range numeric field: {name}")
    return float(value)


def check_contract(contract: dict, protocol: dict) -> None:
    unsigned = {k: v for k, v in contract.items() if k != "contract_sha256"}
    if contract.get("contract_sha256") != canonical_sha256(unsigned):
        raise ValueError("Run contract checksum changed.")
    if contract.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Run contract belongs to another protocol.")
    if contract.get("label_prompt") != [5, 14, 116, 117]:
        raise ValueError("Unexpected automatic renal prompt.")
    if contract.get("kidney_mass_surface_tolerance_mm") != KITS23_KIDNEY_AND_MASS_TOLERANCE_MM:
        raise ValueError("Unexpected surface tolerance.")
    if number(contract["runtime_identity"]["gpu_total_bytes"], "gpu_total_bytes") == 0:
        raise ValueError("GPU total memory must be positive.")


def read_case(run_root: Path, contract: dict, cid: str, historical: set[str]) -> dict:
    case_root = run_root / "cases" / cid
    inputs = read_json(case_root / "inputs.json")
    if set(inputs) != {"case_id", "image_sha256", "ground_truth_sha256"} or inputs["case_id"] != cid:
        raise ValueError("Invalid frozen input identity.")
    for key in ("image_sha256", "ground_truth_sha256"):
        if not isinstance(inputs[key], str) or not re.fullmatch(r"[0-9a-f]{64}", inputs[key]):
            raise ValueError(f"Invalid input digest: {key}")
    receipt = checked_receipt(case_root, {**inputs, "contract_sha256": contract["contract_sha256"]})
    if receipt is None:
        raise ValueError("Missing completion receipt.")
    artifacts = {Path(name).name: case_root / name for name in receipt["artifacts"]}
    payload = read_json(artifacts["supported_renal_metrics.json"])
    if payload.get("schema_version") != "1.1" or payload.get("protocol") != "supported_renal_structures":
        raise ValueError("Unexpected evaluation schema or protocol.")
    result = payload["result"]
    if result["case_id"] != cid or result["full_kits23_score_available"] is not False:
        raise ValueError("Evaluation case/scope mismatch.")
    job, worker = read_json(artifacts["job.json"]), read_json(artifacts["worker.json"])
    if job.get("mode") != "infer" or worker.get("status") != "complete":
        raise ValueError("Incomplete inference job/report.")
    if job.get("label_prompt") != contract["label_prompt"] or job.get("bundle_files") != contract["bundle_files"]:
        raise ValueError("Job does not match frozen bundle/prompt.")
    for key, expected in (
        ("image_sha256", inputs["image_sha256"]),
        ("checkpoint_sha256", contract["checkpoint_sha256"]),
        ("runtime_identity", contract["runtime_identity"]),
    ):
        if job.get(key) != expected or worker.get(key) != expected:
            raise ValueError(f"Job/worker identity mismatch: {key}")
    audit = receipt["prediction_audit"]
    if any(audit.get(k) is not True for k in (
        "affine_matches_input", "shape_matches_input", "integer_valued", "finite",
    )) or audit.get("unexpected_labels") != [] or audit.get("contains_unprocessed_255") is not False:
        raise ValueError("Prediction audit was not valid.")
    prediction_digest = next(d for p, d in receipt["artifacts"].items() if Path(p).name == "prediction.nii.gz")
    if audit["sha256"] != prediction_digest or audit["shape"] != result["shape"]:
        raise ValueError("Prediction audit/evaluation mismatch.")
    shape = result["shape"]
    if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("Invalid stored volume shape.")
    hec = result["kidney_and_mass_hec"]
    if hec["tolerance_mm"] != contract["kidney_mass_surface_tolerance_mm"]:
        raise ValueError("Evaluation surface tolerance mismatch.")
    dice = number(hec["dice"], "dice", upper=1)
    surface = number(hec["surface_dice"], "surface_dice", upper=1)
    distance = hec["diagnostic_hd95_mm"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or math.isnan(distance) or distance < 0:
        raise ValueError("Invalid HD95; only positive infinity is an accepted nonfinite outcome.")
    hd95 = float(distance) if math.isfinite(distance) else None
    expected = {"dice": dice, "surface_dice": surface, "hd95_mm": hd95,
                "hd95_nonfinite": hd95 is None}
    if receipt["kidney_mass"] != expected:
        raise ValueError("Receipt and evaluation metrics disagree.")
    cyst = result["cyst"]
    for field in ("reference_voxels", "prediction_voxels"):
        if type(cyst[field]) is not int or cyst[field] < 0:
            raise ValueError(f"Invalid cyst {field}.")
    row = {
        "case_id": cid,
        "group": GROUPS[0] if cid in historical else GROUPS[1],
        "receipt_sha256": receipt["receipt_sha256"],
        "image_sha256": inputs["image_sha256"],
        "ground_truth_sha256": inputs["ground_truth_sha256"],
        "kidney_mass_dice": dice, "surface_dice": surface,
        "hd95_mm": hd95, "hd95_infinite": hd95 is None,
        "kidney_core_dice": number(result["kidney_core_diagnostic"]["dice"], "kidney_core_dice", upper=1),
        "cyst_positive": cyst["reference_voxels"] > 0,
        "cyst_prediction_present": cyst["prediction_voxels"] > 0,
        "cyst_dice": number(cyst["dice"], "cyst_dice", upper=1),
        "cyst_reference_ml": number(cyst["reference_volume_ml"], "cyst_reference_ml"),
        "cyst_false_positive_ml": number(cyst["false_positive_volume_ml"], "cyst_false_positive_ml"),
        "cyst_false_negative_ml": number(cyst["false_negative_volume_ml"], "cyst_false_negative_ml"),
        "tumor_reference_ml": number(result["gt_tumor_volume_ml"], "tumor_reference_ml"),
        "laterality_resolved": result["ground_truth_laterality_resolved"],
        "input_grid_voxels": math.prod(shape),
    }
    if type(row["laterality_resolved"]) is not bool:
        raise ValueError("Invalid laterality status.")
    for field in ("case_total_seconds", "worker_process_seconds", "bundle_seconds"):
        row[field] = number(receipt["timing"][field], field)
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = number(receipt["memory"][field], field)
        if value != worker[field]:
            raise ValueError(f"Worker/receipt measurement mismatch: {field}")
        row[field.replace("_bytes", "_gib")] = value / 2**30
    if row["bundle_seconds"] != worker["bundle_wall_seconds"]:
        raise ValueError("Worker/receipt timing mismatch.")
    return row


def collect_development(run_root: Path, contract: dict, protocol: dict) -> tuple[list[dict], dict]:
    check_contract(contract, protocol)
    requested = protocol["partition"]["development"]["case_ids"]
    historical = set(protocol["cohorts"]["historical100"]["case_ids"])
    reserved = set(protocol["partition"]["validation"]["case_ids"]) | set(protocol["partition"]["test"]["case_ids"])
    if len(set(requested)) != len(requested) or not historical.issubset(requested) or set(requested) & reserved:
        raise ValueError("Development allocation overlaps reserved cases or is inconsistent.")
    rows, errors = [], {}
    # Never enumerate or open validation/test case outputs.
    for index, cid in enumerate(requested):
        print(f"[{index + 1:03d}/{len(requested):03d}] verify development {cid}", flush=True)
        try:
            rows.append(read_case(run_root, contract, cid, historical))
        except Exception as exc:
            errors[cid] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {errors[cid]}", flush=True)
    audit = {
        "status": "complete" if not errors else "failed",
        "protocol_sha256": protocol["protocol_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "requested_case_ids": requested,
        "verified_case_ids": [r["case_id"] for r in rows],
        "failed_cases": errors,
        "receipt_sha256": {r["case_id"]: r["receipt_sha256"] for r in rows},
        "artifact_verification": "receipt checksum plus prediction, evaluation, worker and job byte hashes",
        "original_ct_gt_bytes_rehashed": False,
        "model_weights_rehashed_or_loaded": False,
        "prediction_voxels_loaded_or_metrics_recomputed": False,
        "validation_test_outputs_opened": False,
        "input_identity_scope": "frozen input digests cross-checked against receipts; raw input files not reopened",
    }
    return rows, audit


def describe(values: list, *, reps: int = 0, seed: int = 0) -> dict:
    finite = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    n = len(finite)
    result = {"n_total": len(values), "n_finite": n, "n_nonfinite": len(values) - n,
              "mean": None, "sd": None, "median": None, "q1": None, "q3": None,
              "p95": None, "min": None, "max": None, "bootstrap_mean_95ci": None}
    if not n:
        return result
    q1, median, q3, p95 = np.percentile(finite, [25, 50, 75, 95], method="linear")
    result.update(mean=float(finite.mean()), sd=float(finite.std(ddof=1)) if n > 1 else None,
                  median=float(median), q1=float(q1), q3=float(q3), p95=float(p95),
                  min=float(finite.min()), max=float(finite.max()))
    if reps and n > 1:
        rng = np.random.default_rng(seed)
        means = []
        for start in range(0, reps, 256):
            indices = rng.integers(0, n, size=(min(256, reps - start), n))
            means.extend(finite[indices].mean(axis=1).tolist())
        result["bootstrap_mean_95ci"] = np.percentile(means, [2.5, 97.5], method="linear").tolist()
    return result


def summarize(rows: list[dict], contract: dict, *, reps: int = 5000, seed: int = 20260905) -> dict:
    if reps < 1 or seed < 0:
        raise ValueError("Bootstrap repetitions must be positive and seed nonnegative.")
    cohorts = {}
    gpu_gib = contract["runtime_identity"]["gpu_total_bytes"] / 2**30
    for group in GROUPS:
        selected = rows if group == GROUPS[2] else [r for r in rows if r["group"] == group]
        positive = [r for r in selected if r["cyst_positive"]]
        negative = [r for r in selected if not r["cyst_positive"]]
        metrics = {}
        for metric in (*PRIMARY, "kidney_core_dice", "cyst_positive_dice"):
            sample = [r["cyst_dice"] for r in positive] if metric == "cyst_positive_dice" else [r[metric] for r in selected]
            metric_seed = int.from_bytes(sha256(f"{seed}|{group}|{metric}".encode()).digest()[:8], "big")
            metrics[metric] = describe(sample, reps=reps, seed=metric_seed)
        cohorts[group] = {
            "n": len(selected), "metrics": metrics,
            "cyst_positive_n": len(positive), "cyst_negative_n": len(negative),
            "cyst_negative_with_prediction_n": sum(r["cyst_prediction_present"] for r in negative),
            "cyst_negative_false_positive_ml": describe([r["cyst_false_positive_ml"] for r in negative]),
            "tumor_positive_n": sum(r["tumor_reference_ml"] > 0 for r in selected),
            "tumor_reference_ml": describe([r["tumor_reference_ml"] for r in selected]),
            "laterality_unresolved_n": sum(not r["laterality_resolved"] for r in selected),
            "runtime": {k: describe([r[k] for r in selected]) for k in (
                "case_total_seconds", "worker_process_seconds", "bundle_seconds",
                "peak_allocated_gib", "peak_reserved_gib",
            )},
            "sum_recorded_successful_case_seconds": sum(r["case_total_seconds"] for r in selected),
            "peak_allocated_above_reported_gpu_total_n": sum(r["peak_allocated_gib"] > gpu_gib for r in selected),
            "peak_reserved_above_reported_gpu_total_n": sum(r["peak_reserved_gib"] > gpu_gib for r in selected),
        }
    rankings = {}
    for label, metric, descending in (
        ("lowest_kidney_mass_dice", "kidney_mass_dice", False),
        ("lowest_surface_dice", "surface_dice", False),
        ("highest_hd95", "hd95_mm", True),
        ("slowest_cases", "case_total_seconds", True),
        ("highest_reserved_memory", "peak_reserved_gib", True),
    ):
        def rank(row):
            value = math.inf if row[metric] is None else row[metric]
            return (-value if descending else value, row["case_id"])
        rankings[label] = [{"case_id": r["case_id"], "group": r["group"], metric: r[metric],
                            "hd95_infinite": r["hd95_infinite"]} for r in sorted(rows, key=rank)[:10]]
    shifts = {}
    for metric in PRIMARY:
        old, new = (cohorts[g]["metrics"][metric]["mean"] for g in GROUPS[:2])
        shifts[metric] = None if old is None or new is None else new - old
    return {
        "schema_version": "1.0", "block": 59, "status": "complete", "scope": SCOPE,
        "contract_sha256": contract["contract_sha256"], "gpu": contract["runtime_identity"],
        "reported_gpu_total_gib": gpu_gib, "cohorts": cohorts, "review_rankings": rankings,
        "additional_minus_historical_mean": shifts,
        "shift_interpretation": "descriptive cohort shift; not a causal comparison or an independent development-vs-historical test",
        "uncertainty": {"method": "case bootstrap percentile CI for the arithmetic mean", "repetitions": reps,
                        "base_seed": seed, "quantile_method": "linear", "sd_ddof": 1,
                        "singleton_ci": "not estimated", "hd95_statistics": "finite subset only; infinite count reported separately"},
        "runtime_scope": "original successful case timings; sum excludes acquisition, probes, failed attempts and later cache verification",
        "memory_scope": "PyTorch allocator counters; not a measurement of physical VRAM residency or shared system RAM",
        "analysis_python": platform.python_version(), "analysis_numpy": np.__version__,
        "analysis_module_sha256": sha256_file(Path(__file__)),
        "gpu_inference_performed": False,
    }


def _fmt(value, digits=4):
    return "NA" if value is None else f"{value:.{digits}f}"


def markdown_report(summary: dict) -> str:
    lines = ["# Full489: development baseline — Block 59", "", SCOPE, "",
             f"Contract: `{summary['contract_sha256']}`", "",
             "## Cohort estimates", "",
             "| Cohort | n | Dice mean (95% bootstrap CI) | Surface Dice mean | HD95 finite median, mm | HD95 infinite n | Cyst-positive Dice mean (n) |",
             "| --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for group, cohort in summary["cohorts"].items():
        metrics = cohort["metrics"]
        dice, hd = metrics["kidney_mass_dice"], metrics["hd95_mm"]
        interval = dice["bootstrap_mean_95ci"]
        ci = "NA" if interval is None else f"{_fmt(interval[0])}–{_fmt(interval[1])}"
        lines.append(f"| {group} | {cohort['n']} | {_fmt(dice['mean'])} ({ci}) | "
                     f"{_fmt(metrics['surface_dice']['mean'])} | {_fmt(hd['median'], 2)} | {hd['n_nonfinite']} | "
                     f"{_fmt(metrics['cyst_positive_dice']['mean'])} ({cohort['cyst_positive_n']}) |")
    lines += ["", "Intervals describe case-sampling uncertainty under an independence assumption. "
              "HD95 summaries exclude positive infinity numerically and retain its count. "
              "Cyst-positive Dice excludes all reference-negative cases. Group differences are exploratory.",
              "", "## Runtime and memory", "",
              f"GPU: {summary['gpu']['gpu_name']}; reported total: {summary['reported_gpu_total_gib']:.2f} GiB.", "",
              "| Cohort | Case median (s) | Case p95 (s) | Case maximum (s) | Maximum reserved (GiB) | Cases reserved above GPU total |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for group, cohort in summary["cohorts"].items():
        timing = cohort["runtime"]["case_total_seconds"]
        lines.append(f"| {group} | {_fmt(timing['median'], 2)} | {_fmt(timing['p95'], 2)} | "
                     f"{_fmt(timing['max'], 2)} | {_fmt(cohort['runtime']['peak_reserved_gib']['max'], 2)} | "
                     f"{cohort['peak_reserved_above_reported_gpu_total_n']} |")
    lines += ["", summary["runtime_scope"] + ".", "", summary["memory_scope"] + ".", "",
              "## Cases to review", "", "Rankings prioritize development review; they do not define clinical failure thresholds."]
    for name, ranking in summary["review_rankings"].items():
        lines += ["", f"### {name}", "", "| Case | Group | Value |", "| --- | --- | ---: |"]
        for row in ranking:
            metric = next(k for k in row if k not in ("case_id", "group", "hd95_infinite"))
            value = "+Infinity" if row[metric] is None and metric == "hd95_mm" else _fmt(row[metric])
            lines.append(f"| {row['case_id']} | {row['group']} | {value} |")
    lines += ["", "## Verification boundary", "",
              "Completion receipts and all four referenced output artifacts were hash-verified. "
              "Frozen input identities were cross-checked; original CT/GT bytes and model weights were not reopened. "
              "No metric was recomputed from voxels. Validation and test outputs were not opened.", "",
              "![Development distributions](development_distributions.png)", ""]
    return "\n".join(lines)


def plot_distributions(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ("#245781", "#c36a28")
    for group, color in zip(GROUPS[:2], colors):
        selected = [r for r in rows if r["group"] == group]
        for axis, metric, title in zip(axes.flat, PRIMARY, ("Kidney-and-mass Dice", "Surface Dice", "HD95 (mm; +inf outside axis)")):
            values = sorted(r[metric] for r in selected if r[metric] is not None)
            if values:
                # Keep infinite outcomes in the ECDF denominator.
                axis.step(values, np.arange(1, len(values) + 1) / len(selected), where="post",
                          label=f"{group} (n={len(selected)})", color=color)
            axis.set(xlabel=title, ylabel="Cumulative fraction", ylim=(0, 1.02))
            axis.grid(alpha=.2)
        if selected:
            axes[1, 1].scatter([r["input_grid_voxels"] / 1e6 for r in selected],
                               [r["case_total_seconds"] for r in selected],
                               label=group, s=20, alpha=.65, color=color)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set(xlabel="Original grid (million voxels)", ylabel="Recorded case total (s)")
    axes[1, 1].grid(alpha=.2)
    fig.suptitle("Development baseline: historical and additional cases")
    fig.savefig(output / "development_distributions.png", dpi=180)
    fig.savefig(output / "development_distributions.pdf")
    plt.close(fig)


def export_report(output: Path, rows: list[dict], audit: dict, contract: dict,
                  *, reps: int = 5000, seed: int = 20260905) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "integrity_audit.json", audit)
    if not rows or audit["status"] != "complete" or len(rows) != len(audit["requested_case_ids"]):
        raise ValueError(f"Incomplete or invalid development results; see {output / 'integrity_audit.json'}")
    summary = summarize(rows, contract, reps=reps, seed=seed)
    atomic_json(output / "source_contract.json", contract)
    with (output / "case_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flat = []
    for group, cohort in summary["cohorts"].items():
        for metric, stats in cohort["metrics"].items():
            interval = stats["bootstrap_mean_95ci"] or [None, None]
            flat.append({"cohort": group, "metric": metric,
                         **{k: v for k, v in stats.items() if k != "bootstrap_mean_95ci"},
                         "mean_ci95_low": interval[0], "mean_ci95_high": interval[1]})
    with (output / "cohort_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (output / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    plot_distributions(rows, output)
    # Publish a complete summary only after all human-readable exports succeed.
    atomic_json(output / "summary.json", summary)
    names = ("integrity_audit.json", "source_contract.json", "case_metrics.csv", "cohort_summary.csv",
             "report.md", "development_distributions.png", "development_distributions.pdf", "summary.json")
    with zipfile.ZipFile(output / "block59_development_report.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(output / name, arcname=name)
    return summary
