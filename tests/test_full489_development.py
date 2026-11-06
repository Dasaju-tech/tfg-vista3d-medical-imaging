from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import subprocess
import zipfile

import nibabel as nib
import pytest

from vista3d_tfg.analysis import full489_development as analysis
from vista3d_tfg.experiments import full489_execution as execution
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.integration.nv_inference import sha256_file
from test_full489_execution import runtime, script, write_case


REAL_SUBPROCESS_RUN = subprocess.run


@pytest.fixture
def completed(runtime, monkeypatch):
    # The runtime fixture fakes Git through subprocess.run. Font discovery in
    # real Matplotlib needs the actual subprocess implementation again.
    monkeypatch.setattr(execution.subprocess, "run", REAL_SUBPROCESS_RUN)
    # Small synthetic groups exercise the same grouping used for the real 100+189.
    runtime.protocol["cohorts"]["historical100"]["case_ids"] = ["case_00000"]
    for number in range(2):
        write_case(runtime.root, number)
        execution.execute_case(runtime.root, runtime.run_root, runtime.contract,
                               f"case_{number:05d}", "vista3d-nv")
    return runtime


def change_evaluation(completed, cid, mutate):
    case_root = completed.run_root / "cases" / cid
    receipt = execution.read_json(case_root / "complete.json")
    key = next(k for k in receipt["artifacts"] if k.endswith("supported_renal_metrics.json"))
    path = case_root / key
    payload = execution.read_json(path)
    mutate(payload["result"], receipt)
    # The old metric format can represent +Infinity; the new summaries cannot.
    path.write_text(json.dumps(payload), encoding="utf-8")
    receipt["artifacts"][key] = sha256_file(path)
    receipt["receipt_sha256"] = canonical_sha256({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    execution.atomic_json(case_root / "complete.json", receipt)


def test_collection_reads_saved_outputs_without_loading_voxels_or_running_gpu(completed, monkeypatch):
    # Reserved output content is deliberately invalid and must never be opened.
    reserved = completed.run_root / "cases/case_00102/complete.json"
    reserved.parent.mkdir(parents=True)
    reserved.write_text("not JSON", encoding="utf-8")
    before = {p: p.read_bytes() for p in completed.run_root.rglob("*") if p.is_file()}
    monkeypatch.setattr(nib, "load", lambda *a, **kw: pytest.fail("Must not read NIfTI voxel arrays"))
    monkeypatch.setattr(execution, "run_worker", lambda *a, **kw: pytest.fail("Must not execute GPU worker"))
    rows, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    assert audit["status"] == "complete" and not audit["validation_test_outputs_opened"]
    assert [r["group"] for r in rows] == list(analysis.GROUPS[:2])
    assert [r["kidney_mass_dice"] for r in rows] == [1., 1.]
    assert all(r["cyst_positive"] for r in rows)
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("artifact", ["prediction.nii.gz", "supported_renal_metrics.json", "job.json", "worker.json", "complete.json"])
def test_corrupt_artifact_is_reported_and_no_aggregate_is_published(completed, tmp_path, artifact):
    case_root = completed.run_root / "cases/case_00001"
    target = next(case_root.rglob(artifact))
    target.write_bytes(target.read_bytes() + b"corrupt")
    rows, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    assert audit["status"] == "failed" and set(audit["failed_cases"]) == {"case_00001"}
    assert len(audit["requested_case_ids"]) == 2 and len(rows) == 1
    output = tmp_path / "failed_export"
    with pytest.raises(ValueError, match="Incomplete or invalid"):
        analysis.export_report(output, rows, audit, completed.contract)
    assert (output / "integrity_audit.json").exists()
    assert not (output / "summary.json").exists()
    assert not (output / "block59_development_report.zip").exists()


def test_missing_receipt_preserves_requested_denominator(completed):
    (completed.run_root / "cases/case_00001/complete.json").unlink()
    rows, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    assert len(rows) == 1 and len(audit["requested_case_ids"]) == 2
    assert "Missing completion" in audit["failed_cases"]["case_00001"]


def test_resigned_but_disagreeing_metrics_are_rejected(completed):
    change_evaluation(completed, "case_00001", lambda result, receipt: result["kidney_and_mass_hec"].update(dice=.2))
    _, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    assert "metrics disagree" in audit["failed_cases"]["case_00001"]


@pytest.mark.parametrize("bad", [float("nan"), float("-inf"), -1.])
def test_invalid_hd95_is_not_silently_filtered(completed, bad):
    change_evaluation(completed, "case_00001", lambda result, receipt: result["kidney_and_mass_hec"].update(diagnostic_hd95_mm=bad))
    _, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    assert "Invalid HD95" in audit["failed_cases"]["case_00001"]


def test_positive_infinite_hd95_is_retained_counted_and_ranked_first(completed):
    def infinite(result, receipt):
        result["kidney_and_mass_hec"]["diagnostic_hd95_mm"] = math.inf
        receipt["kidney_mass"].update(hd95_mm=None, hd95_nonfinite=True)
    change_evaluation(completed, "case_00001", infinite)
    rows, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    assert audit["status"] == "complete"
    summary = analysis.summarize(rows, completed.contract, reps=200)
    hd = summary["cohorts"]["development289"]["metrics"]["hd95_mm"]
    assert (hd["n_total"], hd["n_finite"], hd["n_nonfinite"]) == (2, 1, 1)
    assert hd["median"] == 0.
    assert summary["review_rankings"]["highest_hd95"][0]["case_id"] == "case_00001"
    assert "+Infinity" in analysis.markdown_report(summary)
    json.dumps(summary, allow_nan=False)


def test_cyst_negative_cases_do_not_inflate_positive_dice(completed):
    rows, _ = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    rows[0].update(cyst_positive=True, cyst_dice=.2)
    rows[1].update(cyst_positive=False, cyst_dice=1., cyst_prediction_present=True, cyst_false_positive_ml=2.)
    summary = analysis.summarize(rows, completed.contract, reps=200)
    combined = summary["cohorts"]["development289"]
    assert combined["cyst_positive_n"] == 1
    assert combined["metrics"]["cyst_positive_dice"]["mean"] == .2
    assert combined["cyst_negative_with_prediction_n"] == 1
    assert summary["cohorts"]["additional_development189"]["metrics"]["cyst_positive_dice"]["mean"] is None


def test_descriptive_statistics_bootstrap_reproducibility_and_small_denominators():
    stats = analysis.describe([1., 2., 3., None], reps=1000, seed=7)
    assert stats["mean"] == 2. and stats["sd"] == 1. and stats["median"] == 2.
    assert stats["n_nonfinite"] == 1 and stats["q1"] == 1.5 and stats["q3"] == 2.5
    assert stats == analysis.describe([1., 2., 3., None], reps=1000, seed=7)
    assert 1 <= stats["bootstrap_mean_95ci"][0] < stats["bootstrap_mean_95ci"][1] <= 3
    assert analysis.describe([1.], reps=1000)["bootstrap_mean_95ci"] is None
    assert analysis.describe([None], reps=1000)["mean"] is None


def test_nonoverlapping_group_counts_runtime_scope_and_memory_exceedances(completed):
    rows, _ = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    rows[0].update(kidney_mass_dice=.2, peak_reserved_gib=14.)
    rows[1].update(kidney_mass_dice=.8, peak_reserved_gib=3.)
    summary = analysis.summarize(rows, completed.contract, reps=200)
    cohorts = summary["cohorts"]
    assert [cohorts[g]["n"] for g in analysis.GROUPS] == [1, 1, 2]
    assert cohorts["development289"]["metrics"]["kidney_mass_dice"]["mean"] == .5
    assert summary["additional_minus_historical_mean"]["kidney_mass_dice"] == pytest.approx(.6)
    assert cohorts["development289"]["peak_reserved_above_reported_gpu_total_n"] == 1
    assert cohorts["development289"]["sum_recorded_successful_case_seconds"] == sum(r["case_total_seconds"] for r in rows)


def test_contract_and_partition_guards_fail_before_case_reads(completed, monkeypatch):
    monkeypatch.setattr(analysis, "read_case", lambda *a, **kw: pytest.fail("Should reject before cases"))
    bad = deepcopy(completed.contract)
    bad["runtime_identity"]["gpu_name"] = "changed"
    with pytest.raises(ValueError, match="checksum changed"):
        analysis.collect_development(completed.run_root, bad, completed.protocol)
    bad_protocol = deepcopy(completed.protocol)
    bad_protocol["partition"]["development"]["case_ids"].append("case_00102")
    with pytest.raises(ValueError, match="reserved"):
        analysis.collect_development(completed.run_root, completed.contract, bad_protocol)


def test_export_contains_only_compact_reports_and_figures(completed, tmp_path):
    rows, audit = analysis.collect_development(completed.run_root, completed.contract, completed.protocol)
    output = tmp_path / "export"
    summary = analysis.export_report(output, rows, audit, completed.contract, reps=200)
    assert execution.read_json(output / "summary.json") == summary
    with (output / "case_metrics.csv").open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert (output / "development_distributions.png").stat().st_size > 1000
    assert (output / "development_distributions.pdf").read_bytes().startswith(b"%PDF")
    with zipfile.ZipFile(output / "block59_development_report.zip") as archive:
        assert len(archive.namelist()) == 8
        assert "case_metrics.csv" in archive.namelist()
        assert all("/" not in name and not name.endswith(".nii.gz") for name in archive.namelist())
    with pytest.raises(FileExistsError):
        analysis.export_report(output, rows, audit, completed.contract, reps=200)


def test_cli_export_keeps_inference_identity_and_does_not_start_worker(completed, monkeypatch, capsys):
    module = script("analyze_nv_segment_ct_full489_development")
    monkeypatch.setattr(module, "verified_protocol", lambda root: completed.protocol)
    calls = len(completed.calls)
    contract_bytes = (completed.run_root / "contract.json").read_bytes()
    assert module.main(["--run-id", completed.run_root.name, "--project-root", str(completed.root),
                        "--bootstrap-reps", "1000"]) == 0
    assert "analysis PASSED" in capsys.readouterr().out
    assert len(completed.calls) == calls
    assert (completed.run_root / "contract.json").read_bytes() == contract_bytes
    assert not (completed.root / "outputs/full489/execution.lock").exists()


def test_cli_failure_emits_audit_without_successful_summary(completed, monkeypatch, capsys):
    module = script("analyze_nv_segment_ct_full489_development")
    monkeypatch.setattr(module, "verified_protocol", lambda root: completed.protocol)
    (completed.run_root / "cases/case_00000/complete.json").unlink()
    assert module.main(["--run-id", completed.run_root.name, "--project-root", str(completed.root)]) == 1
    assert "analysis PASSED" not in capsys.readouterr().out
    reports = completed.root / "outputs/reporting/full489_development"
    assert len(list(reports.rglob("integrity_audit.json"))) == 1
    assert not list(reports.rglob("summary.json"))


def test_cli_rejects_run_path_traversal_before_io(tmp_path, monkeypatch):
    module = script("analyze_nv_segment_ct_full489_development")
    monkeypatch.setattr(module, "verified_protocol", lambda *a: pytest.fail("Unexpected I/O"))
    with pytest.raises(SystemExit) as exc:
        module.main(["--run-id", "../../test", "--project-root", str(tmp_path)])
    assert exc.value.code == 2
