from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import zipfile

import numpy as np
import pytest

from vista3d_tfg.analysis import full489_cpu as cpu, full489_review as review
from vista3d_tfg.analysis.full489_development import collect_development
from vista3d_tfg.experiments import full489_execution as execution
from vista3d_tfg.metrics import morphology
from test_full489_development import completed
from test_full489_execution import runtime, script


@pytest.mark.parametrize("mask", [
    np.zeros((0, 2, 3), dtype=bool), np.ones((2, 3, 4), dtype=bool),
    np.arange(120).reshape(4, 5, 6)[::-1, ::2, :] % 2 == 0,
    np.ones((2, 3, 4), dtype=np.uint8), np.zeros((2, 3, 4), dtype=float),
    np.array([0, 1], dtype=object).reshape(1, 1, 2),
    np.array([0, 2]).reshape(1, 1, 2), np.array([0, -.1]).reshape(1, 1, 2),
    np.array([0, math.nan]).reshape(1, 1, 2), np.array([0, math.inf]).reshape(1, 1, 2),
    np.ones((2, 3), dtype=bool), np.ones((1, 2, 3, 4), dtype=bool),
])
def test_validator_matches_original_acceptance_and_errors(mask):
    def outcome(function):
        try:
            function(mask)
            return None
        except Exception as exc:
            return type(exc), str(exc)
    assert outcome(cpu.validate_binary_mask_fast) == outcome(cpu._ORIGINAL_VALIDATOR)


def test_boolean_shortcut_avoids_unique_without_mutating_input(monkeypatch):
    mask = np.ones((5, 6, 7), dtype=bool)[::-1, ::2, :]
    before = mask.copy()
    monkeypatch.setattr(np, "unique", lambda *a, **kw: pytest.fail("Boolean validation must not sort"))
    cpu.validate_binary_mask_fast(mask)
    assert np.array_equal(mask, before)


def test_numeric_arrays_and_subclasses_keep_original_validation(monkeypatch):
    seen = []
    class ArraySubclass(np.ndarray):
        pass
    monkeypatch.setattr(cpu, "_ORIGINAL_VALIDATOR", lambda x: seen.append(x))
    values = [np.ones((2, 3, 4), dtype=int), np.ones((2, 3, 4), dtype=bool).view(ArraySubclass)]
    for value in values:
        cpu.validate_binary_mask_fast(value)
    assert len(seen) == 2 and all(a is b for a, b in zip(seen, values))


def test_context_restores_baseline_on_error_and_interrupt_and_rejects_nesting():
    original = morphology.validate_binary_mask
    for error in (ValueError, KeyboardInterrupt):
        with pytest.raises(error):
            with cpu.evaluation_mode(True):
                assert morphology.validate_binary_mask is cpu.validate_binary_mask_fast
                with pytest.raises(RuntimeError, match="concurrently or nest"):
                    with cpu.evaluation_mode(False):
                        pass
                raise error()
        assert morphology.validate_binary_mask is original
    with cpu.evaluation_mode(False):
        assert morphology.validate_binary_mask is original


@pytest.fixture
def ready(completed, monkeypatch):
    # Block 60 figure rendering is already tested there. This fixture needs only
    # verifiable review artifacts; evaluation and all hashes remain real.
    def render(image, gt, pred, spacing, output, cid):
        for name in ("renal_mass", "cyst"):
            (output / f"{name}_review.png").write_bytes(b"synthetic review artifact")
        return {}
    monkeypatch.setattr(review, "render_review", render)
    data = completed
    rows, audit = collect_development(data.run_root, data.contract, data.protocol)
    assert audit["status"] == "complete"
    review_root, review_plan = review.prepare_review(data.root, rows, data.contract)
    state = review.run_review(data.root, data.run_root, review_root, review_plan, data.contract)
    assert state["status"] == "complete"
    source = cpu.check_completed_review(review_root, rows, data.contract)
    output, plan = cpu.prepare_benchmark(data.root, data.contract, source)
    return data, rows, review_root, output, plan


def test_real_evaluation_balanced_pairs_full_parity_and_baseline_files_unchanged(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    before = {p: p.read_bytes() for folder in (data.run_root, review_root) for p in folder.rglob("*") if p.is_file()}
    monkeypatch.setattr(execution, "run_worker", lambda *a, **kw: pytest.fail("No GPU inference"))
    original = cpu.evaluate_supported_renal_prediction
    modes = []
    def evaluate(**kwargs):
        modes.append(morphology.validate_binary_mask is cpu.validate_binary_mask_fast)
        return original(**kwargs)
    monkeypatch.setattr(cpu, "evaluate_supported_renal_prediction", evaluate)
    cid = plan["case_ids"][0]
    receipt, reused = cpu.benchmark_case(data.root, data.run_root, output, cid, plan, data.contract)
    assert not reused and receipt["metric_parity"] is True
    assert modes == [False, True, True, False]
    report = execution.read_json(output / "cases" / cid / next(iter(receipt["artifacts"])))
    assert len(report["measurements"]) == 4
    assert all(m["result"] == report["measurements"][0]["result"] for m in report["measurements"])
    assert all(m["seconds"] > 0 for m in report["measurements"])
    assert before == {p: p.read_bytes() for p in before}
    assert morphology.validate_binary_mask is cpu._ORIGINAL_VALIDATOR


def test_budget_resume_balances_other_case_and_exports_only_reports(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    first, archive = cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract, max_cases=1)
    assert first["status"] == "partial" and len(first["completed_case_ids"]) == 1
    original = cpu.evaluate_supported_renal_prediction
    modes = []
    def evaluate(**kwargs):
        modes.append(morphology.validate_binary_mask is cpu.validate_binary_mask_fast)
        return original(**kwargs)
    monkeypatch.setattr(cpu, "evaluate_supported_renal_prediction", evaluate)
    second, archive = cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract, max_cases=1)
    assert second["status"] == "complete" and len(second["newly_benchmarked_case_ids"]) == 1
    assert modes == [True, False, False, True]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert sum(n.endswith('/benchmark.json') for n in z.namelist()) == 2
        assert "measurements.csv" in z.namelist() and "case_timings.csv" in z.namelist()
        assert not any(n.endswith((".nii.gz", ".png", ".pt")) for n in z.namelist())


def test_optimized_metric_drift_fails_and_restores_validator(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    original = cpu.evaluate_supported_renal_prediction
    class Result:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    def changed(**kwargs):
        payload = original(**kwargs).to_dict()
        if morphology.validate_binary_mask is cpu.validate_binary_mask_fast:
            payload["kidney_and_mass_hec"]["dice"] = .123
        return Result(payload)
    monkeypatch.setattr(cpu, "evaluate_supported_renal_prediction", changed)
    state, archive = cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract)
    assert state["status"] == "failed" and len(state["failed_cases"]) == 1
    assert len(state["deferred_case_ids"]) == 1 and not state["completed_case_ids"]
    assert not list((output / "cases").glob("*/complete.json"))
    assert morphology.validate_binary_mask is cpu._ORIGINAL_VALIDATOR
    with zipfile.ZipFile(archive) as z:
        assert any(n.endswith('parity_failure.json') for n in z.namelist())


def test_cached_benchmark_rechecks_raw_inputs_and_artifact_hashes(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    cid = plan["case_ids"][0]
    receipt, _ = cpu.benchmark_case(data.root, data.run_root, output, cid, plan, data.contract)
    monkeypatch.setattr(cpu, "evaluate_supported_renal_prediction", lambda **kw: pytest.fail("Cache must not evaluate"))
    assert cpu.benchmark_case(data.root, data.run_root, output, cid, plan, data.contract)[1]
    report = output / "cases" / cid / next(iter(receipt["artifacts"]))
    before = report.read_bytes()
    report.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="artifact changed"):
        cpu.benchmark_case(data.root, data.run_root, output, cid, plan, data.contract)
    report.write_bytes(before)
    image = data.root / "data/raw/kits23" / cid / "imaging.nii.gz"
    image.write_bytes(image.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        cpu.benchmark_case(data.root, data.run_root, output, cid, plan, data.contract)


def test_incomplete_or_corrupt_block60_is_rejected(ready):
    data, rows, review_root, output, plan = ready
    path = review_root / "review_progress.json"
    state = execution.read_json(path)
    execution.atomic_json(path, {**state, "status": "partial"})
    with pytest.raises(ValueError, match="Complete Block 60"):
        cpu.check_completed_review(review_root, rows, data.contract)
    execution.atomic_json(path, state)
    png = next((review_root / "cases").rglob('cyst_review.png'))
    png.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        cpu.check_completed_review(review_root, rows, data.contract)


def test_plan_source_and_environment_changes_fail_before_evaluation(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    monkeypatch.setattr(cpu, "evaluate_supported_renal_prediction", lambda **kw: pytest.fail("Must fail before evaluation"))
    changed = deepcopy(plan)
    changed["case_ids"].append("case_00102")
    with pytest.raises(ValueError, match="checksum"):
        cpu.run_benchmark(data.root, data.run_root, output, changed, data.contract)
    original = cpu.machine_identity
    monkeypatch.setattr(cpu, "machine_identity", lambda: {**original(), "logical_cpus": -1})
    with pytest.raises(ValueError, match="environment changed"):
        cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract)
    monkeypatch.setattr(cpu, "machine_identity", original)
    path = data.root / "src/vista3d_tfg/metrics/morphology.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Evaluation source differs"):
        cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract)


def test_cli_executes_and_reuses_with_spaces_and_no_reserved_output_reads(ready, monkeypatch, capsys):
    data, rows, review_root, output, plan = ready
    reserved = data.run_root / "cases/case_00102/complete.json"
    reserved.parent.mkdir(parents=True)
    reserved.write_text("invalid reserved output")
    module = script("benchmark_nv_segment_ct_full489_cpu")
    monkeypatch.setattr(module, "verified_protocol", lambda root: data.protocol)
    args = ["--run-id", data.run_root.name, "--review-id", review_root.name,
            "--project-root", str(data.root), "--max-cases", "1"]
    assert module.main(args) == 0
    assert "Status: partial" in capsys.readouterr().out
    assert module.main(args) == 0
    assert "Status: complete" in capsys.readouterr().out
    assert not (data.root / "outputs/full489/execution.lock").exists()


@pytest.mark.parametrize("extra", [["--partition", "test"], ["--repeats", "1"], ["--repeats", "3"], ["--max-cases", "0"]])
def test_cli_rejects_reserved_partition_and_invalid_budgets(extra):
    module = script("benchmark_nv_segment_ct_full489_cpu")
    with pytest.raises(SystemExit) as exc:
        module.main(["--run-id", "a" * 16, "--review-id", "b" * 16, *extra])
    assert exc.value.code == 2


def test_nonfinite_storage_is_explicit_and_never_masks_nan():
    assert cpu.json_metrics({"hd95": math.inf, "shape": (1, 2, 3)}) == {"hd95": "+Infinity", "shape": [1, 2, 3]}
    for value in (math.nan, -math.inf):
        with pytest.raises(ValueError, match="Nonfinite"):
            cpu.json_metrics(value)


def test_loaded_package_copy_is_checked_against_original_contract(ready, monkeypatch, tmp_path):
    data, rows, review_root, output, plan = ready
    other = tmp_path / "other_morphology.py"
    other.write_text("# a different installed copy\n")
    monkeypatch.setattr(morphology, "__file__", str(other))
    with pytest.raises(ValueError, match="Loaded evaluation source differs"):
        cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract)


def test_input_change_during_comparison_prevents_completion(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    cid = plan["case_ids"][0]
    image = data.root / "data/raw/kits23" / cid / "imaging.nii.gz"
    original = cpu.evaluate_supported_renal_prediction
    def evaluate(**kwargs):
        result = original(**kwargs)
        image.write_bytes(image.read_bytes() + b"changed")
        return result
    monkeypatch.setattr(cpu, "evaluate_supported_renal_prediction", evaluate)
    state, _ = cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract, max_cases=1)
    assert state["status"] == "failed" and cid in state["failed_cases"]
    assert not (output / "cases" / cid / "complete.json").exists()


def test_continue_on_error_keeps_full_denominator(ready, monkeypatch):
    data, rows, review_root, output, plan = ready
    original = cpu.benchmark_case
    def benchmark(root, run_root, output, cid, plan, contract):
        if cid == plan["case_ids"][0]:
            raise OSError("synthetic read failure")
        return original(root, run_root, output, cid, plan, contract)
    monkeypatch.setattr(cpu, "benchmark_case", benchmark)
    state, _ = cpu.run_benchmark(data.root, data.run_root, output, plan, data.contract, continue_on_error=True)
    assert state["status"] == "failed" and len(state["requested_case_ids"]) == 2
    assert len(state["failed_cases"]) == len(state["completed_case_ids"]) == 1
    assert not state["deferred_case_ids"]
