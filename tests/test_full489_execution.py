from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments import full489_execution as execution
from vista3d_tfg.experiments.expanded100 import selection_manifest_payload
from vista3d_tfg.experiments.full489 import build_protocol, canonical_sha256
from vista3d_tfg.integration.nv_inference import EXPECTED_RENAL_LABELS, sha256_file
from vista3d_tfg.integration.nv_segment import runtime_paths


SOURCE = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_case(root, number=0, shape=(10, 10, 10)):
    path = root / "data/raw/kits23" / f"case_{number:05d}"
    path.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, number, dtype=np.int16)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[1:3, 2:5, 2:5] = 1
    labels[7:9, 2:5, 2:5] = 1
    labels[1, 3, 3] = 2
    labels[7, 3, 3] = 3
    affine = np.diag([1., 1., 2., 1.])
    nib.save(nib.Nifti1Image(image, affine), path / "imaging.nii.gz")
    nib.save(nib.Nifti1Image(labels, affine), path / "segmentation.nii.gz")
    return path


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    root = tmp_path / "project with spaces"
    bundle = runtime_paths(root).ct_bundle_root
    execution.atomic_json(bundle / "configs/inference.json", {
        "checkpointloader": {"load_path": "@checkpoint_path"}, "amp": True,
    })
    execution.atomic_json(bundle / "configs/label_dict.json", EXPECTED_RENAL_LABELS)
    (bundle / "models").mkdir()
    (bundle / "models/model.pt").write_bytes(b"synthetic checkpoint, never a real model")
    for folder in ("data", "metrics"):
        shutil.copytree(SOURCE / "src/vista3d_tfg" / folder,
                        root / "src/vista3d_tfg" / folder, ignore=shutil.ignore_patterns("__pycache__"))
    for relative in (
        "analysis/anatomy.py", "evaluation/nv_kits23.py", "integration/nv_inference.py",
        "integration/nv_segment.py", "experiments/full489.py", "experiments/full489_execution.py",
    ):
        target = root / "src/vista3d_tfg" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / "src/vista3d_tfg" / relative, target)
    (root / "scripts").mkdir()
    shutil.copyfile(SOURCE / execution.WORKER_FILE, root / execution.WORKER_FILE)
    identity = {
        "gpu_name": "synthetic GPU", "gpu_total_bytes": 12 * 2**30,
        "cuda": "synthetic", "torch": "synthetic", "monai": "synthetic",
    }
    calls = []

    def fake_worker(project, environment, job, folder):
        calls.append(deepcopy(job))
        execution.atomic_json(folder / "job.json", job)
        if job["mode"] == "probe":
            result = {"status": "complete", "identity": deepcopy(identity),
                      "fp32_fp16_conv3d_smoke": True}
        else:
            source = nib.load(job["image_path"])
            predicted = np.zeros(source.shape, dtype=np.uint8)
            predicted[1:3, 2:5, 2:5] = 5
            predicted[7:9, 2:5, 2:5] = 14
            predicted[7, 3, 3] = 116
            prediction_path = Path(job["output_dir"]) / "imaging/imaging_trans.nii.gz"
            prediction_path.parent.mkdir(parents=True)
            nib.save(nib.Nifti1Image(predicted, source.affine), prediction_path)
            result = {
                "status": "complete", "runtime_identity": deepcopy(job["runtime_identity"]),
                "checkpoint_sha256": job["checkpoint_sha256"], "image_sha256": job["image_sha256"],
                "bundle_wall_seconds": 2., "peak_allocated_bytes": 2**29,
                "peak_reserved_bytes": 2**30,
            }
        execution.atomic_json(folder / "worker.json", result)
        return {**result, "process_wall_seconds": 4.}

    monkeypatch.setattr(execution, "run_worker", fake_worker)
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="a" * 40 + "\n"))
    protocol = {
        "protocol_sha256": "f" * 64,
        "cohorts": {
            "historical100": {"case_ids": [f"case_{n:05d}" for n in range(100)]},
            "all489": {"case_ids": ["case_00000", "case_00001", "case_00101", "case_00102"]},
        },
        "partition": {
            "development": {"case_ids": ["case_00000", "case_00001"]},
            "validation": {"case_ids": ["case_00101"]},
            "test": {"case_ids": ["case_00102"]},
        },
    }
    run_root, contract, _ = execution.prepare_run(root, protocol, "vista3d-nv")
    return SimpleNamespace(root=root, bundle=bundle, run_root=run_root, contract=contract,
                           protocol=protocol, identity=identity, calls=calls, fake_worker=fake_worker)


@pytest.mark.parametrize("stage,partition", [
    ("run", "test"), ("run", "all"), ("profile", "test"),
    ("profile", "validation"), ("probe", "all"),
])
def test_cli_rejects_reserved_execution_before_any_side_effect(tmp_path, monkeypatch, stage, partition):
    module = script("run_nv_segment_ct_full489")
    monkeypatch.setattr(module, "verified_protocol", lambda *a: pytest.fail("Should reject before I/O"))
    with pytest.raises(SystemExit) as exc:
        module.main(["--stage", stage, "--partition", partition, "--project-root", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


def test_core_rejects_test_even_with_a_profile(runtime):
    with pytest.raises(ValueError, match="sealed"):
        execution.execute_batch(runtime.root, runtime.run_root, runtime.contract, runtime.protocol,
                                "vista3d-nv", phase="run", partition="test")
    assert len(runtime.calls) == 1  # Only the fixture's synthetic probe.
    assert execution.selected_cases(runtime.protocol, "test", acquire=True) == ["case_00102"]


def test_windows_safe_worker_command_keeps_paths_as_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "resolve_conda_executable", lambda: "C:/Conda Folder/conda.exe")
    root = tmp_path / "a folder"
    command = execution.worker_command(root, "vista3d-nv", root / "job.json", root / "worker.json")
    assert command[:6] == ["C:/Conda Folder/conda.exe", "run", "--no-capture-output", "-n",
                           "vista3d-nv", "python"]
    assert command[6] == str((root / execution.WORKER_FILE).resolve())
    assert command[-3:] == [str((root / "job.json").resolve()), "--report",
                            str((root / "worker.json").resolve())]
    assert all("\n" not in arg for arg in command)
    assert "-c" not in command


def test_protocol_requires_same_committed_json_and_preserves_bom(tmp_path):
    numbers = [n for n in range(493) if n not in {324, 345, 396, 398}]
    metadata = json.dumps([{"case_id": f"case_{n:05d}"} for n in numbers]).encode()
    baseline = selection_manifest_payload(official_case_numbers=numbers,
                                         metadata_sha256=sha256(metadata).hexdigest())
    protocol = build_protocol(metadata_raw=metadata, expanded100_raw=json.dumps(baseline).encode())
    execution.atomic_json(tmp_path / "configs/experiments/nv_segment_ct_expanded100_selection.json", baseline)
    snapshot = tmp_path / "outputs/protocols/nv_segment_ct_full489/kits23_metadata.json"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(metadata)
    target = tmp_path / execution.PROTOCOL_FILE
    target.write_text("\ufeff" + json.dumps(protocol), encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    with pytest.raises(ValueError, match="Commit the validated"):
        execution.verified_protocol(tmp_path)
    subprocess.run(["git", "-C", str(tmp_path), "add", execution.PROTOCOL_FILE], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-m", "synthetic freeze"], check=True, capture_output=True)
    assert execution.verified_protocol(tmp_path) == protocol
    original = target.read_bytes()
    snapshot.write_bytes(metadata + b"\n")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        execution.verified_protocol(tmp_path)
    assert target.read_bytes() == original


@pytest.mark.parametrize("change", ["checkpoint", "gpu", "config", "evaluation", "worker"])
def test_runtime_or_model_change_creates_separate_namespace(runtime, change):
    if change == "gpu":
        runtime.identity["gpu_name"] = "another synthetic GPU"
    else:
        path = {
            "checkpoint": runtime.bundle / "models/model.pt",
            "config": runtime.bundle / "configs/label_dict.json",
            "evaluation": runtime.root / "src/vista3d_tfg/metrics/binary.py",
            "worker": runtime.root / execution.WORKER_FILE,
        }[change]
        with path.open("ab") as stream:
            stream.write(b"\n")
    new_root, new_contract, _ = execution.prepare_run(runtime.root, runtime.protocol, "vista3d-nv")
    assert new_root != runtime.run_root
    assert new_contract["contract_sha256"] != runtime.contract["contract_sha256"]
    assert execution.read_json(runtime.run_root / "contract.json") == runtime.contract


def test_unchanged_probe_reuses_namespace(runtime):
    root, contract, _ = execution.prepare_run(runtime.root, runtime.protocol, "vista3d-nv")
    assert root == runtime.run_root
    assert contract == runtime.contract


def test_case_uses_real_geometry_and_metrics_then_verifies_cache(runtime):
    write_case(runtime.root)
    old = runtime.root / "outputs/predictions/nv_segment_ct/case_00000/prediction.nii.gz"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"historical artifact must remain unchanged")
    receipt, reused = execution.execute_case(runtime.root, runtime.run_root, runtime.contract,
                                             "case_00000", "vista3d-nv")
    assert not reused
    assert receipt["kidney_mass"]["dice"] == pytest.approx(1.)
    assert receipt["kidney_mass"]["surface_dice"] == pytest.approx(1.)
    assert receipt["memory"]["peak_reserved_bytes"] == 2**30
    assert receipt["timing"]["worker_process_seconds"] == 4.
    assert "ground_truth_path" not in runtime.calls[-1]
    assert "ground_truth_sha256" not in runtime.calls[-1]
    assert old.read_bytes() == b"historical artifact must remain unchanged"
    count = len(runtime.calls)
    restored, reused = execution.execute_case(runtime.root, runtime.run_root, runtime.contract,
                                              "case_00000", "vista3d-nv")
    assert reused and restored["receipt_sha256"] == receipt["receipt_sha256"]
    assert len(runtime.calls) == count


@pytest.mark.parametrize("changed", ["imaging.nii.gz", "segmentation.nii.gz", "prediction.nii.gz",
                                     "supported_renal_metrics.json", "worker.json", "job.json", "complete.json"])
def test_changed_inputs_or_cached_artifacts_are_not_silently_reused(runtime, changed):
    inputs = write_case(runtime.root)
    receipt, _ = execution.execute_case(runtime.root, runtime.run_root, runtime.contract,
                                        "case_00000", "vista3d-nv")
    case_root = runtime.run_root / "cases/case_00000"
    if changed in ("imaging.nii.gz", "segmentation.nii.gz"):
        target = inputs / changed
    elif changed == "complete.json":
        target = case_root / changed
        data = execution.read_json(target)
        data["kidney_mass"]["dice"] = 0.
        execution.atomic_json(target, data)
    else:
        target = case_root / next(p for p in receipt["artifacts"] if Path(p).name == changed)
    if changed != "complete.json":
        with target.open("ab") as stream:
            stream.write(b"changed")
    count = len(runtime.calls)
    with pytest.raises(ValueError, match="changed|identity"):
        execution.execute_case(runtime.root, runtime.run_root, runtime.contract, "case_00000", "vista3d-nv")
    assert len(runtime.calls) == count


def test_failed_attempt_is_retained_but_never_accepted_as_complete(runtime, monkeypatch):
    write_case(runtime.root)
    def failed_worker(root, environment, job, folder):
        runtime.fake_worker(root, environment, job, folder)
        raise RuntimeError("synthetic CUDA out of memory")
    monkeypatch.setattr(execution, "run_worker", failed_worker)
    with pytest.raises(RuntimeError, match="out of memory"):
        execution.execute_case(runtime.root, runtime.run_root, runtime.contract, "case_00000", "vista3d-nv")
    case_root = runtime.run_root / "cases/case_00000"
    assert not (case_root / "complete.json").exists()
    monkeypatch.setattr(execution, "run_worker", runtime.fake_worker)
    receipt, reused = execution.execute_case(runtime.root, runtime.run_root, runtime.contract,
                                             "case_00000", "vista3d-nv")
    assert not reused and receipt["status"] == "complete"
    assert len(list(case_root.glob("attempt_*"))) == 2


def test_worker_identity_mismatch_never_publishes_success(runtime, monkeypatch):
    write_case(runtime.root)
    def mismatched(*args):
        result = runtime.fake_worker(*args)
        result["checkpoint_sha256"] = "wrong"
        return result
    monkeypatch.setattr(execution, "run_worker", mismatched)
    with pytest.raises(ValueError, match="different checkpoint_sha256"):
        execution.execute_case(runtime.root, runtime.run_root, runtime.contract, "case_00000", "vista3d-nv")
    assert not (runtime.run_root / "cases/case_00000/complete.json").exists()


def test_invalid_prediction_geometry_is_rejected(runtime, monkeypatch):
    write_case(runtime.root)
    def wrong_geometry(*args):
        report = runtime.fake_worker(*args)
        path = Path(args[2]["output_dir"]) / "imaging/imaging_trans.nii.gz"
        image = nib.load(path)
        affine = image.affine.copy()
        affine[0, 3] += 10
        nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), path)
        return report
    monkeypatch.setattr(execution, "run_worker", wrong_geometry)
    with pytest.raises(ValueError, match="Prediction geometry"):
        execution.execute_case(runtime.root, runtime.run_root, runtime.contract, "case_00000", "vista3d-nv")
    assert not (runtime.run_root / "cases/case_00000/complete.json").exists()


def test_profile_is_fixed_uses_large_historical_grids_and_gates_cohort_execution(runtime, monkeypatch):
    for number in range(100):
        write_case(runtime.root, number, shape=(10, 10, 10 + number))
    plan = execution.profile_plan(runtime.root, runtime.protocol)
    assert plan["case_ids"][:5] == [f"case_{n:05d}" for n in range(99, 94, -1)]
    assert len(set(plan["case_ids"])) == 20
    assert set(plan["case_ids"]).issubset(runtime.protocol["cohorts"]["historical100"]["case_ids"])
    assert plan == execution.profile_plan(runtime.root, runtime.protocol)
    with pytest.raises(ValueError, match="Complete the 20-case"):
        execution.require_profile(runtime.root, runtime.run_root, runtime.contract, runtime.protocol)
    kwargs = dict(root=runtime.root, run_root=runtime.run_root, contract=runtime.contract,
                  protocol=runtime.protocol, environment="vista3d-nv", phase="profile")
    state = execution.execute_batch(**kwargs, max_cases=1)
    assert state["status"] == "partial" and state["completed_count"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        execution.require_profile(runtime.root, runtime.run_root, runtime.contract, runtime.protocol)
    state = execution.execute_batch(**kwargs)
    assert state["completed_count"] == 20 and state["status"] == "complete"
    assert len(state["new_inference_case_ids"]) == 19
    execution.require_profile(runtime.root, runtime.run_root, runtime.contract, runtime.protocol)
    summary = execution.read_json(runtime.run_root / "profile_summary.json")
    assert summary["maximum_peak_reserved_gib"] == 1.
    assert summary["statistical_test_or_clinical_validation"] is False
    cid = plan["case_ids"][0]
    (runtime.run_root / "cases" / cid / "complete.json").unlink()
    with pytest.raises(ValueError, match="Missing verified"):
        execution.require_profile(runtime.root, runtime.run_root, runtime.contract, runtime.protocol)


def test_batch_budget_moves_to_next_case_and_failure_keeps_denominator(runtime, monkeypatch):
    for number in range(2):
        write_case(runtime.root, number)
    # This test exercises batching; the separate profile test covers this gate.
    monkeypatch.setattr(execution, "require_profile", lambda *args: None)
    kwargs = dict(root=runtime.root, run_root=runtime.run_root, contract=runtime.contract,
                  protocol=runtime.protocol, environment="vista3d-nv", phase="run", max_cases=1)
    first = execution.execute_batch(**kwargs)
    assert first["status"] == "partial" and first["new_inference_case_ids"] == ["case_00000"]
    second = execution.execute_batch(**kwargs)
    assert second["status"] == "complete" and second["new_inference_case_ids"] == ["case_00001"]
    assert second["requested_count"] == 2
    assert sum(job["mode"] == "infer" for job in runtime.calls) == 2
    runtime.protocol["partition"]["development"]["case_ids"].append("case_00002")
    third = execution.execute_batch(**kwargs, continue_on_error=True)
    assert third["status"] == "failed"
    assert third["requested_count"] == 3 and third["completed_count"] == 2
    assert "case_00002" in third["failed_cases"]


def test_test_acquisition_only_audits_inputs_and_resumes(runtime, monkeypatch):
    downloaded = []
    def downloader(number, directory, **kwargs):
        downloaded.append(number)
        write_case(runtime.root, number)
        return ()
    monkeypatch.setattr(execution, "download_kits23_case", downloader)
    monkeypatch.setattr(execution, "evaluate_supported_renal_prediction",
                        lambda **kwargs: pytest.fail("Reserved metrics must not be calculated"))
    initial_calls = len(runtime.calls)
    first = execution.acquire_partition(runtime.root, runtime.protocol, "test")
    assert first["status"] == "complete"
    assert not first["predictions_metrics_or_visual_qc_generated"]
    assert execution.acquire_partition(runtime.root, runtime.protocol, "test")["status"] == "complete"
    assert downloaded == [102] and len(runtime.calls) == initial_calls
    assert not (runtime.run_root / "cases").exists()
    path = runtime.root / "data/raw/kits23/case_00102/imaging.nii.gz"
    path.write_bytes(path.read_bytes() + b"changed")
    result = execution.acquire_partition(runtime.root, runtime.protocol, "test")
    assert result["status"] == "failed"
    assert "changed" in result["failed_cases"]["case_00102"]
    assert downloaded == [102]


def test_acquisition_budget_resumes_and_reports_duplicate_bytes(runtime, monkeypatch):
    def downloader(number, directory, **kwargs):
        write_case(runtime.root, number)
        if number == 101:
            shutil.copyfile(runtime.root / "data/raw/kits23/case_00000/imaging.nii.gz",
                            runtime.root / "data/raw/kits23/case_00101/imaging.nii.gz")
        return ()
    monkeypatch.setattr(execution, "download_kits23_case", downloader)
    one = execution.acquire_partition(runtime.root, runtime.protocol, "development", max_cases=1)
    assert one["status"] == "partial" and one["completed_case_ids"] == ["case_00000"]
    two = execution.acquire_partition(runtime.root, runtime.protocol, "development", max_cases=1)
    assert two["status"] == "complete" and len(two["completed_case_ids"]) == 2
    duplicate = execution.acquire_partition(runtime.root, runtime.protocol, "validation")
    assert duplicate["status"] == "failed" and "Byte-identical" in duplicate["failed_cases"]["case_00101"]


def test_exclusive_lock_rejects_overlap_and_releases_after_error(tmp_path):
    with pytest.raises(RuntimeError, match="synthetic failure"):
        with execution.execution_lock(tmp_path):
            with pytest.raises(RuntimeError, match="lock already exists"):
                with execution.execution_lock(tmp_path):
                    pytest.fail("Concurrent execution entered the lock")
            raise RuntimeError("synthetic failure")
    assert not (tmp_path / "execution.lock").exists()


@pytest.mark.parametrize("returncode,status", [(1, "complete"), (0, "failed")])
def test_runtime_exit_or_report_failure_cannot_be_masked(tmp_path, monkeypatch, returncode, status):
    folder = tmp_path / "attempt"
    monkeypatch.setattr(execution, "resolve_conda_executable", lambda: "conda.exe")
    def invoke(command, root, timeout):
        execution.atomic_json(folder / "worker.json", {"status": status})
        return returncode
    monkeypatch.setattr(execution, "invoke_worker", invoke)
    with pytest.raises(RuntimeError):
        execution.run_worker(tmp_path, "vista3d-nv", {"mode": "probe"}, folder)


def test_timeout_terminates_worker_process(tmp_path):
    # Exercise the real process cleanup on the host, without Conda or a GPU.
    worker = tmp_path / "waiting_worker.py"
    worker.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    with pytest.raises(subprocess.TimeoutExpired):
        execution.invoke_worker([sys.executable, str(worker)], tmp_path, timeout=0.1)
    assert execution.invoke_worker([sys.executable, "-c", "raise SystemExit(0)"],
                                   tmp_path, timeout=5) == 0


def test_worker_calls_monai_with_only_image_prompt_and_frozen_checkpoint(runtime, monkeypatch):
    module = script("nv_segment_ct_full489_worker")
    path = write_case(runtime.root) / "imaging.nii.gz"
    job = {
        "mode": "infer", "bundle_root": str(runtime.bundle),
        "bundle_files": runtime.contract["bundle_files"],
        "checkpoint_sha256": runtime.contract["checkpoint_sha256"],
        "runtime_identity": runtime.identity, "image_path": str(path),
        "image_sha256": sha256_file(path), "label_prompt": [5, 14, 116, 117],
        "output_dir": str(runtime.run_root / "synthetic raw"),
    }
    monkeypatch.setattr(module, "runtime_snapshot", lambda **kw: {"identity": runtime.identity})
    # Preserve process state around the same chdir/sys.path operation used inside Conda.
    monkeypatch.chdir(runtime.root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda _: None, synchronize=lambda: None,
                           max_memory_allocated=lambda _: 100, max_memory_reserved=lambda _: 200)
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "monai.bundle", SimpleNamespace(run=lambda **kw: calls.append(kw)))
    result = module.execute(job)
    assert result["status"] == "complete" and result["peak_reserved_bytes"] == 200
    assert calls == [{"config_file": str(runtime.bundle / "configs/inference.json"),
                      "input_dict": {"image": path.as_posix(), "label_prompt": [5, 14, 116, 117]},
                      "output_dir": Path(job["output_dir"]).as_posix(),
                      "checkpoint_path": (runtime.bundle / "models/model.pt").as_posix()}]
    (runtime.bundle / "models/model.pt").write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        module.execute(job)
    assert len(calls) == 1


def test_cli_returns_failure_for_incomplete_case_and_prints_report(runtime, monkeypatch, capsys):
    module = script("run_nv_segment_ct_full489")
    monkeypatch.setattr(module, "verified_protocol", lambda root: runtime.protocol)
    monkeypatch.setattr(execution, "require_profile", lambda *args: None)
    code = module.main(["--stage", "run", "--project-root", str(runtime.root)])
    assert code == 1
    assert "Status: failed" in capsys.readouterr().out
    assert not (runtime.root / "outputs/full489/execution.lock").exists()
