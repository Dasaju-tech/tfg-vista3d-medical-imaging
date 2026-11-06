from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PureWindowsPath
import shutil

import pytest

from vista3d_tfg.data.acquisition import download_kits23_case
from vista3d_tfg.experiments import full489_acquisition as acquisition
from vista3d_tfg.experiments import full489_execution as execution
from test_full489_execution import runtime, script, write_case


@pytest.fixture
def cohort(tmp_path):
    for number in range(2):
        write_case(tmp_path, number)
    ids = ["case_00000", "case_00001"]
    protocol = {
        "protocol_sha256": "synthetic",
        "cohorts": {"all489": {"case_ids": ids}},
        "partition": {name: {"case_ids": ids} for name in ("development", "validation", "test")},
    }
    return tmp_path, protocol


def test_real_cached_downloader_records_are_json_serializable_and_resumable(cohort, monkeypatch):
    root, protocol = cohort
    records = download_kits23_case(0, root / "data/raw/kits23")
    assert len(records) == 2 and all(isinstance(row.path, Path) for row in records)
    before = {path: path.read_bytes() for path in (root / "data").rglob("*.nii.gz")}
    first = acquisition.acquire_partition(root, protocol, "development", max_cases=1)
    assert first["status"] == "partial" and first["completed_case_ids"] == ["case_00000"]
    ledger_path = root / "outputs/full489/acquisition.json"
    ledger = execution.read_json(ledger_path)
    saved = ledger["cases"]["case_00000"]["downloads"]
    assert len(saved) == 2
    for actual, expected in zip(saved, records):
        assert actual["path"] == str(expected.path)
        assert actual["sha256"] == expected.sha256
        assert actual["size_bytes"] == expected.size_bytes
        assert actual["status"] == "reused"
    second = acquisition.acquire_partition(root, protocol, "development", max_cases=1)
    assert second["status"] == "complete" and len(second["completed_case_ids"]) == 2
    monkeypatch.setattr(acquisition, "download_kits23_case", lambda *a, **kw: pytest.fail("Already verified"))
    assert acquisition.acquire_partition(root, protocol, "development")["status"] == "complete"
    assert before == {path: path.read_bytes() for path in before}


def test_windows_path_record_round_trips_exactly_on_any_host(cohort, monkeypatch):
    root, protocol = cohort
    records = download_kits23_case(0, root / "data/raw/kits23")
    windows = PureWindowsPath("C:/Users/Daniel/a folder/imaging.nii.gz")
    records = (replace(records[0], path=windows), records[1])
    monkeypatch.setattr(acquisition, "download_kits23_case", lambda *a, **kw: records)
    state = acquisition.acquire_partition(root, protocol, "development", max_cases=1)
    assert not state["failed_cases"]
    ledger = execution.read_json(root / "outputs/full489/acquisition.json")
    assert ledger["cases"]["case_00000"]["downloads"][0]["path"] == str(windows)


@pytest.mark.parametrize("continue_on_error", [False, True])
def test_write_failure_never_counts_or_commits_the_failed_case(cohort, monkeypatch, continue_on_error):
    root, protocol = cohort
    ledger_path = root / "outputs/full489/acquisition.json"
    original = execution.atomic_json
    def interrupted_write(path, payload):
        if path == ledger_path and "case_00000" in payload["cases"]:
            raise OSError("synthetic disk write failure")
        original(path, payload)
    monkeypatch.setattr(acquisition, "atomic_json", interrupted_write)
    state = acquisition.acquire_partition(root, protocol, "development", continue_on_error=continue_on_error)
    assert state["status"] == "failed" and set(state["failed_cases"]) == {"case_00000"}
    assert "case_00000" not in state["completed_case_ids"]
    if continue_on_error:
        assert state["completed_case_ids"] == ["case_00001"]
        assert set(execution.read_json(ledger_path)["cases"]) == {"case_00001"}
    else:
        assert not ledger_path.exists()
        assert state["completed_case_ids"] == []
        assert state["deferred_case_ids"] == ["case_00001"]
    monkeypatch.setattr(acquisition, "atomic_json", original)
    resumed = acquisition.acquire_partition(root, protocol, "development")
    assert resumed["status"] == "complete" and len(resumed["completed_case_ids"]) == 2


def test_existing_ledger_survives_a_later_write_failure(cohort, monkeypatch):
    root, protocol = cohort
    acquisition.acquire_partition(root, protocol, "development", max_cases=1)
    ledger_path = root / "outputs/full489/acquisition.json"
    before = ledger_path.read_bytes()
    original = execution.atomic_json
    def reject_ledger(path, payload):
        if path == ledger_path:
            raise OSError("synthetic failure")
        original(path, payload)
    monkeypatch.setattr(acquisition, "atomic_json", reject_ledger)
    state = acquisition.acquire_partition(root, protocol, "development")
    assert state["completed_case_ids"] == ["case_00000"]
    assert set(state["failed_cases"]) == {"case_00001"}
    assert ledger_path.read_bytes() == before


def test_duplicate_bytes_remain_blocked_in_the_repaired_acquisition(cohort):
    root, protocol = cohort
    data = root / "data/raw/kits23"
    shutil.copyfile(data / "case_00000/imaging.nii.gz", data / "case_00001/imaging.nii.gz")
    state = acquisition.acquire_partition(root, protocol, "development", continue_on_error=True)
    assert state["completed_case_ids"] == ["case_00000"]
    assert "Byte-identical" in state["failed_cases"]["case_00001"]


@pytest.mark.parametrize("partition", ["test", "all"])
def test_repaired_acquisition_of_reserved_inputs_does_not_infer(cohort, monkeypatch, partition):
    root, protocol = cohort
    monkeypatch.setattr(execution, "run_worker", lambda *a, **kw: pytest.fail("GPU must stay idle"))
    monkeypatch.setattr(execution, "evaluate_supported_renal_prediction",
                        lambda *a, **kw: pytest.fail("No metrics for reserved inputs"))
    result = acquisition.acquire_partition(root, protocol, partition)
    assert result["status"] == "complete"
    assert result["predictions_metrics_or_visual_qc_generated"] is False
    assert not list((root / "outputs").rglob("prediction.nii.gz"))


def test_cli_routes_to_repair_and_keeps_existing_runtime_identity(runtime, monkeypatch, capsys):
    write_case(runtime.root)
    module = script("run_nv_segment_ct_full489")
    assert module.acquire_partition is acquisition.acquire_partition
    monkeypatch.setattr(module, "verified_protocol", lambda root: runtime.protocol)
    before = (runtime.run_root / "contract.json").read_bytes()
    call_count = len(runtime.calls)
    assert module.main(["--stage", "acquire", "--max-cases", "1",
                        "--project-root", str(runtime.root)]) == 0
    assert "Completed: 1/2" in capsys.readouterr().out
    assert len(runtime.calls) == call_count
    assert (runtime.run_root / "contract.json").read_bytes() == before
    run_root, contract, _ = execution.prepare_run(runtime.root, runtime.protocol, "vista3d-nv")
    assert run_root == runtime.run_root and contract == runtime.contract
