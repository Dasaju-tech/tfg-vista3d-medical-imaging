from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.analysis import full489_review as review
from vista3d_tfg.analysis.full489_development import collect_development
from vista3d_tfg.experiments import full489_execution as execution
from test_full489_development import completed
from test_full489_execution import runtime, script


@pytest.fixture
def planned(completed):
    rows, audit = collect_development(completed.run_root, completed.contract, completed.protocol)
    assert audit["status"] == "complete"
    output, plan = review.prepare_review(completed.root, rows, completed.contract)
    return completed, rows, output, plan


def test_plan_is_deterministic_deduplicated_and_bound_to_sources(planned):
    data, rows, output, plan = planned
    assert plan == review.build_plan(rows, data.contract)
    assert len(plan["case_ids"]) == len(set(plan["case_ids"])) == 2
    assert set(plan["case_ids"]) <= set(data.protocol["partition"]["development"]["case_ids"])
    assert len(plan["selection_reasons"]["case_00001"]) > 1
    assert review.prepare_review(data.root, rows, data.contract) == (output, plan)
    changed = deepcopy(rows)
    changed[0]["receipt_sha256"] = "a" * 64
    assert review.build_plan(changed, data.contract)["plan_sha256"] != plan["plan_sha256"]


@pytest.mark.parametrize("positive,present,dice,expected", [
    (False, False, 1., "reference_negative_no_prediction"),
    (False, True, 0., "reference_negative_prediction"),
    (True, False, 0., "no_cyst_prediction"),
    (True, True, 0., "zero_overlap_with_prediction"),
])
def test_zero_overlap_is_separated_from_absent_prediction(planned, positive, present, dice, expected):
    row = deepcopy(planned[1][0])
    row.update(cyst_positive=positive, cyst_prediction_present=present, cyst_dice=dice)
    assert review.screen_case(row)["cyst_error_category"] == expected


def test_cyst_volume_bins_are_case_totals_with_explicit_boundaries(planned):
    base = planned[1][0]
    rows = []
    for index, volume in enumerate((.1, 1., 5., 20.)):
        rows.append({**base, "case_id": f"case_{index:05d}", "cyst_reference_ml": volume,
                     "cyst_false_negative_ml": volume, "cyst_dice": 0.})
    summary = review.cohort_screen(rows)["groups"]["development289"]
    assert [b["n"] for b in summary["cyst_volume_bins"]] == [1, 1, 1, 1]
    assert summary["cyst_positive_zero_overlap_n"] == 4


def test_component_overlap_diagnostics_include_reference_negative_predictions():
    reference = np.zeros((6, 6, 6), dtype=bool)
    reference[0, 0, 0] = reference[5, 5, 5] = True
    predicted = np.zeros_like(reference)
    predicted[0, 0, 0] = predicted[3, 3, 3] = True
    info = review.component_summary(reference, predicted, .002)
    assert info["reference"]["component_count"] == 2
    assert info["reference"]["components_without_overlap"] == 1
    assert info["prediction"]["components_without_overlap"] == 1
    assert info["prediction"]["largest_component_ml"] == .002
    info = review.component_summary(np.zeros_like(reference), predicted, .002)
    assert info["reference"]["component_count"] == 0
    assert info["prediction"]["components_without_overlap"] == 2


def test_parity_handles_tuple_json_lists_infinity_and_rejects_nan():
    assert review.differences({"shape": [1, 2], "hd95": math.inf}, {"shape": (1, 2), "hd95": math.inf}) == []
    assert review.differences({"dice": .9}, {"dice": .9 + 1e-11}) == []
    assert review.differences({"dice": .9}, {"dice": .8})
    assert review.differences(math.nan, math.nan)
    assert review.differences(True, 1)


def test_review_profiles_real_cpu_evaluation_renders_and_reuses_without_inference(planned, monkeypatch):
    data, rows, output, plan = planned
    before = {p: p.read_bytes() for p in data.run_root.rglob("*") if p.is_file()}
    monkeypatch.setattr(execution, "run_worker", lambda *a, **kw: pytest.fail("No GPU inference"))
    cid = plan["case_ids"][0]
    receipt, reused = review.review_case(data.root, data.run_root, output, cid, plan, data.contract)
    assert not reused and receipt["metric_parity"] is True
    assert "evaluation_with_cprofile" in receipt["stage_seconds"]
    assert all(t >= 0 for t in receipt["stage_seconds"].values())
    folder = output / "cases" / cid
    for name in receipt["artifacts"]:
        assert (folder / name).stat().st_size > 0
    monkeypatch.setattr(review, "evaluate_supported_renal_prediction", lambda **kw: pytest.fail("Verified cache"))
    restored, reused = review.review_case(data.root, data.run_root, output, cid, plan, data.contract)
    assert reused and restored["receipt_sha256"] == receipt["receipt_sha256"]
    assert before == {p: p.read_bytes() for p in before}
    png = folder / next(n for n in receipt["artifacts"] if n.endswith("cyst_review.png"))
    png.write_bytes(png.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        review.review_case(data.root, data.run_root, output, cid, plan, data.contract)


def test_parity_failure_keeps_diagnostics_and_does_not_publish_completion(planned, monkeypatch):
    data, rows, output, plan = planned
    evaluator = review.evaluate_supported_renal_prediction
    class ChangedResult:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    def changed(**kwargs):
        payload = evaluator(**kwargs).to_dict()
        payload["kidney_and_mass_hec"]["dice"] = .1
        return ChangedResult(payload)
    monkeypatch.setattr(review, "evaluate_supported_renal_prediction", changed)
    cid = plan["case_ids"][0]
    with pytest.raises(ValueError, match="Recomputed metrics differ"):
        review.review_case(data.root, data.run_root, output, cid, plan, data.contract)
    folder = output / "cases" / cid
    assert not (folder / "complete.json").exists()
    assert len(list(folder.rglob("parity_failure.json"))) == 1
    assert len(list(folder.rglob("failure.json"))) == 1


def test_max_cases_resumes_and_zip_preserves_relative_image_links(planned):
    data, rows, output, plan = planned
    first = review.run_review(data.root, data.run_root, output, plan, data.contract, max_cases=1)
    assert first["status"] == "partial" and len(first["completed_case_ids"]) == 1
    second = review.run_review(data.root, data.run_root, output, plan, data.contract, max_cases=1)
    assert second["status"] == "complete" and len(second["newly_reviewed_case_ids"]) == 1
    archive = review.package_review(output, plan, second)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert sum(n.endswith("_review.png") for n in names) == 4
        assert all(not n.endswith((".nii.gz", ".pt", ".prof")) for n in names)
        assert "report.md" in names and "stage_timings.csv" in names


def test_failures_keep_denominator_and_continue_to_other_cases(planned, monkeypatch):
    data, rows, output, plan = planned
    original = review.review_case
    def fail_first(root, run_root, folder, cid, plan, contract):
        if cid == plan["case_ids"][0]:
            raise OSError("synthetic read error")
        return original(root, run_root, folder, cid, plan, contract)
    monkeypatch.setattr(review, "review_case", fail_first)
    state = review.run_review(data.root, data.run_root, output, plan, data.contract, continue_on_error=True)
    assert state["status"] == "failed" and len(state["requested_case_ids"]) == 2
    assert len(state["failed_cases"]) == len(state["completed_case_ids"]) == 1
    assert not state["deferred_case_ids"]


def test_source_and_plan_drift_are_rejected_before_review(planned, monkeypatch):
    data, rows, output, plan = planned
    monkeypatch.setattr(review, "review_case", lambda *a, **kw: pytest.fail("Must reject before review"))
    changed = deepcopy(plan)
    changed["case_ids"].append("case_00102")
    with pytest.raises(ValueError, match="plan checksum"):
        review.run_review(data.root, data.run_root, output, changed, data.contract)
    source = data.root / "src/vista3d_tfg/evaluation/nv_kits23.py"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Evaluation source differs"):
        review.run_review(data.root, data.run_root, output, plan, data.contract)


def test_canonical_review_loading_aligns_flipped_axes(tmp_path):
    affine = np.diag([-2., 3., 4., 1.])
    array = np.zeros((4, 5, 6), dtype=np.int16)
    array[0, 2, 3] = 1
    paths = [tmp_path / f"{n}.nii.gz" for n in ("image", "gt", "prediction")]
    for path in paths:
        nib.save(nib.Nifti1Image(array, affine), path)
    image, gt, pred, spacing = review.load_review_arrays(*paths)
    assert gt[-1, 2, 3] == 1 and gt[0, 2, 3] == 0
    assert np.array_equal(image, gt) and np.array_equal(gt, pred)
    assert spacing == (2., 3., 4.)


def test_plan_cli_does_not_load_raw_volumes_or_evaluate(planned, monkeypatch, capsys):
    data, rows, output, plan = planned
    module = script("review_nv_segment_ct_full489_development")
    monkeypatch.setattr(module, "verified_protocol", lambda root: data.protocol)
    monkeypatch.setattr(nib, "load", lambda *a, **kw: pytest.fail("Plan must not load voxels"))
    monkeypatch.setattr(review, "evaluate_supported_renal_prediction", lambda **kw: pytest.fail("Plan must not evaluate"))
    assert module.main(["--stage", "plan", "--run-id", data.run_root.name, "--project-root", str(data.root)]) == 0
    assert "Status: planned" in capsys.readouterr().out
    assert not (data.root / "outputs/full489/execution.lock").exists()


def test_cli_has_no_reserved_partition_option(tmp_path):
    module = script("review_nv_segment_ct_full489_development")
    with pytest.raises(SystemExit) as exc:
        module.main(["--stage", "review", "--run-id", "a" * 16, "--partition", "test", "--project-root", str(tmp_path)])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())
