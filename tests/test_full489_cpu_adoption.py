from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import zipfile

import pytest

from vista3d_tfg.analysis import full489_cpu as cpu, full489_cpu_adoption as adoption
from vista3d_tfg.evaluation import nv_kits23_cpu_v1 as evaluator
from vista3d_tfg.experiments import full489_execution as execution
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.metrics import morphology
from test_full489_cpu import ready
from test_full489_development import completed
from test_full489_execution import runtime, script


@pytest.fixture
def prepared(ready):
    data, rows, review_root, benchmark_root, benchmark_plan = ready
    state, _ = cpu.run_benchmark(data.root, data.run_root, benchmark_root, benchmark_plan, data.contract)
    assert state["status"] == "complete"
    evidence = adoption.completed_benchmark(data.root, data.run_root, benchmark_root, rows, data.contract)
    output, plan = adoption.prepare_adoption(data.root, rows, data.contract, data.protocol, evidence)
    return data, rows, benchmark_root, output, plan


def test_versioned_entrypoint_matches_original_and_restores_validator(prepared):
    data, rows, benchmark_root, output, plan = prepared
    cid = plan["case_ids"][0]
    case, _, paths = cpu.source_case(data.root, data.run_root, cid, plan, data.contract)
    before = morphology.validate_binary_mask
    result = evaluator.evaluate_supported_renal_prediction_cpu_v1(case_id=cid,
        ground_truth_path=case.segmentation_path, prediction_path=paths["prediction.nii.gz"])
    saved = execution.read_json(paths["supported_renal_metrics.json"])["result"]
    assert cpu.json_metrics(saved) == cpu.json_metrics(result.to_dict())
    assert morphology.validate_binary_mask is before
    assert evaluator.EVALUATOR_VERSION == "supported-renal-cpu-bool-v1"


def test_full_development_one_call_each_preserves_originals_and_certifies(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    before = {p: p.read_bytes() for folder in (data.run_root, benchmark_root) for p in folder.rglob('*') if p.is_file()}
    monkeypatch.setattr(execution, "run_worker", lambda *a, **kw: pytest.fail("No GPU inference"))
    original = evaluator.evaluate_supported_renal_prediction_cpu_v1
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs['case_id'])
        return original(**kwargs)
    monkeypatch.setattr(evaluator, "evaluate_supported_renal_prediction_cpu_v1", evaluate)
    state, archive = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)
    assert calls == data.protocol['partition']['development']['case_ids']
    assert state['status'] == 'complete' and state['exact_metric_matches'] == len(rows)
    certificate = execution.read_json(output / 'development_validation.json')
    assert certificate['n_cases'] == len(rows) and set(certificate['case_receipts']) == set(calls)
    assert certificate['certificate_sha256'] == canonical_sha256({k:v for k,v in certificate.items() if k != 'certificate_sha256'})
    assert before == {p:p.read_bytes() for p in before}
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and 'development_validation.json' in z.namelist()
        assert sum(n.endswith('/evaluation.json') for n in z.namelist()) == len(rows)
        assert not any(n.endswith(('.png','.nii.gz','.pt')) for n in z.namelist())


def test_partial_has_no_certificate_and_resume_only_runs_pending(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    state, archive = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol, max_cases=1)
    assert state['status'] == 'partial' and len(state['completed_case_ids']) == 1
    assert not (output / 'development_validation.json').exists()
    with zipfile.ZipFile(archive) as z: assert 'development_validation.json' not in z.namelist()
    original = evaluator.evaluate_supported_renal_prediction_cpu_v1
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs['case_id'])
        return original(**kwargs)
    monkeypatch.setattr(evaluator, "evaluate_supported_renal_prediction_cpu_v1", evaluate)
    state, _ = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol, max_cases=1)
    assert state['status'] == 'complete' and calls == [plan['case_ids'][1]]
    assert state['newly_evaluated_case_ids'] == calls


@pytest.mark.parametrize('bad', [.123, math.nan])
def test_metric_drift_or_nan_fails_without_certificate(prepared, monkeypatch, bad):
    data, rows, benchmark_root, output, plan = prepared
    original = evaluator.evaluate_supported_renal_prediction_cpu_v1
    class Result:
        def __init__(self, payload): self.payload = payload
        def to_dict(self): return self.payload
    def evaluate(**kwargs):
        payload = original(**kwargs).to_dict()
        payload['kidney_and_mass_hec']['dice'] = bad
        return Result(payload)
    monkeypatch.setattr(evaluator, 'evaluate_supported_renal_prediction_cpu_v1', evaluate)
    state, archive = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)
    assert state['status'] == 'failed' and not state['completed_case_ids']
    assert not (output / 'development_validation.json').exists()
    assert not list(output.glob('cases/*/complete.json'))
    with zipfile.ZipFile(archive) as z:
        assert any(n.endswith('parity_failure.json') for n in z.namelist())


def test_partial_benchmark_and_corrupt_benchmark_payload_rejected(prepared):
    data, rows, benchmark_root, output, plan = prepared
    path = benchmark_root / 'progress.json'
    state = execution.read_json(path)
    execution.atomic_json(path, {**state, 'status':'partial'})
    with pytest.raises(ValueError, match='Complete Block 61'):
        adoption.completed_benchmark(data.root, data.run_root, benchmark_root, rows, data.contract)
    execution.atomic_json(path, state)
    payload = next(benchmark_root.glob('cases/*/attempt_*/benchmark.json'))
    payload.write_bytes(payload.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        adoption.completed_benchmark(data.root, data.run_root, benchmark_root, rows, data.contract)


def test_resigned_benchmark_metric_drift_is_detected(prepared):
    data, rows, benchmark_root, output, plan = prepared
    path = next(benchmark_root.glob('cases/*/complete.json'))
    receipt = execution.read_json(path)
    relative = next(iter(receipt['artifacts']))
    report_path = path.parent / relative
    report = execution.read_json(report_path)
    report['measurements'][0]['result']['kidney_and_mass_hec']['dice'] = .123
    execution.atomic_json(report_path, report)
    receipt['artifacts'][relative] = cpu.sha256_file(report_path)
    receipt['receipt_sha256'] = canonical_sha256({k:v for k,v in receipt.items() if k != 'receipt_sha256'})
    execution.atomic_json(path, receipt)
    with pytest.raises(ValueError, match='measurement or metric parity'):
        adoption.completed_benchmark(data.root, data.run_root, benchmark_root, rows, data.contract)


def test_scope_requires_all_development_and_rejects_reserved(prepared):
    data, rows, benchmark_root, output, plan = prepared
    changed = deepcopy(plan)
    changed['case_ids'] = changed['case_ids'][:1]
    changed['plan_sha256'] = canonical_sha256({k:v for k,v in changed.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError, match='selection differs'):
        adoption.run_adoption(data.root, data.run_root, output, changed, data.contract, data.protocol)
    changed_protocol = deepcopy(data.protocol)
    changed_protocol['partition']['test']['case_ids'].append(plan['case_ids'][0])
    with pytest.raises(ValueError, match='selection differs'):
        adoption.check_adoption(plan, data.contract, changed_protocol)


def test_adoption_selection_expands_beyond_benchmark_subset(prepared):
    data, rows, benchmark_root, output, plan = prepared
    evidence = {'benchmark_plan_sha256': plan['benchmark_plan_sha256'],
                'benchmark_receipts': {plan['case_ids'][0]: plan['benchmark_receipts'][plan['case_ids'][0]]}}
    _, expanded = adoption.prepare_adoption(data.root, rows, data.contract, data.protocol, evidence)
    assert len(expanded['benchmark_receipts']) == 1
    assert expanded['case_ids'] == data.protocol['partition']['development']['case_ids']
    assert len(expanded['case_ids']) == 2


def test_input_change_during_evaluation_prevents_completion(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    cid = plan['case_ids'][0]
    image = data.root / 'data/raw/kits23' / cid / 'imaging.nii.gz'
    original = evaluator.evaluate_supported_renal_prediction_cpu_v1
    def evaluate(**kwargs):
        result = original(**kwargs)
        image.write_bytes(image.read_bytes() + b'changed')
        return result
    monkeypatch.setattr(evaluator, 'evaluate_supported_renal_prediction_cpu_v1', evaluate)
    state, _ = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)
    assert state['status'] == 'failed' and cid in state['failed_cases']
    assert not (output / 'cases' / cid / 'complete.json').exists()


def test_versioned_entrypoint_restores_validator_after_interruption(monkeypatch):
    def interrupt(**kwargs):
        assert morphology.validate_binary_mask is cpu.validate_binary_mask_fast
        raise KeyboardInterrupt()
    monkeypatch.setattr(evaluator, '_baseline', interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluator.evaluate_supported_renal_prediction_cpu_v1(case_id='synthetic', ground_truth_path=Path('gt'), prediction_path=Path('pred'))
    assert morphology.validate_binary_mask is cpu._ORIGINAL_VALIDATOR


def test_continue_on_error_never_certifies_incomplete_coverage(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    original = adoption.adoption_case
    def evaluate(root, run_root, output, cid, plan, contract, protocol):
        if cid == plan['case_ids'][0]:
            raise OSError('synthetic read failure')
        return original(root, run_root, output, cid, plan, contract, protocol)
    monkeypatch.setattr(adoption, 'adoption_case', evaluate)
    state, _ = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol, continue_on_error=True)
    assert state['status'] == 'failed' and len(state['completed_case_ids']) == len(state['failed_cases']) == 1
    assert not state['deferred_case_ids'] and not (output / 'development_validation.json').exists()


def test_cache_rechecks_inputs_and_removes_readiness_on_failure(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)
    assert (output / 'development_validation.json').exists()
    monkeypatch.setattr(evaluator, 'evaluate_supported_renal_prediction_cpu_v1', lambda **kw: pytest.fail('No repeated evaluation'))
    cid = plan['case_ids'][0]
    image = data.root / 'data/raw/kits23' / cid / 'imaging.nii.gz'
    image.write_bytes(image.read_bytes() + b'changed')
    state, archive = adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)
    assert state['status'] == 'failed' and cid in state['failed_cases']
    assert not (output / 'development_validation.json').exists()
    with zipfile.ZipFile(archive) as z: assert 'development_validation.json' not in z.namelist()


def test_corrupt_cached_evaluation_is_rejected(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    cid = plan['case_ids'][0]
    adoption.adoption_case(data.root, data.run_root, output, cid, plan, data.contract, data.protocol)
    path = next(output.glob('cases/*/attempt_*/evaluation.json'))
    path.write_bytes(path.read_bytes() + b'changed')
    monkeypatch.setattr(evaluator, 'evaluate_supported_renal_prediction_cpu_v1', lambda **kw: pytest.fail('Cache must fail'))
    with pytest.raises(ValueError, match='artifact changed'):
        adoption.adoption_case(data.root, data.run_root, output, cid, plan, data.contract, data.protocol)


def test_evaluator_source_and_baseline_drift_rejected(prepared, monkeypatch):
    data, rows, benchmark_root, output, plan = prepared
    original = adoption.implementation_identity
    monkeypatch.setattr(adoption, 'implementation_identity', lambda: {**original(), 'evaluator_version':'different'})
    with pytest.raises(ValueError, match='code or environment changed'):
        adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)
    monkeypatch.setattr(adoption, 'implementation_identity', original)
    path = data.root / 'src/vista3d_tfg/metrics/morphology.py'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='Evaluation source differs'):
        adoption.run_adoption(data.root, data.run_root, output, plan, data.contract, data.protocol)


def test_cli_progress_and_reserved_outputs_untouched(prepared, monkeypatch, capsys):
    data, rows, benchmark_root, output, plan = prepared
    reserved = data.run_root / 'cases/case_00102/complete.json'
    reserved.parent.mkdir(parents=True)
    reserved.write_text('invalid reserved output')
    module = script('validate_nv_segment_ct_full489_cpu_v1')
    monkeypatch.setattr(module, 'verified_protocol', lambda root: data.protocol)
    args = ['--project-root',str(data.root),'--run-id',data.run_root.name,'--benchmark-id',benchmark_root.name,'--max-cases','1']
    assert module.main(args) == 0
    assert 'Status: partial' in capsys.readouterr().out
    assert module.main(args) == 0
    assert 'Status: complete' in capsys.readouterr().out
    assert reserved.read_text() == 'invalid reserved output'
    assert not (data.root / 'outputs/full489/execution.lock').exists()


@pytest.mark.parametrize('extra', [['--partition','test'], ['--max-cases','0'], ['--benchmark-id','../bad']])
def test_cli_rejects_bad_arguments(extra):
    module = script('validate_nv_segment_ct_full489_cpu_v1')
    with pytest.raises(SystemExit) as exc:
        module.main(['--run-id','a'*16,'--benchmark-id','b'*16,*extra])
    assert exc.value.code == 2
