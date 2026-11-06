from __future__ import annotations

from copy import deepcopy
import math
import zipfile

import numpy as np
import pytest

from vista3d_tfg.analysis import full489_cyst_confusion as confusion, full489_cpu_adoption as adoption
from vista3d_tfg.evaluation import nv_kits23 as baseline
from vista3d_tfg.experiments import full489_execution as execution
from vista3d_tfg.experiments.full489 import canonical_sha256
from test_full489_cpu_adoption import prepared
from test_full489_cpu import ready
from test_full489_development import completed
from test_full489_execution import runtime, script


def test_every_reference_prediction_pair_and_slab_partition_conserve_voxels():
    gt = np.repeat(np.arange(4), 5).reshape(2, 2, 5)
    pred = np.tile(confusion.PRED_VALUES, 4).reshape(2, 2, 5)
    expected = np.ones((4, 5), dtype=np.int64)
    for depth in (1, 2, 16):
        assert np.array_equal(confusion.confusion_counts(gt, pred, slab_depth=depth), expected)
    assert np.array_equal(confusion.confusion_counts(gt[::-1], pred[::-1]), expected)
    assert confusion.PRED_NAMES[1:3] == ('right_kidney', 'left_kidney')


@pytest.mark.parametrize('gt,pred', [
    (np.ones((2, 2, 2))*1., np.ones((2, 2, 2), dtype=int)),
    (np.full((2, 2, 2), 4), np.zeros((2, 2, 2), dtype=int)),
    (np.full((2, 2, 2), -1), np.zeros((2, 2, 2), dtype=int)),
    (np.zeros((2, 2, 2), dtype=int), np.full((2, 2, 2), 118)),
    (np.zeros((2, 2, 2), dtype=int), np.full((2, 2, 2), 6)),
    (np.zeros((2, 2, 2), dtype=int), np.full((2, 2, 2), -1)),
    (np.zeros((2, 2, 2), dtype=int), np.zeros((2, 2), dtype=int)),
])
def test_invalid_labels_dtype_and_geometry_are_rejected(gt, pred):
    with pytest.raises(ValueError): confusion.confusion_counts(gt, pred)


def test_matrix_binary_metrics_match_independent_voxel_comparisons():
    rng = np.random.default_rng(63)
    gt = rng.integers(0, 4, size=(7, 8, 9))
    pred = np.asarray(confusion.PRED_VALUES)[rng.integers(0, 5, size=gt.shape)]
    counts = confusion.confusion_counts(gt, pred, slab_depth=3)
    spacing = (.7, .8, 2.5)
    result = confusion.derived_metrics(counts, spacing)
    from dataclasses import asdict
    assert result['cyst'] == asdict(baseline.compare_binary_masks(gt == 3, np.isin(pred, (116,117)), spacing_mm=spacing))
    assert result['kidney_core_diagnostic'] == asdict(baseline.compare_binary_masks(gt == 1, np.isin(pred, (5,14)), spacing_mm=spacing))
    assert result['kidney_mass_dice'] == baseline.compare_binary_masks(gt > 0, pred > 0, spacing_mm=spacing).dice
    assert counts.sum() == gt.size


def test_directional_breakdown_has_explicit_denominators_and_envelope_omissions():
    volumes = np.zeros((4,5))
    volumes[0,3] = 2.;volumes[1,4] = 3.;volumes[2,3] = 5.;volumes[3,3] = 10.
    volumes[3,0] = 7.;volumes[3,1] = 11.;volumes[3,2] = 13.
    b = confusion.breakdown(volumes)
    assert b['cyst_fp_by_reference_ml'] == {'background':2.,'kidney':3.,'tumor':5.}
    assert b['cyst_fn_by_prediction_ml'] == {'background':7.,'right_kidney':11.,'left_kidney':13.}
    assert b['cyst_fp_total_ml'] == 10. and b['cyst_fn_total_ml'] == 31.
    assert sum(b['predicted_cyst_composition'].values()) == 1.
    assert math.isclose(sum(b['reference_cyst_destinations'].values()), 1.)
    empty = confusion.breakdown(np.zeros((4,5)))
    assert all(v is None for k in ('fp_composition','fn_composition','predicted_cyst_composition','reference_cyst_destinations') for v in empty[k].values())


def test_volume_pooling_respects_spacing_and_macro_excludes_undefined_cases():
    reports = []
    for cid, factor, index in [('a',.001,0),('b',.01,2),('empty',.002,None)]:
        matrix = np.zeros((4,5), dtype=int)
        if index is not None: matrix[index,3] = 100
        volume = matrix * factor
        reports.append({'case_id':cid,'group':'historical100','counts':matrix.tolist(),'volumes_ml':volume.tolist(),'breakdown':confusion.breakdown(volume)})
    summary = confusion.summarize_reports(reports)['groups']['development289']
    assert summary['pooled_counts'][0][3] == summary['pooled_counts'][2][3] == 100
    b = summary['pooled_volume_breakdown']
    assert math.isclose(b['fp_composition']['tumor'], 1/1.1)
    macro = summary['per_case_fraction_summaries']['fp_composition']['tumor']
    assert macro['mean'] == .5 and macro['n_finite'] == 2 and macro['n_nonfinite'] == 1


@pytest.fixture
def analysis_ready(prepared):
    data, rows, benchmark_root, validation_root, validation_plan = prepared
    state, _ = adoption.run_adoption(data.root, data.run_root, validation_root, validation_plan, data.contract, data.protocol)
    assert state['status'] == 'complete'
    evidence = confusion.completed_validation(data.root, data.run_root, validation_root, rows, data.contract, data.protocol)
    output, plan = confusion.prepare_analysis(data.root, rows, data.contract, data.protocol, evidence)
    return data, rows, validation_root, output, plan


def test_analysis_does_not_run_gpu_or_surface_evaluation_and_preserves_sources(analysis_ready, monkeypatch):
    data, rows, validation_root, output, plan = analysis_ready
    before = {p:p.read_bytes() for folder in (data.run_root, validation_root) for p in folder.rglob('*') if p.is_file()}
    monkeypatch.setattr(execution,'run_worker',lambda *a,**kw:pytest.fail('No inference'))
    monkeypatch.setattr(baseline,'evaluate_supported_renal_prediction',lambda **kw:pytest.fail('No full evaluation'))
    monkeypatch.setattr(baseline.surface_metrics,'compute_surface_distances',lambda *a,**kw:pytest.fail('No distances'))
    state, archive = confusion.run_analysis(data.root,data.run_root,output,plan,data.contract,data.protocol)
    assert state['status'] == 'complete' and len(state['completed_case_ids']) == len(rows)
    assert before == {p:p.read_bytes() for p in before}
    for cid in plan['case_ids']:
        receipt = confusion.verified_case(output,cid,plan)
        report = execution.read_json(output/'cases'/cid/next(iter(receipt['artifacts'])))
        assert np.asarray(report['counts']).sum() == math.prod(report['shape'])
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and 'confusion_cells.csv' in z.namelist()
        assert not any(n.endswith(('.nii.gz','.pt','.png')) for n in z.namelist())


def test_budget_resume_and_partial_summary_denominator(analysis_ready, monkeypatch):
    data, rows, validation_root, output, plan = analysis_ready
    state,_ = confusion.run_analysis(data.root,data.run_root,output,plan,data.contract,data.protocol,max_cases=1)
    summary = execution.read_json(output/'summary.json')
    assert state['status'] == summary['status'] == 'partial'
    assert summary['n_analyzed'] == 1 and summary['n_requested'] == len(rows)
    original = confusion._load_pair
    calls = []
    def load(*a):calls.append(a);return original(*a)
    monkeypatch.setattr(confusion,'_load_pair',load)
    state,_ = confusion.run_analysis(data.root,data.run_root,output,plan,data.contract,data.protocol,max_cases=1)
    assert state['status'] == 'complete' and len(calls) == 1


def test_wrong_counts_fail_before_success_receipt(analysis_ready, monkeypatch):
    data, rows, validation_root, output, plan = analysis_ready
    original = confusion.confusion_counts
    def wrong(*a,**kw):
        counts = original(*a,**kw);counts[3,3] += 1
        return counts
    monkeypatch.setattr(confusion,'confusion_counts',wrong)
    state,archive = confusion.run_analysis(data.root,data.run_root,output,plan,data.contract,data.protocol)
    assert state['status'] == 'failed' and not state['completed_case_ids']
    assert not list(output.glob('cases/*/complete.json'))
    with zipfile.ZipFile(archive) as z:assert any(n.endswith('failure.json') for n in z.namelist())


def test_invalid_or_changed_block62_certificate_and_payload_are_rejected(analysis_ready):
    data, rows, validation_root, output, plan = analysis_ready
    path = validation_root/'development_validation.json'
    original = path.read_bytes();cert=execution.read_json(path)
    cert['n_cases'] -= 1;execution.atomic_json(path,cert)
    with pytest.raises(ValueError,match='certificate'):
        confusion.completed_validation(data.root,data.run_root,validation_root,rows,data.contract,data.protocol)
    path.write_bytes(original)
    result = next(validation_root.glob('cases/*/attempt_*/evaluation.json'))
    result.write_bytes(result.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        confusion.completed_validation(data.root,data.run_root,validation_root,rows,data.contract,data.protocol)


def test_cached_confusion_rechecks_original_inputs_and_output_hashes(analysis_ready, monkeypatch):
    data, rows, validation_root, output, plan = analysis_ready
    cid = plan['case_ids'][0]
    receipt,_ = confusion.analysis_case(data.root,data.run_root,output,cid,plan,data.contract,data.protocol)
    monkeypatch.setattr(confusion,'_load_pair',lambda *a:pytest.fail('Cache must not load arrays'))
    assert confusion.analysis_case(data.root,data.run_root,output,cid,plan,data.contract,data.protocol)[1]
    result = output/'cases'/cid/next(iter(receipt['artifacts']));original=result.read_bytes()
    result.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        confusion.analysis_case(data.root,data.run_root,output,cid,plan,data.contract,data.protocol)
    result.write_bytes(original)
    gt = data.root/'data/raw/kits23'/cid/'segmentation.nii.gz'
    gt.write_bytes(gt.read_bytes()+b'changed')
    with pytest.raises(ValueError):confusion.analysis_case(data.root,data.run_root,output,cid,plan,data.contract,data.protocol)


def test_count_parity_rejects_geometry_spacing_or_labels(analysis_ready):
    data, rows, validation_root, output, plan = analysis_ready
    cid=plan['case_ids'][0]
    case,_,artifacts=confusion.cpu.source_case(data.root,data.run_root,cid,plan,data.contract)
    gt,pred,spacing,_=confusion._load_pair(case.segmentation_path,artifacts['prediction.nii.gz'])
    counts=confusion.confusion_counts(gt,pred)
    saved=execution.read_json(artifacts['supported_renal_metrics.json'])['result']
    confusion.check_count_parity(counts,gt.shape,spacing,saved)
    for key,value in [('shape',[1,2,3]),('prediction_labels_present',[0,5])]:
        changed=deepcopy(saved);changed[key]=value
        with pytest.raises(ValueError,match='parity'):confusion.check_count_parity(counts,gt.shape,spacing,changed)
    with pytest.raises(ValueError):confusion.check_count_parity(counts,gt.shape,(-1,1,1),saved)


def test_cli_processes_only_development_and_exports_expected_progress(analysis_ready,monkeypatch,capsys):
    data,rows,validation_root,output,plan=analysis_ready
    reserved=data.run_root/'cases/case_00102/complete.json';reserved.parent.mkdir(parents=True);reserved.write_text('invalid reserved')
    module=script('analyze_nv_segment_ct_full489_cyst_confusion')
    monkeypatch.setattr(module,'verified_protocol',lambda root:data.protocol)
    args=['--project-root',str(data.root),'--run-id',data.run_root.name,'--validation-id',validation_root.name,'--max-cases','1']
    assert module.main(args)==0 and 'Status: partial' in capsys.readouterr().out
    assert module.main(args)==0 and 'Status: complete' in capsys.readouterr().out
    assert reserved.read_text()=='invalid reserved'
    assert not (data.root/'outputs/full489/execution.lock').exists()


@pytest.mark.parametrize('extra',[['--partition','test'],['--max-cases','0'],['--validation-id','../bad']])
def test_cli_rejects_invalid_arguments(extra):
    with pytest.raises(SystemExit) as exc:
        script('analyze_nv_segment_ct_full489_cyst_confusion').main(['--run-id','a'*16,'--validation-id','b'*16,*extra])
    assert exc.value.code==2


def test_scope_and_code_drift_rejected_before_processing(analysis_ready,monkeypatch):
    data,rows,validation_root,output,plan=analysis_ready
    changed=deepcopy(plan);changed['case_ids']=changed['case_ids'][:1]
    changed['plan_sha256']=canonical_sha256({k:v for k,v in changed.items() if k!='plan_sha256'})
    with pytest.raises(ValueError,match='selection differs'):
        confusion.run_analysis(data.root,data.run_root,output,changed,data.contract,data.protocol)
    changed_protocol=deepcopy(data.protocol)
    changed_protocol['partition']['test']['case_ids'].append(plan['case_ids'][0])
    with pytest.raises(ValueError,match='selection differs'):
        confusion.check_analysis(plan,data.contract,changed_protocol)
    original=confusion.implementation
    monkeypatch.setattr(confusion,'implementation',lambda:{**original(),'module_sha256':'changed'})
    with pytest.raises(ValueError,match='code or environment changed'):
        confusion.run_analysis(data.root,data.run_root,output,plan,data.contract,data.protocol)


def test_continue_on_error_preserves_denominator_and_excludes_failed_case(analysis_ready,monkeypatch):
    data,rows,validation_root,output,plan=analysis_ready
    original=confusion.analysis_case
    def analyze(root,run_root,output,cid,plan,contract,protocol):
        if cid==plan['case_ids'][0]:raise OSError('synthetic input failure')
        return original(root,run_root,output,cid,plan,contract,protocol)
    monkeypatch.setattr(confusion,'analysis_case',analyze)
    state,_=confusion.run_analysis(data.root,data.run_root,output,plan,data.contract,data.protocol,continue_on_error=True)
    summary=execution.read_json(output/'summary.json')
    assert state['status']=='failed' and len(state['failed_cases'])==len(state['completed_case_ids'])==1
    assert not state['deferred_case_ids']
    assert summary['n_analyzed']==1 and summary['n_requested']==2
