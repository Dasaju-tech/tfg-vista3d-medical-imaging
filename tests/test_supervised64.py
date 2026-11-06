from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments import supervised64 as experiment
from vista3d_tfg.experiments.full489 import canonical_sha256
from vista3d_tfg.experiments.full489_execution import atomic_json, input_identity, read_json
from vista3d_tfg.evaluation.nv_kits23 import KITS23_KIDNEY_AND_MASS_TOLERANCE_MM
from vista3d_tfg.integration.nv_inference import sha256_file
from vista3d_tfg.training import supervised64 as training
from test_full489_execution import script


def save_pair(folder, *, affine=None):
    folder.mkdir(parents=True, exist_ok=True)
    gt = np.zeros((24, 24, 24), dtype=np.uint8)
    gt[2:10, 2:10, 2:10] = 1
    gt[12:20, 2:10, 2:10] = 2
    gt[12:20, 12:20, 12:20] = 3
    ct = (gt.astype(np.int16) * 150) - 200
    affine = np.diag([1.5, 1.5, 1.5, 1.]) if affine is None else affine
    image_path, label_path = folder / 'imaging.nii.gz', folder / 'segmentation.nii.gz'
    nib.save(nib.Nifti1Image(ct, affine), image_path)
    nib.save(nib.Nifti1Image(gt, affine), label_path)
    return image_path, label_path


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / 'project with spaces'
    source = Path(__file__).resolve().parents[1]
    for name in experiment.SOURCES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    ids = [f'case_{n:05d}' for n in range(3)]
    protocol = {'partition': {k: {'case_ids': v} for k, v in
                            [('development', ids), ('validation', ['case_00010']), ('test', ['case_00011'])]}}
    protocol['protocol_sha256'] = canonical_sha256(protocol)
    contract = {'protocol_sha256': protocol['protocol_sha256'], 'label_prompt': [5,14,116,117],
                'kidney_mass_surface_tolerance_mm': KITS23_KIDNEY_AND_MASS_TOLERANCE_MM,
                'runtime_identity': {'gpu_total_bytes': 12*2**30}}
    contract['contract_sha256'] = canonical_sha256(contract)
    run = root / 'outputs/full489' / contract['contract_sha256'][:16]
    atomic_json(run/'contract.json', contract)
    source_plan = {'case_ids': ids, 'contract_sha256': contract['contract_sha256'],
                   'protocol_sha256': protocol['protocol_sha256'],
                   'source_receipts': {c: 'a'*64 for c in ids}, 'validation_receipts': {c: 'b'*64 for c in ids},
                   'implementation': experiment.confusion.implementation()}
    source_plan['plan_sha256'] = canonical_sha256(source_plan)
    folder = root/'outputs/reporting/full489_block63'/run.name/source_plan['plan_sha256'][:16]
    atomic_json(folder/'plan.json', source_plan)
    atomic_json(folder/'progress.json', {'status':'complete','completed_case_ids':ids,'requested_case_ids':ids,
                                        'failed_cases':{},'deferred_case_ids':[], 'plan_sha256':source_plan['plan_sha256']})
    for cid in ids:
        image, label = save_pair(root/'data/raw/kits23'/cid)
        atomic_json(run/'cases'/cid/'inputs.json', input_identity(root,cid)[1])
        counts = np.zeros((4,5),dtype=int)
        counts[:,0] = np.bincount(np.asarray(nib.load(label).dataobj).ravel(), minlength=4)
        report = {'case_id':cid, 'plan_sha256':source_plan['plan_sha256'], 'count_parity':True,
                  'original_receipt_sha256':'a'*64,'validation_receipt_sha256':'b'*64,
                  'shape':[24,24,24], 'counts':counts.tolist()}
        path = folder/'cases'/cid/'attempt_test/confusion.json'
        atomic_json(path, report)
        receipt = {'case_id':cid,'plan_sha256':source_plan['plan_sha256'],'status':'complete','count_parity':True,
                   'artifacts':{'attempt_test/confusion.json':sha256_file(path)}}
        receipt['receipt_sha256'] = canonical_sha256(receipt)
        atomic_json(folder/'cases'/cid/'complete.json',receipt)
    output, plan = experiment.prepare(root,protocol,run.name,folder.name)
    return root, protocol, run, folder, output, plan


def test_prepare_freezes_without_gpu_and_keeps_reserved_paths_unread(frozen, monkeypatch):
    root, protocol, run, folder, output, plan = frozen
    monkeypatch.setattr(experiment,'call_worker',lambda *a:pytest.fail('prepare must not use GPU'))
    reserved = root/'data/raw/kits23/case_00010/imaging.nii.gz'
    reserved.parent.mkdir(parents=True);reserved.write_bytes(b'invalid reserved volume')
    before = {p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert experiment.prepare(root,protocol,run.name,folder.name) == (output,plan)
    assert before == {p:p.read_bytes() for p in before}
    assert set(plan['smoke_case_ids']) == set(protocol['partition']['development']['case_ids'])


@pytest.mark.parametrize('change', ['partial','artifact','inputs'])
def test_prepare_rejects_incomplete_or_changed_evidence(frozen, change):
    root,protocol,run,folder,output,plan = frozen
    if change == 'partial':
        state=read_json(folder/'progress.json');state['completed_case_ids'].pop();atomic_json(folder/'progress.json',state)
    elif change == 'artifact':
        next(folder.glob('cases/*/attempt_*/confusion.json')).write_text('changed')
    else:
        next((root/'data/raw/kits23').glob('*/imaging.nii.gz')).write_bytes(b'changed')
    with pytest.raises(ValueError):experiment.prepare(root,protocol,run.name,folder.name)


@pytest.mark.parametrize('change',['reserved','settings','code','digest'])
def test_changed_training_scope_or_code_is_rejected(frozen, change):
    root,protocol,run,folder,output,plan=frozen
    plan=deepcopy(plan)
    if change == 'reserved':plan['smoke_case_ids'][0]='case_00010'
    if change == 'settings':plan['settings']['steps_per_case']=800
    if change == 'code':(root/experiment.WORKER).write_text('changed')
    if change == 'digest':plan['scope']='changed without rehash'
    else:plan['experiment_sha256']=canonical_sha256({k:v for k,v in plan.items() if k!='experiment_sha256'})
    with pytest.raises(ValueError):experiment.check_plan(root,plan,protocol)


def test_selection_uses_reference_presence_not_prediction_error():
    ids=[f'case_{n:05d}' for n in range(20)]
    reports=[{'case_id':c,'counts':[[1,0,0,0,0] for _ in range(4)],'shape':[1,2,2]} for c in ids]
    selected=experiment.select_smoke(reports,ids)
    for r in reports:r['counts']=[[0,0,0,0,1] for _ in range(4)]
    assert experiment.select_smoke(reports,ids)==selected
    with pytest.raises(ValueError):experiment.select_smoke(reports[:2],ids[:2])


def test_resampling_preserves_world_alignment_classes_and_hu_window(tmp_path):
    affine=np.diag([-1.5,1.5,1.5,1.]);affine[0,3]=34.5
    image,label=save_pair(tmp_path,affine=affine)
    ct,gt,info=training.preprocess(image,label,experiment.settings())
    assert ct.dtype==np.float32 and gt.dtype==np.uint8
    assert np.array_equal(gt,np.asarray(nib.load(label).dataobj)[::-1])
    assert np.allclose(ct,(gt.astype(float)*150)*2/500-1)
    assert nib.aff2axcodes(np.asarray(info['resampled_affine']))==('R','A','S')
    config=experiment.settings();config['spacing_mm']=[2.,2.,2.]
    ct,gt,info=training.preprocess(image,label,config)
    assert ct.shape==gt.shape and sorted(np.unique(gt))==[0,1,2,3]
    assert np.allclose(nib.affines.voxel_sizes(np.asarray(info['resampled_affine'])),2.)


@pytest.mark.parametrize('change',['affine','fractional','unknown','nan','ram'])
def test_preprocess_rejects_invalid_geometry_labels_or_memory_budget(tmp_path,change):
    image,label=save_pair(tmp_path)
    config=experiment.settings()
    n=nib.load(label);a=n.affine.copy();values=n.get_fdata(dtype=np.float32)
    if change=='affine':a[0,3]+=2
    if change=='fractional':values[0,0,0]=1.5
    if change=='unknown':values[0,0,0]=117
    if change=='nan':values[0,0,0]=np.nan
    if change=='ram':config['max_resampled_voxels']=1
    nib.save(nib.Nifti1Image(values,a),label)
    with pytest.raises(ValueError):training.preprocess(image,label,config)


def test_class_center_sampling_padding_and_reproducibility():
    gt=np.arange(8,dtype=np.uint8).reshape(2,2,2)%4
    ct=gt.astype(np.float32)
    for k in range(4):
        a=training.sample_patch(ct,gt,(16,16,16),k,np.random.default_rng(64))
        b=training.sample_patch(ct,gt,(16,16,16),k,np.random.default_rng(64))
        assert np.array_equal(a[0],b[0]) and a[2]==b[2]
        assert a[1][8,8,8]==k and a[0].shape==(16,16,16)
        assert np.count_nonzero(a[0]==-1)==16**3-8
    with pytest.raises(ValueError):training.sample_patch(ct,gt,(16,16,16),5,np.random.default_rng(1))


def synthetic_worker(root,environment,job,attempt):
    # Orchestration fixture only. The real tensor-training test below runs on CPU.
    atomic_json(attempt/'job.json',job)
    atomic_json(attempt/'trace.json',{'synthetic_orchestration_fixture':True})
    atomic_json(attempt/'runtime.json',{'synthetic_orchestration_fixture':True})
    (attempt/'checkpoint.pt').write_bytes(b'synthetic orchestration checkpoint')
    report={'status':'complete','experiment_sha256':job['plan']['experiment_sha256'],
            'case_ids':job['plan']['smoke_case_ids'],'optimizer_steps':24,'parameters_changed':True,
            'checkpoint_reload_exact':True,'validation_test_reads':False,'observed_target_labels':[0,1,2,3],
            'losses':[1.]*24,'device':'cuda','quality_evaluation_performed':False,
            'checkpoint_sha256':sha256_file(attempt/'checkpoint.pt')}
    atomic_json(attempt/'worker.json',report)
    return report


def test_success_cache_archive_and_checkpoint_tamper(frozen,monkeypatch):
    root,protocol,run,folder,output,plan=frozen
    monkeypatch.setattr(experiment,'call_worker',synthetic_worker)
    state,archive=experiment.run_smoke(root,protocol,'vista3d-nv')
    assert state['status']=='complete'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and 'complete.json' in z.namelist()
        assert not any(n.endswith(('.pt','.nii.gz')) for n in z.namelist())
    monkeypatch.setattr(experiment,'call_worker',lambda *a:pytest.fail('Cache must not retrain'))
    assert experiment.run_smoke(root,protocol,'vista3d-nv')[0]['cached']
    (output/state['attempt']/'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):experiment.run_smoke(root,protocol,'vista3d-nv')


def test_failed_training_keeps_evidence_and_restarts_in_new_attempt(frozen,monkeypatch):
    root,protocol,run,folder,output,plan=frozen
    def fail(*a):raise RuntimeError('synthetic CUDA OOM')
    monkeypatch.setattr(experiment,'call_worker',fail)
    failed,archive=experiment.run_smoke(root,protocol,'vista3d-nv')
    assert failed['status']=='failed' and not (output/'complete.json').exists()
    with zipfile.ZipFile(archive) as z:assert any(n.endswith('failure.json') for n in z.namelist())
    monkeypatch.setattr(experiment,'call_worker',synthetic_worker)
    success,_=experiment.run_smoke(root,protocol,'vista3d-nv')
    assert success['status']=='complete' and success['attempt']!=failed['attempt']
    assert (output/failed['attempt']/'failure.json').exists()


def test_input_change_blocks_cache_and_post_training_certification(frozen,monkeypatch):
    root,protocol,run,folder,output,plan=frozen
    cid=plan['smoke_case_ids'][0]
    source=root/'data/raw/kits23'/cid/'imaging.nii.gz'
    def changed_during_training(*args):
        report=synthetic_worker(*args)
        source.write_bytes(source.read_bytes()+b'changed')
        return report
    monkeypatch.setattr(experiment,'call_worker',changed_during_training)
    state,_=experiment.run_smoke(root,protocol,'vista3d-nv')
    assert state['status']=='failed' and not (output/'complete.json').exists()
    monkeypatch.setattr(experiment,'call_worker',lambda *a:pytest.fail('Inputs must be checked first'))
    with pytest.raises(ValueError,match='input changed'):experiment.run_smoke(root,protocol,'vista3d-nv')


def test_gpu_worker_imports_without_cpu_reporting_dependencies():
    root=Path(__file__).resolve().parents[1]
    code='''
import importlib.abc, runpy, sys
class BlockReporting(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('surface_distance', 'matplotlib', 'vista3d_tfg.evaluation')):
            raise ImportError('Reporting dependency unavailable in GPU environment: '+fullname)
sys.meta_path.insert(0, BlockReporting())
sys.path.insert(0, sys.argv[1])
runpy.run_path(sys.argv[2], run_name='worker_import_test')
'''
    result=subprocess.run([sys.executable,'-c',code,str(root/'scripts'),str(root/experiment.WORKER)],
                           capture_output=True,text=True,timeout=30)
    assert result.returncode==0, result.stderr


def test_cli_prepare_and_smoke(frozen,monkeypatch,capsys):
    root,protocol,run,folder,output,plan=frozen
    cli=script('run_kits23_supervised64')
    monkeypatch.setattr(cli,'verified_protocol',lambda r:protocol)
    monkeypatch.setattr(experiment,'call_worker',synthetic_worker)
    args=['--project-root',str(root),'--run-id',run.name,'--confusion-id',folder.name]
    assert cli.main(args+['--stage','prepare'])==0
    assert cli.main(args+['--stage','smoke'])==0
    assert 'Optimizer steps: 24/24' in capsys.readouterr().out
    with pytest.raises(SystemExit):cli.main(args+['--stage','test'])


def test_real_cpu_tensor_updates_and_exact_checkpoint_reload(tmp_path):
    torch=pytest.importorskip('torch')
    pytest.importorskip('monai')
    old_threads=torch.get_num_threads()
    old_deterministic=torch.are_deterministic_algorithms_enabled()
    old_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark=torch.backends.cudnn.benchmark
    old_cudnn=torch.backends.cudnn.deterministic
    try:
        torch.set_num_threads(2)
        config=experiment.settings();config['patch_size']=[16,16,16]
        pairs=[(f'case_{i:05d}',*save_pair(tmp_path/f'case_{i:05d}')) for i in range(3)]
        plan={'smoke_case_ids':[p[0] for p in pairs],'settings':config,'experiment_sha256':'c'*64}
        report=training.train_smoke(plan,pairs,tmp_path/'training',device='cpu')
        assert report['optimizer_steps']==24 and report['parameters_changed']
        assert report['checkpoint_reload_exact'] and report['observed_target_labels']==[0,1,2,3]
        assert report['device']=='cpu' and report['peak_reserved_bytes'] is None
        assert all(np.isfinite(report['losses']))
        # A real CPU test must never be certified as the user's CUDA smoke.
        with pytest.raises(ValueError):experiment.validate_result(report,plan)
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic,warn_only=old_warn)
        torch.backends.cudnn.benchmark=old_benchmark
        torch.backends.cudnn.deterministic=old_cudnn
