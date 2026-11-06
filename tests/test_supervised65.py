from __future__ import annotations

from copy import deepcopy
from importlib.metadata import version
from pathlib import Path
import shutil
import zipfile

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.training import contract65 as c, cache65 as cache, engine65 as engine, validation65 as validation
from vista3d_tfg.experiments import supervised65 as experiment, supervised64 as old
from test_supervised64 import frozen, save_pair, synthetic_worker
from test_full489_execution import script


@pytest.fixture
def prepared(frozen,monkeypatch):
    root,protocol,run,folder,output,oldplan=frozen
    source=Path(__file__).resolve().parents[1]
    for name in c.SOURCES:
        destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,destination)
    def fake(*args):
        report=synthetic_worker(*args)
        report['runtime_identity']={'numpy':version('numpy'),'packages':{'nibabel':version('nibabel'),'scipy':version('scipy')}}
        attempt=args[-1]
        c.atomic_json(attempt/'worker.json',report)
        c.atomic_json(attempt/'runtime.json',{'identity':report['runtime_identity'],'fp32_fp16_conv3d_smoke':True})
        return report
    monkeypatch.setattr(old,'call_worker',fake)
    assert old.run_smoke(root,protocol,'vista3d-nv')[0]['status']=='complete'
    # Unique CT bytes for each case; synchronize synthetic baseline identities.
    # Block 64 inputs must stay unchanged, so rewrite only a newly added validation case.
    image,label=save_pair(root/'data/raw/kits23/case_00010')
    n=nib.load(image);nib.save(nib.Nifti1Image(n.get_fdata(dtype=np.float32)+10,n.affine),image)
    # The old synthetic fixture has identical CTs: give each training NIfTI a
    # distinct header without changing voxel arrays, then freeze fresh Block 64 evidence.
    for i,cid in enumerate(oldplan['smoke_case_ids']):
        image=root/'data/raw/kits23'/cid/'imaging.nii.gz';n=nib.load(image)
        n.header['descrip']=f'synthetic fixture {i}';nib.save(n,image)
        identity=c.input_identity(root,cid)
        oldplan['inputs'][cid]=identity;c.atomic_json(run/'cases'/cid/'inputs.json',identity)
    oldplan['experiment_sha256']=c.canonical_sha256({k:v for k,v in oldplan.items() if k!='experiment_sha256'})
    c.atomic_json(root/old.CONFIG,oldplan)
    assert old.run_smoke(root,protocol,'vista3d-nv')[0]['status']=='complete'
    output,plan=experiment.prepare(root,protocol)
    return root,protocol,output,plan


def test_prepare_freezes_all_inputs_and_never_reads_test(prepared):
    root,protocol,output,plan=prepared
    sentinel=root/'data/raw/kits23/case_00011/imaging.nii.gz'
    sentinel.parent.mkdir(parents=True);sentinel.write_bytes(b'sealed test sentinel')
    assert experiment.prepare(root,protocol)==(output,plan)
    assert set(plan['inputs'])==set(c.partition_ids(plan,'development')+c.partition_ids(plan,'validation'))
    assert plan['settings']['epochs']==50 and plan['estimated_cache_bytes']>0
    assert sentinel.read_bytes()==b'sealed test sentinel'


@pytest.mark.parametrize('change',['test','settings','code','input'])
def test_scope_and_input_drift_rejected(prepared,change):
    root,protocol,output,plan=prepared;altered=deepcopy(plan)
    if change=='test':altered['partition']['development']['case_ids'][0]='case_00011'
    if change=='settings':altered['settings']['epochs']=1000
    if change=='code':(root/c.WORKER).write_text('changed')
    if change=='input':
        cid=c.partition_ids(plan,'development')[0]
        (root/'data/raw/kits23'/cid/'imaging.nii.gz').write_bytes(b'changed')
        with pytest.raises(ValueError):c.verify_input(root,plan,'development',cid)
        return
    altered['experiment_sha256']=c.canonical_sha256({k:v for k,v in altered.items() if k!='experiment_sha256'})
    with pytest.raises(ValueError):c.check_plan(root,altered,protocol)


def test_duplicate_ct_rejected_before_freezing(prepared):
    root,protocol,output,plan=prepared
    a=root/'data/raw/kits23/case_00000/imaging.nii.gz';b=root/'data/raw/kits23/case_00010/imaging.nii.gz'
    b.write_bytes(a.read_bytes())
    with pytest.raises(ValueError,match='Byte-identical'):experiment.prepare(root,protocol)


def test_absent_cyst_class_and_validation_image_independent_of_labels(tmp_path):
    image,label=save_pair(tmp_path)
    n=nib.load(label);gt=np.asarray(n.dataobj).copy();gt[gt==3]=0
    nib.save(nib.Nifti1Image(gt,n.affine),label)
    arrays,meta=cache.preprocess(image,label,c.settings(),training=True)
    assert meta['present_classes']==[0,1,2] and arrays['slice_counts.npy'][3].sum()==0
    first,info=cache.preprocess(image,label,c.settings(),training=False)
    gt[:]=0;nib.save(nib.Nifti1Image(gt,n.affine),label)
    second,_=cache.preprocess(image,label,c.settings(),training=False)
    assert set(first)=={'image.npy'} and np.array_equal(first['image.npy'],second['image.npy'])


def test_cache_budget_resume_and_corruption(prepared,monkeypatch):
    root,protocol,output,plan=prepared
    state=cache.prepare_cache(root,plan,'development',max_cases=1)
    assert state['status']=='partial' and len(state['completed_case_ids'])==1
    first=state['completed_case_ids'][0]
    original=cache.preprocess;calls=[]
    def record(*a,**kw):calls.append(a[0].parent.name);return original(*a,**kw)
    monkeypatch.setattr(cache,'preprocess',record)
    assert cache.prepare_cache(root,plan,'development')['status']=='complete'
    assert first not in calls
    assert cache.prepare_cache(root,plan,'validation')['status']=='complete'
    index=cache.CacheIndex(root,plan)
    cache_file=c.cache_root(root,plan)/'development'/first/'image.npy'
    cache_file.write_bytes(cache_file.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):index.load('development',first)
    with pytest.raises(ValueError,match='artifact changed'):cache.verified_cache(root,plan,'development',first)
    with pytest.raises(ValueError,match='sealed'):cache.prepare_cache(root,plan,'test')


def test_sampling_only_available_classes_and_aligned_flip(tmp_path):
    image,label=save_pair(tmp_path)
    arrays,_=cache.preprocess(image,label,c.settings(),training=True)
    gt=arrays['label.npy'];gt[gt==3]=2
    counts=np.array([[np.count_nonzero(gt[:,:,z]==k) for z in range(gt.shape[2])] for k in range(4)])
    config=c.settings();config['patch_size']=[16]*3
    config['augmentation']={'flip_probability_per_axis':1.,'intensity_scale':[1.,1.],'intensity_shift':[0.,0.]}
    # Encode class in image so independent image/label flips would be detected.
    rng=np.random.default_rng(65)
    for _ in range(20):
        x,y,k=engine.augmented_patch(gt.astype(np.float32)/3,gt,counts,config,rng)
        assert k in (0,1,2) and np.allclose(x[x>=0],y[x>=0]/3.)


def test_empty_class_policy_and_reference_negative_false_positives():
    gt=np.zeros((3,3,3),dtype=np.uint8);pred=gt.copy()
    metric=validation.voxel_metrics(gt,pred,np.eye(4))
    assert metric['cyst']['dice']==1.
    pred[0,0,0]=3
    metric=validation.voxel_metrics(gt,pred,np.eye(4))
    assert metric['cyst']['dice']==0. and metric['cyst']['false_positive_ml']==.001
    summary=validation.summarize([{'case_id':'a','metrics':metric}],['a'])
    assert summary['classes']['cyst']['reference_positive_mean_dice'] is None
    assert summary['classes']['cyst']['reference_negative_with_prediction_n']==1
    with pytest.raises(ValueError):validation.summarize([{'case_id':'a','metrics':metric}],['a','b'])


def test_full_volume_prediction_covers_original_grid_without_reference(tmp_path):
    torch=pytest.importorskip('torch');pytest.importorskip('monai')
    class OracleIntensity(torch.nn.Module):
        def forward(self,x):return torch.cat((-x,x,x*0-10,x*0-10),dim=1)
    image=np.ones((17,19,21),dtype=np.float32)
    image[:8]=-1
    meta={'affine':np.eye(4).tolist(),'original_shape':list(image.shape),'original_affine':np.eye(4).tolist()}
    config=c.settings();config['inference']['roi_size']=[16]*3
    result=validation.full_volume_prediction(OracleIntensity(),image,meta,config,'cpu')
    assert result.shape==image.shape and np.array_equal(result,(image>0).astype(np.uint8))


@pytest.fixture
def small_training(prepared):
    root,protocol,output,plan=prepared
    # Reduced tensors/epochs exercise the real engine; production CLI rejects
    # any changed settings before entering this internal function.
    plan=deepcopy(plan)
    plan['settings']['patch_size']=[16]*3;plan['settings']['inference']['roi_size']=[16]*3
    plan['settings']['epochs']=2;plan['settings']['validation_epochs']=[1,2];plan['settings']['patches_per_case']=2
    for partition in ('development','validation'):cache.prepare_cache(root,plan,partition)
    return root,protocol,output,plan


def test_real_training_resume_matches_uninterrupted_and_preserves_epochs(small_training,tmp_path,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('monai')
    root,protocol,output,plan=small_training
    other=tmp_path/'uninterrupted';shutil.copytree(root,other)
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        first=engine.train(root,plan,{'test':'cpu'},epochs=1,device='cpu')
        assert first['status']=='partial' and first['completed_epochs']==1
        saved=engine.epoch_receipt(output,plan,1);saved_bytes=(output/'epochs/epoch_001'/saved['file']).read_bytes()
        resumed=engine.train(root,plan,{'test':'cpu'},epochs=1,device='cpu')
        continuous=engine.train(other,plan,{'test':'cpu'},epochs=2,device='cpu')
        assert resumed['status']==continuous['status']=='complete'
        assert resumed['optimizer_steps']==12 and resumed['selection']==continuous['selection']
        def load(base):
            out=c.output_root(base,plan);r=engine.epoch_receipt(out,plan,2)
            return torch.load(out/'epochs/epoch_002'/r['file'],map_location='cpu',weights_only=True)
        a,b=load(root),load(other)
        assert a['model_state_sha256']==b['model_state_sha256']
        assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
        assert (output/'epochs/epoch_001'/saved['file']).read_bytes()==saved_bytes
        with pytest.raises(ValueError,match='Frozen artifact'):engine.train(root,plan,{'test':'changed'},epochs=1,device='cpu')
        archive=experiment.export(root,plan)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None and 'selection.json' in z.namelist()
            assert not any(n.endswith(('.pt','.nii.gz','.npy')) for n in z.namelist())
    finally:torch.set_num_threads(old_threads)


def test_interrupted_epoch_repeats_without_overwriting_previous_checkpoint(small_training,monkeypatch):
    torch=pytest.importorskip('torch');pytest.importorskip('monai')
    root,protocol,output,plan=small_training
    old_threads=torch.get_num_threads();torch.set_num_threads(2)
    original=engine.augmented_patch;count=0
    def fail(*a,**kw):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('synthetic interruption')
        return original(*a,**kw)
    monkeypatch.setattr(engine,'augmented_patch',fail)
    try:
        with pytest.raises(RuntimeError,match='synthetic'):engine.train(root,plan,{'test':'cpu'},epochs=1,device='cpu')
        assert engine.epoch_receipt(output,plan,0) and engine.epoch_receipt(output,plan,1) is None
        monkeypatch.setattr(engine,'augmented_patch',original)
        assert engine.train(root,plan,{'test':'cpu'},epochs=1,device='cpu')['completed_epochs']==1
    finally:torch.set_num_threads(old_threads)


@pytest.mark.parametrize('args', [['--stage','train','--partition','test'],['--stage','train','--max-cases','1'],
                                 ['--stage','cache','--max-cases','0'],['--stage','train','--epochs','0']])
def test_cli_rejects_test_and_partial_training_cohorts(args):
    with pytest.raises(SystemExit) as e:script('run_kits23_supervised65').main(args)
    assert e.value.code==2
