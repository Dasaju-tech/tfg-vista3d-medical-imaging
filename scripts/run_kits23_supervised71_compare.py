#!/usr/bin/env python
r"""Block 71: paired CT+rotation versus CT+rotation+soft-edge training.

NEW file for scripts/. Requires the unchanged, completed Block 69 script/run.
Run from the project root with the vista3d-nv environment:
  python scripts/run_kits23_supervised71_compare.py --stage self-test
  python scripts/run_kits23_supervised71_compare.py --stage prepare
  python scripts/run_kits23_supervised71_compare.py --stage train --arm both --epochs 1
  python scripts/run_kits23_supervised71_compare.py --stage train --arm both --epochs 100
  python scripts/run_kits23_supervised71_compare.py --stage report

--epochs is an ADDITIONAL budget PER ARM, capped at the frozen maximum of 100.
Arms run sequentially, not simultaneously. Repeating a stopped arm verifies it
without restarting it. Partial training epochs repeat from their previous
checkpoint. Committed epochs with interrupted validation resume at missing cases.

Both arms: same seed, cases, centers, angles, flips and intensity jitter; 8 patches
per case; class sampling uniform among present classes; max 100, patience 10,
min_delta 1e-4, validation every epoch. Parent Block 68's optimizer, architecture,
patch size, normalization and cosine schedule are preserved.
Selection and early stopping maximize mean per-patient renal-envelope Dice
(union of labels 1, 2, 3) on all 80 validation cases. This metric does not
distinguish tissue classes; tumor/cyst metrics must be interpreted separately.
This is a fresh exploratory experiment following the Block 70 pilot; no pilot
checkpoints or stopping counters are imported. Existing files are unchanged.
Only B has the extra edge input. Shared parameters copy the same fresh random
initialization; extra input kernels are zero, so initial predictions agree.

DiceCE excludes padding outside the actual CT in BOTH terms. It equals MONAI's
foreground Dice + four-class CE on fully valid patches. This padding rule plus
rotations distinguishes BOTH arms from historical Block 68. Block 68 is context,
not the controlled comparator. No case is excluded due to padding or empty class.

Validation is unaugmented full volume on the original grid; no test execution
route. Soft edges are image-only, sigma 1.5 mm, generated in slabs with Gaussian
halo and explicit CT=-1 padding at true scan boundaries. They are cached once
per validation case, in a NEW output folder. No training edge cache: recompute
AFTER all sampled image transformations. Old source caches are read-only.

Epoch checkpoints and per-case validation metrics are retained; NIfTI predictions
are not retained every epoch. Reports contain JSON/CSV/HTML, not weights/volumes.
CPU stitching/argmax need host RAM; derived edge cache disk estimate is printed.
A single seeded comparison is exploratory, not independent test evidence.

References: https://monai.readthedocs.io/en/1.5.2/inferers.html
https://monai.readthedocs.io/en/1.5.2/losses.html#diceceloss
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import gc
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from uuid import uuid4
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import run_kits23_supervised69_edges_smoke as b
from vista3d_tfg.training import contract65 as c
from vista3d_tfg.training.cache65 import CacheIndex
from vista3d_tfg.training.validation65 import voxel_metrics, summarize as legacy_summarize
from vista3d_tfg.training.amp65b import optimizer_step

CONFIG = 'configs/experiments/kits23_supervised_block71_compare.json'
REVISION = 'block71-renal-envelope-selection-v1'
SELECTION_METRIC = 'mean_per_patient_renal_envelope_dice'
ARMS = ('ct','ct_edges')
SOURCES = [
    'scripts/run_kits23_supervised71_compare.py', 'scripts/run_kits23_supervised69_edges_smoke.py',
    'src/vista3d_tfg/training/contract65.py', 'src/vista3d_tfg/training/cache65.py',
    'src/vista3d_tfg/training/contract64.py', 'src/vista3d_tfg/training/supervised64.py',
    'src/vista3d_tfg/training/validation65.py', 'src/vista3d_tfg/training/amp65b.py',
]


def study_root(root,plan):
    return root/'outputs/reporting/full489_block71'/plan['experiment_sha256'][:16]


def arm_root(root,plan,arm):
    if arm not in ARMS: raise ValueError('Unknown arm.')
    return study_root(root,plan)/arm


@contextmanager
def execution_lock(root):
    # Reuse the crash-safe OS lock without changing the old module or old runs.
    folder=root/'outputs/reporting/full489_block71';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a+b') as f:
        f.seek(0,2)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('Another Block 71 invocation is active.') from e
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


def source_context(root):
    parent,source,lineage=b.context(root)
    old=b.read(root/b.CONFIG69);b.check(old,'experiment_sha256')
    if old['source']!=lineage or old['script_sha256']!=b.sha(root/'scripts/run_kits23_supervised69_edges_smoke.py'):
        raise ValueError('Block 69 source identity/code changed.')
    folder=root/'outputs/reporting/full489_block69'/old['experiment_sha256'][:16]
    done=b.verify_completed(folder,old)
    if done is None:raise ValueError('Complete Block 69 first.')
    if old['test_outputs_opened'] is not False:raise ValueError('Test must remain sealed.')
    return parent,source,{'block68':lineage,'block69_sha256':old['experiment_sha256'],
                          'block69_complete_sha256':b.sha(folder/'complete.json')}


def build_plan(root,parent,source,lineage):
    cfg=deepcopy(source['settings'])
    cfg['epochs']=100;cfg['validation_epochs']=list(range(1,101))
    cfg['selection_metric']=SELECTION_METRIC
    cfg['selection_definition']='unweighted mean of per-case Dice for union labels 1,2,3; all frozen validation cases'
    cfg['selection_ties']='earliest exact maximum; patience resets only above reference + min_delta'
    cfg['early_stopping']={'patience':10,'min_delta':1e-4,'mode':'max','max_epochs':100}
    cfg['rotation_degrees_per_world_axis']=10.0
    cfg['edge']={'sigma_mm':1.5,'truncate':4.0,'scale':'clip(sigma_mm*|grad_world CT|,0,1)',
                 'volume_boundary':'explicit CT=-1 halo','slab_depth':16}
    cfg['loss']='foreground soft Dice + four-class CE, weights 1/1; exclude CT padding in both terms'
    cfg['inference']['postprocessing']='none';cfg['inference']['tta']=False
    cfg['sampling']='uniform present class, uniform class voxel; same schedule in both arms; no padding-based case/patch rejection'
    cfg['validity']='nearest-neighbor voxel-cell support; explicit one-voxel air border for interpolation stability'
    cfg['schedule']='base_lr*0.5*(1+cos(pi*(epoch-1)/100))'
    cfg['initialization']='fresh shared weights; extra input channel kernels zero; equal initial predictions'
    plan={'block':71,'revision':REVISION,'partition':deepcopy(parent['partition']),
          'inputs':deepcopy(parent['inputs']),'cache_parent_experiment_sha256':parent['experiment_sha256'],
          'source':lineage,'settings':cfg,'arms':list(ARMS),'runtime':b.runtime(),
          'implementation':{name:b.sha(root/name) for name in SOURCES},
          'test_outputs_opened':False,'validation_for_model_selection':True,
          'paired_claim':'same patches/augmentations per common epoch, shared initial prediction; different input-channel capacity',
          'comparability_to_68':'historical only: both arms add rotations and valid-voxel loss masking'}
    plan['experiment_sha256']=b.digest(plan)
    return plan


def load_plan(root,*,create=False):
    parent,source,lineage=source_context(root)
    expected=build_plan(root,parent,source,lineage)
    path=root/CONFIG
    if path.exists():
        plan=b.read(path);b.check(plan,'experiment_sha256')
        if plan!=expected:raise ValueError('Block 71 config/code/runtime changed. Preserve the frozen run; do not overwrite.')
    elif create:
        plan=expected;b.freeze(path,plan)
    else:raise FileNotFoundError('Run --stage prepare first.')
    out=study_root(root,plan);b.freeze(out/'protocol.json',plan)
    return parent,plan


def verify_index(root,parent,plan):
    index=CacheIndex(root,parent)
    grids={};edge_bytes=0
    for part in ('development','validation'):
        for cid in plan['partition'][part]['case_ids']:
            folder=c.cache_root(root,parent)/part/cid;meta=b.read(folder/'meta.json')
            _,spacing=b.geometry(meta['affine'])
            if not np.allclose(spacing,plan['settings']['spacing_mm'],rtol=0,atol=1e-4):raise ValueError(f'Unexpected spacing: {cid}')
            if part=='validation':edge_bytes+=4*math.prod(meta['shape'])
            grids[cid]={'shape':meta['shape'],'affine':meta['affine'],'receipt_sha256':index.receipts[cid]}
    identity={'cache_parent_sha256':parent['experiment_sha256'],'cache_receipts':index.receipts,
              'grids_sha256':b.digest(grids),'edge_cache_estimated_bytes':edge_bytes}
    b.freeze(study_root(root,plan)/'input_audit.json',identity)
    print(f'Validation edge cache estimate: {edge_bytes/2**30:.2f} GiB (generated once for ct_edges).',flush=True)
    return index


def initialized_model(config,arm,seed):
    """Copy every shared weight, including shortcut convs; zero edge kernels."""
    import torch
    if arm not in ARMS:raise ValueError('Unknown arm.')
    torch.manual_seed(seed)
    one=deepcopy(config);one['in_channels']=1
    base=b.model_from_config(one)
    if arm=='ct':return base
    two=deepcopy(config);two['in_channels']=2
    model=b.model_from_config(two);state=model.state_dict();original=base.state_dict()
    if set(state)!=set(original):raise ValueError('Unexpected input-dependent parameter names.')
    for key,value in state.items():
        src=original[key]
        if src.shape==value.shape:value.copy_(src)
        elif src.ndim==5 and src.shape[1]==1 and value.shape[1]==2 and src.shape[0]==value.shape[0] and src.shape[2:]==value.shape[2:]:
            value.zero_();value[:,0:1].copy_(src)
        else:raise ValueError(f'Uncontrolled architecture difference: {key}')
    model.load_state_dict(state,strict=True)
    return model


def masked_dice_ce(logits,target,valid):
    import torch
    import torch.nn.functional as F
    logits=logits.float();y=target[:,0].long();w=valid[:,0].float()
    if logits.shape[1]!=4 or target.shape!=valid.shape or logits.shape[2:]!=target.shape[2:]:raise ValueError('Loss shape mismatch.')
    if not torch.all((w==0)|(w==1)).item() or not torch.all(w.flatten(1).sum(1)>0).item():raise ValueError('Invalid or empty CT validity mask.')
    ce=(F.cross_entropy(logits,y,reduction='none')*w).flatten(1).sum(1)/w.flatten(1).sum(1)
    ref=F.one_hot(y,num_classes=4).movedim(-1,1).float()
    prob=torch.softmax(logits,dim=1)
    mask=w[:,None];dims=tuple(range(2,logits.ndim))
    inter=(prob[:,1:]*ref[:,1:]*mask).sum(dims)
    denom=((prob[:,1:]+ref[:,1:])*mask).sum(dims)
    dice=1.-(2.*inter+1e-5)/(denom+1e-5)
    return dice.mean()+ce.mean()


def rotated_region(image,label,affine,center,shape,degrees):
    """Explicit air border prevents roundoff at a selected scan-edge voxel.

    Validity follows nearest-neighbor voxel-cell support, not a strict floating
    coordinate comparison at voxel centers. The same support is used for GT.
    """
    from scipy.ndimage import affine_transform
    import itertools
    A,_=b.geometry(affine);shape=np.asarray(shape,dtype=int)
    matrix=np.linalg.solve(A,b.rotation(degrees)@A)
    offset=np.asarray(center,dtype=float)-matrix@(shape//2)
    corners=np.array(list(itertools.product(*[(0,n-1) for n in shape])))
    coords=corners@matrix.T+offset
    lower=np.maximum(0,np.floor(coords.min(0)).astype(int)-2)
    upper=np.minimum(image.shape,np.ceil(coords.max(0)).astype(int)+3)
    if np.any(upper<=lower):raise ValueError('Rotation misses the CT.')
    sl=tuple(slice(int(lo),int(hi)) for lo,hi in zip(lower,upper))
    ct=np.asarray(image[sl],dtype=np.float32);gt=np.asarray(label[sl],dtype=np.uint8)
    if not np.isfinite(ct).all() or ct.min() < -1.0001 or ct.max()>1.0001 or not np.isin(gt,[0,1,2,3]).all():
        raise ValueError('Invalid source CT or labels in sampled region.')
    valid=np.pad(np.ones(ct.shape,dtype=np.uint8),1,constant_values=0)
    ct=np.pad(ct,1,constant_values=-1.);gt=np.pad(gt,1,constant_values=0)
    kw={'matrix':matrix,'offset':offset-lower+1,'output_shape':tuple(shape),'mode':'constant','prefilter':False}
    return (affine_transform(ct,order=1,cval=-1.,**kw),affine_transform(gt,order=0,cval=0,**kw),
            affine_transform(valid,order=0,cval=0,**kw))


def training_patch(image,label,counts,meta,cfg,rng,arm):
    present=np.flatnonzero(counts.sum(axis=1));k=int(rng.choice(present))
    center=b.center_for_class(label,counts,k,rng)
    degrees=rng.uniform(-cfg['rotation_degrees_per_world_axis'],cfg['rotation_degrees_per_world_axis'],3)
    A,spacing=b.geometry(meta['affine']);edgecfg=cfg['edge']
    halo=np.ceil(edgecfg['truncate']*edgecfg['sigma_mm']/spacing).astype(int)+2
    size=np.asarray(cfg['patch_size'],int)
    x,y,valid=rotated_region(image,label,meta['affine'],center,size+2*halo,degrees)
    aug=cfg['augmentation'];flips=[bool(rng.random()<aug['flip_probability_per_axis']) for _ in range(3)]
    scale=float(rng.uniform(*aug['intensity_scale']));shift=float(rng.uniform(*aug['intensity_shift']))
    x=np.clip(x*scale+shift,-1.,1.).astype(np.float32);x[valid==0]=-1.
    # Orthogonal grids make magnitude invariant to axis reflections. Recompute
    # the gradient from the final transformed CT, not from a rotated edge map.
    for axis,flip in enumerate(flips):
        if flip:x=np.flip(x,axis);y=np.flip(y,axis);valid=np.flip(valid,axis)
    x=np.ascontiguousarray(x)
    sl=tuple(slice(int(h),int(h+n)) for h,n in zip(halo,size))
    if arm=='ct':inputs=x[sl][None]
    elif arm=='ct_edges':
        edge,_=b.soft_edges(x,meta['affine'],edgecfg['sigma_mm'],edgecfg['truncate'])
        inputs=np.stack([x[sl],edge[sl]])
    else:raise ValueError('Unknown arm.')
    target=np.ascontiguousarray(y[sl],dtype=np.int64);mask=np.ascontiguousarray(valid[sl],dtype=np.uint8)
    selected=tuple(int(n//2-1 if flip else n//2) for n,flip in zip(size,flips))
    if target[selected]!=k or mask[selected]!=1:raise ValueError('Transformed center changed or is outside CT.')
    info={'center_class':k,'center':center,'angles_degrees':degrees.tolist(),'flips':flips,
          'intensity_scale':scale,'intensity_shift':shift,'padding_fraction':float(1-mask.mean())}
    return np.ascontiguousarray(inputs,dtype=np.float32),target,mask,info


def fill_edges(image,meta,cfg,result):
    """Bounded-RAM slabs, mathematically matching a fully padded image filter."""
    _,spacing=b.geometry(meta['affine']);e=cfg['edge']
    halo=np.ceil(e['truncate']*e['sigma_mm']/spacing).astype(int)+2
    shape=image.shape
    for start in range(0,shape[2],e['slab_depth']):
        end=min(start+e['slab_depth'],shape[2]);lo=max(0,start-halo[2]);hi=min(shape[2],end+halo[2])
        slab=np.asarray(image[:,:,lo:hi],dtype=np.float32)
        if not np.isfinite(slab).all() or float(slab.min())< -1.0001 or float(slab.max())>1.0001:raise ValueError('Invalid validation CT cache.')
        pad=((int(halo[0]),int(halo[0])),(int(halo[1]),int(halo[1])),
             (int(halo[2]-(start-lo)),int(halo[2]-(hi-end))))
        extended=np.pad(slab,pad,mode='constant',constant_values=-1.)
        edge,_=b.soft_edges(extended,meta['affine'],e['sigma_mm'],e['truncate'])
        result[:,:,start:end]=edge[halo[0]:halo[0]+shape[0],halo[1]:halo[1]+shape[1],halo[2]:halo[2]+end-start]


def edge_array(image,meta,cfg,destination):
    result=np.lib.format.open_memmap(destination,mode='w+',dtype=np.float32,shape=image.shape)
    fill_edges(image,meta,cfg,result)
    result.flush();del result


def ensure_edge(root,plan,index,cid,image,meta):
    if cid not in plan['partition']['validation']['case_ids']:raise ValueError('Validation-only derived edge cache.')
    folder=study_root(root,plan)/'edge_cache'/cid;marker=folder/'complete.json'
    identity={'experiment_sha256':plan['experiment_sha256'],'case_id':cid,'cache_receipt':index.receipts[cid],
              'policy':plan['settings']['edge'],'shape':list(image.shape),'affine':meta['affine']}
    if marker.exists():
        r=b.read(marker);b.check(r,'receipt_sha256')
        if r['identity']!=identity or r['sha256']!=b.sha(folder/'edge.npy'):raise ValueError(f'Derived edge cache changed: {cid}')
    else:
        folder.mkdir(parents=True,exist_ok=True)
        need=4*math.prod(image.shape)+2*2**30
        if shutil.disk_usage(folder).free<need:raise OSError(f'Need {need/2**30:.2f} GiB free for edge cache + reserve.')
        tmp=folder/('edge_'+uuid4().hex+'.npy')
        try:
            edge_array(image,meta,plan['settings'],tmp)
            os.replace(tmp,folder/'edge.npy')
        finally:tmp.unlink(missing_ok=True)
        r={'identity':identity,'sha256':b.sha(folder/'edge.npy')};r['receipt_sha256']=b.digest(r);b.write(marker,r)
    result=np.load(folder/'edge.npy',mmap_mode='r',allow_pickle=False)
    if result.shape!=image.shape or result.dtype!=np.float32:raise ValueError('Invalid edge array shape/dtype.')
    return result,r['receipt_sha256']


def predict_original(model,image,meta,cfg,device,edge=None):
    import torch
    import nibabel as nib
    from nibabel.processing import resample_from_to
    from monai.inferers import sliding_window_inference
    model.eval();shape=image.shape
    # CT=-1 padding when a scan is smaller than the ROI. In that case edges
    # are computed on the padded CT too, matching the training halo rule.
    roi=cfg['inference']['roi_size'];pad=[max(0,r-n) for r,n in zip(roi,shape)]
    before=[p//2 for p in pad];after=[p-p//2 for p in pad]
    padded=tuple(n+p for n,p in zip(shape,pad));channels=1 if edge is None else 2
    arr=np.empty((channels,*padded),dtype=np.float32);arr[0].fill(-1.)
    if edge is not None:arr[1].fill(0.)
    sl=tuple(slice(start,start+n) for start,n in zip(before,shape));arr[(0,*sl)]=image
    if edge is not None:
        if any(pad):fill_edges(arr[0],meta,cfg,arr[1])
        else:arr[(1,*sl)]=edge
    def predictor(patches):
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):logits=model(patches)
        if not torch.isfinite(logits).all().item():raise ValueError('Non-finite validation logits.')
        return logits.float()
    with torch.inference_mode():
        x=torch.from_numpy(arr)[None]
        logits=sliding_window_inference(x,roi_size=roi,sw_batch_size=cfg['inference']['sw_batch_size'],predictor=predictor,
                    overlap=cfg['inference']['overlap'],mode='gaussian',sw_device=device,device='cpu',progress=False)
        if logits.shape!=(1,4,*padded):raise ValueError('Invalid stitched shape.')
        # Chunked argmax avoids a full-volume int64 temporary.
        pred=np.empty(shape,dtype=np.uint8)
        for z in range(0,shape[2],16):
            end=min(shape[2],z+16)
            window=logits[0,:,before[0]:before[0]+shape[0],before[1]:before[1]+shape[1],before[2]+z:before[2]+end]
            pred[:,:,z:end]=window.argmax(0).numpy().astype(np.uint8)
    del x,arr,logits
    restored=resample_from_to(nib.Nifti1Image(pred,np.asarray(meta['affine'])),
                (tuple(meta['original_shape']),np.asarray(meta['original_affine'])),order=0,cval=0)
    return np.asarray(restored.dataobj,dtype=np.uint8)


def checkpoint_receipt(out,plan,arm,epoch):
    folder=out/'epochs'/f'epoch_{epoch:03d}';path=folder/'complete.json'
    if not path.exists():return None
    r=b.read(path);b.check(r,'receipt_sha256')
    if (r['experiment_sha256']!=plan['experiment_sha256'] or r['arm']!=arm or r['epoch']!=epoch
        or not re.fullmatch(r'checkpoint_[0-9a-f]{12}\.pt',r['file']) or b.sha(folder/r['file'])!=r['sha256']):
        raise ValueError('Checkpoint receipt/bytes changed.')
    return r


def history(out,plan,arm):
    records=[];last=plan['settings']['epochs']
    for epoch in range(last+1):
        r=checkpoint_receipt(out,plan,arm,epoch)
        if r is None:
            if any((out/'epochs'/f'epoch_{e:03d}'/'complete.json').exists() for e in range(epoch+1,last+1)):
                raise ValueError('Gap in committed checkpoint history.')
            break
        records.append(r)
    return records


def commit(out,plan,arm,model,opt,scaler,epoch,stats,device):
    import torch
    folder=out/'epochs'/f'epoch_{epoch:03d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():raise ValueError('Refusing to replace a completed checkpoint.')
    name='checkpoint_'+uuid4().hex[:12]+'.pt';path=folder/name
    payload={'experiment_sha256':plan['experiment_sha256'],'arm':arm,'epoch':epoch,'stats':stats,'runtime':plan['runtime'],
             'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
             'model_state_sha256':b.state_digest(model),'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []}
    torch.save(payload,path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if any(not torch.equal(t.detach().cpu(),restored['model'][n]) for n,t in model.state_dict().items()):raise ValueError('Checkpoint tensor round-trip failed.')
    del restored
    r={'experiment_sha256':plan['experiment_sha256'],'arm':arm,'epoch':epoch,'file':name,'sha256':b.sha(path),'stats':stats}
    r['receipt_sha256']=b.digest(r);b.write(folder/'complete.json',r)
    return r


def restore(out,plan,arm,record,model,opt,scaler,device):
    import torch
    p=torch.load(out/'epochs'/f"epoch_{record['epoch']:03d}"/record['file'],map_location='cpu',weights_only=True)
    if any(p[key]!=value for key,value in [('experiment_sha256',plan['experiment_sha256']),('arm',arm),('epoch',record['epoch']),('stats',record['stats']),('runtime',plan['runtime'])]):
        raise ValueError('Resume payload mismatch.')
    model.load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer']);scaler.load_state_dict(p['scaler'])
    if b.state_digest(model)!=p['model_state_sha256']:raise ValueError('Restored model digest mismatch.')
    torch.set_rng_state(p['torch_rng'])
    if device=='cuda':torch.cuda.set_rng_state_all(p['cuda_rng'])


def summarize(reports, ids):
    """Aggregate complete validation; absent cysts never directly raise selection."""
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Expected nonempty, unique frozen validation case IDs.')
    for report in reports:
        for label in ('kidney', 'tumor', 'cyst', 'renal_envelope'):
            value = report['metrics'][label]['dice']
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid validation Dice.')
        if report['metrics']['renal_envelope']['reference_voxels'] <= 0:
            raise ValueError('Selection requires a reference renal envelope in every case.')
    result = legacy_summarize(reports, ids)
    result['legacy_macro_dice_diagnostic_only'] = result.pop('selection_score')
    result['selection_score'] = result['classes']['renal_envelope']['mean_dice']
    result['selection_metric'] = SELECTION_METRIC
    for values in result['classes'].values():
        n = values['reference_negative_n']
        values['reference_negative_mean_false_positive_ml'] = (
            values['reference_negative_false_positive_ml'] / n if n else None)
    return result


def curve_metrics(summary):
    """Keep lesion detection and absent-class FP visible beside selection."""
    row = {'selection_metric': SELECTION_METRIC}
    for label, values in summary['classes'].items():
        for key in ('mean_dice', 'reference_positive_n', 'reference_positive_mean_dice',
                    'reference_negative_n', 'reference_negative_with_prediction_n',
                    'reference_negative_mean_false_positive_ml'):
            row[label + '_' + key] = values[key]
    return row


def validation_identity(plan,index,arm,record,cid):
    if cid not in plan['partition']['validation']['case_ids']:raise ValueError('Validation-only metric route.')
    return {'experiment_sha256':plan['experiment_sha256'],'arm':arm,'epoch':record['epoch'],'case_id':cid,
            'checkpoint_sha256':record['sha256'],'input_identity':plan['inputs'][cid],'cache_receipt':index.receipts[cid]}


def case_metrics(folder,identity):
    if not (folder/'complete.json').exists():return None
    r=b.read(folder/'complete.json');b.check(r,'receipt_sha256')
    if r['identity']!=identity or r['metrics_sha256']!=b.sha(folder/'metrics.json'):raise ValueError('Validation case receipt changed.')
    report=b.read(folder/'metrics.json')
    if report['identity']!=identity or report['case_id']!=identity['case_id']:raise ValueError('Validation case identity mismatch.')
    return report,r['receipt_sha256']


def verified_summary(out,plan,index,arm,record):
    if record['epoch']==0:return None
    folder=out/'validation'/f"epoch_{record['epoch']:03d}"
    if not (folder/'summary.json').exists():return None
    s=b.read(folder/'summary.json');b.check(s,'summary_sha256')
    if s['experiment_sha256']!=plan['experiment_sha256'] or s['arm']!=arm or s['epoch']!=record['epoch'] or s['checkpoint_sha256']!=record['sha256']:
        raise ValueError('Validation summary identity mismatch.')
    reports=[];receipts={};ids=plan['partition']['validation']['case_ids']
    for cid in ids:
        result=case_metrics(folder/cid,validation_identity(plan,index,arm,record,cid))
        if result is None:raise ValueError('Summary has missing validation evidence.')
        reports.append(result[0]);receipts[cid]=result[1]
    expected=summarize(reports,ids)
    if any(s[k]!=v for k,v in expected.items()) or s['case_receipts']!=receipts or s['status']!='complete':raise ValueError('Validation aggregate does not match cases.')
    return s


def validate(root,plan,index,model,arm,record,device):
    import nibabel as nib
    out=arm_root(root,plan,arm);epoch=record['epoch'];folder=out/'validation'/f'epoch_{epoch:03d}'
    previous=verified_summary(out,plan,index,arm,record)
    if previous is not None:return previous
    reports=[];receipts={};ids=plan['partition']['validation']['case_ids']
    for number,cid in enumerate(ids,1):
        c.verify_input(root,index.plan,'validation',cid)
        ident=validation_identity(plan,index,arm,record,cid);casefolder=folder/cid
        result=case_metrics(casefolder,ident)
        if result is None:
            print(f'{arm} validation epoch {epoch:03d} [{number:03d}/{len(ids):03d}] {cid}',flush=True)
            started=time.perf_counter();image,meta=index.load('validation',cid)
            edge=None;edge_id=None
            if arm=='ct_edges':edge,edge_id=ensure_edge(root,plan,index,cid,image,meta)
            predicted=predict_original(model,image,meta,plan['settings'],device,edge=edge)
            _,gt_path=c.case_paths(root,index.plan,'validation',cid);reference=nib.load(gt_path)
            if list(reference.shape)!=meta['original_shape'] or not np.allclose(reference.affine,meta['original_affine'],rtol=0,atol=1e-5):
                raise ValueError('Original-grid reference geometry changed.')
            values=voxel_metrics(np.asanyarray(reference.dataobj),predicted,meta['original_affine'])
            report={'case_id':cid,'identity':ident,'metrics':values,'prediction_array_sha256':array_sha(predicted),
                    'edge_receipt_sha256':edge_id,'seconds':time.perf_counter()-started}
            c.verify_input(root,index.plan,'validation',cid)
            b.write(casefolder/'metrics.json',report)
            r={'identity':ident,'metrics_sha256':b.sha(casefolder/'metrics.json')};r['receipt_sha256']=b.digest(r);b.write(casefolder/'complete.json',r)
            result=report,r['receipt_sha256']
            del predicted,image,edge,reference;gc.collect()
        reports.append(result[0]);receipts[cid]=result[1]
        b.write(folder/'progress.json',{'completed':number,'requested':len(ids),'case_ids':[r['case_id'] for r in reports]})
    result={'experiment_sha256':plan['experiment_sha256'],'arm':arm,'epoch':epoch,'checkpoint_sha256':record['sha256'],
            'status':'complete','case_receipts':receipts,**summarize(reports,ids)}
    result['summary_sha256']=b.digest(result);b.freeze(folder/'summary.json',result)
    return result


def early_state(summaries,cfg):
    best=None;reference=None;bad=0;stop=None;es=cfg['early_stopping']
    for s in summaries:
        score=s['selection_score']
        if not math.isfinite(score):raise ValueError('Non-finite selection score.')
        if best is None or score>best['selection_score']:best=s
        if reference is None or score>reference['selection_score']+es['min_delta']:reference=s;bad=0
        else:bad+=1
        if bad>=es['patience']:stop=s['epoch'];break
    return {'best_epoch':None if best is None else best['epoch'],'best_score':None if best is None else best['selection_score'],
            'no_improve':bad,'patience':es['patience'],'min_delta':es['min_delta'],'stopped_epoch':stop,'should_stop':stop is not None}


def write_progress(out,plan,arm,records,summaries):
    early=early_state(summaries,plan['settings'])
    committed=records[-1]['epoch'] if records else -1
    valid=summaries[-1]['epoch'] if summaries else 0
    if early['should_stop'] and committed>early['stopped_epoch']:
        raise ValueError('Checkpoint history extends past the frozen early-stop decision.')
    if early['should_stop']:status='early_stopped'
    elif valid>=plan['settings']['epochs']:status='max_epochs_reached'
    else:status='partial'
    if committed>valid:status='pending_validation'
    selection={'best_epoch':early['best_epoch'],'best_score':early['best_score'],'selection_metric':SELECTION_METRIC,'test_reads':False}
    if early['best_epoch'] is not None:
        rec=next(r for r in records if r['epoch']==early['best_epoch'])
        selection.update(checkpoint_file=f"epochs/epoch_{rec['epoch']:03d}/{rec['file']}",checkpoint_sha256=rec['sha256'])
    result={'status':status,'experiment_sha256':plan['experiment_sha256'],'arm':arm,'committed_epochs':max(0,committed),
            'validated_epochs':valid,'planned_max_epochs':plan['settings']['epochs'],
            'optimizer_steps':max(0,committed)*len(plan['partition']['development']['case_ids'])*plan['settings']['patches_per_case'],
            'selection_metric':SELECTION_METRIC,'selection':selection,'early_stopping':early,'test_reads':False}
    b.write(out/'training_progress.json',result);b.write(out/'selection.json',selection)
    return result


def train_arm(root,plan,index,arm,budget,device='cuda'):
    import torch
    if arm not in ARMS or budget<1:raise ValueError('Invalid arm or epoch budget.')
    out=arm_root(root,plan,arm);cfg=plan['settings'];records=history(out,plan,arm);summaries=[]
    # Verify every historical summary's receipts, not just its JSON digest.
    for rec in [r for r in records if r['epoch']>0]:
        summary=verified_summary(out,plan,index,arm,rec)
        if summary is None:
            if rec is not records[-1]:raise ValueError('Only the latest committed epoch may have pending validation.')
        else:summaries.append(summary)
    progress=write_progress(out,plan,arm,records,summaries)
    if progress['status'] in ('early_stopped','max_epochs_reached'):
        print(f'{arm}: already {progress["status"]}; best={progress["selection"]}',flush=True);return progress
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True);torch.manual_seed(cfg['seed'])
    if device=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
    model=initialized_model(cfg['model'],arm,cfg['seed']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['lr'],weight_decay=cfg['optimizer']['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda',init_scale=128.)
    if not records:
        stats={'epoch':0,'role':'fresh_paired_initialization','state_sha256':b.state_digest(model),'parameter_count':sum(p.numel() for p in model.parameters())}
        records.append(commit(out,plan,arm,model,opt,scaler,0,stats,device))
    else:restore(out,plan,arm,records[-1],model,opt,scaler,device)
    start_valid=summaries[-1]['epoch'] if summaries else 0
    target=min(cfg['epochs'],start_valid+budget)
    if records[-1]['epoch']>start_valid:
        summaries.append(validate(root,plan,index,model,arm,records[-1],device))
        progress=write_progress(out,plan,arm,records,summaries)
        if progress['early_stopping']['should_stop']:return progress
    ids=plan['partition']['development']['case_ids']
    for epoch in range(records[-1]['epoch']+1,target+1):
        started=time.perf_counter();model.train()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        rng=np.random.default_rng(cfg['seed']+epoch);ordered=[ids[int(i)] for i in rng.permutation(len(ids))]
        lr=cfg['optimizer']['lr']*.5*(1+math.cos(math.pi*(epoch-1)/cfg['epochs']))
        for group in opt.param_groups:group['lr']=lr
        losses=[];case_losses=[];class_centers={str(k):0 for k in range(4)};padding=[];schedule=hashlib.sha256();events=[];trace=[]
        for number,cid in enumerate(ordered,1):
            c.verify_input(root,index.plan,'development',cid)
            image,label,counts,meta=index.load('development',cid);local=[]
            for patch in range(cfg['patches_per_case']):
                x_np,y_np,mask_np,info=training_patch(image,label,counts,meta,cfg,rng,arm)
                identity={'case_id':cid,'patch':patch,**info};schedule.update(bytes.fromhex(b.digest(identity)))
                x=torch.from_numpy(x_np)[None].to(device);y=torch.from_numpy(y_np)[None,None].to(device);mask=torch.from_numpy(mask_np)[None,None].to(device)
                def record_retry(event):
                    entry={'epoch':epoch,'case_id':cid,'patch':patch,**event};events.append(entry)
                    b.write(out/'amp_events'/('event_'+uuid4().hex+'.json'),entry)
                    print(f'  {arm} AMP retry same patch: scale {event["scale_before"]} -> {event["scale_after"]}',flush=True)
                value,retries=optimizer_step(model,opt,scaler,lambda logits,target:masked_dice_ce(logits,target,mask),x,y,cfg['max_gradient_norm'],device,on_retry=record_retry)
                losses.append(value);local.append(value);class_centers[str(info['center_class'])]+=1;padding.append(info['padding_fraction'])
                if len(trace)<12:trace.append(identity)
                del x,y,mask
            case_losses.append({'case_id':cid,'mean_loss':float(np.mean(local))})
            print(f'{arm} epoch {epoch:03d}/100 [{number:03d}/{len(ids):03d}] {cid}; loss={np.mean(local):.6f}',flush=True)
            b.write(out/'active.json',{'status':'training','epoch':epoch,'processed_cases':number,'last_committed_epoch':epoch-1})
            del image,label,counts
        if device=='cuda':torch.cuda.synchronize()
        stats={'epoch':epoch,'optimizer_steps':epoch*len(ids)*cfg['patches_per_case'],'steps_this_epoch':len(losses),
               'mean_loss':float(np.mean(losses)),'lr':lr,'class_centers':class_centers,'case_order':ordered,'case_losses':case_losses,
               'sample_schedule_sha256':schedule.hexdigest(),'first_patch_records':trace,'amp_retries':len(events),
               'padding_mean':float(np.mean(padding)),'padding_max':max(padding),'padding_excluded_from_loss':True,
               'wall_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
               'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}
        other='ct_edges' if arm=='ct' else 'ct';otherpath=arm_root(root,plan,other)/'epochs'/f'epoch_{epoch:03d}'/'complete.json'
        if otherpath.exists():
            otherrec=checkpoint_receipt(arm_root(root,plan,other),plan,other,epoch)
            if stats['sample_schedule_sha256']!=otherrec['stats']['sample_schedule_sha256']:raise ValueError('Paired augmentation schedule differs between arms.')
        records.append(commit(out,plan,arm,model,opt,scaler,epoch,stats,device))
        write_progress(out,plan,arm,records,summaries)
        summaries.append(validate(root,plan,index,model,arm,records[-1],device))
        progress=write_progress(out,plan,arm,records,summaries)
        e=progress['early_stopping']
        print(f'{arm} epoch {epoch:03d}: score={summaries[-1]["selection_score"]:.6f}; best={e["best_score"]:.6f} @ {e["best_epoch"]}; no-improve={e["no_improve"]}/{e["patience"]}',flush=True)
        if e['should_stop']:break
    progress=write_progress(out,plan,arm,records,summaries)
    b.write(out/'active.json',{'status':progress['status'],'last_committed_epoch':records[-1]['epoch']})
    del model,opt,scaler;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return progress


def report(root,plan,index):
    out=study_root(root,plan);summary={'experiment_sha256':plan['experiment_sha256'],'arms':{},'test_reads':False,
          'selection_metric':SELECTION_METRIC,
          'interpretation':'Revised selection after Block 70 pilot. Renal-envelope Dice does not measure tissue discrimination. Single-seed exploratory validation comparison, not an independent test estimate. Historical Block 68 is contextual only.'}
    selected={};curves=[];common={}
    for arm in ARMS:
        folder=arm_root(root,plan,arm);records=history(folder,plan,arm);summaries=[]
        for rec in records:
            s=verified_summary(folder,plan,index,arm,rec)
            if s is None and rec['epoch']>0 and rec is not records[-1]:raise ValueError('Historical epoch lacks validation evidence.')
            if s is not None:
                summaries.append(s);curves.append({'arm':arm,'epoch':s['epoch'],'selection_score':s['selection_score'],'train_loss':rec['stats']['mean_loss'],**curve_metrics(s)})
        state=write_progress(folder,plan,arm,records,summaries);summary['arms'][arm]=state
        common[arm]={r['epoch']:r['stats'].get('sample_schedule_sha256') for r in records if r['epoch']>0}
        best=state['selection']['best_epoch']
        if best is not None:
            selected[arm]=b.read(folder/'validation'/f'epoch_{best:03d}'/'summary.json')
            summary['arms'][arm]['selected_validation']=selected[arm]
    shared=sorted(set(common['ct'])&set(common['ct_edges']))
    if any(common['ct'][e]!=common['ct_edges'][e] for e in shared):raise ValueError('Paired schedule audit failed.')
    summary['paired_schedule_epochs_verified']=shared
    summary['comparison_complete']=all(summary['arms'][a]['status'] in ('early_stopped','max_epochs_reached') for a in ARMS)
    if len(selected)==2:
        summary['delta_score_edges_minus_ct']=selected['ct_edges']['selection_score']-selected['ct']['selection_score']
        differences=[]
        for cid in plan['partition']['validation']['case_ids']:
            a=b.read(arm_root(root,plan,'ct')/'validation'/f"epoch_{selected['ct']['epoch']:03d}"/cid/'metrics.json')
            d=b.read(arm_root(root,plan,'ct_edges')/'validation'/f"epoch_{selected['ct_edges']['epoch']:03d}"/cid/'metrics.json')
            row={'case_id':cid}
            for k in ('kidney','tumor','cyst','renal_envelope'):
                row[k+'_delta_dice']=d['metrics'][k]['dice']-a['metrics'][k]['dice']
                row[k+'_reference_voxels']=a['metrics'][k]['reference_voxels']
            differences.append(row)
        summary['worst_case_changes']={k:sorted(differences,key=lambda row:row[k+'_delta_dice'])[:10] for k in ('kidney','tumor','cyst','renal_envelope')}
        write_csv(out/'paired_validation.csv',differences)
    b.write(out/'comparison.json',summary);write_csv(out/'learning_curves.csv',curves)
    rows=''.join('<tr><td>'+a+'</td><td>'+html.escape(s['status'])+'</td><td>'+str(s['validated_epochs'])+'</td><td>'+str(s['selection']['best_epoch'])+'</td><td>'+str(s['selection']['best_score'])+'</td></tr>' for a,s in summary['arms'].items())
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>Bloque 71</title><style>body{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px}td,th{padding:12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>Bloque 71: CT frente a CT + bordes</h1><p>Rotaciones y perdida Dice + CE con exclusion del relleno en ambos modelos. Seleccion y early stopping: media por paciente del Dice de la union rinon + tumor + quiste. Validacion para seleccion; test120 reservado.</p><p>El Dice conjunto no distingue confusiones entre tejidos. Revisar tumor, quistes positivos y FP en casos negativos en learning_curves.csv y el detalle. Inicio desde cero; bloque 70 conservado como piloto.</p><p>Comparacion terminada: '+str(summary['comparison_complete'])+'. Una semilla; no es evaluacion independiente.</p><table><tr><th>Modelo</th><th>Estado</th><th>Epocas validadas</th><th>Mejor epoca</th><th>Dice conjunto renal</th></tr>'+rows+'</table><h2>Detalle y limitaciones</h2><pre>'+html.escape(json.dumps(summary,indent=2,ensure_ascii=False))+'</pre></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    files=[out/n for n in ('protocol.json','input_audit.json','comparison.json','report.html','learning_curves.csv','paired_validation.csv','last_error.json') if (out/n).exists()]
    for arm in ARMS:
        folder=arm_root(root,plan,arm)
        files.extend(folder/n for n in ('training_progress.json','selection.json','active.json') if (folder/n).exists())
        for sub in ('epochs','validation','amp_events'):files.extend(sorted((folder/sub).rglob('*.json')))
    target=out/'block71_comparison_report.zip';temp=target.with_suffix('.tmp')
    manifest={p.relative_to(out).as_posix():b.sha(p) for p in files}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(out).as_posix())
        z.writestr('archive_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    os.replace(temp,target)
    print(f'Share this ZIP: {target}',flush=True)
    print('Validation80 is used for model selection. Test120 remains sealed.',flush=True)
    return summary,target


def array_sha(array):
    h=hashlib.sha256()
    for i in range(0,array.shape[0],8):h.update(np.ascontiguousarray(array[i:i+8]).tobytes())
    return h.hexdigest()


def write_csv(path,rows):
    if not rows:return
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    path.write_text(stream.getvalue(),encoding='utf-8')


def selection_self_test():
    # Same foreground union, different tissue labels: same NEW score.
    gt = np.array([1, 2, 3, 0], dtype=np.uint8).reshape(1, 1, 4)
    wrong_tissue = np.array([1, 1, 1, 0], dtype=np.uint8).reshape(gt.shape)
    def report(pred):
        return {'case_id': 'synthetic', 'metrics': voxel_metrics(gt, pred, np.eye(4))}
    a = summarize([report(gt)], ['synthetic'])
    d = summarize([report(wrong_tissue)], ['synthetic'])
    assert a['selection_score'] == d['selection_score'] == 1.
    assert d['classes']['cyst']['reference_positive_mean_dice'] == 0.
    assert a['legacy_macro_dice_diagnostic_only'] > d['legacy_macro_dice_diagnostic_only']
    # Improving union wins even when the old absent-class macro metric falls.
    seq = [{'epoch': 1, 'selection_score': .4, 'legacy_macro_dice_diagnostic_only': .8},
           {'epoch': 2, 'selection_score': .6, 'legacy_macro_dice_diagnostic_only': .3}]
    state = early_state(seq, {'early_stopping': {'patience': 10, 'min_delta': 1e-4}})
    assert state['best_epoch'] == 2 and state['no_improve'] == 0
    print('PASS: renal-envelope selection; tissue errors remain visible; legacy score diagnostic only')


def self_test():
    import tempfile
    import torch
    from monai.losses import DiceCELoss
    torch.set_num_threads(2)
    modelcfg={'spatial_dims':3,'in_channels':1,'out_channels':4,'channels':[4,8,16],'strides':[2,2],'num_res_units':1,'norm':'INSTANCE','dropout':0.}
    a=initialized_model(modelcfg,'ct',6501).eval();d=initialized_model(modelcfg,'ct_edges',6501).eval()
    x=torch.randn(1,2,16,16,16)
    with torch.no_grad():assert torch.allclose(a(x[:,:1]),d(x),rtol=1e-5,atol=1e-5)
    print('PASS: shared initialization and equal initial predictions')
    logits=torch.randn(1,4,16,16,16,requires_grad=True);y=torch.randint(0,4,(1,1,16,16,16));valid=torch.ones_like(y)
    standard=DiceCELoss(include_background=False,to_onehot_y=True,softmax=True)(logits,y)
    masked=masked_dice_ce(logits,y,valid);assert torch.allclose(standard,masked,atol=2e-6)
    grad1=torch.autograd.grad(standard,logits,retain_graph=True)[0];grad2=torch.autograd.grad(masked,logits)[0]
    assert torch.allclose(grad1,grad2,atol=1e-7)
    valid[:,:,:4]=0;loss=masked_dice_ce(logits,y,valid);g=torch.autograd.grad(loss,logits)[0];assert torch.count_nonzero(g[:,:,:4])==0
    print('PASS: DiceCE value/gradient parity; padding contributes zero loss gradient')
    cfg={'patch_size':[16]*3,'rotation_degrees_per_world_axis':10.,'augmentation':{'flip_probability_per_axis':.5,'intensity_scale':[.9,1.1],'intensity_shift':[-.1,.1]},
         'edge':{'sigma_mm':1.5,'truncate':4.,'slab_depth':5},'inference':{'roi_size':[16]*3,'sw_batch_size':2,'overlap':.25}}
    rng=np.random.default_rng(2);image=rng.uniform(-1,1,(27,25,29)).astype(np.float32);label=rng.integers(0,4,image.shape,dtype=np.uint8)
    counts=np.array([[np.count_nonzero(label[:,:,z]==k) for z in range(image.shape[2])] for k in range(4)],dtype=np.int64)
    meta={'shape':list(image.shape),'affine':np.diag([1.5,1.5,1.5,1]).tolist(),'original_shape':list(image.shape),'original_affine':np.diag([1.5,1.5,1.5,1]).tolist()}
    pa=training_patch(image,label,counts,meta,cfg,np.random.default_rng(32),'ct');pb=training_patch(image,label,counts,meta,cfg,np.random.default_rng(32),'ct_edges')
    assert np.array_equal(pa[0][0],pb[0][0]) and np.array_equal(pa[1],pb[1]) and np.array_equal(pa[2],pb[2]) and pa[3]==pb[3]
    print('PASS: identical CT, targets, rotations, flips and intensity transforms across arms')
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'edge.npy';edge_array(image,meta,cfg,path);edge=np.load(path)
        full,_=b.soft_edges(np.pad(image,6,constant_values=-1.),meta['affine'],1.5,4.)
        assert np.allclose(edge,full[6:-6,6:-6,6:-6],atol=1e-6)
        pred=predict_original(a,image,meta,cfg,'cpu');pred2=predict_original(d,image,meta,cfg,'cpu',edge=edge)
        assert np.array_equal(pred,pred2) and pred.shape==image.shape
    print('PASS: slab/full-volume Gaussian parity and paired original-grid inference')
    cfg['early_stopping']={'patience':3,'min_delta':1e-4};seq=[{'epoch':i+1,'selection_score':v} for i,v in enumerate([.4,.5,.499,.49,.48])]
    e=early_state(seq,cfg);assert e['should_stop'] and e['best_epoch']==2 and e['stopped_epoch']==5
    print('PASS: early stopping; best checkpoint retained')
    selection_self_test()
    print('Block 71 self-test PASSED. Synthetic CPU only; no patient data read.')


def execute(root,args):
    parent,plan=load_plan(root,create=args.stage=='prepare');index=None
    try:
        index=verify_index(root,parent,plan)
        if args.stage=='train':
            for arm in ARMS if args.arm=='both' else (args.arm,):
                result=train_arm(root,plan,index,arm,args.epochs)
                print(f'{arm}: {result["status"]}; validated epochs={result["validated_epochs"]}; best={result["selection"]}',flush=True)
                report(root,plan,index)
        elif args.stage=='prepare':print('Block 71 frozen; all development/validation caches verified; no training yet.',flush=True)
        else:report(root,plan,index)
        load_plan(root)  # implementation/runtime/source are still frozen
        # A successful invocation resolves the current error marker, while
        # retaining its historical evidence. The active report then reflects it.
        last=study_root(root,plan)/'last_error.json'
        if last.exists():
            archive=last.parent/'error_history';archive.mkdir(exist_ok=True)
            os.replace(last,archive/('error_'+uuid4().hex+'.json'))
            if args.stage in ('train','report'):report(root,plan,index)
    except BaseException as exc:
        b.write(study_root(root,plan)/'last_error.json',{'error':f'{type(exc).__name__}: {exc}','stage':args.stage,'arm':args.arm,'test_reads':False})
        if index is not None:
            try:report(root,plan,index)
            except Exception as report_error:print(f'Report check failed too: {report_error}',file=sys.stderr)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage',choices=['self-test','prepare','train','report'],required=True)
    parser.add_argument('--arm',choices=['both',*ARMS],default='both')
    parser.add_argument('--epochs',type=int,default=1,help='Additional validated epochs per requested arm, default 1.')
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args(argv)
    if not 1<=args.epochs<=100:parser.error('--epochs must be between 1 and 100.')
    root=args.project_root.resolve()
    if args.stage!='self-test' and root!=ROOT:parser.error('Copy this script into the target checkout scripts directory; --project-root must match it.')
    try:
        if args.stage=='self-test':self_test();return 0
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Use the vista3d-nv CUDA environment.')
        with execution_lock(root):execute(root,args)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Rerun the same command: committed checkpoints and completed validation cases are preserved.',file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Block 71 FAILED: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
