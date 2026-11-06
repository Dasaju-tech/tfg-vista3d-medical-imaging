"""Case-atomic inference using the selected checkpoint and existing CT cache."""
import re
import time
from uuid import uuid4
import nibabel as nib
import numpy as np
from vista3d_tfg.training import contract65 as c, contract66 as a
from vista3d_tfg.training.cache65 import verified_cache
from vista3d_tfg.training.supervised64 import build_model, state_digest
from vista3d_tfg.training.validation65 import full_volume_prediction
from vista3d_tfg.analysis.supervised66 import analyze_arrays, verify_report, summarize


def completed_case(root, plan, parent, cid):
    identity = a.case_identity(plan, parent, cid)
    folder = a.output_root(root, plan)/'cases'/cid
    if not (folder/'complete.json').exists(): return None
    r = c.read_json(folder/'complete.json'); c.check_digest(r, 'receipt_sha256')
    if (r['identity'] != identity or not re.fullmatch(r'attempt_[0-9a-f]{12}', r['attempt'])
            or set(r['artifacts']) != {'prediction.nii.gz', 'metrics.json'}):
        raise ValueError(f'Changed completion identity: {cid}')
    attempt = folder/r['attempt']
    for name, digest in r['artifacts'].items():
        if c.sha256_file(attempt/name) != digest: raise ValueError(f'Changed audit artifact: {cid}/{name}')
    report = c.read_json(attempt/'metrics.json')
    if report['identity'] != identity or report['case_id'] != cid: raise ValueError('Changed metric identity.')
    verify_report(report)
    return report, r, attempt


def verify_case_inputs(root, plan, parent, cid):
    a.case_identity(plan, parent, cid)
    c.verify_input(root, parent, 'development', cid)
    receipt = verified_cache(root, parent, 'development', cid)
    if receipt is None or receipt['receipt_sha256'] != plan['cache_receipts'][cid]:
        raise ValueError(f'Development cache changed: {cid}')
    folder = c.cache_root(root, parent)/'development'/cid
    paths = [folder/n for n in [*receipt['artifacts'], 'complete.json']]
    return folder, {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}


def load_selected_model(root, plan, source, runtime, device):
    import torch
    selected = plan['selected_checkpoint']; path = root/selected['path']
    if c.sha256_file(path) != selected['sha256']: raise ValueError('Selected checkpoint changed.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['epoch'] != selected['epoch'] or checkpoint['experiment_sha256'] != selected['experiment_sha256']
            or checkpoint['runtime'] != runtime):
        raise ValueError('Loaded checkpoint identity or runtime changed.')
    model = build_model(source['settings']); model.load_state_dict(checkpoint['model'], strict=True)
    if state_digest(model) != checkpoint['model_state_sha256']: raise ValueError('Model state digest mismatch.')
    del checkpoint
    return model.to(device).eval()


def run_case(root, plan, source, parent, cid, model, device):
    identity = a.case_identity(plan, parent, cid)
    folder, stamps = verify_case_inputs(root, plan, parent, cid)
    meta = c.read_json(folder/'meta.json')
    image = np.load(folder/'image.npy', mmap_mode='r', allow_pickle=False)
    if image.dtype != np.float32 or list(image.shape) != meta['shape']: raise ValueError('Invalid cached CT.')
    image_path, gt_path = c.case_paths(root, parent, 'development', cid)
    ct, gt_image = nib.load(image_path), nib.load(gt_path)
    if (list(ct.shape) != meta['original_shape'] or ct.shape != gt_image.shape
            or not np.isfinite(ct.affine).all() or not np.isfinite(gt_image.affine).all()
            or not np.allclose(ct.affine, meta['original_affine'], atol=1e-5, rtol=0)
            or not np.allclose(ct.affine, gt_image.affine, atol=1e-5, rtol=0)):
        raise ValueError('Original CT/reference/cache geometry differs.')
    started = time.perf_counter()
    # The predictor receives only the CT cache and CT geometry, never labels.
    pred = full_volume_prediction(model, image, meta, source['settings'], device)
    inference_seconds = time.perf_counter()-started
    gt = np.asanyarray(gt_image.dataobj)
    report = {'case_id': cid, 'identity': identity, 'shape': list(ct.shape), 'affine': ct.affine.tolist(),
              **analyze_arrays(gt, pred, ct.affine), 'inference_seconds': inference_seconds,
              'seconds': time.perf_counter()-started, 'timing_scope': 'inference and counts; excludes I/O audits and saving',
              'scope': 'development diagnostic; training case'}
    name = 'attempt_'+uuid4().hex[:12]; case_folder = a.output_root(root, plan)/'cases'/cid
    attempt = case_folder/name; attempt.mkdir(parents=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8, copy=False), ct.affine), attempt/'prediction.nii.gz')
    c.atomic_json(attempt/'metrics.json', report)
    c.verify_input(root, parent, 'development', cid)
    if any((p.stat().st_size, p.stat().st_mtime_ns) != s for p, s in stamps.items()):
        raise ValueError('Cache changed during inference.')
    receipt = {'identity': identity, 'attempt': name,
               'artifacts': {n: c.sha256_file(attempt/n) for n in ('prediction.nii.gz', 'metrics.json')}}
    receipt['receipt_sha256'] = c.canonical_sha256(receipt)
    c.frozen_json(case_folder/'complete.json', receipt)
    return report


def audit(root, plan, source, parent, runtime, *, max_cases=None, device='cuda'):
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('New-case budget must be positive.')
    model = None; attempted = 0; reports = []; failed = []
    output = a.output_root(root, plan)
    def progress():
        done = [r['case_id'] for r in reports]
        state = {'status': 'failed' if failed else 'complete' if done == plan['case_ids'] else 'partial',
                 'audit_sha256': plan['audit_sha256'], 'completed_case_ids': done,
                 'requested_case_ids': plan['case_ids'], 'failed': failed,
                 'deferred': len(plan['case_ids'])-len(done)-len(failed), 'new_cases_this_call': attempted,
                 'validation_test_inference': False, 'training': False}
        c.atomic_json(output/'progress.json', state)
        c.atomic_json(output/'summary.json', summarize(reports, plan['case_ids']))
        return state
    for number, cid in enumerate(plan['case_ids']):
        try:
            done = completed_case(root, plan, parent, cid)
            if done:
                c.verify_input(root, parent, 'development', cid)
                reports.append(done[0]); continue
            if max_cases is not None and attempted >= max_cases: continue
            attempted += 1
            print(f'[{number+1:03d}/{len(plan["case_ids"]):03d}] development audit {cid}', flush=True)
            if model is None: model = load_selected_model(root, plan, source, runtime, device)
            report = run_case(root, plan, source, parent, cid, model, device); reports.append(report)
            print('  count parity passed; '+ '; '.join(f'{k} Dice={report["metrics"][k]["dice"]:.4f}'
                  for k in ('kidney', 'tumor', 'cyst')), flush=True)
            progress()
        except (Exception, KeyboardInterrupt) as exc:
            failed.append({'case_id': cid, 'error': f'{type(exc).__name__}: {exc}'})
            print(f'  FAILED: {failed[-1]["error"]}', flush=True)
            return progress()
    return progress()
