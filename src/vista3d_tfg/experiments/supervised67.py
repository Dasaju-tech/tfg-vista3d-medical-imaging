"""Resumable paired CPU evaluation; no prediction or source artifact is overwritten."""
import re
from uuid import uuid4
import nibabel as nib
import numpy as np
from vista3d_tfg.training import contract65 as c, contract66 as a
from vista3d_tfg.experiments import contract67 as contract
from vista3d_tfg.analysis import sensitivity67 as sensitivity


def completed(root, plan, source, parent, cid):
    identity = contract.case_identity(plan, source, parent, cid)
    folder = contract.output_root(root, plan)/'cases'/cid
    if not (folder/'complete.json').exists(): return None
    receipt = c.read_json(folder/'complete.json'); c.check_digest(receipt, 'receipt_sha256')
    if (receipt['identity'] != identity or not re.fullmatch(r'attempt_[0-9a-f]{12}', receipt['attempt'])
            or set(receipt['artifacts']) != {'sensitivity.json'}):
        raise ValueError('Changed Block 67 receipt identity.')
    attempt = folder/receipt['attempt']
    if c.sha256_file(attempt/'sensitivity.json') != receipt['artifacts']['sensitivity.json']:
        raise ValueError('Changed Block 67 result artifact.')
    result = c.read_json(attempt/'sensitivity.json')
    if result['identity'] != identity or result['case_id'] != cid: raise ValueError('Changed result identity.')
    if (c.canonical_sha256(result['baseline']) != plan['source_cases'][cid]['baseline_sha256']
            or result.get('baseline_parity') is not True):
        raise ValueError('Frozen Block 66 baseline changed.')
    sensitivity.verify_result(result, plan['policy'])
    return result, receipt, attempt


def analyze_case(root, plan, source, parent, cid):
    identity = contract.case_identity(plan, source, parent, cid)
    original, receipt, original_folder = contract.source_case(root, plan, source, parent, cid)
    c.verify_input(root, parent, 'development', cid)
    _, gt_path = c.case_paths(root, parent, 'development', cid)
    gt_image, prediction_image = nib.load(gt_path), nib.load(original_folder/'prediction.nii.gz')
    affine = np.asarray(original['affine'], dtype=np.float64)
    for image in (gt_image, prediction_image):
        if (list(image.shape) != original['shape'] or not np.isfinite(image.affine).all()
                or not np.allclose(image.affine, affine, atol=1e-5, rtol=0)):
            raise ValueError('Original reference/prediction geometry changed.')
    gt, prediction = np.asanyarray(gt_image.dataobj), np.asanyarray(prediction_image.dataobj)
    result = sensitivity.evaluate(gt, prediction, affine, plan['policy'], announce=True)
    if any(result['baseline'][key] != original[key] for key in result['baseline']):
        raise ValueError('Exact Block 66 baseline parity failed.')
    result.update(case_id=cid, identity=identity, baseline_parity=True)
    del gt, prediction
    c.verify_input(root, parent, 'development', cid)
    if contract.source_case(root, plan, source, parent, cid)[1] != receipt:
        raise ValueError('Source changed during analysis.')
    case_root = contract.output_root(root, plan)/'cases'/cid
    attempt = case_root/('attempt_'+uuid4().hex[:12]); attempt.mkdir(parents=True)
    c.atomic_json(attempt/'sensitivity.json', result)
    record = {'identity': identity, 'attempt': attempt.name,
              'artifacts': {'sensitivity.json': c.sha256_file(attempt/'sensitivity.json')}}
    record['receipt_sha256'] = c.canonical_sha256(record)
    c.frozen_json(case_root/'complete.json', record)
    return result


def run(root, plan, source, parent, *, max_cases=None):
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('New-case budget must be positive.')
    rows, failures, attempted = [], [], 0
    out = contract.output_root(root, plan)
    def progress():
        done = [r['case_id'] for r in rows]
        state = {'status': 'failed' if failures else 'complete' if done == plan['case_ids'] else 'partial',
                 'experiment_sha256': plan['experiment_sha256'], 'requested_case_ids': plan['case_ids'],
                 'completed_case_ids': done, 'failed': failures, 'deferred': len(plan['case_ids'])-len(done)-len(failures),
                 'new_cases_this_call': attempted, 'gpu_inference': False, 'training': False,
                 'validation_test_volume_reads': False}
        c.atomic_json(out/'progress.json', state)
        c.atomic_json(out/'summary.json', sensitivity.summarize(rows, plan['case_ids'], plan['source_review_rows'], plan['policy']))
        return state
    for index, cid in enumerate(plan['case_ids']):
        try:
            saved = completed(root, plan, source, parent, cid)
            if saved:
                contract.source_case(root, plan, source, parent, cid)
                c.verify_input(root, parent, 'development', cid)
                rows.append(saved[0]); continue
            if max_cases is not None and attempted >= max_cases: continue
            attempted += 1
            print(f'[{index+1:03d}/{len(plan["case_ids"]):03d}] component sensitivity {cid}', flush=True)
            result = analyze_case(root, plan, source, parent, cid); rows.append(result)
            print(f'  baseline parity passed; 12 variants; {result["seconds"]:.2f} s', flush=True)
            progress()
        except (Exception, KeyboardInterrupt) as exc:
            failures.append({'case_id': cid, 'error': f'{type(exc).__name__}: {exc}'})
            print(f'  FAILED: {failures[-1]["error"]}', flush=True)
            return progress()
    return progress()


def dispatch(root, protocol, *, stage, max_cases=None):
    plan = c.read_json(root/contract.CONFIG); contract.check_plan(root, plan, protocol)
    source, parent = c.read_json(root/a.CONFIG), c.read_json(root/c.CONFIG)
    try:
        if stage == 'run': state = run(root, plan, source, parent, max_cases=max_cases)
        elif stage == 'report': state = {'status': 'report'}
        else: raise ValueError('Unknown stage.')
        contract.check_plan(root, plan, protocol)
    except (Exception, KeyboardInterrupt) as exc:
        state = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    if stage != 'report': c.atomic_json(contract.output_root(root, plan)/'last_action.json', {'stage': stage, **state})
    from vista3d_tfg.reporting.supervised67 import export
    archive, summary, issues = export(root, plan, source, parent)
    if issues:
        state = {**state, 'status': 'failed', 'error': str(issues)}
    elif stage == 'report': state = {'status': summary['status'], 'completed': summary['n_cases']}
    return state, archive
