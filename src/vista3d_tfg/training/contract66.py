"""Immutable development audit of the final Block 65b selected checkpoint."""
from pathlib import Path
import math
import re
from vista3d_tfg.training import contract65 as c, contract65b as b

CONFIG = 'configs/experiments/kits23_supervised_block66.json'
WORKER = 'scripts/kits23_supervised66_worker.py'
SOURCES = (WORKER, 'scripts/run_kits23_supervised66.py',
           'src/vista3d_tfg/training/contract66.py', 'src/vista3d_tfg/training/inference66.py',
           'src/vista3d_tfg/analysis/supervised66.py', 'src/vista3d_tfg/visualization/supervised66.py',
           'src/vista3d_tfg/experiments/supervised66.py')
POLICY = {'scope': 'development diagnostic; training cases, not independent performance',
          'labels': ['background', 'kidney', 'tumor', 'cyst'],
          'confusion': 'rows=reference, columns=prediction; original CT grid',
          'dice_empty': 'both empty=1; exactly one empty=0',
          'precision_empty_prediction': None, 'recall_empty_reference': None,
          'postprocessing': 'none', 'review_top_per_rule': 3, 'review_controls': 2,
          'review': 'outcome-guided slices; closest RAS voxel axes, native obliquity',
          'validation_test_inference': False}


def output_root(root, plan):
    return root/'outputs/reporting/full489_block66'/plan['audit_sha256'][:16]


def source_snapshot(root, protocol):
    source = c.read_json(root/b.CONFIG)
    parent = b.check_plan(root, source, protocol)
    out = c.output_root(root, source)
    if c.read_json(out/'protocol.json') != source:
        raise ValueError('Block 65b protocol copy changed.')
    selection = c.read_json(out/'selection.json')
    progress = c.read_json(out/'training_progress.json')
    total = source['settings']['epochs']
    steps = len(c.partition_ids(source, 'development'))*source['settings']['patches_per_case']
    if (progress.get('status') != 'complete' or progress.get('completed_epochs') != total
            or progress.get('planned_epochs') != total or progress.get('optimizer_steps') != total*steps
            or progress.get('test_reads') is not False or progress.get('selection') != selection
            or selection.get('final') is not True or selection.get('test_reads') is not False
            or selection.get('experiment_sha256') != source['experiment_sha256']
            or selection.get('evaluated_epochs') != source['settings']['validation_epochs']):
        raise ValueError('Block 66 requires complete, final Block 65b training/selection.')
    files = [root/b.CONFIG, out/'protocol.json', out/'selection.json', out/'training_progress.json']
    summaries, checkpoints = [], {}
    for epoch in range(total+1):
        origin = parent if epoch <= source['parent_last_epoch'] else source
        folder = c.output_root(root, origin)/'epochs'/f'epoch_{epoch:03d}'
        r = c.read_json(folder/'complete.json'); c.check_digest(r, 'receipt_sha256')
        if (r['epoch'] != epoch or r['experiment_sha256'] != origin['experiment_sha256']
                or not re.fullmatch(r'checkpoint_[0-9a-f]{12}\.pt', r['file'])
                or r['stats']['optimizer_steps'] != epoch*steps):
            raise ValueError('Invalid committed checkpoint history.')
        if epoch > source['parent_last_epoch'] and (
                r['stats'].get('recovery_revision') != b.REVISION
                or r['stats'].get('parent_experiment_sha256') != parent['experiment_sha256']):
            raise ValueError('Continuation checkpoint lineage changed.')
        files.append(folder/'complete.json')
        checkpoints[epoch] = (folder/r['file'], r, origin)
        if epoch in source['settings']['validation_epochs']:
            path = c.output_root(root, origin)/'validation'/f'epoch_{epoch:03d}'/'summary.json'
            s = c.read_json(path); c.check_digest(s, 'summary_sha256')
            scores = [s['classes'][k]['mean_dice'] for k in ('kidney', 'tumor', 'cyst')]
            score = sum(scores)/3
            if (s['epoch'] != epoch or s['experiment_sha256'] != origin['experiment_sha256']
                    or s['checkpoint_sha256'] != r['sha256'] or s['status'] != 'complete'
                    or s['n_cases'] != len(c.partition_ids(source, 'validation'))
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores)
                    or not math.isclose(score, s['selection_score'], abs_tol=1e-12, rel_tol=0)):
                raise ValueError('Invalid historical selection summary.')
            summaries.append(s); files.append(path)
    best = max(summaries, key=lambda s: (s['selection_score'], -s['epoch']))
    path, receipt, origin = checkpoints[best['epoch']]
    if (selection['best_epoch'] != best['epoch'] or selection['best_score'] != best['selection_score']
            or selection['best_checkpoint_sha256'] != receipt['sha256']
            or selection['best_checkpoint_experiment_sha256'] != origin['experiment_sha256']
            or c.sha256_file(path) != receipt['sha256']):
        raise ValueError('Selected checkpoint changed or disagrees with final selection.')
    files.append(path)
    selected = {'epoch': best['epoch'], 'sha256': receipt['sha256'],
                'experiment_sha256': origin['experiment_sha256'], 'path': path.relative_to(root).as_posix()}
    return source, parent, selected, {p.relative_to(root).as_posix(): c.sha256_file(p) for p in files}


def cache_receipts(root, parent):
    result = {}
    runtime = {'numpy': parent['runtime_identity']['numpy'],
               **{k: parent['runtime_identity']['packages'][k] for k in ('nibabel', 'scipy')}}
    for cid in c.partition_ids(parent, 'development'):
        r = c.read_json(c.cache_root(root, parent)/'development'/cid/'complete.json')
        c.check_digest(r, 'receipt_sha256')
        if (r['case_id'] != cid or r['partition'] != 'development'
                or r['experiment_sha256'] != parent['experiment_sha256']
                or r['input_identity'] != parent['inputs'][cid] or r['runtime'] != runtime
                or set(r['artifacts']) != {'image.npy', 'label.npy', 'slice_counts.npy', 'meta.json'}):
            raise ValueError(f'Development cache receipt changed: {cid}')
        result[cid] = r['receipt_sha256']
    return result


def build_plan(root, protocol):
    source, parent, selected, files = source_snapshot(root, protocol)
    plan = {'block': 66, 'protocol_sha256': source['protocol_sha256'],
            'source_experiment_sha256': source['experiment_sha256'],
            'cache_experiment_sha256': parent['experiment_sha256'], 'selected_checkpoint': selected,
            'case_ids': c.partition_ids(source, 'development'), 'source_files': files,
            'cache_receipts': cache_receipts(root, parent), 'policy': dict(POLICY),
            'implementation': {**b.code_identity(root), **{n: c.sha256_file(root/n) for n in SOURCES}}}
    plan['audit_sha256'] = c.canonical_sha256(plan)
    return plan


def freeze(root, protocol):
    plan = build_plan(root, protocol)
    c.frozen_json(root/CONFIG, plan)
    c.frozen_json(output_root(root, plan)/'protocol.json', plan)
    return plan


def check_plan(root, plan, protocol):
    c.check_digest(plan, 'audit_sha256')
    if plan != build_plan(root, protocol):
        raise ValueError('Block 66 source selection, cache receipts, code or policy changed.')
    if c.read_json(output_root(root, plan)/'protocol.json') != plan:
        raise ValueError('Block 66 protocol copy changed.')


def case_identity(plan, parent, cid):
    if cid not in plan['case_ids'] or cid not in c.partition_ids(parent, 'development'):
        raise ValueError('Only frozen development cases are allowed; validation/test are reserved.')
    return {'audit_sha256': plan['audit_sha256'], 'case_id': cid,
            'checkpoint_sha256': plan['selected_checkpoint']['sha256'],
            'input_identity': parent['inputs'][cid], 'cache_receipt_sha256': plan['cache_receipts'][cid]}
