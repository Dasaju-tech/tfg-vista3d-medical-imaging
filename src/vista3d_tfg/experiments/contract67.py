"""Freeze a completed Block 66 source and CPU-only sensitivity experiment."""
import platform
from importlib.metadata import version
from vista3d_tfg.training import contract65 as c, contract66 as a
from vista3d_tfg.training.inference66 import completed_case
from vista3d_tfg.analysis.supervised66 import review_selection
from vista3d_tfg.experiments.supervised66 import checked_review
from vista3d_tfg.analysis.components67 import POLICY, variants

CONFIG = 'configs/experiments/kits23_supervised_block67.json'
SOURCES = ('scripts/run_kits23_supervised67.py', 'src/vista3d_tfg/experiments/contract67.py',
           'src/vista3d_tfg/experiments/supervised67.py', 'src/vista3d_tfg/analysis/components67.py',
           'src/vista3d_tfg/analysis/sensitivity67.py', 'src/vista3d_tfg/reporting/supervised67.py')


def runtime_identity():
    return {'python': platform.python_version(), 'system': platform.system(),
            **{k: version(k) for k in ('numpy', 'scipy', 'nibabel')}}


def output_root(root, plan):
    return root/'outputs/reporting/full489_block67'/plan['experiment_sha256'][:16]


def case_fingerprint(report, receipt):
    keys = ('confusion_voxels', 'voxel_ml', 'metrics', 'predicted_cyst_reference_tissue_ml', 'reference_cyst_predicted_tissue_ml')
    return {'receipt_sha256': receipt['receipt_sha256'], 'artifacts': receipt['artifacts'],
            'baseline_sha256': c.canonical_sha256({k: report[k] for k in keys})}


def source_snapshot(root, protocol):
    source = c.read_json(root/a.CONFIG); a.check_plan(root, source, protocol)
    parent = c.read_json(root/c.CONFIG); out = a.output_root(root, source)
    progress = c.read_json(out/'progress.json'); review_progress = c.read_json(out/'review_progress.json')
    integrity = c.read_json(out/'export_integrity.json')
    if (progress.get('audit_sha256') != source['audit_sha256']
            or review_progress.get('audit_sha256') != source['audit_sha256']
            or progress.get('status') != 'complete' or progress.get('completed_case_ids') != source['case_ids']
            or progress.get('requested_case_ids') != source['case_ids'] or progress.get('failed') != []
            or progress.get('deferred') != 0 or progress.get('training') is not False
            or progress.get('validation_test_inference') is not False
            or integrity.get('status') != 'complete' or integrity.get('issues') != []
            or integrity.get('exported_cases') != source['case_ids']):
        raise ValueError('Block 67 requires complete, verified Block 66 development inference.')
    reports, cases = [], {}
    for cid in source['case_ids']:
        done = completed_case(root, source, parent, cid)
        if done is None: raise ValueError(f'Missing Block 66 case: {cid}')
        report, receipt, _ = done; reports.append(report)
        cases[cid] = case_fingerprint(report, receipt)
    rows = review_selection(reports, source['case_ids'], source['policy'])
    ids = [r['case_id'] for r in rows]
    if (c.read_json(out/'review_selection.json') != {'audit_sha256': source['audit_sha256'], 'rows': rows}
            or review_progress.get('status') != 'complete' or review_progress.get('completed_case_ids') != ids
            or review_progress.get('requested_case_ids') != ids or review_progress.get('failed') != []
            or review_progress.get('deferred') != 0 or review_progress.get('gpu_inference') is not False
            or review_progress.get('validation_test_inference') is not False
            or integrity.get('exported_reviews') != ids):
        raise ValueError('Block 67 requires the complete Block 66 visual review.')
    reviews = {}
    for row in rows:
        cid = row['case_id']; record = checked_review(out, source, row, cases[cid])
        if record is None: raise ValueError('Missing Block 66 review receipt.')
        reviews[cid] = record[0]['receipt_sha256']
    return source, parent, cases, rows, reviews


def build_plan(root, protocol):
    source, parent, cases, rows, reviews = source_snapshot(root, protocol)
    plan = {'block': 67, 'protocol_sha256': protocol['protocol_sha256'],
            'source_audit_sha256': source['audit_sha256'], 'case_ids': source['case_ids'],
            'source_cases': cases, 'source_review_rows': rows, 'source_review_receipts': reviews,
            'selected_checkpoint': source['selected_checkpoint'], 'policy': POLICY, 'variants': variants(),
            'runtime': runtime_identity(), 'implementation': {n: c.sha256_file(root/n) for n in SOURCES}}
    plan['experiment_sha256'] = c.canonical_sha256(plan)
    return plan


def freeze(root, protocol):
    plan = build_plan(root, protocol)
    c.frozen_json(root/CONFIG, plan)
    c.frozen_json(output_root(root, plan)/'protocol.json', plan)
    return plan


def check_plan(root, plan, protocol):
    c.check_digest(plan, 'experiment_sha256')
    if plan != build_plan(root, protocol): raise ValueError('Block 67 source, policy, code or CPU runtime changed.')
    if c.read_json(output_root(root, plan)/'protocol.json') != plan: raise ValueError('Block 67 protocol copy changed.')


def case_identity(plan, source, parent, cid):
    a.case_identity(source, parent, cid)
    if cid not in plan['case_ids'] or plan['source_audit_sha256'] != source['audit_sha256']:
        raise ValueError('Case is outside the frozen development experiment.')
    return {'experiment_sha256': plan['experiment_sha256'], 'case_id': cid,
            'source_receipt_sha256': plan['source_cases'][cid]['receipt_sha256']}


def source_case(root, plan, source, parent, cid):
    case_identity(plan, source, parent, cid)
    value = completed_case(root, source, parent, cid)
    if value is None or case_fingerprint(value[0], value[1]) != plan['source_cases'][cid]:
        raise ValueError('Frozen Block 66 case changed.')
    return value
