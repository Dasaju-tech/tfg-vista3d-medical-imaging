#!/usr/bin/env python
"""Block 72B: separate native-grid Validation80 metrics for frozen Block 71.

No training, test route, new loss or augmentation. Uses each arm's selected
checkpoint and the original inference pipeline. Stores new resumable JSON/CSV
and a report ZIP under outputs/reporting/full489_block72_metrics/.

Distances: voxel-center surfaces (6-connected erosion), exact nearest neighbors
in physical RAS coordinates. Euclidean L2 and Manhattan L1 are reported in mm.
HD95 = max of the two directed 95th percentiles (NumPy linear percentile).
ASSD = pooled mean of the two directed surface-voxel distance arrays.
Empty surfaces have undefined distance, recorded as null plus explicit counts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from uuid import uuid4

import numpy as np
from scipy.ndimage import binary_erosion, find_objects, generate_binary_structure
from scipy.spatial import cKDTree

LABELS = {'kidney': (1,), 'tumor': (2,), 'cyst': (3,), 'renal_envelope': (1, 2, 3)}
DISTANCES = tuple(f'{metric}_{norm}_mm' for norm in ('euclidean', 'manhattan')
                  for metric in ('hd', 'hd95', 'assd'))
REVISION = 'block72-native-ras-surfaces-v1'
DEFINITION = {
    'surface': 'centers of foreground voxels removed by 6-connected binary erosion; outside image is background',
    'geometry': 'original NIfTI voxel coordinates transformed by full affine to physical RAS mm; obliquity/shear preserved',
    'euclidean': 'exact cKDTree nearest-neighbor query p=2, eps=0, workers=1',
    'manhattan': 'exact cKDTree nearest-neighbor query p=1, eps=0, workers=1; L1 in fixed world RAS axes, orientation dependent',
    'hd': 'maximum of both directed distance arrays',
    'hd95': 'maximum of separate directed 95th percentiles, NumPy method=linear',
    'assd': 'pooled mean of both directed surface-voxel arrays; voxel weighted, not physical surface-area weighted',
    'empty': 'both empty: Dice/IoU=1, distances null; one empty: Dice/IoU=0, distances null; coverage/failure counts reported',
    'distance_aggregation': 'conditional on both masks nonempty; assess together with missing-prediction and absent-reference FP counts',
    'precision_recall': 'undefined when own denominator is zero; null, with defined-n in aggregates',
    'references': ['https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.spatial.cKDTree.query.html',
                   'https://monai.readthedocs.io/en/1.5.1/metrics.html#hausdorff-distance'],
}


def check_affine(affine):
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1]):
        raise ValueError('Expected a finite 4x4 affine.')
    if abs(np.linalg.det(a[:3, :3])) < 1e-12:
        raise ValueError('Singular affine.')
    return a


def surface_points(mask, affine):
    # Crop a single foreground bounding box; coordinates retain their offset.
    bounds = find_objects(mask, max_label=1)[0]
    local = mask[bounds]
    surface = local & ~binary_erosion(local, structure=generate_binary_structure(3, 1), border_value=0)
    indices = np.argwhere(surface).astype(np.float64)
    indices += np.array([s.start for s in bounds])
    return indices @ affine[:3, :3].T + affine[:3, 3]


def nearest_distances(source, target, p):
    tree = cKDTree(target)
    distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), 100000):
        end = min(start + 100000, len(source))
        distances[start:end] = tree.query(source[start:end], k=1, p=p, eps=0, workers=1)[0]
    return distances


def binary_metrics(reference, prediction, affine):
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError('Masks must occupy the same 3D grid.')
    a = check_affine(affine)
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    n = int(np.count_nonzero(reference))
    m = int(np.count_nonzero(prediction))
    tp = int(np.count_nonzero(reference & prediction))
    union = n + m - tp
    volume = abs(float(np.linalg.det(a[:3, :3]))) / 1000
    status = 'both_nonempty' if n and m else 'both_empty' if not n and not m else 'missing_prediction' if n else 'absent_reference_false_positive'
    result = {'reference_voxels': n, 'predicted_voxels': m, 'intersection_voxels': tp,
              'dice': 2 * tp / (n + m) if n + m else 1.0,
              'iou': tp / union if union else 1.0,
              'precision': tp / m if m else None, 'recall': tp / n if n else None,
              'false_positive_ml': (m - tp) * volume, 'false_negative_ml': (n - tp) * volume,
              'distance_status': status, **{name: None for name in DISTANCES}}
    if not n or not m:
        return result
    ref_points = surface_points(reference, a)
    pred_points = surface_points(prediction, a)
    for norm, p in (('euclidean', 2), ('manhattan', 1)):
        forward = nearest_distances(pred_points, ref_points, p)
        backward = nearest_distances(ref_points, pred_points, p)
        result[f'hd_{norm}_mm'] = float(max(forward.max(), backward.max()))
        result[f'hd95_{norm}_mm'] = float(max(np.percentile(forward, 95, method='linear'),
                                               np.percentile(backward, 95, method='linear')))
        result[f'assd_{norm}_mm'] = float((forward.sum() + backward.sum()) / (len(forward) + len(backward)))
    return result


def segmentation_metrics(reference, prediction, affine):
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    for array in (reference, prediction):
        if array.ndim != 3 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError('Expected 3D integer label arrays.')
        if array.min() < 0 or array.max() > 3:
            raise ValueError('Unexpected label outside 0,1,2,3.')
    return {label: binary_metrics(np.isin(reference, ids), np.isin(prediction, ids), affine)
            for label, ids in LABELS.items()}


def summarize_cases(reports):
    if not reports:
        raise ValueError('No complete validation reports.')
    result = {}
    scalar_names = ('dice', 'iou', 'precision', 'recall', 'false_positive_ml', 'false_negative_ml', *DISTANCES)
    for label in LABELS:
        values = [report['metrics'][label] for report in reports]
        positive = [v for v in values if v['reference_voxels'] > 0]
        stats = {'cases': len(values), 'reference_positive_n': len(positive),
                 'reference_positive_mean_dice': float(np.mean([v['dice'] for v in positive])) if positive else None,
                 'reference_positive_missing_prediction_n': sum(v['distance_status'] == 'missing_prediction' for v in values),
                 'reference_negative_with_prediction_n': sum(v['distance_status'] == 'absent_reference_false_positive' for v in values),
                 'both_empty_n': sum(v['distance_status'] == 'both_empty' for v in values),
                 'distance_defined_n': sum(v['distance_status'] == 'both_nonempty' for v in values),
                 'scalars': {}}
        for name in scalar_names:
            array = np.array([v[name] for v in values if v[name] is not None], dtype=float)
            if not np.isfinite(array).all():
                raise ValueError('Non-finite aggregate input.')
            stats['scalars'][name] = {'defined_n': len(array), 'mean': float(array.mean()) if len(array) else None,
                                     'std': float(array.std(ddof=1)) if len(array) > 1 else None,
                                     'median': float(np.median(array)) if len(array) else None}
        result[label] = stats
    return result


@contextmanager
def checked_source(root):
    import run_kits23_supervised71_compare as m
    import run_kits23_supervised71_resume as legacy
    if m.ROOT.resolve() != root:
        raise ValueError('Place the script in the project scripts directory.')
    frozen = m.b.read(root / m.CONFIG)
    m.b.check(frozen, 'experiment_sha256')
    original = m.build_plan
    observed = []
    def builder(*args, **kwargs):
        plan = original(*args, **kwargs)
        observed.append(deepcopy(plan['runtime']))
        return legacy.compatible_plan(plan, frozen)
    m.build_plan = builder
    try:
        parent, plan = m.load_plan(root)
        yield m, parent, plan, observed[-1]
    finally:
        m.build_plan = original


def selected_records(root, m, plan, index, arms):
    selected = {}
    for arm in arms:
        out = m.arm_root(root, plan, arm)
        records = m.history(out, plan, arm)
        candidates = []
        for record in records:
            if record['epoch'] > 0:
                summary = m.verified_summary(out, plan, index, arm, record)
                if summary is not None:
                    candidates.append((summary, record))
                elif record is not records[-1]:
                    raise ValueError('Historical epoch lacks completed validation.')
        if not candidates:
            raise ValueError(f'No selected validation checkpoint for {arm}.')
        summary, record = max(candidates, key=lambda pair: (pair[0]['selection_score'], -pair[0]['epoch']))
        selection = m.b.read(out / 'selection.json')
        if (selection['best_epoch'] != record['epoch'] or selection['best_score'] != summary['selection_score']
                or selection['checkpoint_sha256'] != record['sha256']
                or selection['checkpoint_file'] != f"epochs/epoch_{record['epoch']:03d}/{record['file']}"):
            raise ValueError('Saved selection disagrees with verified validation history.')
        selected[arm] = {'record': record, 'selection': selection,
                         'training_status': m.b.read(out / 'training_progress.json')['status']}
    return selected


def execute(root, args):
    import nibabel as nib
    import torch
    from run_kits23_supervised71_resume_io import atomic_json_write
    if not torch.cuda.is_available():
        raise RuntimeError('Use the original vista3d-nv CUDA environment.')
    with checked_source(root) as (m, parent, plan, runtime), m.execution_lock(root):
        index = m.verify_index(root, parent, plan)
        arms = m.ARMS if args.arm == 'both' else (args.arm,)
        selected = selected_records(root, m, plan, index, arms)
        protocol = {'revision': REVISION, 'source_experiment_sha256': plan['experiment_sha256'],
                    'selected': {arm: item['selection'] for arm, item in selected.items()},
                    'validation_case_ids': plan['partition']['validation']['case_ids'],
                    'definition': DEFINITION, 'runtime': plan['runtime'], 'test_reads': False,
                    'implementation': {name: hashlib.sha256((root / 'scripts' / name).read_bytes()).hexdigest()
                                       for name in ('run_kits23_supervised72_metrics.py', 'run_kits23_supervised71_resume_io.py',
                                                    'run_kits23_supervised71_resume.py')},
                    'interpretation': 'Exploratory validation snapshot; source arms may be incomplete. Conditional surface distances are not sufficient alone to rank models.'}
        protocol['analysis_sha256'] = m.b.digest(protocol)
        destination = root / 'outputs/reporting/full489_block72_metrics' / protocol['analysis_sha256'][:16]
        print(f'Block 72B metrics folder: {destination}', flush=True)
        print('Validation80 only; Test120 remains sealed. Selected checkpoint per requested arm:', flush=True)
        for arm, item in selected.items():
            print(f"  {arm}: epoch {item['record']['epoch']}; training status={item['training_status']}", flush=True)
        if args.stage == 'check':
            print('Block 72B check PASSED. No inference performed.', flush=True)
            return 0
        if (destination / 'protocol.json').exists():
            if m.b.read(destination / 'protocol.json') != protocol:
                raise ValueError('Metrics protocol changed.')
        else:
            atomic_json_write(destination / 'protocol.json', protocol)
        # Telemetry can vary slightly without changing any inference setting.
        # Record it per invocation outside the stable checkpoint-analysis hash.
        atomic_json_write(destination / 'runtime_audit' / (uuid4().hex + '.json'),
                          {'analysis_sha256': protocol['analysis_sha256'],
                           'observed_runtime': runtime, 'frozen_runtime': plan['runtime'],
                           'source_training_status': {arm: item['training_status'] for arm, item in selected.items()},
                           'compatibility': 'original narrow device_total_bytes exception; all other frozen fields checked',
                           'test_reads': False})
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        reports = {}
        for arm, item in selected.items():
            record = item['record']
            out = m.arm_root(root, plan, arm)
            payload = torch.load(out / 'epochs' / f"epoch_{record['epoch']:03d}" / record['file'], map_location='cpu', weights_only=True)
            if any(payload[key] != value for key, value in (('experiment_sha256', plan['experiment_sha256']),
                       ('arm', arm), ('epoch', record['epoch']), ('stats', record['stats']), ('runtime', plan['runtime']))):
                raise ValueError('Selected checkpoint payload mismatch.')
            model = m.initialized_model(plan['settings']['model'], arm, plan['settings']['seed']).to('cuda')
            model.load_state_dict(payload['model'], strict=True)
            if m.b.state_digest(model) != payload['model_state_sha256']:
                raise ValueError('Model state digest mismatch.')
            del payload
            model.eval()
            reports[arm] = []
            for number, cid in enumerate(protocol['validation_case_ids'], 1):
                m.c.verify_input(root, index.plan, 'validation', cid)
                casefolder = destination / arm / cid
                identity = {'analysis_sha256': protocol['analysis_sha256'], 'arm': arm, 'case_id': cid,
                            'source_identity': m.validation_identity(plan, index, arm, record, cid)}
                path = casefolder / 'metrics.json'
                if path.exists():
                    report = m.b.read(path)
                    m.b.check(report, 'report_sha256')
                    if report['identity'] != identity:
                        raise ValueError('Metrics receipt identity changed.')
                else:
                    print(f'{arm} extra metrics [{number:03d}/080] {cid}', flush=True)
                    started = time.perf_counter()
                    image, meta = index.load('validation', cid)
                    edge = None
                    edge_path = casefolder / 'edge_work.npy'
                    try:
                        if arm == 'ct_edges':
                            casefolder.mkdir(parents=True, exist_ok=True)
                            m.edge_array(image, meta, plan['settings'], edge_path)
                            edge = np.load(edge_path, mmap_mode='r')
                        predicted = m.predict_original(model, image, meta, plan['settings'], 'cuda', edge=edge)
                    finally:
                        if edge is not None:
                            edge._mmap.close()
                        edge_path.unlink(missing_ok=True)
                    _, gt_path = m.c.case_paths(root, index.plan, 'validation', cid)
                    reference = nib.load(gt_path)
                    if reference.header.get_xyzt_units()[0] != 'mm':
                        raise ValueError('Native NIfTI spatial units must be mm for these distance labels.')
                    if not np.allclose(reference.affine, meta['original_affine'], atol=1e-5, rtol=0):
                        raise ValueError('Reference/native-grid affine mismatch.')
                    ground_truth = np.asanyarray(reference.dataobj)
                    if ground_truth.shape != predicted.shape:
                        raise ValueError('Reference/prediction grid mismatch.')
                    old = m.case_metrics(out / 'validation' / f"epoch_{record['epoch']:03d}" / cid,
                                         identity['source_identity'])
                    if old is None:
                        raise ValueError('Missing original selected-epoch validation evidence.')
                    prediction_hash = m.array_sha(predicted)
                    report = {'identity': identity, 'case_id': cid, 'metrics': segmentation_metrics(ground_truth, predicted, meta['original_affine']),
                              'prediction_array_sha256': prediction_hash,
                              'source_prediction_array_sha256': old[0]['prediction_array_sha256'],
                              'matches_source_prediction': prediction_hash == old[0]['prediction_array_sha256'],
                              'seconds': time.perf_counter() - started, 'test_reads': False}
                    if not report['matches_source_prediction']:
                        print(f'WARNING: recomputed mask differs for {arm}/{cid}; all Block 72B metrics use this recomputed mask.', file=sys.stderr, flush=True)
                    m.c.verify_input(root, index.plan, 'validation', cid)
                    report['report_sha256'] = m.b.digest(report)
                    atomic_json_write(path, report)
                    del image, ground_truth, predicted, reference
                    gc.collect()
                reports[arm].append(report)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        summary = {'analysis_sha256': protocol['analysis_sha256'], 'test_reads': False,
                   'arms': {arm: {'training_status': selected[arm]['training_status'],
                                  'matches_source_prediction_n': sum(r['matches_source_prediction'] for r in rows),
                                  'classes': summarize_cases(rows)} for arm, rows in reports.items()},
                   'interpretation': protocol['interpretation'], 'distance_aggregation': DEFINITION['distance_aggregation']}
        atomic_json_write(destination / 'summary.json', summary)
        rows = [{'arm': arm, 'epoch': selected[arm]['record']['epoch'], 'case_id': r['case_id'],
                 'label': label, 'matches_source_prediction': r['matches_source_prediction'], **values}
                for arm, cases in reports.items() for r in cases for label, values in r['metrics'].items()]
        target_csv = destination / 'per_case_metrics.csv'
        with target_csv.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        target = destination / 'block72_validation_metrics_report.zip'
        temporary = target.with_name(target.name + '.' + uuid4().hex + '.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in [destination / 'protocol.json', destination / 'summary.json', target_csv,
                         *[destination / arm / cid / 'metrics.json' for arm in arms for cid in protocol['validation_case_ids']]]:
                archive.write(path, str(path.relative_to(destination)))
        os.replace(temporary, target)
        for arm, item in summary['arms'].items():
            envelope = item['classes']['renal_envelope']
            print(f"{arm}: renal-envelope HD95 L2 mean={envelope['scalars']['hd95_euclidean_mm']['mean']}; "
                  f"distance coverage={envelope['distance_defined_n']}/80; identical source masks={item['matches_source_prediction_n']}/80", flush=True)
        print(f'Share this ZIP: {target}', flush=True)
        print('Block 72B complete. No training; Test120 remains sealed.', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('self-test', 'check', 'metrics'), required=True)
    parser.add_argument('--arm', choices=('both', 'ct', 'ct_edges'), default='both')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    sys.path.insert(0, str(root / 'src'))
    from run_kits23_supervised71_resume_io import run_self_test
    if args.stage == 'self-test':
        return run_self_test(root, 'test_block72b_metrics.py')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return execute(root, args)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Rerun the same metrics command to reuse completed case receipts.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'Block 72B FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
