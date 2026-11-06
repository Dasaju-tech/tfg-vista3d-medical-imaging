"""Original-grid counts and development diagnostics, no surface recomputation."""
import math
import statistics
import numpy as np

LABELS = ('background', 'kidney', 'tumor', 'cyst')
CLASSES = ('kidney', 'tumor', 'cyst', 'renal_envelope')


def metrics_from_confusion(matrix, voxel_ml):
    a = np.asarray(matrix)
    if (a.shape != (4, 4) or not np.issubdtype(a.dtype, np.integer) or (a < 0).any()
            or not math.isfinite(voxel_ml) or voxel_ml <= 0):
        raise ValueError('Invalid confusion counts or voxel volume.')
    result = {}
    for name, labels in zip(CLASSES, ([1], [2], [3], [1, 2, 3])):
        ref, pred = int(a[labels, :].sum()), int(a[:, labels].sum())
        tp = int(a[np.ix_(labels, labels)].sum())
        result[name] = {'reference_voxels': ref, 'predicted_voxels': pred, 'intersection_voxels': tp,
                        'dice': 2*tp/(ref+pred) if ref+pred else 1.,
                        'precision': tp/pred if pred else None, 'recall': tp/ref if ref else None,
                        'reference_ml': ref*voxel_ml, 'predicted_ml': pred*voxel_ml,
                        'false_positive_ml': (pred-tp)*voxel_ml, 'false_negative_ml': (ref-tp)*voxel_ml}
    return result


def analyze_arrays(gt, pred, affine):
    if gt.shape != pred.shape or len(gt.shape) != 3 or any(n < 1 for n in gt.shape):
        raise ValueError('Invalid original-grid label shapes.')
    affine = np.asarray(affine)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Invalid original-grid affine.')
    voxel_ml = abs(float(np.linalg.det(affine[:3, :3])))/1000
    counts = np.zeros((4, 4), dtype=np.int64)
    for start in range(0, gt.shape[0], 8):
        ref, prediction = gt[start:start+8], pred[start:start+8]
        if not np.isin(ref, (0, 1, 2, 3)).all() or not np.isin(prediction, (0, 1, 2, 3)).all():
            raise ValueError('Non-integer, non-finite or unsupported labels.')
        encoded = ref.astype(np.uint8)*4 + prediction.astype(np.uint8)
        counts += np.bincount(encoded.ravel(), minlength=16).reshape(4, 4)
    if int(counts.sum()) != math.prod(gt.shape): raise ValueError('Voxel count parity failed.')
    return {'confusion_voxels': counts.tolist(), 'voxel_ml': voxel_ml,
            'metrics': metrics_from_confusion(counts, voxel_ml),
            'predicted_cyst_reference_tissue_ml': {n: int(counts[i, 3])*voxel_ml for i, n in enumerate(LABELS)},
            'reference_cyst_predicted_tissue_ml': {n: int(counts[3, i])*voxel_ml for i, n in enumerate(LABELS)}}


def verify_report(r):
    expected = metrics_from_confusion(r['confusion_voxels'], r['voxel_ml'])
    shape = r['shape']
    if (len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape)
            or expected != r['metrics'] or sum(map(sum, r['confusion_voxels'])) != math.prod(shape)):
        raise ValueError('Saved metric/count parity failed.')
    counts = r['confusion_voxels']; volume = r['voxel_ml']
    if (r['predicted_cyst_reference_tissue_ml'] != {n: counts[i][3]*volume for i, n in enumerate(LABELS)}
            or r['reference_cyst_predicted_tissue_ml'] != {n: counts[3][i]*volume for i, n in enumerate(LABELS)}):
        raise ValueError('Saved cyst confusion parity failed.')
    if not math.isfinite(r['seconds']) or r['seconds'] < 0: raise ValueError('Invalid timing.')


def describe(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def summarize(reports, requested_ids):
    ids = [r['case_id'] for r in reports]
    if len(ids) != len(set(ids)) or ids != [cid for cid in requested_ids if cid in set(ids)]:
        raise ValueError('Duplicate, unordered or non-development report IDs.')
    for report in reports: verify_report(report)
    classes = {}
    for name in CLASSES:
        values = [r['metrics'][name] for r in reports]
        positive = [v for v in values if v['reference_voxels'] > 0]
        negative = [v for v in values if v['reference_voxels'] == 0]
        classes[name] = {'dice_all': describe([v['dice'] for v in values]),
                         'dice_reference_positive': describe([v['dice'] for v in positive]),
                         'precision_defined': describe([v['precision'] for v in values]),
                         'recall_reference_positive': describe([v['recall'] for v in positive]),
                         'reference_negative_n': len(negative),
                         'reference_negative_with_prediction_n': sum(v['predicted_voxels'] > 0 for v in negative),
                         'reference_negative_false_positive_ml': describe([v['false_positive_ml'] for v in negative])}
    counts = np.zeros((4, 4), dtype=np.int64); volumes = np.zeros((4, 4), dtype=np.float64)
    for r in reports:
        a = np.asarray(r['confusion_voxels'], dtype=np.int64); counts += a; volumes += a*r['voxel_ml']
    return {'status': 'complete' if ids == requested_ids else 'partial', 'n_cases': len(ids),
            'requested_n': len(requested_ids), 'scope': 'development diagnostic; training cases',
            'classes': classes, 'confusion_voxels_sum': counts.tolist(), 'confusion_ml_sum': volumes.tolist(),
            'inference_and_counts_seconds': describe([r['seconds'] for r in reports])}


def review_selection(reports, requested_ids, policy):
    if [r['case_id'] for r in reports] != requested_ids:
        raise ValueError('Review ranking requires the complete frozen development cohort.')
    def ranked(name, positive, metric='dice', reverse=False):
        rows = [r for r in reports if (r['metrics'][name]['reference_voxels'] > 0) == positive]
        return sorted(rows, key=lambda r: ((-1 if reverse else 1)*r['metrics'][name][metric], r['case_id']))
    rules = [('cyst_fp_reference_negative', ranked('cyst', False, 'false_positive_ml', True), 'cyst_fp'),
             ('cyst_fp_reference_positive', ranked('cyst', True, 'false_positive_ml', True), 'cyst_fp'),
             ('tumor_low_dice', ranked('tumor', True), 'tumor_fn'),
             ('cyst_low_dice', ranked('cyst', True), 'cyst_fn'),
             ('renal_low_dice', ranked('renal_envelope', True), 'renal_error'),
             ('cyst_positive_control', ranked('cyst', True, reverse=True), 'cyst_reference')]
    selected = {}
    for reason, rows, focus in rules:
        if focus == 'cyst_fp': rows = [r for r in rows if r['metrics']['cyst']['false_positive_ml'] > 0]
        limit = policy['review_controls'] if reason == 'cyst_positive_control' else policy['review_top_per_rule']
        for r in rows[:limit]:
            cid = r['case_id']
            selected.setdefault(cid, {'case_id': cid, 'reasons': [], 'focus': focus})['reasons'].append(reason)
    return list(selected.values())
