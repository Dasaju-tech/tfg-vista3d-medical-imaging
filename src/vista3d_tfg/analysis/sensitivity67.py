"""Paired component-filter evaluation, with explicit removal benefit and harm."""
import math
import time
import numpy as np
from vista3d_tfg.analysis import components67 as kernel, supervised66 as old


def component_reference_counts(components, n, gt):
    counts = np.zeros((n+1, 4), dtype=np.int64)
    for start in range(0, gt.shape[0], 8):
        ids = components[start:start+8].ravel()
        reference = gt[start:start+8].ravel()
        selected = ids > 0
        codes = ids[selected].astype(np.int64)*4 + reference[selected].astype(np.int64)
        counts += np.bincount(codes, minlength=(n+1)*4).reshape(n+1, 4)
    return counts


def comparison(baseline, matrix, voxel_ml):
    metrics = old.metrics_from_confusion(matrix, voxel_ml)
    original = old.metrics_from_confusion(baseline, voxel_ml)
    differences = {}
    for name in old.CLASSES:
        before, after = original[name], metrics[name]
        tp_loss = before['intersection_voxels']-after['intersection_voxels']
        fp_before = before['predicted_voxels']-before['intersection_voxels']
        fp_after = after['predicted_voxels']-after['intersection_voxels']
        differences[name] = {'delta_dice': after['dice']-before['dice'],
                             'delta_recall': None if before['recall'] is None else after['recall']-before['recall'],
                             'removed_tp_voxels': tp_loss, 'removed_tp_ml': tp_loss*voxel_ml,
                             'removed_fp_voxels': fp_before-fp_after, 'removed_fp_ml': (fp_before-fp_after)*voxel_ml}
    return metrics, differences


def evaluate(gt, pred, affine, policy=kernel.POLICY, *, announce=False):
    started = time.perf_counter()
    affine = kernel.validate_prediction(pred, affine)
    baseline = old.analyze_arrays(gt, pred, affine)
    original = np.asarray(baseline['confusion_voxels'], dtype=np.int64)
    matrices = {v['name']: original.copy() for v in kernel.variants(policy)}
    removals = {name: np.zeros((4, 4), dtype=np.int64) for name in matrices}
    if announce: print('  kidney anchors in physical coordinates...', flush=True)
    trees, anchor_info = kernel.kidney_anchors(pred, affine, policy)
    components_info = {}
    for label in (2, 3):
        if announce: print(f'  class {label}: components, distances and paired counts...', flush=True)
        components, n, retained = kernel.lesion_components(pred, affine, trees, label, policy)
        reference = component_reference_counts(components, n, gt)
        components_info[str(label)] = {'components': n, 'removed_components': {}}
        for name, keep in retained.items():
            removed = reference[~keep].sum(axis=0)
            matrices[name][:, label] -= removed; matrices[name][:, 0] += removed
            removals[name][:, label] = removed
            components_info[str(label)]['removed_components'][name] = int(np.count_nonzero(~keep))
        del components, reference, retained
    results = {}
    for name, matrix in matrices.items():
        metrics, differences = comparison(original, matrix, baseline['voxel_ml'])
        results[name] = {'confusion_voxels': matrix.tolist(), 'metrics': metrics, 'changes': differences,
                         'removed_reference_by_prediction_voxels': removals[name].tolist(),
                         'removed_reference_foreground_ml': int(removals[name][1:, :].sum())*baseline['voxel_ml']}
    result = {'shape': list(pred.shape), 'voxel_ml': baseline['voxel_ml'],
              'baseline': baseline, 'variants': results, 'anchors': anchor_info, 'components': components_info,
              'seconds': time.perf_counter()-started}
    verify_result(result, policy)
    return result


def verify_result(r, policy=kernel.POLICY):
    matrix = np.asarray(r['baseline']['confusion_voxels'], dtype=np.int64)
    shape = r['shape']; ml = r['voxel_ml']
    if (len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape)
            or int(matrix.sum()) != math.prod(shape) or ml != r['baseline']['voxel_ml']
            or old.metrics_from_confusion(matrix, ml) != r['baseline']['metrics']
            or set(r['variants']) != {v['name'] for v in kernel.variants(policy)}):
        raise ValueError('Baseline/variant count parity failed.')
    for variant in kernel.variants(policy):
        v = r['variants'][variant['name']]
        removal = np.asarray(v['removed_reference_by_prediction_voxels'])
        if (removal.shape != (4, 4) or not np.issubdtype(removal.dtype, np.integer)
                or (removal < 0).any() or (removal > matrix).any() or removal[:, :2].any()
                or variant['target'] == 'cyst' and removal[:, 2].any()):
            raise ValueError('Invalid removal ledger.')
        expected = matrix-removal; expected[:, 0] += removal.sum(axis=1)
        metrics, changes = comparison(matrix, expected, ml)
        if (expected.tolist() != v['confusion_voxels'] or metrics != v['metrics'] or changes != v['changes']
                or v['removed_reference_foreground_ml'] != int(removal[1:, :].sum())*ml):
            raise ValueError('Paired metric/removal parity failed.')
        if r['anchors']['no_anchor_fallback'] and removal.any(): raise ValueError('No-anchor fallback violated.')
    if not math.isfinite(r['seconds']) or r['seconds'] < 0: raise ValueError('Invalid timing.')


def summarize(rows, ids, review_rows, policy=kernel.POLICY):
    present = [r['case_id'] for r in rows]
    if len(set(present)) != len(present) or present != [cid for cid in ids if cid in set(present)]:
        raise ValueError('Duplicate, unordered or non-development results.')
    for r in rows: verify_result(r, policy)
    def class_stats(values):
        pos = [v for v in values if v['reference_voxels'] > 0]
        neg = [v for v in values if v['reference_voxels'] == 0]
        return {'dice_all': old.describe([v['dice'] for v in values]),
                'dice_positive': old.describe([v['dice'] for v in pos]),
                'recall_positive': old.describe([v['recall'] for v in pos]),
                'reference_negative_n': len(neg), 'negative_with_prediction_n': sum(v['predicted_voxels'] > 0 for v in neg),
                'negative_fp_ml': old.describe([v['false_positive_ml'] for v in neg])}
    baseline = {k: class_stats([r['baseline']['metrics'][k] for r in rows]) for k in old.CLASSES}
    output = {}
    for variant in kernel.variants(policy):
        name = variant['name']; classes = {}
        for k in old.CLASSES:
            entries = [r['variants'][name] for r in rows]; changes = [v['changes'][k] for v in entries]
            delta = [v['delta_dice'] for v in changes]
            classes[k] = {**class_stats([v['metrics'][k] for v in entries]),
                          'delta_dice': old.describe(delta),
                          'improved_n': sum(d > 1e-12 for d in delta), 'worse_n': sum(d < -1e-12 for d in delta),
                          'dice_loss_over_001_n': sum(d < -.01 for d in delta),
                          'recall_loss_over_001_n': sum(v['delta_recall'] is not None and v['delta_recall'] < -.01 for v in changes),
                          'removed_tp_ml_sum': sum(v['removed_tp_ml'] for v in changes),
                          'removed_fp_ml_sum': sum(v['removed_fp_ml'] for v in changes),
                          'worst_dice_changes': [{'case_id': r['case_id'], 'delta_dice': r['variants'][name]['changes'][k]['delta_dice']}
                              for r in sorted(rows, key=lambda r: (r['variants'][name]['changes'][k]['delta_dice'], r['case_id']))[:10]]}
        output[name] = {'classes': classes, 'removed_reference_foreground_ml_sum': sum(
            r['variants'][name]['removed_reference_foreground_ml'] for r in rows)}
    review_ids = {r['case_id'] for r in review_rows}
    return {'status': 'complete' if present == ids else 'partial', 'n_cases': len(rows), 'requested_n': len(ids),
            'scope': policy['cohort'], 'baseline': baseline, 'variants': output,
            'no_anchor_case_ids': [r['case_id'] for r in rows if r['anchors']['no_anchor_fallback']],
            'block66_review_cases_present': [cid for cid in present if cid in review_ids],
            'selected_variant': None, 'selection_note': 'No automatic winner or promotion; development sensitivity only.',
            'seconds': old.describe([r['seconds'] for r in rows])}
