"""Prediction-only component support in physical coordinates (no reference labels)."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

POLICY = {
    'connectivity': 6, 'thresholds_mm': [5., 10., 20.], 'boundary_tolerance_mm': 1e-7,
    'anchors': ['all_kidney', 'largest2_kidney'], 'targets': ['cyst', 'tumor_cyst'],
    'distance': 'minimum Euclidean distance between voxel centres using the full affine',
    'component_rule': 'retain whole class-specific component when any voxel is within threshold',
    'anchor_tie': 'voxel count descending, component id ascending in array scan order',
    'no_kidney': 'identity fallback; do not remove lesions',
    'kidney_output': 'unchanged; largest2 restricts anchors only',
    'replacement_label': 0, 'query_batch': 131072, 'query_workers': 1,
    'cohort': 'development only; exploratory training-data sensitivity analysis',
    'automatic_selection': False,
}


def variants(policy=POLICY):
    return [{'name': f'{target}_{anchor}_{int(radius):02d}mm', 'target': target,
             'anchor': anchor, 'radius_mm': radius}
            for target in policy['targets'] for anchor in policy['anchors'] for radius in policy['thresholds_mm']]


def validate_prediction(pred, affine):
    if pred.ndim != 3 or any(n < 1 for n in pred.shape) or pred.size >= 2**31:
        raise ValueError('Expected a positive 3D grid with int32-addressable components.')
    affine = np.asarray(affine, dtype=np.float64)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or abs(float(np.linalg.det(affine[:3, :3]))) == 0):
        raise ValueError('Invalid affine for physical distances.')
    for start in range(0, pred.shape[0], 8):
        if not np.isin(pred[start:start+8], (0, 1, 2, 3)).all():
            raise ValueError('Invalid prediction labels.')
    return affine


def slab_points(labels, selected=None, batch=131072):
    """Yield bounded coordinate batches, with labels optionally restricted by lookup."""
    for start in range(0, labels.shape[0], 8):
        values = labels[start:start+8]
        mask = values > 0 if selected is None else selected[values]
        coordinates = np.argwhere(mask)
        coordinates[:, 0] += start
        for offset in range(0, len(coordinates), batch):
            yield coordinates[offset:offset+batch]


def world(points, affine):
    return points @ affine[:3, :3].T + affine[:3, 3]


def kidney_anchors(pred, affine, policy):
    labels, n = ndimage.label(pred == 1, structure=ndimage.generate_binary_structure(3, 1))
    counts = np.zeros(n+1, dtype=np.int64)
    for start in range(0, pred.shape[0], 8):
        counts += np.bincount(labels[start:start+8].ravel(), minlength=n+1)
    order = sorted(range(1, n+1), key=lambda i: (-int(counts[i]), i))
    top = np.zeros(n+1, dtype=bool); top[order[:2]] = True
    pieces, membership = [], []
    for points in slab_points(labels, batch=policy['query_batch']):
        pieces.append(world(points, affine))
        membership.append(top[labels[tuple(points.T)]])
    if not pieces:
        return {'all_kidney': None, 'largest2_kidney': None}, {'kidney_components': 0,
                'kidney_voxels': 0, 'largest2_anchor_voxels': 0, 'no_anchor_fallback': True}
    coordinates = np.concatenate(pieces); keep = np.concatenate(membership)
    del labels, pieces, membership
    all_tree = cKDTree(coordinates)
    top_tree = all_tree if keep.all() else cKDTree(coordinates[keep])
    return {'all_kidney': all_tree, 'largest2_kidney': top_tree}, {
        'kidney_components': n, 'kidney_voxels': int(counts[1:].sum()),
        'largest2_anchor_voxels': int(counts[order[:2]].sum()), 'no_anchor_fallback': False}


def lesion_components(pred, affine, trees, label, policy=POLICY):
    """No GT argument: decisions depend only on prediction and image geometry."""
    labels, n = ndimage.label(pred == label, structure=ndimage.generate_binary_structure(3, 1))
    minima = {anchor: np.full(n+1, np.inf) for anchor in policy['anchors']}
    cap = max(policy['thresholds_mm']) + policy['boundary_tolerance_mm']
    for points in slab_points(labels, batch=policy['query_batch']):
        component_ids = labels[tuple(points.T)]; coordinates = world(points, affine)
        previous = {}
        for anchor, tree in trees.items():
            if tree is None: continue
            # Reuse the query when both anchor policies have exactly the same tree.
            key = id(tree)
            if key not in previous:
                previous[key] = tree.query(coordinates, k=1, eps=0., p=2,
                    distance_upper_bound=np.nextafter(cap, np.inf), workers=policy['query_workers'])[0]
            np.minimum.at(minima[anchor], component_ids, previous[key])
    keep = {}
    for variant in variants(policy):
        unchanged = (label == 2 and variant['target'] == 'cyst') or trees[variant['anchor']] is None
        retained = np.ones(n+1, dtype=bool) if unchanged else (
            minima[variant['anchor']] <= variant['radius_mm']+policy['boundary_tolerance_mm'])
        retained[0] = True  # Background in this component map is never removed.
        keep[variant['name']] = retained
    return labels, n, keep


def materialize_variant(pred, affine, variant_name, policy=POLICY):
    """Reference reconstruction helper; the cohort sweep stores counts, not 12 volumes."""
    if variant_name not in {v['name'] for v in variants(policy)}: raise ValueError('Unknown variant.')
    affine = validate_prediction(pred, affine)
    trees, _ = kidney_anchors(pred, affine, policy)
    result = pred.copy()
    for label in (2, 3):
        components, _, keep = lesion_components(pred, affine, trees, label, policy)
        for start in range(0, pred.shape[0], 8):
            result[start:start+8][~keep[variant_name][components[start:start+8]]] = 0
        del components
    return result
