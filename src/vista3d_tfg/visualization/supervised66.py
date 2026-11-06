"""Outcome-guided development slices in closest RAS voxel planes."""
import nibabel as nib
import numpy as np


def canonical_arrays(image_path, gt_path, prediction_path):
    volumes = [nib.load(p) for p in (image_path, gt_path, prediction_path)]
    image = volumes[0]
    if len(image.shape) != 3 or any(v.shape != image.shape or not np.isfinite(v.affine).all()
            or not np.allclose(v.affine, image.affine, atol=1e-5, rtol=0) for v in volumes):
        raise ValueError('Review CT/reference/prediction geometry differs.')
    arrays = []
    for i, v in enumerate(volumes):
        data = v.get_fdata(dtype=np.float32, caching='unchanged') if i == 0 else np.asanyarray(v.dataobj)
        if not np.isfinite(data).all() or i and not np.isin(data, (0, 1, 2, 3)).all():
            raise ValueError('Invalid review labels or voxels.')
        canonical = nib.as_closest_canonical(nib.Nifti1Image(data, v.affine))
        if nib.aff2axcodes(canonical.affine) != ('R', 'A', 'S'): raise ValueError('Cannot construct RAS axes.')
        arrays.append(np.asarray(canonical.dataobj))
    return arrays, nib.affines.voxel_sizes(canonical.affine)


def focus_mask(gt, pred, focus):
    if focus == 'cyst_fp': mask = (pred == 3) & (gt != 3)
    elif focus == 'cyst_fn': mask = (gt == 3) & (pred != 3)
    elif focus == 'tumor_fn': mask = (gt == 2) & (pred != 2)
    elif focus == 'renal_error': mask = (gt > 0) != (pred > 0)
    elif focus == 'cyst_reference': mask = gt == 3
    else: raise ValueError('Unknown review focus.')
    if not mask.any(): mask = (gt != pred) & ((gt > 0) | (pred > 0))
    if not mask.any(): mask = gt > 0
    return mask


def render(image_path, gt_path, prediction_path, destination, row, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    (ct, gt, pred), spacing = canonical_arrays(image_path, gt_path, prediction_path)
    target = focus_mask(gt, pred, row['focus']); has_target = target.any()
    indices = [int(np.argmax(target.sum(axis=tuple(i for i in range(3) if i != axis))))
               if has_target else gt.shape[axis]//2 for axis in range(3)]
    del target
    colors = ['black', '#40d8e8', '#ffd34d', '#ef69d4']
    errors_colors = ['black', '#fa6969', '#609cff', '#ffa64d']
    directions = [('Sagittal', 'P', 'A', 'I', 'S'), ('Coronal', 'L', 'R', 'I', 'S'), ('Axial', 'L', 'R', 'P', 'A')]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10), facecolor='#101720')
    try:
        for axis, index in enumerate(indices):
            image, ref, prediction = [np.take(v, index, axis=axis).T for v in (ct, gt, pred)]
            other = [i for i in range(3) if i != axis]; aspect = spacing[other[1]]/spacing[other[0]]
            errors = np.zeros(ref.shape, dtype=np.uint8)
            errors[(ref == 0) & (prediction > 0)] = 1
            errors[(ref > 0) & (prediction == 0)] = 2
            errors[(ref > 0) & (prediction > 0) & (ref != prediction)] = 3
            for col in range(4):
                ax = axes[axis, col]; ax.set_facecolor('black')
                ax.imshow(image, cmap='gray', vmin=-200, vmax=300, origin='lower', aspect=aspect)
                if col:
                    overlay = ref if col == 1 else prediction if col == 2 else errors
                    ax.imshow(np.ma.masked_equal(overlay, 0), cmap=ListedColormap(errors_colors if col == 3 else colors),
                              vmin=0, vmax=3, alpha=.65, origin='lower', aspect=aspect, interpolation='nearest')
                name, left, right, bottom, top = directions[axis]
                ax.set_title(f'{name} i={index} | '+('CT', 'Reference', 'Prediction', 'Errors')[col], color='white', fontsize=9)
                for x, y, text in ((.02, .5, left), (.98, .5, right), (.5, .02, bottom), (.5, .98, top)):
                    ax.text(x, y, text, transform=ax.transAxes, ha='center', va='center', color='white', fontsize=8,
                            bbox={'facecolor': 'black', 'alpha': .65, 'edgecolor': 'none', 'pad': 1})
                ax.set_xticks([]); ax.set_yticks([])
        values = '; '.join(f'{k} Dice {metrics[k]["dice"]:.3f}' for k in ('kidney', 'tumor', 'cyst'))
        fig.suptitle(f'Block 66 | {row["case_id"]} | {values}\nDevelopment diagnostic | focus: {row["focus"]}', color='white', fontsize=12, y=.99)
        handles = [Patch(color=colors[i], label=('Kidney', 'Tumor', 'Cyst')[i-1]) for i in (1, 2, 3)]
        handles += [Patch(color=errors_colors[i], label=('Foreground FP', 'Foreground FN', 'Wrong foreground class')[i-1]) for i in (1, 2, 3)]
        fig.legend(handles=handles, loc='lower center', ncol=3, facecolor='#101720', labelcolor='white', fontsize=9)
        fig.text(.5, .071, 'Closest RAS voxel planes; native obliquity retained. Outcome-guided slices, not a complete volume review.',
                 ha='center', color='#b8c8d8', fontsize=8)
        fig.subplots_adjust(left=.02, right=.98, top=.91, bottom=.1, wspace=.06, hspace=.22)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=120, facecolor=fig.get_facecolor())
    finally: plt.close(fig)
    return {'focus': row['focus'], 'canonical_slice_indices': indices, 'spacing_mm': [float(v) for v in spacing]}
