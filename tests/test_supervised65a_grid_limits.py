from copy import deepcopy
import math

import nibabel as nib
import numpy as np
import pytest

from vista3d_tfg.experiments import supervised65 as experiment
from vista3d_tfg.training import cache65 as cache, contract65 as c
from test_supervised65 import prepared
from test_supervised64 import frozen


def test_large_header_reaches_freeze_without_allocating_voxels(prepared, monkeypatch):
    root, protocol, output, previous = prepared
    # Only replace the NIfTI header interface: no giant array is allocated.
    class LargeHeader:
        shape = (1024, 1024, 600)
        affine = np.diag([1.5, 1.5, 1.5, 1.])
        @property
        def dataobj(self):
            pytest.fail('Prepare must not load image voxels')
    monkeypatch.setattr(experiment.nib, 'load', lambda *a, **kw: LargeHeader())
    # Fixture-only previous freeze has a different geometry. Preserve its bytes
    # elsewhere before simulating this user's first successful prepare.
    (root/c.CONFIG).rename(root/'fixture_previous_protocol.json')
    _, plan = experiment.prepare(root, protocol)
    count = 1024*1024*600
    assert count > 512000000 and count > 64000000
    assert plan['settings']['max_original_voxels'] == count
    assert plan['settings']['max_resampled_voxels'] == count
    assert plan['settings']['patch_size'] == [96]*3 and plan['settings']['spacing_mm'] == [1.5]*3
    shape, _ = cache.grid(LargeHeader(), plan['settings'])
    assert math.prod(shape) == count
    c.check_plan(root, plan, protocol)


def test_limits_cover_later_validation_cases_and_preserve_hyperparameters():
    audit = {
        'development': {'original_shape':[512,512,900], 'resampled_shape':[300,300,500]},
        'validation_early': {'original_shape':[512,512,250], 'resampled_shape':[400,400,501]},
        'validation_later': {'original_shape':[1024,1024,600], 'resampled_shape':[500,500,600]},
    }
    config = c.settings_for_grids(audit, list(audit))
    assert config['max_original_voxels'] == 1024*1024*600
    assert config['max_resampled_voxels'] == 500*500*600
    for key, value in c.settings().items():
        if key not in ('max_original_voxels', 'max_resampled_voxels'):
            assert config[key] == value


@pytest.mark.parametrize('shape', [[0,2,3],[-1,2,3],[1,2], [1.5,2,3], [True,2,3], [2**63,2,3]])
def test_invalid_shapes_cannot_relax_guard(shape):
    audit={'a':{'original_shape':shape,'resampled_shape':[1,2,3]}}
    with pytest.raises(ValueError): c.settings_for_grids(audit,['a'])


def test_missing_or_reserved_case_and_manually_changed_limit_rejected(prepared):
    root, protocol, output, plan = prepared
    audit=deepcopy(plan['grid_audit'])
    with pytest.raises(ValueError): c.settings_for_grids(audit,list(audit)+['reserved'])
    changed=deepcopy(plan);changed['settings']['max_resampled_voxels']+=1
    changed['experiment_sha256']=c.canonical_sha256({k:v for k,v in changed.items() if k!='experiment_sha256'})
    with pytest.raises(ValueError): c.check_plan(root,changed,protocol)


def test_actual_grid_still_rejects_shapes_above_frozen_maximum():
    class Header:
        shape=(2,3,5)
        affine=np.diag([1.5,1.5,1.5,1.])
    config=c.settings_for_grids({'a':{'original_shape':[2,3,4],'resampled_shape':[2,3,4]}},['a'])
    with pytest.raises(ValueError,match='grid exceeds'): cache.grid(Header(),config)
