from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pytest
from vista3d_tfg.analysis import components67 as kernel, sensitivity67 as sensitivity, supervised66 as old
from vista3d_tfg.experiments import contract67 as contract, supervised67 as experiment, supervised66 as e66
from vista3d_tfg.training import contract65 as c, contract66 as a, inference66 as i66
from vista3d_tfg.reporting.supervised67 import export
from test_supervised64 import frozen
from test_supervised65 import prepared, small_training
from test_supervised65b import parent_run, snapshot
from test_supervised66 import completed_training, audit_plan
from test_full489_execution import script


def synthetic():
    pred = np.zeros((24, 24, 24), dtype=np.uint8)
    pred[2:5, 2:5, 2:5] = 1
    pred[14:17, 2:5, 2:5] = 1
    pred[21, 21, 21] = 1  # Small remote kidney anchor: retained only by all-kidney policy.
    pred[5:13, 3, 3] = 3  # Long component with one supported end must remain whole.
    pred[21, 21, 22] = 3
    pred[2, 20:23, 20] = 2
    gt = np.zeros_like(pred); gt[pred == 1] = 1
    gt[5:7, 3, 3] = 3; gt[2, 20, 20] = 2
    gt[2, 21, 20] = 3  # Removal harms renal foreground despite being a tumor FP.
    return gt, pred


@pytest.mark.parametrize('geometry', ['anisotropic', 'rotated', 'sheared'])
def test_every_variant_matches_materialized_masks_and_legacy_metrics(geometry):
    gt, pred = synthetic(); affine = np.diag([2., 1.5, 3., 1.]); affine[:3, 3] = [-80., 12., 90.]
    if geometry == 'rotated':
        angle = .6; rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        affine[:3, :3] = rotation @ affine[:3, :3]
    if geometry == 'sheared': affine[0, 1] = .7
    result = sensitivity.evaluate(gt, pred, affine)
    assert len(result['variants']) == 12
    assert any(v['removed_reference_foreground_ml'] > 0 for v in result['variants'].values())
    for variant in kernel.variants():
        mask = kernel.materialize_variant(pred, affine, variant['name'])
        expected = old.analyze_arrays(gt, mask, affine); got = result['variants'][variant['name']]
        assert got['metrics'] == expected['metrics'] and got['confusion_voxels'] == expected['confusion_voxels']
        assert got['metrics']['kidney'] == result['baseline']['metrics']['kidney']
        assert np.array_equal(mask == 1, pred == 1)
        assert np.all((mask == pred) | ((pred > 1) & (mask == 0)))


def test_physical_distance_matches_bruteforce_under_shear():
    gt, pred = synthetic(); affine = np.diag([2., 3., 4., 1.]); affine[0, 1] = 1.2
    trees, _ = kernel.kidney_anchors(pred, affine, kernel.POLICY)
    labels, n, decisions = kernel.lesion_components(pred, affine, trees, 3)
    anchors = kernel.world(np.argwhere(pred == 1), affine)
    for component in range(1, n+1):
        points = kernel.world(np.argwhere(labels == component), affine)
        distance = np.linalg.norm(points[:, None, :]-anchors[None, :, :], axis=2).min()
        for radius in (5, 10, 20):
            assert decisions[f'cyst_all_kidney_{radius:02d}mm'][component] == (distance <= radius+1e-7)


def test_whole_components_top2_anchor_and_cyst_only_semantics():
    _, pred = synthetic(); affine = np.eye(4)
    all_mask = kernel.materialize_variant(pred, affine, 'cyst_all_kidney_05mm')
    two_mask = kernel.materialize_variant(pred, affine, 'cyst_largest2_kidney_05mm')
    assert all_mask[21, 21, 22] == 3 and two_mask[21, 21, 22] == 0
    assert two_mask[21, 21, 21] == 1  # Anchor choice never edits kidney output.
    assert np.all(two_mask[5:13, 3, 3] == 3)  # Whole connected extension retained.
    assert np.array_equal(two_mask == 2, pred == 2)


def test_boundary_and_six_connectivity():
    pred = np.zeros((8, 8, 8), dtype=np.uint8); pred[0, 0, 0] = 1
    pred[3, 4, 0] = 3; pred[4, 5, 0] = 3  # Diagonal neighbours are separate.
    output = kernel.materialize_variant(pred, np.eye(4), 'cyst_all_kidney_05mm')
    assert output[3, 4, 0] == 3 and output[4, 5, 0] == 0


def test_no_kidney_and_empty_lesions_are_identity():
    gt, pred = synthetic(); pred[pred == 1] = 0
    r = sensitivity.evaluate(gt, pred, np.eye(4))
    assert r['anchors']['no_anchor_fallback']
    assert all(v['confusion_voxels'] == r['baseline']['confusion_voxels'] for v in r['variants'].values())
    pred[:] = 0; pred[0, 0, 0] = 1
    r = sensitivity.evaluate(gt, pred, np.eye(4))
    assert all(v['confusion_voxels'] == r['baseline']['confusion_voxels'] for v in r['variants'].values())


def test_decisions_do_not_depend_on_reference(monkeypatch):
    gt, pred = synthetic(); decisions = []; original = kernel.lesion_components
    def record(*args, **kw):
        value = original(*args, **kw); decisions.append({k: v.copy() for k, v in value[2].items()}); return value
    monkeypatch.setattr(kernel, 'lesion_components', record)
    sensitivity.evaluate(gt, pred, np.eye(4)); sensitivity.evaluate(np.zeros_like(gt), pred, np.eye(4))
    assert all(np.array_equal(decisions[i][k], decisions[i+2][k]) for i in (0, 1) for k in decisions[i])


@pytest.mark.parametrize('change', ['label', 'affine', 'shape', 'ledger', 'metric'])
def test_invalid_input_or_saved_result_rejected(change):
    gt, pred = synthetic(); affine = np.eye(4)
    if change == 'label': pred[0, 0, 0] = 4
    if change == 'affine': affine[0, 0] = 0
    if change == 'shape': gt = gt[:-1]
    if change in ('label', 'affine', 'shape'):
        with pytest.raises(ValueError): sensitivity.evaluate(gt, pred, affine)
    else:
        r = sensitivity.evaluate(gt, pred, affine); v = next(iter(r['variants'].values()))
        if change == 'ledger': v['removed_reference_by_prediction_voxels'][0][1] = 1
        else: v['metrics']['cyst']['dice'] += .01
        with pytest.raises(ValueError): sensitivity.verify_result(r)


@pytest.fixture
def finished66(audit_plan):
    root, protocol, parent, source65, source = audit_plan
    assert i66.audit(root, source, source65, parent, {'test': 'cpu'}, device='cpu')['status'] == 'complete'
    assert e66.review(root, source, parent)['status'] == 'complete'
    e66.export(root, source, parent)
    for name in contract.SOURCES:
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[1]/name, target)
    return root, protocol, parent, source


@pytest.fixture
def plan67(finished66):
    root, protocol, parent, source = finished66
    return root, protocol, parent, source, contract.freeze(root, protocol)


def test_freeze_is_idempotent_and_reserved_volumes_are_never_opened(finished66, monkeypatch):
    root, protocol, parent, source = finished66
    original = Path.open
    def guard(path, *args, **kw):
        text = path.as_posix()
        if '/data/' in text and ('case_00010' in text or 'case_00011' in text): pytest.fail('Reserved volume/cache read')
        return original(path, *args, **kw)
    monkeypatch.setattr(Path, 'open', guard)
    plan = contract.freeze(root, protocol)
    assert contract.freeze(root, protocol) == plan
    assert len(plan['variants']) == 12
    assert experiment.run(root, plan, source, parent, max_cases=1)['status'] == 'partial'
    for cid in ['case_00010', 'case_00011', '../case_00000']:
        with pytest.raises(ValueError): contract.case_identity(plan, source, parent, cid)


@pytest.mark.parametrize('change', ['incomplete_review', 'source_prediction', 'runtime', 'code', 'policy'])
def test_changed_contract_or_source_rejected(plan67, change, monkeypatch):
    root, protocol, parent, source, plan = plan67
    if change == 'incomplete_review':
        path = a.output_root(root, source)/'review_progress.json'; r = c.read_json(path); r['status'] = 'partial'; c.atomic_json(path, r)
    if change == 'source_prediction': i66.completed_case(root, source, parent, source['case_ids'][0])[2].joinpath('prediction.nii.gz').write_bytes(b'changed')
    if change == 'runtime': monkeypatch.setattr(contract, 'runtime_identity', lambda: {'changed': True})
    if change == 'code': (root/contract.SOURCES[0]).write_text('changed')
    if change == 'policy':
        plan = deepcopy(plan); plan['policy']['thresholds_mm'][0] = 7.
        plan['experiment_sha256'] = c.canonical_sha256({k: v for k, v in plan.items() if k != 'experiment_sha256'})
    with pytest.raises(ValueError): contract.check_plan(root, plan, protocol)


def test_resume_preserves_sources_and_exports_complete_paired_results(plan67, monkeypatch):
    root, protocol, parent, source, plan = plan67
    before = snapshot(a.output_root(root, source)); inputs_before = snapshot(root/'data')
    first, archive = experiment.dispatch(root, protocol, stage='run', max_cases=1)
    assert first['status'] == 'partial' and len(first['completed_case_ids']) == 1
    original = sensitivity.evaluate; calls = []
    def track(*args, **kw): calls.append(1); return original(*args, **kw)
    monkeypatch.setattr(sensitivity, 'evaluate', track)
    second, archive = experiment.dispatch(root, protocol, stage='run')
    assert second['status'] == 'complete' and len(calls) == 2
    monkeypatch.setattr(sensitivity, 'evaluate', lambda *a, **kw: pytest.fail('Repeated analysis'))
    assert experiment.dispatch(root, protocol, stage='run')[0]['status'] == 'complete'
    assert experiment.dispatch(root, protocol, stage='report')[0]['status'] == 'complete'
    assert before == snapshot(a.output_root(root, source)) and inputs_before == snapshot(root/'data')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('archive_manifest.json'))
        assert set(manifest) == set(z.namelist())-{'archive_manifest.json'}
        assert all(sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
        assert not any(n.endswith(('.npy', '.nii.gz', '.pt')) for n in z.namelist())
        summary = json.loads(z.read('summary.json'))
        assert summary['n_cases'] == 3 and len(summary['variants']) == 12 and summary['selected_variant'] is None
        assert 'GT+' in z.read('report.html').decode()


def test_interruption_and_corrupt_saved_result_fail_closed(plan67, monkeypatch):
    root, _, parent, source, plan = plan67
    original = sensitivity.evaluate
    monkeypatch.setattr(sensitivity, 'evaluate', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('synthetic interruption')))
    assert experiment.run(root, plan, source, parent, max_cases=1)['status'] == 'failed'
    assert experiment.completed(root, plan, source, parent, plan['case_ids'][0]) is None
    monkeypatch.setattr(sensitivity, 'evaluate', original)
    assert experiment.run(root, plan, source, parent, max_cases=1)['status'] == 'partial'
    saved = experiment.completed(root, plan, source, parent, plan['case_ids'][0])
    (saved[2]/'sensitivity.json').write_bytes(b'changed')
    assert experiment.run(root, plan, source, parent, max_cases=1)['status'] == 'failed'
    assert export(root, plan, source, parent)[2]


def test_summary_reports_cross_class_harm_and_positive_case_controls():
    gt, pred = synthetic(); r = sensitivity.evaluate(gt, pred, np.eye(4)); r['case_id'] = 'a'
    summary = sensitivity.summarize([r], ['a', 'b'], [{'case_id': 'a', 'reasons': ['cyst_positive_control']}])
    assert summary['status'] == 'partial' and summary['block66_review_cases_present'] == ['a']
    variant = r['variants']['tumor_cyst_largest2_kidney_05mm']
    assert variant['changes']['tumor']['removed_fp_voxels'] > 0
    assert variant['removed_reference_foreground_ml'] > variant['changes']['tumor']['removed_tp_ml']
    assert summary['selected_variant'] is None
    with pytest.raises(ValueError): sensitivity.summarize([r, r], ['a', 'b'], [])


@pytest.mark.parametrize('args', [['--stage', 'run', '--partition', 'test'], ['--stage', 'run', '--max-cases', '0'],
                                  ['--stage', 'prepare', '--max-cases', '3'], ['--stage', 'train']])
def test_cli_rejects_invalid_scope(args):
    with pytest.raises(SystemExit) as exc: script('run_kits23_supervised67').main(args)
    assert exc.value.code == 2
