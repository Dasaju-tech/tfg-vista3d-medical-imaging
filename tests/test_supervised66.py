from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import zipfile
import nibabel as nib
import numpy as np
import pytest
from vista3d_tfg.training import contract65 as c, contract65b as b, contract66 as a
from vista3d_tfg.training import engine65b, inference66 as infer
from vista3d_tfg.training.validation65 import voxel_metrics
from vista3d_tfg.analysis import supervised66 as metrics
from vista3d_tfg.experiments import supervised66 as experiment
from vista3d_tfg.visualization import supervised66 as visual
from test_supervised64 import frozen
from test_supervised65 import prepared, small_training
from test_supervised65b import parent_run, snapshot
from test_full489_execution import script


@pytest.fixture
def completed_training(parent_run):
    root, protocol, _, parent = parent_run
    source = b.freeze(root, protocol)
    result = engine65b.train_until(root, source, {'test': 'cpu'}, until_epoch=3, device='cpu')
    assert result['status'] == 'complete'
    for name in a.SOURCES:
        dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[1]/name, dest)
    return root, protocol, parent, source


@pytest.fixture
def audit_plan(completed_training):
    root, protocol, parent, source = completed_training
    return root, protocol, parent, source, a.freeze(root, protocol)


def run(fixture, **kwargs):
    root, _, parent, source, plan = fixture
    return infer.audit(root, plan, source, parent, {'test': 'cpu'}, device='cpu', **kwargs)


@pytest.mark.parametrize('kind', ['random', 'empty', 'false_positive', 'false_negative', 'cross_class'])
def test_confusion_exact_parity_with_block65(kind):
    rng = np.random.default_rng(66)
    gt = rng.integers(0, 4, (19, 11, 7), dtype=np.uint8); pred = rng.integers(0, 4, gt.shape, dtype=np.uint8)
    if kind != 'random':
        gt[:] = 0; pred[:] = 0
        if kind in ('false_positive', 'cross_class'): pred[0, 0, 0] = 3
        if kind == 'false_negative': gt[0, 0, 0] = 3
        if kind == 'cross_class': gt[0, 0, 0] = 2
    affine = np.diag([-2., 1.5, 3., 1.]); affine[0, 1] = .2
    result = metrics.analyze_arrays(gt, pred, affine)
    for name, oracle in voxel_metrics(gt, pred, affine).items():
        assert {k: result['metrics'][name][k] for k in oracle} == oracle
    assert sum(map(sum, result['confusion_voxels'])) == gt.size
    if kind == 'cross_class':
        assert result['confusion_voxels'][2][3] == 1
        assert result['metrics']['renal_envelope']['dice'] == 1
        assert result['metrics']['cyst']['dice'] == 0
    if kind == 'empty':
        assert result['metrics']['cyst']['precision'] is None
        assert result['metrics']['cyst']['recall'] is None


@pytest.mark.parametrize('bad', [4, -1, .5, np.nan, np.inf])
def test_invalid_labels_rejected(bad):
    gt = np.zeros((2, 3, 4)); pred = gt.copy(); pred[0, 0, 0] = bad
    with pytest.raises(ValueError, match='labels'): metrics.analyze_arrays(gt, pred, np.eye(4))


def test_invalid_geometry_and_saved_metric_rejected():
    gt = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError): metrics.analyze_arrays(gt, gt[:, :, :1], np.eye(4))
    with pytest.raises(ValueError): metrics.analyze_arrays(gt, gt, np.zeros((4, 4)))
    r = {'case_id': 'a', 'shape': list(gt.shape), 'seconds': 0., **metrics.analyze_arrays(gt, gt, np.eye(4))}
    metrics.verify_report(r)
    r['metrics']['cyst']['dice'] = 0
    with pytest.raises(ValueError, match='parity'): metrics.verify_report(r)


def test_freeze_uses_final_selection_preserves_sources_and_never_opens_reserved_inputs(completed_training, monkeypatch):
    root, protocol, parent, source = completed_training
    before = snapshot(root)
    original = Path.open
    def guarded(path, *args, **kwargs):
        normalized = path.as_posix()
        if ('case_00010' in normalized or 'case_00011' in normalized) and '/data/' in normalized:
            pytest.fail('Reserved raw/cache path accessed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    plan = a.freeze(root, protocol)
    assert a.freeze(root, protocol) == plan
    selected = c.read_json(c.output_root(root, source)/'selection.json')
    assert plan['selected_checkpoint']['epoch'] == selected['best_epoch']
    assert plan['selected_checkpoint']['sha256'] == selected['best_checkpoint_sha256']
    assert plan['case_ids'] == c.partition_ids(parent, 'development')
    # No original file was modified; inspect only the original snapshot paths.
    assert all(c.sha256_file(root/name) == digest for name, digest in before.items()
               if '/data/' not in '/'+name or 'case_00010' not in name)
    for cid in ('case_00010', 'case_00011', '../case_00000'):
        with pytest.raises(ValueError, match='development'): a.case_identity(plan, parent, cid)


@pytest.mark.parametrize('change', ['incomplete', 'selection', 'checkpoint', 'code', 'cache_receipt'])
def test_source_drift_rejected(audit_plan, change):
    root, protocol, parent, source, plan = audit_plan
    out = c.output_root(root, source)
    if change == 'incomplete':
        p = c.read_json(out/'training_progress.json'); p['status'] = 'partial'; c.atomic_json(out/'training_progress.json', p)
    if change == 'selection':
        selection = c.read_json(out/'selection.json'); selection['best_score'] += .01
        progress = c.read_json(out/'training_progress.json'); progress['selection'] = selection
        c.atomic_json(out/'selection.json', selection); c.atomic_json(out/'training_progress.json', progress)
    if change == 'checkpoint': (root/plan['selected_checkpoint']['path']).write_bytes(b'changed')
    if change == 'code': (root/a.WORKER).write_text('changed')
    if change == 'cache_receipt':
        path = c.cache_root(root, parent)/'development'/plan['case_ids'][0]/'complete.json'
        r = c.read_json(path); r['artifacts']['image.npy'] = '0'*64
        r['receipt_sha256'] = c.canonical_sha256({k: v for k, v in r.items() if k != 'receipt_sha256'})
        c.atomic_json(path, r)
    with pytest.raises(ValueError): a.check_plan(root, plan, protocol)


def test_inference_resume_does_not_repeat_and_preserves_training(audit_plan, monkeypatch):
    root, protocol, parent, source, plan = audit_plan
    before = snapshot(c.output_root(root, source)); cache_before = snapshot(c.cache_root(root, parent))
    first = run(audit_plan, max_cases=1)
    assert first['status'] == 'partial' and len(first['completed_case_ids']) == 1
    with pytest.raises(ValueError, match='complete'): experiment.review(root, plan, parent)
    prediction = infer.full_volume_prediction; calls = []
    def tracked(*args): calls.append(1); return prediction(*args)
    monkeypatch.setattr(infer, 'full_volume_prediction', tracked)
    assert run(audit_plan)['status'] == 'complete' and len(calls) == 2
    monkeypatch.setattr(infer, 'load_selected_model', lambda *args: pytest.fail('Cached run loaded model'))
    assert run(audit_plan)['status'] == 'complete'
    assert snapshot(c.output_root(root, source)) == before
    assert snapshot(c.cache_root(root, parent)) == cache_before
    a.check_plan(root, plan, protocol)
    for cid in plan['case_ids']:
        r, _, folder = infer.completed_case(root, plan, parent, cid)
        gt = nib.load(root/'data/raw/kits23'/cid/'segmentation.nii.gz')
        pred = nib.load(folder/'prediction.nii.gz')
        assert pred.shape == gt.shape and np.allclose(pred.affine, gt.affine)
        oracle = voxel_metrics(np.asarray(gt.dataobj), np.asarray(pred.dataobj), gt.affine)
        assert all({k: r['metrics'][name][k] for k in values} == values for name, values in oracle.items())


def test_failed_case_has_no_completion_and_resumes(audit_plan, monkeypatch):
    original = infer.full_volume_prediction
    monkeypatch.setattr(infer, 'full_volume_prediction', lambda *args: (_ for _ in ()).throw(RuntimeError('synthetic failure')))
    result = run(audit_plan, max_cases=1)
    assert result['status'] == 'failed' and len(result['failed']) == 1
    root, _, parent, _, plan = audit_plan
    assert infer.completed_case(root, plan, parent, plan['case_ids'][0]) is None
    monkeypatch.setattr(infer, 'full_volume_prediction', original)
    assert run(audit_plan, max_cases=1)['status'] == 'partial'


@pytest.mark.parametrize('change', ['raw', 'cache', 'prediction', 'metrics'])
def test_changed_case_input_or_artifact_fails_closed(audit_plan, change):
    root, _, parent, _, plan = audit_plan; cid = plan['case_ids'][0]
    if change in ('prediction', 'metrics'):
        assert run(audit_plan, max_cases=1)['status'] == 'partial'
        folder = infer.completed_case(root, plan, parent, cid)[2]
        path = folder/('prediction.nii.gz' if change == 'prediction' else 'metrics.json')
    elif change == 'cache': path = c.cache_root(root, parent)/'development'/cid/'image.npy'
    else: path = root/'data/raw/kits23'/cid/'imaging.nii.gz'
    path.write_bytes(path.read_bytes()+b'changed')
    assert run(audit_plan, max_cases=1)['status'] == 'failed'


def test_review_resume_and_archive_are_verified(audit_plan, monkeypatch, tmp_path):
    root, _, parent, _, plan = audit_plan
    assert run(audit_plan)['status'] == 'complete'
    first = experiment.review(root, plan, parent, max_cases=1)
    assert len(first['completed_case_ids']) == 1
    original = visual.render; calls = []
    def track(*args): calls.append(args[-2]['case_id']); return original(*args)
    monkeypatch.setattr(visual, 'render', track)
    state = experiment.review(root, plan, parent)
    assert state['status'] == 'complete' and first['completed_case_ids'][0] not in calls
    monkeypatch.setattr(visual, 'render', lambda *args: pytest.fail('Review repeated'))
    assert experiment.review(root, plan, parent)['status'] == 'complete'
    archive = experiment.export(root, plan, parent)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.pt', '.nii.gz', '.npy')) for n in z.namelist())
        manifest = json.loads(z.read('archive_manifest.json'))
        assert set(manifest) == set(z.namelist())-{'archive_manifest.json'}
        assert all(sha256(z.read(n)).hexdigest() == digest for n, digest in manifest.items())
        assert json.loads(z.read('summary.json'))['n_cases'] == 3
        assert 'Training-data diagnostic' in z.read('report.html').decode()
        assert sum(n.endswith('.png') for n in z.namelist()) == len(state['completed_case_ids'])
        image = next(n for n in z.namelist() if n.endswith('.png'))
        (tmp_path/'review_preview.png').write_bytes(z.read(image))
    row = c.read_json(a.output_root(root, plan)/'review_selection.json')['rows'][0]
    record = infer.completed_case(root, plan, parent, row['case_id'])
    _, path = experiment.checked_review(a.output_root(root, plan), plan, row, record[1])
    path.write_bytes(b'changed')
    assert experiment.review(root, plan, parent)['status'] == 'failed'
    experiment.export(root, plan, parent)
    assert c.read_json(a.output_root(root, plan)/'export_integrity.json')['status'] == 'failed'


def test_ranking_ties_empty_classes_and_distinct_controls():
    gt = np.zeros((3, 3, 3), dtype=np.uint8); gt[0, 0, 0] = 3
    good = {'shape': list(gt.shape), 'seconds': 0., **metrics.analyze_arrays(gt, gt, np.eye(4))}
    rows = [{'case_id': cid, **deepcopy(good)} for cid in ('a', 'b', 'c', 'd')]
    selected = metrics.review_selection(rows, ['a', 'b', 'c', 'd'], a.POLICY)
    assert [v['case_id'] for v in selected] == ['a', 'b', 'c']
    assert [v['case_id'] for v in selected if 'cyst_positive_control' in v['reasons']] == ['a', 'b']
    with pytest.raises(ValueError): metrics.review_selection(rows[:1], ['a', 'b'], a.POLICY)
    with pytest.raises(ValueError): metrics.summarize(rows+rows, ['a', 'b', 'c', 'd'])
    assert metrics.summarize([], ['a'])['classes']['cyst']['dice_all']['mean'] is None


def test_review_ras_and_physical_spacing(tmp_path):
    shape = (5, 7, 9); gt = np.zeros(shape, dtype=np.uint8); gt[1, 2, 3] = 3
    affine = np.diag([-2., -3., 4., 1.])
    paths = [tmp_path/n for n in ('ct.nii.gz', 'gt.nii.gz', 'pred.nii.gz')]
    for p, data in zip(paths, (gt.astype(np.float32), gt, gt)): nib.save(nib.Nifti1Image(data, affine), p)
    arrays, spacing = visual.canonical_arrays(*paths)
    assert list(spacing) == [2., 3., 4.]
    assert arrays[1][3, 4, 3] == 3 and np.array_equal(arrays[1], arrays[2])
    assert visual.focus_mask(arrays[1], arrays[2], 'cyst_reference')[3, 4, 3]
    nib.save(nib.Nifti1Image(gt, np.eye(4)), paths[-1])
    with pytest.raises(ValueError, match='geometry'): visual.canonical_arrays(*paths)


@pytest.mark.parametrize('args', [['--stage', 'run', '--partition', 'test'], ['--stage', 'run', '--max-cases', '0'],
                                  ['--stage', 'prepare', '--max-cases', '3']])
def test_cli_rejects_invalid_scope_before_io(args):
    with pytest.raises(SystemExit) as exc: script('run_kits23_supervised66').main(args)
    assert exc.value.code == 2


def test_dispatch_uses_argument_list_and_exports_partial_then_complete(audit_plan, monkeypatch):
    root, protocol, parent, source, plan = audit_plan
    calls = []
    monkeypatch.setattr(experiment, 'resolve_conda_executable', lambda: 'C:/Conda Folder/conda.exe')
    def worker(command, project, timeout):
        assert isinstance(command, list) and command[0] == 'C:/Conda Folder/conda.exe'
        assert project == root and timeout > 0
        job = c.read_json(Path(command[command.index('--job')+1])); calls.append(job)
        result = infer.audit(root, plan, source, parent, {'test': 'cpu'}, device='cpu', max_cases=job['max_cases'])
        c.atomic_json(Path(command[command.index('--report')+1]), result)
        return 0
    monkeypatch.setattr(experiment, 'invoke_worker', worker)
    state, archive = experiment.dispatch(root, protocol, stage='run', max_cases=1)
    assert state['status'] == 'partial' and archive.is_file()
    assert calls[0]['root'] == str(root) and calls[0]['max_cases'] == 1
    state, archive = experiment.dispatch(root, protocol, stage='run')
    assert state['status'] == 'complete'
    monkeypatch.setattr(experiment, 'invoke_worker', lambda *a, **kw: pytest.fail('Report used GPU'))
    assert experiment.dispatch(root, protocol, stage='report')[0]['status'] == 'complete'


def test_worker_rejects_runtime_drift_before_inference(audit_plan, monkeypatch):
    root, protocol, _, _, plan = audit_plan
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module = script('kits23_supervised66_worker')
    monkeypatch.setattr(module, 'ROOT', root)
    c.atomic_json(root/'configs/experiments/nv_segment_ct_full489_protocol.json', protocol)
    monkeypatch.setattr(module, 'runtime_snapshot', lambda **kw: {'identity': {'changed': True}})
    monkeypatch.setattr(infer, 'audit', lambda *a, **kw: pytest.fail('Inference before runtime gate'))
    with pytest.raises(ValueError, match='runtime'): module.execute({'root': str(root), 'plan': plan})
    with pytest.raises(ValueError, match='checkout'): module.execute({'root': str(root.parent), 'plan': plan})
