from pathlib import Path
from contextlib import contextmanager, redirect_stdout, nullcontext
import hashlib
import io
import json
import math
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_kits23_supervised72_metrics as metrics72


class GeometryMetricTests(unittest.TestCase):
    def test_identical_nonempty_masks_have_zero_distances(self):
        mask = np.zeros((9, 9, 9), bool)
        mask[2:6, 3:7, 1:5] = True
        values = metrics72.binary_metrics(mask, mask, np.diag([0.7, 0.9, 3.0, 1]))
        self.assertEqual(values['dice'], 1)
        self.assertEqual(values['iou'], 1)
        for name in metrics72.DISTANCES:
            self.assertEqual(values[name], 0)

    def test_single_voxel_physical_shift_and_distinct_l1_l2(self):
        reference = np.zeros((5, 5, 5), bool)
        prediction = reference.copy()
        reference[1, 1, 1] = True
        prediction[2, 2, 1] = True
        affine = np.diag([2, 3, 5, 1])
        values = metrics72.binary_metrics(reference, prediction, affine)
        for name in ('hd', 'hd95', 'assd'):
            self.assertAlmostEqual(values[f'{name}_euclidean_mm'], math.sqrt(13))
            self.assertEqual(values[f'{name}_manhattan_mm'], 5)
        self.assertAlmostEqual(values['false_negative_ml'], 0.03)

    def test_oblique_sheared_affine_matches_brute_force_world_coordinates(self):
        reference = np.zeros((8, 8, 8), bool)
        prediction = reference.copy()
        reference[1, 2, 1] = reference[4, 3, 5] = reference[6, 6, 6] = True
        prediction[2, 1, 1] = prediction[4, 4, 5] = True
        a = np.array([[1.2, -0.3, 0.4, 10], [0.5, 1.8, 0.2, -30],
                      [0.2, 0.0, 3.0, 7], [0, 0, 0, 1]], dtype=float)
        r = np.array([[1, 2, 1, 1], [4, 3, 5, 1], [6, 6, 6, 1]]) @ a.T
        p = np.array([[2, 1, 1, 1], [4, 4, 5, 1]]) @ a.T
        values = metrics72.binary_metrics(reference, prediction, a)
        for norm, metric in (('euclidean', 'euclidean'), ('manhattan', 'cityblock')):
            distances = cdist(p[:, :3], r[:, :3], metric=metric)
            forward, backward = distances.min(axis=1), distances.min(axis=0)
            self.assertAlmostEqual(values[f'hd_{norm}_mm'], max(forward.max(), backward.max()))
            self.assertAlmostEqual(values[f'hd95_{norm}_mm'], max(np.percentile(forward, 95), np.percentile(backward, 95)))
            self.assertAlmostEqual(values[f'assd_{norm}_mm'], (forward.sum() + backward.sum()) / 5)

    def test_hd95_uses_separate_directed_percentiles(self):
        reference = np.zeros((9, 9, 9), bool)
        prediction = reference.copy()
        reference[1, 1, 1] = prediction[1, 1, 1] = prediction[7, 7, 7] = True
        values = metrics72.binary_metrics(reference, prediction, np.eye(4))
        distance = math.sqrt(108)
        self.assertAlmostEqual(values['hd95_euclidean_mm'], 0.95 * distance)
        self.assertAlmostEqual(values['hd_euclidean_mm'], distance)
        self.assertAlmostEqual(values['assd_euclidean_mm'], distance / 3)

    def test_crop_and_image_boundary_preserve_surface_coordinates(self):
        mask = np.zeros((10, 10, 10), bool)
        mask[7:10, 7:10, 7:10] = True
        points = metrics72.surface_points(mask, np.eye(4))
        self.assertEqual(len(points), 26)
        self.assertFalse(np.any(np.all(points == [8, 8, 8], axis=1)))
        self.assertTrue(np.any(np.all(points == [9, 9, 9], axis=1)))

    def test_empty_masks_remain_undefined_distances_with_explicit_status(self):
        empty = np.zeros((4, 4, 4), bool)
        present = empty.copy()
        present[1, 1, 1] = True
        for reference, prediction, status, dice in (
            (empty, empty, 'both_empty', 1),
            (present, empty, 'missing_prediction', 0),
            (empty, present, 'absent_reference_false_positive', 0)):
            values = metrics72.binary_metrics(reference, prediction, np.eye(4))
            self.assertEqual(values['distance_status'], status)
            self.assertEqual(values['dice'], dice)
            for name in metrics72.DISTANCES:
                self.assertIsNone(values[name])

    def test_overlap_counts_and_label_union(self):
        reference = np.zeros((4, 4, 4), np.uint8)
        prediction = reference.copy()
        reference[0, 0, 0] = prediction[0, 0, 0] = 1
        reference[0, 0, 1] = prediction[0, 0, 1] = 2
        reference[0, 0, 2] = 3
        prediction[0, 0, 2] = 2
        values = metrics72.segmentation_metrics(reference, prediction, np.eye(4))
        self.assertEqual(values['renal_envelope']['dice'], 1)
        self.assertAlmostEqual(values['tumor']['dice'], 2 / 3)
        self.assertEqual(values['tumor']['iou'], 0.5)
        self.assertEqual(values['tumor']['precision'], 0.5)
        self.assertEqual(values['tumor']['recall'], 1)
        self.assertEqual(values['cyst']['distance_status'], 'missing_prediction')

    def test_missing_predictions_are_counted_in_conditional_aggregate(self):
        empty = np.zeros((4, 4, 4), np.uint8)
        present = empty.copy()
        present[1, 1, 1] = 3
        reports = [{'metrics': metrics72.segmentation_metrics(r, p, np.eye(4))}
                   for r, p in ((present, present), (present, empty), (empty, present), (empty, empty))]
        values = metrics72.summarize_cases(reports)['cyst']
        self.assertEqual(values['cases'], 4)
        self.assertEqual(values['distance_defined_n'], 1)
        self.assertEqual(values['reference_positive_missing_prediction_n'], 1)
        self.assertEqual(values['reference_negative_with_prediction_n'], 1)
        self.assertEqual(values['both_empty_n'], 1)
        self.assertEqual(values['reference_positive_mean_dice'], 0.5)
        self.assertEqual(values['scalars']['hd95_euclidean_mm']['defined_n'], 1)

    def test_mismatched_grids_and_invalid_labels_fail(self):
        with self.assertRaises(ValueError):
            metrics72.binary_metrics(np.ones((3, 3, 3)), np.ones((4, 3, 3)), np.eye(4))
        with self.assertRaises(ValueError):
            metrics72.segmentation_metrics(np.full((3, 3, 3), 4, dtype=np.uint8), np.zeros((3, 3, 3), dtype=np.uint8), np.eye(4))
        with self.assertRaises(ValueError):
            metrics72.segmentation_metrics(np.zeros((3, 3, 3), dtype=float), np.zeros((3, 3, 3), dtype=np.uint8), np.eye(4))

    def test_invalid_affines_fail(self):
        for affine in (np.zeros((4, 4)), np.full((4, 4), float('nan')), np.eye(3)):
            with self.assertRaises(ValueError):
                metrics72.binary_metrics(np.ones((3, 3, 3)), np.ones((3, 3, 3)), affine)


class EvaluatorOrchestrationTests(unittest.TestCase):
    """Test validation-only routing and resumable receipts with a synthetic GPU API.

    This exercises the collector, not actual CUDA/MONAI execution.
    """
    def fixture(self, root):
        ids = [f'case_{i:05d}' for i in range(80)]
        plan = {'experiment_sha256': 'f' * 64, 'runtime': {'gpu': 'synthetic'},
                'settings': {'model': {}, 'seed': 6501},
                'partition': {'validation': {'case_ids': ids}, 'test': {'case_ids': ['sealed_case']}}}
        record = {'epoch': 1, 'file': 'checkpoint_000000000000.pt', 'sha256': 'c' * 64, 'stats': {}}
        selection = {'best_epoch': 1, 'best_score': 1.0, 'checkpoint_sha256': record['sha256'],
                     'checkpoint_file': 'epochs/epoch_001/' + record['file']}
        array = np.zeros((4, 4, 4), np.uint8)
        array[1, 1, 1], array[1, 1, 2], array[1, 2, 1] = 1, 2, 3
        loads, accesses = [], []
        digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        def check(obj, field):
            if obj[field] != digest({k: v for k, v in obj.items() if k != field}):
                raise ValueError('Corrupt receipt')
        def read(path):
            if Path(path).name == 'selection.json':
                return dict(selection)
            if Path(path).name == 'training_progress.json':
                return {'status': 'partial'}
            return json.loads(Path(path).read_text())
        def verify(root, source, role, cid):
            self.assertEqual(role, 'validation')
            self.assertIn(cid, ids)
            accesses.append((role, cid))
        def load(role, cid):
            self.assertEqual(role, 'validation')
            self.assertIn(cid, ids)
            loads.append(cid)
            return np.zeros(array.shape, np.float32), {'original_affine': np.eye(4)}
        def case_paths(root, source, role, cid):
            self.assertEqual(role, 'validation')
            self.assertIn(cid, ids)
            return Path('synthetic_ct'), Path('synthetic_gt')
        class Model:
            def to(self, device): return self
            def load_state_dict(self, state, strict): pass
            def eval(self): pass
        index = SimpleNamespace(plan=plan, load=load)
        array_sha = lambda value: hashlib.sha256(value.tobytes()).hexdigest()
        m = SimpleNamespace(ARMS=('ct', 'ct_edges'),
            b=SimpleNamespace(read=read, digest=digest, check=check, state_digest=lambda _: 'state'),
            c=SimpleNamespace(verify_input=verify, case_paths=case_paths),
            execution_lock=lambda _: nullcontext(), verify_index=lambda *args: index,
            arm_root=lambda root, plan, arm: root / 'source' / arm,
            history=lambda *args: [dict(record)],
            verified_summary=lambda *args: {'selection_score': 1.0, 'epoch': 1},
            initialized_model=lambda *args: Model(), array_sha=array_sha,
            validation_identity=lambda plan, index, arm, record, cid: {'case_id': cid, 'arm': arm},
            case_metrics=lambda *args: ({'prediction_array_sha256': array_sha(array)}, 'receipt'),
            predict_original=lambda *args, **kwargs: array.copy())
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=False, deterministic=True)),
            use_deterministic_algorithms=lambda *args, **kwargs: None,
            load=lambda *args, **kwargs: {'experiment_sha256': plan['experiment_sha256'], 'arm': 'ct',
                    'epoch': 1, 'stats': {}, 'runtime': plan['runtime'], 'model': {}, 'model_state_sha256': 'state'})
        nifti = SimpleNamespace(affine=np.eye(4), dataobj=array,
                    header=SimpleNamespace(get_xyzt_units=lambda: ('mm', 'sec')))
        nib = SimpleNamespace(load=lambda _: nifti)
        @contextmanager
        def checked_source(_):
            yield m, plan, plan, {**plan['runtime'], 'device_total_bytes': 100}
        (root / 'scripts').mkdir()
        for name in ('run_kits23_supervised72_metrics.py', 'run_kits23_supervised71_resume_io.py', 'run_kits23_supervised71_resume.py'):
            (root / 'scripts' / name).write_text('# synthetic fixture\n')
        return checked_source, torch, nib, loads, accesses, selection

    def test_all_80_validation_cases_resume_without_inference_and_test_is_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checked, torch, nib, loads, accesses, selection = self.fixture(root)
            args = SimpleNamespace(stage='metrics', arm='ct')
            with patch.object(metrics72, 'checked_source', checked), patch.dict(sys.modules, {'torch': torch, 'nibabel': nib}), redirect_stdout(io.StringIO()):
                self.assertEqual(metrics72.execute(root, args), 0)
                self.assertEqual(len(loads), 80)
                self.assertEqual(metrics72.execute(root, args), 0)
                self.assertEqual(len(loads), 80)
            self.assertEqual(set(role for role, _ in accesses), {'validation'})
            archives = list(root.rglob('block72_validation_metrics_report.zip'))
            self.assertEqual(len(archives), 1)
            summary = json.loads(archives[0].with_name('summary.json').read_text())
            self.assertEqual(summary['arms']['ct']['classes']['cyst']['cases'], 80)

    def test_check_stage_does_not_run_inference_and_selection_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checked, torch, nib, loads, accesses, selection = self.fixture(root)
            args = SimpleNamespace(stage='check', arm='ct')
            with patch.object(metrics72, 'checked_source', checked), patch.dict(sys.modules, {'torch': torch, 'nibabel': nib}), redirect_stdout(io.StringIO()):
                self.assertEqual(metrics72.execute(root, args), 0)
                self.assertEqual(loads, [])
                selection['checkpoint_sha256'] = 'changed'
                with self.assertRaises(ValueError):
                    metrics72.execute(root, args)

    def test_corrupt_saved_case_receipt_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checked, torch, nib, loads, accesses, selection = self.fixture(root)
            args = SimpleNamespace(stage='metrics', arm='ct')
            with patch.object(metrics72, 'checked_source', checked), patch.dict(sys.modules, {'torch': torch, 'nibabel': nib}), redirect_stdout(io.StringIO()):
                metrics72.execute(root, args)
                case = next(root.glob('outputs/reporting/full489_block72_metrics/*/ct/case_00000/metrics.json'))
                value = json.loads(case.read_text())
                value['metrics']['cyst']['dice'] = 0.123
                case.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    metrics72.execute(root, args)
            self.assertEqual(len(loads), 80)


if __name__ == '__main__':
    unittest.main()
