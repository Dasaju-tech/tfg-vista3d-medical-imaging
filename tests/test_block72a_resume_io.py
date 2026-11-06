from pathlib import Path
import errno
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_kits23_supervised71_resume_io as io72


def denied(number=5):
    exc = PermissionError(errno.EACCES, 'simulated Windows lock')
    exc.winerror = number
    return exc


class AtomicIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.path = self.folder / 'active.json'
        self.path.write_text('{"old": true}\n', encoding='utf-8')
        self.old = self.path.read_bytes()

    def test_transient_lock_eventually_replaces_complete_json(self):
        calls = []
        waits = []
        def replace(src, dst):
            calls.append(Path(dst).read_bytes())
            if len(calls) < 3:
                raise denied()
            os.replace(src, dst)
        counters = {}
        self.assertTrue(io72.atomic_json_write(self.path, {'new': 23}, replace=replace,
                        sleep=waits.append, windows=True, stats=counters))
        self.assertEqual(calls, [self.old] * 3)
        self.assertEqual(json.loads(self.path.read_text()), {'new': 23})
        self.assertEqual(counters['replacement_retries'], 2)
        self.assertEqual(waits, list(io72.RETRY_DELAYS[:2]))
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_permanent_essential_failure_keeps_destination_and_pending_json(self):
        calls = []
        def replace(src, dst):
            calls.append(src)
            raise denied(32)
        with self.assertRaises(PermissionError):
            io72.atomic_json_write(self.path, {'pending': 23}, replace=replace,
                                   sleep=lambda _: None, windows=True)
        self.assertEqual(len(calls), len(io72.RETRY_DELAYS) + 1)
        self.assertEqual(self.path.read_bytes(), self.old)
        pending = list(self.folder.glob('*.tmp'))
        self.assertEqual(len(pending), 1)
        self.assertEqual(json.loads(pending[0].read_text()), {'pending': 23})

    def test_blocked_advisory_update_does_not_abort_training(self):
        def replace(src, dst):
            raise denied(33)
        counters = {}
        with patch('sys.stderr'):
            result = io72.atomic_json_write(self.path, {'new': 23}, advisory=True,
                replace=replace, sleep=lambda _: None, windows=True, stats=counters)
        self.assertFalse(result)
        self.assertEqual(self.path.read_bytes(), self.old)
        self.assertEqual(counters['skipped_active_updates'], 1)
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_non_windows_denial_is_never_hidden_or_retried(self):
        calls = []
        def replace(src, dst):
            calls.append(src)
            raise denied()
        with self.assertRaises(PermissionError):
            io72.atomic_json_write(self.path, {}, advisory=True, replace=replace,
                                   sleep=lambda _: self.fail('Unexpected delay'), windows=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.path.read_bytes(), self.old)

    def test_persistent_advisory_lock_avoids_full_wait_per_patient(self):
        calls = []
        waits = []
        def replace(src, dst):
            calls.append(src)
            raise denied()
        counters = {}
        with patch('sys.stderr'):
            for _ in range(2):
                io72.atomic_json_write(self.path, {}, advisory=True, replace=replace,
                                       sleep=waits.append, windows=True, stats=counters)
        self.assertEqual(len(calls), len(io72.RETRY_DELAYS) + 2)
        self.assertEqual(len(waits), len(io72.RETRY_DELAYS))
        self.assertEqual(counters['skipped_active_updates'], 2)

    def test_advisory_marker_recovers_when_lock_disappears(self):
        def replace(src, dst):
            raise denied()
        counters = {}
        with patch('sys.stderr'):
            io72.atomic_json_write(self.path, {}, advisory=True, replace=replace,
                                   sleep=lambda _: None, windows=True, stats=counters)
        self.assertTrue(io72.atomic_json_write(self.path, {'recovered': True}, advisory=True, stats=counters))
        self.assertEqual(counters['blocked_active_paths'], [])
        self.assertEqual(json.loads(self.path.read_text()), {'recovered': True})

    def test_disk_full_is_not_hidden_for_advisory_marker(self):
        def replace(src, dst):
            raise OSError(errno.ENOSPC, 'disk full')
        with self.assertRaises(OSError):
            io72.atomic_json_write(self.path, {}, advisory=True, replace=replace,
                                   sleep=lambda _: self.fail('Unexpected delay'), windows=True)
        self.assertEqual(self.path.read_bytes(), self.old)

    def test_invalid_json_does_not_touch_destination(self):
        with self.assertRaises(ValueError):
            io72.atomic_json_write(self.path, {'bad': float('nan')})
        self.assertEqual(self.path.read_bytes(), self.old)
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_only_exact_arm_active_paths_are_advisory(self):
        study = self.folder / 'run'
        for arm in ('ct', 'ct_edges'):
            self.assertTrue(io72.is_advisory(study / arm / 'active.json', study))
        for path in (study / 'selection.json', study / 'ct' / 'selection.json',
                     study / 'ct' / 'epochs' / 'active.json', study / 'active.json'):
            self.assertFalse(io72.is_advisory(path, study))

    def test_writer_restored_on_exception_and_other_studies_delegate(self):
        calls = []
        def original(path, value):
            calls.append((path, value))
        module = SimpleNamespace(write=original)
        study = self.folder / 'run'
        outside = self.folder / 'other' / 'active.json'
        with self.assertRaises(RuntimeError):
            with io72.scoped_writer(module, study, {}):
                module.write(outside, {'unchanged': True})
                raise RuntimeError('exercise cleanup')
        self.assertIs(module.write, original)
        self.assertEqual(calls, [(outside, {'unchanged': True})])


if __name__ == '__main__':
    unittest.main()
