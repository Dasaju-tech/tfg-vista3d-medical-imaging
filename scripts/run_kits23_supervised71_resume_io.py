#!/usr/bin/env python
"""Block 72A: bounded Windows JSON replacement retries for frozen Block 71.

Add this file to scripts/, beside the existing compare and resume adapters.
No frozen source or protocol is edited. Only Block 71 JSON writes are adapted
in memory, then restored. active.json is advisory: after bounded Windows lock
retries it may stay stale. Checkpoints, receipts and selections must still save.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0)
WINDOWS_LOCK_ERRORS = (5, 32, 33)


def windows_lock_error(exc, windows=None):
    return (os.name == 'nt' if windows is None else windows) and isinstance(
        exc, OSError) and getattr(exc, 'winerror', None) in WINDOWS_LOCK_ERRORS


def atomic_json_write(path, value, *, advisory=False, stats=None,
                      replace=None, sleep=None, windows=None):
    """Return True after replacement, False only for a blocked advisory marker.

    The old destination is never unlinked. Essential failed temporary files are
    retained for diagnosis. Invalid JSON, disk-full and non-Windows errors fail.
    Dependencies may be injected for deterministic fault tests.
    """
    path = Path(path)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    replace = os.replace if replace is None else replace
    sleep = time.sleep if sleep is None else sleep
    counters = stats if stats is not None else {}
    blocked = counters.setdefault('blocked_active_paths', []) if advisory else []
    delays = () if str(path) in blocked else RETRY_DELAYS
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(len(delays) + 1):
        try:
            replace(temporary, path)
            if str(path) in blocked:
                blocked.remove(str(path))
            return True
        except OSError as exc:
            retryable = windows_lock_error(exc, windows)
            if retryable and attempt < len(delays):
                counters['replacement_retries'] = counters.get('replacement_retries', 0) + 1
                sleep(delays[attempt])
                continue
            if advisory and retryable:
                counters['skipped_active_updates'] = counters.get('skipped_active_updates', 0) + 1
                if str(path) not in blocked:
                    blocked.append(str(path))
                # One warning per invocation; avoid obscuring the training log.
                if counters['skipped_active_updates'] == 1:
                    print(f'WARNING: {path} remains stale after Windows lock retries. '
                          'Continuing: resume uses verified epoch checkpoints, not active.json.',
                          file=sys.stderr, flush=True)
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    counters['leftover_advisory_temporaries'] = counters.get('leftover_advisory_temporaries', 0) + 1
                return False
            exc.add_note(f'Atomic replacement failed; old destination preserved; pending JSON: {temporary}')
            raise
    raise AssertionError('Unreachable retry state')


def is_advisory(path, study):
    return path in {study / arm / 'active.json' for arm in ('ct', 'ct_edges')}


@contextmanager
def scoped_writer(module, study, stats):
    """Install the I/O adaptation only inside this frozen study's output tree."""
    study = Path(study).resolve()
    original = module.write

    def adapted(path, value):
        destination = Path(path).resolve()
        if not destination.is_relative_to(study):
            return original(path, value)
        return atomic_json_write(destination, value,
                                 advisory=is_advisory(destination, study), stats=stats)

    module.write = adapted
    try:
        yield
    finally:
        module.write = original


def run_self_test(root, filename):
    import importlib.util
    import unittest
    path = root / 'tests' / filename
    if not path.is_file():
        raise FileNotFoundError(f'Extract the complete package into the project root; missing {path}')
    spec = importlib.util.spec_from_file_location('block72_self_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('self-test', 'check', 'train', 'report'), required=True)
    parser.add_argument('--arm', choices=('both', 'ct', 'ct_edges'), default='both')
    parser.add_argument('--epochs', type=int, default=1, help='Additional epochs per arm; original maximum/patience apply.')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100:
        parser.error('--epochs must be between 1 and 100')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    sys.path.insert(0, str(root / 'src'))
    if args.stage == 'self-test':
        return run_self_test(root, 'test_block72a_resume_io.py')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import run_kits23_supervised71_compare as comparison
    import run_kits23_supervised71_resume as legacy
    if comparison.ROOT.resolve() != root:
        raise ValueError('Place this file in the project scripts/ directory.')
    frozen = comparison.b.read(root / comparison.CONFIG)
    comparison.b.check(frozen, 'experiment_sha256')
    study = comparison.study_root(root, frozen)
    stats = {'replacement_retries': 0, 'skipped_active_updates': 0}
    audit = {'started_utc': datetime.now(timezone.utc).isoformat(),
             'experiment_sha256': frozen['experiment_sha256'],
             'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'legacy_adapter_sha256': hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
             'arguments': vars(args), 'test_reads': False,
             'io_policy': 'Only frozen study JSON writes; Windows 5/32/33 bounded retries; only ct/ct_edges active.json advisory',
             'status': 'started', 'counters': stats}
    audit_path = study / 'runtime_io_resume_audit' / (uuid4().hex + '.json')
    print('Block 72A: bounded JSON retries; verified checkpoint/selection saves remain mandatory.', flush=True)
    code = 1
    with scoped_writer(comparison.b, study, stats):
        atomic_json_write(audit_path, audit)
        try:
            code = legacy.main(['--stage', args.stage, '--arm', args.arm, '--epochs', str(args.epochs)])
            audit['status'] = 'finished' if code == 0 else 'failed_or_interrupted'
            audit['exit_code'] = code
        except BaseException as exc:
            audit['status'] = 'failed_or_interrupted'
            audit['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            audit['finished_utc'] = datetime.now(timezone.utc).isoformat()
            try:
                atomic_json_write(audit_path, audit)
            except OSError as exc:
                print(f'Could not finish optional I/O audit: {exc}', file=sys.stderr)
            print(f'Block 72A I/O counters: {stats}', flush=True)
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Resume from verified committed checkpoints.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'Block 72A FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        for note in getattr(exc, '__notes__', ()):
            print(note, file=sys.stderr)
        raise SystemExit(1)
