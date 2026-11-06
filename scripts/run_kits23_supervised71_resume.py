#!/usr/bin/env python
"""Block 71 continuation adapter for a small change in reported CUDA total memory.

Place beside run_kits23_supervised71_compare.py in scripts/.
--stage check checks compatibility only. Train/report arguments go to Block 71.
The original protocol, source code and checkpoints are never rewritten.
Actual runtime metadata and this adapter's hash are recorded in runtime_resume_audit/.
This compatibility exception does not claim bitwise GPU reproducibility.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

FIELD = 'device_total_bytes'
MAX_BYTES = 64 * 1024 * 1024
MAX_FRACTION = 0.005


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def compatible_plan(observed, frozen):
    """Allow only a small total-memory telemetry difference; compare ALL else."""
    for plan in (observed, frozen):
        body = deepcopy(plan)
        stored = body.pop('experiment_sha256')
        if digest(body) != stored:
            raise ValueError('Invalid protocol digest.')
    old = frozen['runtime'][FIELD]
    new = observed['runtime'][FIELD]
    if type(old) is not int or type(new) is not int or min(old, new) <= 0:
        raise ValueError('Invalid CUDA memory metadata.')
    if abs(old-new) > min(MAX_BYTES, old * MAX_FRACTION):
        raise ValueError('Reported GPU memory difference exceeds the narrow compatibility limit.')
    result = deepcopy(observed)
    result['runtime'][FIELD] = old
    result.pop('experiment_sha256')
    result['experiment_sha256'] = digest(result)
    if result != frozen:
        raise ValueError('Another config/code/runtime field changed. No compatibility exception applied.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'train', 'report'), required=True)
    parser.add_argument('--arm', choices=('both', 'ct', 'ct_edges'), default='both')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Additional epochs per arm, capped by the original protocol.')
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 100:
        parser.error('--epochs must be between 1 and 100')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    sys.path.insert(0, str(root / 'src'))
    import run_kits23_supervised71_compare as m
    if m.ROOT.resolve() != root:
        raise ValueError('Place this adapter in the project scripts directory.')
    frozen = m.b.read(root / m.CONFIG)
    m.b.check(frozen, 'experiment_sha256')
    original_builder = m.build_plan
    observations = []

    def build_checked(*a, **kw):
        observed = original_builder(*a, **kw)
        result = compatible_plan(observed, frozen)
        runtime = observed['runtime']
        if runtime not in observations:
            observations.append(deepcopy(runtime))
        return result

    m.build_plan = build_checked
    audit_path = None
    audit = None
    try:
        # All original source, digest and output-protocol checks remain active.
        m.load_plan(root)
        old = frozen['runtime'][FIELD]
        new = observations[-1][FIELD]
        print(f'CUDA total-memory compatibility: frozen={old}; actual={new}; delta={(new-old)/2**20:.3f} MiB.', flush=True)
        print('All other protocol/code/runtime fields match. Original files preserved.', flush=True)
        folder = m.study_root(root, frozen) / 'runtime_resume_audit'
        folder.mkdir(parents=True, exist_ok=True)
        audit_path = folder / (uuid4().hex + '.json')
        audit = {
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'experiment_sha256': frozen['experiment_sha256'],
            'adapter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'compatibility_exception': 'Only runtime/device_total_bytes; max 64 MiB AND 0.5 percent',
            'frozen_runtime': frozen['runtime'], 'observed_runtimes': observations,
            'arguments': vars(args), 'status': 'started',
        }
        m.b.write(audit_path, audit)
        print(f'Actual-runtime audit: {audit_path}', flush=True)
        if args.stage == 'check':
            code = 0
            print('Block 71 compatibility check PASSED. No training performed.', flush=True)
        else:
            code = m.main(['--stage', args.stage, '--arm', args.arm,
                           '--epochs', str(args.epochs)])
        audit['status'] = 'finished' if code == 0 else 'failed_or_interrupted'
        audit['exit_code'] = code
        return code
    except BaseException as exc:
        if audit is not None:
            audit['status'] = 'failed_or_interrupted'
            audit['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        m.build_plan = original_builder
        if audit_path is not None:
            audit['finished_utc'] = datetime.now(timezone.utc).isoformat()
            audit['observed_runtimes'] = observations
            m.b.write(audit_path, audit)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed Block 71 checkpoints remain available.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'Block 71 resume FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
