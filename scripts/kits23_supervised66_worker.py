"""GPU development inference only; reporting runs in the main environment."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from vista3d_tfg.training import contract65 as c, contract65b as b, contract66 as a
from nv_segment_ct_full489_worker import runtime_snapshot


def execute(job):
    root = Path(job['root']).resolve(); plan = job['plan']
    if root != ROOT: raise ValueError('Worker belongs to another checkout.')
    protocol = c.read_json(root/'configs/experiments/nv_segment_ct_full489_protocol.json')
    a.check_plan(root, plan, protocol)
    source, parent = c.read_json(root/b.CONFIG), c.read_json(root/c.CONFIG)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    runtime = runtime_snapshot(test_kernel=True)['identity']
    if runtime != source['runtime_identity']: raise ValueError('Frozen CUDA runtime changed.')
    import torch
    torch.manual_seed(source['settings']['seed']); torch.cuda.manual_seed_all(source['settings']['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    from vista3d_tfg.training.inference66 import audit
    result = audit(root, plan, source, parent, runtime, max_cases=job.get('max_cases'))
    if runtime_snapshot()['identity'] != runtime: raise ValueError('CUDA runtime changed during audit.')
    a.check_plan(root, plan, protocol)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try: report = execute(c.read_json(args.job))
    except (Exception, KeyboardInterrupt) as exc:
        report = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        print(report['error'], file=sys.stderr, flush=True)
    c.atomic_json(args.report, report)
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__': raise SystemExit(main())
