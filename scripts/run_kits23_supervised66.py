"""Block 66: selected-checkpoint development diagnostics and visual review."""
import argparse
from pathlib import Path
import sys
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, verified_protocol
from vista3d_tfg.training import contract66 as a
from vista3d_tfg.experiments.supervised66 import dispatch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'run', 'review', 'report'))
    parser.add_argument('--max-cases', type=int, help='New cases per call; verified completions do not consume the budget.')
    parser.add_argument('--environment', default='vista3d-nv')
    parser.add_argument('--project-root', type=Path)
    args = parser.parse_args(argv)
    if args.max_cases is not None and (args.max_cases < 1 or args.stage not in ('run', 'review')):
        parser.error('--max-cases must be positive and is only valid with run/review.')
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        with execution_lock(root/'outputs/full489'):
            if args.stage == 'prepare':
                plan = a.freeze(root, protocol)
                print(f'Block 66 freeze PASSED: {plan["audit_sha256"]}')
                print(f'Selected epoch: {plan["selected_checkpoint"]["epoch"]}; development cases: {len(plan["case_ids"])}')
                print(f'Output: {a.output_root(root, plan)}')
                print('Training-data diagnostic. No inference in prepare. Test120 remains sealed.')
                return 0
            state, archive = dispatch(root, protocol, stage=args.stage, environment=args.environment, max_cases=args.max_cases)
        print(f'Status: {state["status"]}')
        if 'completed_case_ids' in state:
            print(f'Completed: {len(state["completed_case_ids"])}/{len(state["requested_case_ids"])}')
            print(f'Failed: {len(state["failed"])}; deferred: {state["deferred"]}')
        if 'error' in state: print(f'FAILED: {state["error"]}')
        print(f'Share this ZIP: {archive}')
        print('Development diagnostic only. No retraining or new validation/test inference. Test120 remains sealed.')
        return 1 if state['status'] == 'failed' else 0
    except KeyboardInterrupt:
        print('Interrupted; repeat the same stage to resume.', file=sys.stderr); return 130
    except Exception as exc:
        print(f'Block 66 FAILED: {type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
