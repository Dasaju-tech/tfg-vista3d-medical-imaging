"""Block 67: paired development-only CPU component-filter sensitivity."""
import argparse
from pathlib import Path
import sys
from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489_execution import execution_lock, verified_protocol
from vista3d_tfg.experiments import contract67 as c, supervised67 as experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'run', 'report'))
    parser.add_argument('--max-cases', type=int, help='New cases per call; completed cases are verified and reused.')
    parser.add_argument('--project-root', type=Path)
    args = parser.parse_args(argv)
    if args.max_cases is not None and (args.max_cases < 1 or args.stage != 'run'):
        parser.error('--max-cases must be positive and is valid only for run.')
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        protocol = verified_protocol(root)
        with execution_lock(root/'outputs/full489'):
            print('Verifying completed Block 66 source...', flush=True)
            if args.stage == 'prepare':
                plan = c.freeze(root, protocol)
                print(f'Block 67 freeze PASSED: {plan["experiment_sha256"]}')
                print(f'Development cases: {len(plan["case_ids"])}; fixed variants: {len(plan["variants"])}')
                print(f'Output: {c.output_root(root, plan)}')
                print('CPU protocol only; no filtering in prepare. Test120 remains sealed.')
                return 0
            state, archive = experiment.dispatch(root, protocol, stage=args.stage, max_cases=args.max_cases)
        print(f'Status: {state["status"]}')
        if 'completed_case_ids' in state:
            print(f'Completed: {len(state["completed_case_ids"])}/{len(state["requested_case_ids"])}')
            print(f'Failed: {len(state["failed"])}; deferred: {state["deferred"]}')
        if 'error' in state: print(f'FAILED: {state["error"]}')
        print(f'Share this ZIP: {archive}')
        print('CPU sensitivity only. GPU inference / retraining / validation-test volume reads: none. Test120 remains sealed.')
        return 1 if state['status'] == 'failed' else 0
    except KeyboardInterrupt:
        print('Interrupted; repeat the same run command to resume.', file=sys.stderr); return 130
    except Exception as exc:
        print(f'Block 67 FAILED: {type(exc).__name__}: {exc}', file=sys.stderr); return 1


if __name__ == '__main__': raise SystemExit(main())
