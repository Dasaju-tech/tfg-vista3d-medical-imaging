"""Dispatch, verified development reporting and resumable CPU visual review."""
from html import escape
import platform
import re
from uuid import uuid4
import zipfile
from vista3d_tfg.training import contract65 as c, contract65b as b, contract66 as a
from vista3d_tfg.training.inference66 import completed_case
from vista3d_tfg.analysis.supervised66 import summarize, review_selection, CLASSES, LABELS
from vista3d_tfg.experiments.full489_execution import invoke_worker
from vista3d_tfg.integration.nv_segment import resolve_conda_executable


def collect(root, plan, parent):
    records, failures = {}, []
    for cid in plan['case_ids']:
        try:
            value = completed_case(root, plan, parent, cid)
            if value is not None: records[cid] = value
        except Exception as exc:
            failures.append({'case_id': cid, 'error': f'{type(exc).__name__}: {exc}'})
    summary = summarize([v[0] for v in records.values()], plan['case_ids'])
    summary['integrity_failures'] = failures
    if failures: summary['status'] = 'failed'
    return records, summary


def checked_review(output, plan, row, case_receipt):
    folder = output/'review'/row['case_id']; path = folder/'complete.json'
    if not path.exists(): return None
    r = c.read_json(path); c.check_digest(r, 'receipt_sha256')
    if (r['audit_sha256'] != plan['audit_sha256'] or r['selection'] != row
            or r['case_receipt_sha256'] != case_receipt['receipt_sha256']
            or not re.fullmatch(r'attempt_[0-9a-f]{12}', r['attempt'])
            or set(r['artifacts']) != {'review.png'}):
        raise ValueError('Changed review identity.')
    image = folder/r['attempt']/'review.png'
    if c.sha256_file(image) != r['artifacts']['review.png']: raise ValueError('Changed review image.')
    return r, image


def review(root, plan, parent, *, max_cases=None):
    if max_cases is not None and (type(max_cases) is not int or max_cases < 1):
        raise ValueError('New-case budget must be positive.')
    records, summary = collect(root, plan, parent)
    if summary['status'] != 'complete': raise ValueError('Review requires complete, verified development inference.')
    rows = review_selection([v[0] for v in records.values()], plan['case_ids'], plan['policy'])
    output = a.output_root(root, plan)
    selection = {'audit_sha256': plan['audit_sha256'], 'rows': rows}
    c.frozen_json(output/'review_selection.json', selection)
    done, failures, attempted = [], [], 0
    def progress():
        state = {'status': 'failed' if failures else 'complete' if len(done) == len(rows) else 'partial',
                 'audit_sha256': plan['audit_sha256'], 'requested_case_ids': [r['case_id'] for r in rows],
                 'completed_case_ids': done, 'failed': failures, 'deferred': len(rows)-len(done)-len(failures),
                 'new_cases_this_call': attempted, 'gpu_inference': False, 'validation_test_inference': False}
        c.atomic_json(output/'review_progress.json', state)
        return state
    for number, row in enumerate(rows):
        cid = row['case_id']; report, receipt, attempt = records[cid]
        try:
            c.verify_input(root, parent, 'development', cid)
            if checked_review(output, plan, row, receipt): done.append(cid); continue
            if max_cases is not None and attempted >= max_cases: continue
            attempted += 1
            print(f'[{number+1:02d}/{len(rows):02d}] visual review {cid}', flush=True)
            from vista3d_tfg.visualization.supervised66 import render
            import matplotlib
            import nibabel
            import numpy
            name = 'attempt_'+uuid4().hex[:12]
            folder = output/'review'/cid/name
            image_path, gt_path = c.case_paths(root, parent, 'development', cid)
            detail = render(image_path, gt_path, attempt/'prediction.nii.gz', folder/'review.png', row, report['metrics'])
            c.verify_input(root, parent, 'development', cid)
            if completed_case(root, plan, parent, cid)[1] != receipt: raise ValueError('Case changed during review.')
            r = {'audit_sha256': plan['audit_sha256'], 'selection': row, 'attempt': name,
                 'case_receipt_sha256': receipt['receipt_sha256'], 'render': detail,
                 'runtime': {'python': platform.python_version(), 'matplotlib': matplotlib.__version__,
                             'numpy': numpy.__version__, 'nibabel': nibabel.__version__},
                 'artifacts': {'review.png': c.sha256_file(folder/'review.png')}}
            r['receipt_sha256'] = c.canonical_sha256(r)
            c.frozen_json(folder.parent/'complete.json', r)
            done.append(cid); progress()
        except (Exception, KeyboardInterrupt) as exc:
            failures.append({'case_id': cid, 'error': f'{type(exc).__name__}: {exc}'})
            return progress()
    return progress()


def html_report(plan, summary, records, gallery, issues):
    def fmt(v): return 'undefined' if v is None else f'{v:.4f}'
    def table(headers, rows):
        return '<table><thead><tr>'+''.join('<th>'+escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    content = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Block 66 development audit</title>',
               '<style>body{font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#172434}h1,h2{color:#17486a}table{border-collapse:collapse;margin:20px 0;width:100%}td,th{border:1px solid #c5cdd5;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e9f0f5}img{width:100%;height:auto}code{overflow-wrap:anywhere}.note{background:#fff1ce;padding:16px}details{margin:18px 0}a{color:#075889}</style>',
               '<h1>Block 66 · Development audit</h1>',
               '<p class="note"><strong>Training-data diagnostic.</strong> These cases participated in training. This report does not estimate independent performance. No new validation or test inference, no retraining, no postprocessing and no surface-distance recomputation.</p>',
               f'<p>Saved-output integrity: <strong>{escape(summary["status"])}</strong> · {summary["n_cases"]}/{summary["requested_n"]} cases · selected checkpoint epoch {plan["selected_checkpoint"]["epoch"]}</p>',
               '<p>Audit <code>'+escape(plan['audit_sha256'])+'</code><br>Checkpoint <code>'+escape(plan['selected_checkpoint']['sha256'])+'</code></p>']
    if issues: content.append('<p class="note">Action/integrity errors: '+escape(str(issues))+'</p>')
    content += ['<h2>Case-wise metrics</h2>', '<p>Dice: both masks empty = 1; only one empty = 0. Precision is undefined without predicted voxels; recall is undefined without reference voxels. Positive-case means exclude empty references. Renal envelope combines labels 1–3.</p>']
    content.append(table(['Class', 'Mean Dice (all)', 'Mean Dice (GT+)', 'GT+ n', 'Precision (defined)', 'Recall (GT+)', 'GT− with prediction / GT−'], [
        [k, fmt(v['dice_all']['mean']), fmt(v['dice_reference_positive']['mean']), v['dice_reference_positive']['n'],
         fmt(v['precision_defined']['mean']), fmt(v['recall_reference_positive']['mean']),
         f'{v["reference_negative_with_prediction_n"]}/{v["reference_negative_n"]}'] for k, v in summary['classes'].items()]))
    content += ['<h2>Tissue confusion</h2>', '<p>Rows = reference; columns = prediction. Values sum physical volume (mL) across cases, not a case-wise average. Cyst column shows the reference tissue assigned to cyst; cyst row shows where reference cyst was assigned.</p>',
                table(['Reference / prediction', *LABELS], [[label, *[fmt(v) for v in row]] for label, row in zip(LABELS, summary['confusion_ml_sum'])])]
    cyst = summary['classes']['cyst']['reference_negative_false_positive_ml']
    content.append(f'<p>Cyst-negative cases: n={cyst["n"]}; false-positive cyst volume mean {fmt(cyst["mean"])} mL; median {fmt(cyst["median"])} mL; maximum {fmt(cyst["max"])} mL.</p>')
    content += ['<h2>Visual review</h2>', '<p>Up to 17 distinct cases, selected after all development cases complete: largest cyst false positives in GT-negative and GT-positive cases; lowest positive-case tumor, cyst and renal-envelope Dice; and highest positive-case cyst Dice controls. Rules may overlap. Slices maximize the first applicable focus; fallback is disagreement, reference foreground, then volume center. These selected slices are not a full-volume review.</p>']
    requested_reviews = len(review_selection([v[0] for v in records.values()], plan['case_ids'], plan['policy'])) if summary['status'] == 'complete' else None
    content.append(f'<p>Reviewed: {len(gallery)}/{requested_reviews if requested_reviews is not None else "pending complete cohort"} selected cases.</p>')
    if not gallery: content.append('<p>No completed visual reviews yet. Run the review stage after complete inference.</p>')
    for row, name in gallery:
        content.append('<details><summary>'+escape(row['case_id']+' · '+', '.join(row['reasons']))+'</summary><a href="'+name+'"><img loading="lazy" src="'+name+'" alt="'+escape(row['case_id'])+' development review"></a></details>')
    content += ['<h2>Individual cases</h2>', table(['Case', *[k+' Dice' for k in CLASSES], 'Cyst GT mL', 'Cyst prediction mL', 'Cyst FP mL', 'Cyst FN mL'], [
        [cid, *[fmt(r[0]['metrics'][k]['dice']) for k in CLASSES], *[fmt(r[0]['metrics']['cyst'][k]) for k in ('reference_ml', 'predicted_ml', 'false_positive_ml', 'false_negative_ml')]] for cid, r in records.items()]),
        '<p>Timing JSON describes inference plus count calculation, excluding verification and saving. This is not a speed benchmark. JSON files retain exact voxel counts, nullable metrics and provenance. Extract the ZIP before opening this HTML.</p></html>']
    return '\n'.join(content)


def export(root, plan, parent):
    output = a.output_root(root, plan)
    records, summary = collect(root, plan, parent)
    c.atomic_json(output/'summary.json', summary)
    files = {'protocol.json': output/'protocol.json', 'summary.json': output/'summary.json'}
    for name in ('progress.json', 'review_progress.json', 'last_action.json'):
        if (output/name).exists(): files[name] = output/name
    issues = list(summary['integrity_failures']); gallery = []
    if (output/'last_action.json').exists():
        last = c.read_json(output/'last_action.json')
        if last.get('status') == 'failed': issues.append(last)
    for cid, (_, receipt, attempt) in records.items():
        files[f'cases/{cid}/complete.json'] = attempt.parent/'complete.json'
        files[f'cases/{cid}/{attempt.name}/metrics.json'] = attempt/'metrics.json'
    if summary['status'] == 'complete':
        rows = review_selection([v[0] for v in records.values()], plan['case_ids'], plan['policy'])
        selection = {'audit_sha256': plan['audit_sha256'], 'rows': rows}
        c.frozen_json(output/'review_selection.json', selection)
        files['review_selection.json'] = output/'review_selection.json'
        for row in rows:
            cid = row['case_id']
            try:
                value = checked_review(output, plan, row, records[cid][1])
                if value:
                    receipt, image = value
                    name = image.relative_to(output).as_posix()
                    files[name] = image; files[f'review/{cid}/complete.json'] = image.parent.parent/'complete.json'
                    gallery.append((row, name))
            except Exception as exc: issues.append({'case_id': cid, 'error': str(exc)})
    c.atomic_json(output/'export_integrity.json', {'status': 'failed' if issues else summary['status'],
                  'issues': issues, 'exported_cases': list(records), 'exported_reviews': [r['case_id'] for r, _ in gallery]})
    files['export_integrity.json'] = output/'export_integrity.json'
    (output/'report.html').write_text(html_report(plan, summary, records, gallery, issues), encoding='utf-8')
    files['report.html'] = output/'report.html'
    c.atomic_json(output/'archive_manifest.json', {name: c.sha256_file(path) for name, path in files.items()})
    files['archive_manifest.json'] = output/'archive_manifest.json'
    archive = output/'block66_development_audit_report.zip'; temporary = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items(): z.write(path, name)
    temporary.replace(archive)
    return archive


def dispatch(root, protocol, *, stage, environment='vista3d-nv', max_cases=None):
    plan = c.read_json(root/a.CONFIG); a.check_plan(root, plan, protocol)
    parent = c.read_json(root/c.CONFIG); output = a.output_root(root, plan)
    try:
        if stage == 'review': state = review(root, plan, parent, max_cases=max_cases)
        elif stage == 'report':
            _, summary = collect(root, plan, parent)
            state = {'status': summary['status'], 'completed': summary['n_cases'], 'requested': summary['requested_n']}
        elif stage == 'run':
            attempt = output/'attempts'/('attempt_'+uuid4().hex[:12]); attempt.mkdir(parents=True)
            c.atomic_json(attempt/'job.json', {'root': str(root), 'plan': plan, 'max_cases': max_cases})
            command = [resolve_conda_executable(), 'run', '--no-capture-output', '-n', environment,
                       'python', str(root/a.WORKER), '--job', str(attempt/'job.json'), '--report', str(attempt/'worker.json')]
            code = invoke_worker(command, root, timeout=7*24*3600)
            state = c.read_json(attempt/'worker.json')
            if code or state.get('status') not in ('partial', 'complete'):
                raise RuntimeError(state.get('error', str(state.get('failed', f'Worker exit {code}'))))
        else: raise ValueError('Unknown Block 66 stage.')
        a.check_plan(root, plan, protocol)
    except (Exception, KeyboardInterrupt) as exc:
        state = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
    if stage != 'report': c.atomic_json(output/'last_action.json', {'stage': stage, **state})
    archive = export(root, plan, parent)
    integrity = c.read_json(output/'export_integrity.json')
    if integrity['status'] == 'failed' and state['status'] != 'failed':
        state = {**state, 'status': 'failed', 'error': str(integrity['issues'])}
    return state, archive
