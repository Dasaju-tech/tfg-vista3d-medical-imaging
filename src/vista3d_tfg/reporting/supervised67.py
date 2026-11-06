"""Portable paired sensitivity tables; no automatic winner is declared."""
from html import escape
import zipfile
from vista3d_tfg.training import contract65 as c
from vista3d_tfg.experiments import contract67 as contract
from vista3d_tfg.analysis import sensitivity67 as sensitivity


def html_report(plan, summary, records, issues):
    def number(v): return 'undefined' if v is None else f'{v:.4f}'
    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    output = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Block 67 component sensitivity</title>',
              '<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#172434}h1,h2{color:#17486a}.note{background:#fff0ce;padding:16px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{border:1px solid #cad4dd;padding:8px;text-align:right}th{background:#eaf1f6}td:first-child,th:first-child{text-align:left;overflow-wrap:anywhere}details{margin:20px 0}code{overflow-wrap:anywhere}</style>',
              '<h1>Block 67 · Component sensitivity</h1>',
              '<p class="note"><strong>Exploratory development analysis on training cases.</strong> Twelve fixed filters are compared with the unchanged Block 66 predictions. No retraining, new inference, automatic selection or test evaluation. Reference labels are used only to measure outcomes.</p>',
              f'<p>Saved-output status: <strong>{escape(summary["status"])}</strong> · {summary["n_cases"]}/{summary["requested_n"]} cases · checkpoint epoch {plan["selected_checkpoint"]["epoch"]}</p>',
              '<p>Experiment <code>'+escape(plan['experiment_sha256'])+'</code><br>Source audit <code>'+escape(plan['source_audit_sha256'])+'</code></p>']
    if issues: output.append('<p class="note">Execution/integrity issues: '+escape(str(issues))+'</p>')
    output += ['<h2>Fixed experiment</h2>',
               '<p>Connected components use six neighbours and are computed separately for tumor and cyst. Retain the whole component when its minimum voxel-centre distance to a predicted kidney anchor is at most 5, 10 or 20 mm (tolerance 1e-7 mm). Anchors are either all predicted kidney voxels or the two largest kidney components. The kidney output itself is unchanged. A component beyond the threshold becomes background. Without a kidney prediction, every filter falls back to the original prediction.</p>',
               '<p>Distances use the complete affine in physical coordinates. There is no minimum lesion size and no relabelling between tumor and cyst. A connection near an anchor retains even a large remote extension. These hypotheses may remove true tissue and may retain false positives near a kidney.</p>',
               '<h2>Baseline</h2>', table(['Class', 'Mean Dice (all)', 'Mean Dice (GT+)', 'Mean recall (GT+)', 'GT− with prediction / GT−'], [
                   [k, number(v['dice_all']['mean']), number(v['dice_positive']['mean']), number(v['recall_positive']['mean']),
                    f'{v["negative_with_prediction_n"]}/{v["reference_negative_n"]}'] for k, v in summary['baseline'].items()]),
               '<p>Both empty masks give Dice 1; exactly one empty gives 0. Positive-case means exclude empty reference masks. Removing false positives can raise all-case cyst Dice through empty cases, so inspect positive-case Dice and recall as well.</p>',
               '<h2>Cyst benefit and harm</h2>']
    output.append(table(['Variant', 'Dice all', 'Dice GT+', 'Δ Dice all', 'GT− with prediction', 'FP removed (mL)', 'TP lost (mL)', 'Dice loss >0.01 (cases)'], [
        [name, number(v['classes']['cyst']['dice_all']['mean']), number(v['classes']['cyst']['dice_positive']['mean']),
         number(v['classes']['cyst']['delta_dice']['mean']), v['classes']['cyst']['negative_with_prediction_n'],
         number(v['classes']['cyst']['removed_fp_ml_sum']), number(v['classes']['cyst']['removed_tp_ml_sum']),
         v['classes']['cyst']['dice_loss_over_001_n']] for name, v in summary['variants'].items()]))
    output += ['<h2>Tumor and combined renal effects</h2>', table(['Variant', 'Tumor Δ Dice', 'Tumor TP lost (mL)', 'Renal Δ Dice', 'Renal recall loss >0.01 (cases)', 'Reference foreground removed (mL)'], [
        [name, number(v['classes']['tumor']['delta_dice']['mean']), number(v['classes']['tumor']['removed_tp_ml_sum']),
         number(v['classes']['renal_envelope']['delta_dice']['mean']), v['classes']['renal_envelope']['recall_loss_over_001_n'],
         number(v['removed_reference_foreground_ml_sum'])] for name, v in summary['variants'].items()]),
        '<p>FP/TP volumes are sums across cases; Dice and recall are case-wise means. Foreground removed includes true foreground originally assigned the wrong lesion class. Thus a deletion can reduce class-specific FP while damaging the combined renal mask. Filtering cannot recover missing true positives.</p>',
        '<h2>Case-level safeguards</h2>',
        f'<p>Original Block 66 review cases included so far: {len(summary["block66_review_cases_present"])}/{len(plan["source_review_rows"])}. No-anchor fallbacks: {len(summary["no_anchor_case_ids"])}.</p>']
    for variant in plan['variants']:
        name = variant['name']; values = summary['variants'][name]['classes']
        output.append('<details><summary>'+escape(name)+' · losses and original review cases</summary>')
        output.append(table(['Class', 'Cases improved', 'Cases worse', 'Worst Δ Dice', 'Recall loss >0.01'], [
            [k, v['improved_n'], v['worse_n'], number(v['delta_dice']['min']), v['recall_loss_over_001_n']] for k, v in values.items()]))
        rows = []
        for row in plan['source_review_rows']:
            if row['case_id'] not in records: continue
            r = records[row['case_id']][0]; v = r['variants'][name]
            rows.append([row['case_id'], ', '.join(row['reasons']), number(v['changes']['cyst']['delta_dice']),
                         number(v['changes']['tumor']['delta_dice']), number(v['changes']['renal_envelope']['delta_dice']),
                         number(v['removed_reference_foreground_ml'])])
        output.append(table(['Case', 'Block 66 reason', 'Cyst Δ Dice', 'Tumor Δ Dice', 'Renal Δ Dice', 'Foreground removed mL'], rows))
        for key in ('cyst', 'tumor', 'renal_envelope'):
            output.append('<p>'+escape(key)+' · worst case-wise Dice changes: '+escape(', '.join(
                f'{r["case_id"]} ({r["delta_dice"]:+.4f})' for r in values[key]['worst_dice_changes'][:5]))+'</p>')
        output.append('</details>')
    output += ['<h2>Interpretation</h2>', '<p>No variant is promoted by this report. Complete all 289 cases and inspect individual harm, including the positive cyst controls, before freezing a candidate for a separate validation experiment. Test120 remains sealed. Small components are voxel-connected regions, not adjudicated clinical lesions.</p>',
               '<p>JSON records retain exact paired confusion matrices, removal counts and all metrics. The sweep stores no extra NIfTI volumes or model weights. Timings cover counts and component analysis, exclude file verification and saving, and are not a controlled speed benchmark.</p></html>']
    return '\n'.join(output)


def export(root, plan, source, parent):
    from vista3d_tfg.experiments.supervised67 import completed
    output = contract.output_root(root, plan); records, issues = {}, []
    for cid in plan['case_ids']:
        try:
            value = completed(root, plan, source, parent, cid)
            if value is not None: records[cid] = value
        except Exception as exc: issues.append({'case_id': cid, 'error': f'{type(exc).__name__}: {exc}'})
    summary = sensitivity.summarize([v[0] for v in records.values()], plan['case_ids'], plan['source_review_rows'], plan['policy'])
    if (output/'last_action.json').exists():
        last = c.read_json(output/'last_action.json')
        if last.get('status') == 'failed': issues.append(last)
    if issues: summary['status'] = 'failed'
    summary['integrity_issues'] = issues
    c.atomic_json(output/'summary.json', summary)
    (output/'report.html').write_text(html_report(plan, summary, records, issues), encoding='utf-8')
    files = {n: output/n for n in ('protocol.json', 'summary.json', 'report.html')}
    for n in ('progress.json', 'last_action.json'):
        if (output/n).exists(): files[n] = output/n
    for cid, (_, receipt, attempt) in records.items():
        files[f'cases/{cid}/complete.json'] = attempt.parent/'complete.json'
        files[f'cases/{cid}/{attempt.name}/sensitivity.json'] = attempt/'sensitivity.json'
    c.atomic_json(output/'archive_manifest.json', {n: c.sha256_file(p) for n, p in files.items()})
    files['archive_manifest.json'] = output/'archive_manifest.json'
    archive = output/'block67_component_sensitivity_report.zip'; temporary = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, p in files.items(): z.write(p, n)
    temporary.replace(archive)
    return archive, summary, issues
