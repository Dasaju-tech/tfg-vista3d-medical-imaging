from __future__ import annotations

import csv, json
from pathlib import Path
import matplotlib.pyplot as plt

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.evaluation.hd95_tail_audit import analyze_hd95_tail, read_ranked_cases


def _write_csv(rows: list[dict[str,object]], path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=['case_id','dice','surface_dice','hd95_mm','cyst_reference_volume_ml','tumor_volume_ml','kidney_and_mass_components','anatomy_qc_flags','hd95_outlier','failure_mode']
    with path.open('w',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for row in rows:
            copy=dict(row); copy['anatomy_qc_flags']=';'.join(copy['anatomy_qc_flags']); w.writerow(copy)


def _scatter(x,y,xlabel,ylabel,title,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6.8,4.6)); ax.scatter(x,y,alpha=.75); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def _bar(payload,path):
    strata=payload['tumor_strata']; labels=[]; values=[]
    for name in ('no_tumor','low_tumor','medium_tumor','high_tumor'):
        item=strata[name]
        if item['count']:
            labels.append(name.replace('_','\n')); values.append(item['outlier_fraction'])
    path.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.bar(labels,values); ax.set_ylim(0,1); ax.set_ylabel('HD95 outlier fraction'); ax.set_title('HD95 outlier rate by tumor burden'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main()->int:
    root=discover_project_root()
    src=root/'outputs'/'analyses'/'nv_segment_ct_expanded100_failure_stratification'/'ranked_cases.csv'
    out=root/'outputs'/'analyses'/'nv_segment_ct_expanded100_hd95_tail'
    rows=read_ranked_cases(src); payload=analyze_hd95_tail(rows); out.mkdir(parents=True,exist_ok=True)
    (out/'hd95_tail_audit.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    _write_csv(payload['case_taxonomy'],out/'case_taxonomy.csv')
    figs=out/'figures'
    _scatter([r.tumor_volume_ml for r in rows],[r.hd95_mm for r in rows],'Ground-truth tumor volume (mL)','HD95 (mm)','Tumor burden vs HD95',figs/'tumor_volume_vs_hd95.png')
    _scatter([r.dice for r in rows],[r.hd95_mm for r in rows],'Kidney-and-mass Dice','HD95 (mm)','Overlap vs HD95 tail',figs/'dice_vs_hd95.png')
    _bar(payload,figs/'hd95_outlier_fraction_by_tumor_burden.png')

    tail=payload['tail_concentration']; modes=payload['failure_mode_counts']; all_s=payload['hd95_all_cases']; non_s=payload['hd95_non_outliers']
    print('NV-Segment-CT expanded100 HD95 tail audit')
    print(f"HD95 outliers: {payload['hd95_outlier_count']}/100 (>{payload['hd95_outlier_threshold_mm']:.2f} mm)")
    print(f"HD95 mean all/non-outliers: {all_s['hd95_mean_mm']:.2f} / {non_s['hd95_mean_mm']:.2f} mm")
    print(f"Top-5 / top-10 share of total HD95: {100*tail['top_5_share_of_total_hd95']:.1f}% / {100*tail['top_10_share_of_total_hd95']:.1f}%")
    print('Failure modes:')
    print(f"  catastrophic overlap: {modes['catastrophic_overlap_failure']}")
    print(f"  distant boundary: {modes['distant_boundary_failure']}")
    print(f"  mixed HD95: {modes['mixed_hd95_failure']}")
    rr=payload['high_tumor_outlier_risk_ratio_vs_rest']
    print('High-tumor outlier risk ratio vs rest: '+(f'{rr:.2f}x' if rr is not None else 'undefined'))
    print(f"JSON: {out/'hd95_tail_audit.json'}")
    print(f"CSV:  {out/'case_taxonomy.csv'}")
    print('Expanded100 HD95 tail audit PASSED.')
    return 0


if __name__=='__main__': raise SystemExit(main())
