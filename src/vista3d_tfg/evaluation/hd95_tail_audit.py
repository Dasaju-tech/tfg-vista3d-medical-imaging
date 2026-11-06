from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TailCase:
    case_id: str
    dice: float
    surface_dice: float
    hd95_mm: float
    cyst_reference_volume_ml: float
    tumor_volume_ml: float
    kidney_and_mass_components: int
    anatomy_qc_flags: tuple[str, ...]


def read_ranked_cases(path: Path) -> tuple[TailCase, ...]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=100:
        raise ValueError(f'Expected 100 expanded100 rows, got {len(rows)}.')
    ids=[row['case_id'] for row in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate case IDs in ranked_cases.csv.')
    return tuple(
        TailCase(
            case_id=row['case_id'],
            dice=float(row['kidney_and_mass_dice']),
            surface_dice=float(row['kidney_and_mass_surface_dice']),
            hd95_mm=float(row['kidney_and_mass_hd95_mm']),
            cyst_reference_volume_ml=float(row['cyst_reference_volume_ml']),
            tumor_volume_ml=float(row['gt_tumor_volume_ml']),
            kidney_and_mass_components=int(row['kidney_and_mass_components']),
            anatomy_qc_flags=tuple(
                flag for flag in row.get('anatomy_qc_flags','').split(';') if flag
            ),
        )
        for row in rows
    )


def hd95_outlier_threshold(rows: Iterable[TailCase]) -> float:
    values=np.asarray([r.hd95_mm for r in rows], dtype=float)
    if values.size==0:
        raise ValueError('Cannot define HD95 threshold on empty rows.')
    q1,q3=np.percentile(values,[25.0,75.0])
    return float(q3+1.5*(q3-q1))


def classify_failure_mode(row: TailCase, *, threshold_mm: float) -> str:
    if row.hd95_mm<=threshold_mm:
        return 'non_hd95_outlier'
    if row.dice<0.60:
        return 'catastrophic_overlap_failure'
    if row.dice>=0.85:
        return 'distant_boundary_failure'
    return 'mixed_hd95_failure'


def _summary(rows: tuple[TailCase,...], threshold: float) -> dict[str, object]:
    if not rows:
        return {'count':0,'hd95_mean_mm':None,'hd95_median_mm':None,'dice_mean':None,'surface_dice_mean':None,'outlier_count':0,'outlier_fraction':None}
    hd=np.asarray([r.hd95_mm for r in rows],dtype=float)
    dice=np.asarray([r.dice for r in rows],dtype=float)
    surf=np.asarray([r.surface_dice for r in rows],dtype=float)
    out=sum(r.hd95_mm>threshold for r in rows)
    return {
        'count':len(rows),
        'hd95_mean_mm':float(hd.mean()),
        'hd95_median_mm':float(np.median(hd)),
        'dice_mean':float(dice.mean()),
        'surface_dice_mean':float(surf.mean()),
        'outlier_count':out,
        'outlier_fraction':out/len(rows),
    }


def analyze_hd95_tail(rows: Iterable[TailCase]) -> dict[str, object]:
    rows=tuple(rows)
    if len(rows)!=100:
        raise ValueError(f'Expected expanded100 N=100, got {len(rows)}.')
    threshold=hd95_outlier_threshold(rows)
    outliers=tuple(r for r in rows if r.hd95_mm>threshold)
    non_outliers=tuple(r for r in rows if r.hd95_mm<=threshold)

    positive=np.asarray([r.tumor_volume_ml for r in rows if r.tumor_volume_ml>0],dtype=float)
    if positive.size:
        q33,q67=np.percentile(positive,[100/3,200/3])
    else:
        q33=q67=0.0
    strata={
        'no_tumor':tuple(r for r in rows if r.tumor_volume_ml==0),
        'low_tumor':tuple(r for r in rows if 0<r.tumor_volume_ml<=q33),
        'medium_tumor':tuple(r for r in rows if q33<r.tumor_volume_ml<=q67),
        'high_tumor':tuple(r for r in rows if r.tumor_volume_ml>q67),
    }
    strata_payload={k:_summary(v,threshold) for k,v in strata.items()}

    high=strata['high_tumor']
    rest=tuple(r for r in rows if r not in high)
    high_frac=_summary(high,threshold)['outlier_fraction'] if high else None
    rest_frac=_summary(rest,threshold)['outlier_fraction'] if rest else None
    risk_ratio=(high_frac/rest_frac) if high_frac is not None and rest_frac not in (None,0) else None

    sorted_rows=sorted(rows,key=lambda r:(-r.hd95_mm,r.case_id))
    total=sum(r.hd95_mm for r in rows)
    share=lambda k: sum(r.hd95_mm for r in sorted_rows[:k])/total

    taxonomy=[]
    counts={k:0 for k in ('catastrophic_overlap_failure','distant_boundary_failure','mixed_hd95_failure','non_hd95_outlier')}
    for row in rows:
        mode=classify_failure_mode(row,threshold_mm=threshold)
        counts[mode]+=1
        taxonomy.append({**asdict(row),'hd95_outlier':row.hd95_mm>threshold,'failure_mode':mode})

    return {
        'schema_version':'1.0',
        'experiment':'nv_segment_ct_expanded100_hd95_tail_audit',
        'case_count':100,
        'hd95_outlier_threshold_mm':threshold,
        'hd95_outlier_count':len(outliers),
        'hd95_all_cases':_summary(rows,threshold),
        'hd95_non_outliers':_summary(non_outliers,threshold),
        'tail_concentration':{
            'top_5_share_of_total_hd95':share(5),
            'top_10_share_of_total_hd95':share(10),
            'top_outlier_share_of_total_hd95':share(len(outliers)),
        },
        'tumor_strata_thresholds':{'positive_q33_ml':float(q33),'positive_q67_ml':float(q67)},
        'tumor_strata':strata_payload,
        'high_tumor_outlier_risk_ratio_vs_rest':risk_ratio,
        'failure_mode_counts':counts,
        'case_taxonomy':taxonomy,
        'interpretation_boundary':'Post-hoc descriptive audit used to design the next controlled experiment; no causal claim is made.',
    }
