from __future__ import annotations
import pytest
from vista3d_tfg.evaluation.hd95_tail_audit import TailCase, analyze_hd95_tail, classify_failure_mode, hd95_outlier_threshold


def row(i:int,*,dice=.9,hd95=None,tumor=None)->TailCase:
    return TailCase(f'case_{i:05d}',dice,.75,float(hd95 if hd95 is not None else 3+i/100),0.0,float(tumor if tumor is not None else i),2,())


def test_threshold_detects_tail():
    rows=[row(i) for i in range(99)]+[row(99,hd95=100)]
    t=hd95_outlier_threshold(rows); assert 3<t<100


def test_failure_modes():
    assert classify_failure_mode(row(1,dice=.5,hd95=20),threshold_mm=10)=='catastrophic_overlap_failure'
    assert classify_failure_mode(row(2,dice=.9,hd95=20),threshold_mm=10)=='distant_boundary_failure'
    assert classify_failure_mode(row(3,dice=.75,hd95=20),threshold_mm=10)=='mixed_hd95_failure'


def test_requires_100_cases():
    with pytest.raises(ValueError,match='N=100'):
        analyze_hd95_tail(row(i) for i in range(99))


def test_tail_concentration_and_modes():
    rows=[]
    for i in range(100):
        hd95=3+i/100; dice=.92
        if i==97: hd95,dice=80,.50
        if i==98: hd95,dice=90,.90
        if i==99: hd95,dice=100,.75
        rows.append(row(i,dice=dice,hd95=hd95,tumor=i))
    result=analyze_hd95_tail(rows)
    assert result['hd95_outlier_count']>=3
    assert result['tail_concentration']['top_5_share_of_total_hd95']>.20
    assert result['failure_mode_counts']['catastrophic_overlap_failure']>=1
    assert result['failure_mode_counts']['distant_boundary_failure']>=1
    assert result['failure_mode_counts']['mixed_hd95_failure']>=1
