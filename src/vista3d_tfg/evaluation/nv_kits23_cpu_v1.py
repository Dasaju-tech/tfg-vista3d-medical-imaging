"""Explicit, opt-in version of the Block 61 optimized supported-renal evaluator.

Serial use only: the scoped validator substitution is restored on exit. Existing
entry points continue using the original evaluator and original contracts.
"""
from __future__ import annotations

from pathlib import Path

from vista3d_tfg.analysis.full489_cpu import evaluation_mode
from vista3d_tfg.evaluation.nv_kits23 import (
    SupportedRenalComparison, evaluate_supported_renal_prediction as _baseline,
)


EVALUATOR_VERSION = "supported-renal-cpu-bool-v1"


def evaluate_supported_renal_prediction_cpu_v1(
    *, case_id: str, ground_truth_path: Path, prediction_path: Path,
) -> SupportedRenalComparison:
    """Return the original result schema with the verified Boolean fast path."""
    with evaluation_mode(True):
        return _baseline(case_id=case_id, ground_truth_path=ground_truth_path,
                         prediction_path=prediction_path)
