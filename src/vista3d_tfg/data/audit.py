from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .kits23 import KiTS23Case
from .nifti import KITS23_LABEL_NAMES, NiftiMetadata, inspect_nifti, read_segmentation_labels


@dataclass(frozen=True, slots=True)
class KiTS23CaseAudit:
    case_id: str
    image: NiftiMetadata
    segmentation: NiftiMetadata
    labels: tuple[int, ...]
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class KiTS23DatasetAudit:
    cases: tuple[KiTS23CaseAudit, ...]

    @property
    def valid_cases(self) -> int:
        return sum(case.is_valid for case in self.cases)

    @property
    def invalid_cases(self) -> int:
        return len(self.cases) - self.valid_cases

    @property
    def is_valid(self) -> bool:
        return self.invalid_cases == 0


def _metadata_to_dict(metadata: NiftiMetadata) -> dict[str, object]:
    return {
        "path": str(metadata.path),
        "shape": list(metadata.shape),
        "spacing_mm": list(metadata.spacing_mm),
        "orientation": list(metadata.orientation),
        "dtype": metadata.dtype,
        "affine": [list(row) for row in metadata.affine],
    }


def audit_case(case: KiTS23Case) -> KiTS23CaseAudit:
    image = inspect_nifti(case.imaging_path)
    segmentation = inspect_nifti(case.segmentation_path)
    labels = read_segmentation_labels(case.segmentation_path)

    issues: list[str] = []

    if image.shape != segmentation.shape:
        issues.append(
            f"shape_mismatch:image={image.shape},segmentation={segmentation.shape}"
        )

    if not np.allclose(
        np.asarray(image.affine),
        np.asarray(segmentation.affine),
        rtol=0.0,
        atol=1e-5,
    ):
        issues.append("affine_mismatch")

    if not np.allclose(
        image.spacing_mm,
        segmentation.spacing_mm,
        rtol=0.0,
        atol=1e-6,
    ):
        issues.append(
            "spacing_mismatch:"
            f"image={image.spacing_mm},segmentation={segmentation.spacing_mm}"
        )

    if image.orientation != segmentation.orientation:
        issues.append(
            "orientation_mismatch:"
            f"image={image.orientation},segmentation={segmentation.orientation}"
        )

    allowed_labels = set(KITS23_LABEL_NAMES)
    unexpected_labels = sorted(set(labels) - allowed_labels)
    if unexpected_labels:
        issues.append(f"unexpected_labels:{unexpected_labels}")

    return KiTS23CaseAudit(
        case_id=case.case_id,
        image=image,
        segmentation=segmentation,
        labels=labels,
        issues=tuple(issues),
    )


def audit_cases(cases: tuple[KiTS23Case, ...]) -> KiTS23DatasetAudit:
    return KiTS23DatasetAudit(
        cases=tuple(audit_case(case) for case in cases),
    )


def audit_to_dict(audit: KiTS23DatasetAudit) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset": "KiTS23",
        "label_mapping": {
            str(label): name for label, name in KITS23_LABEL_NAMES.items()
        },
        "number_of_cases": len(audit.cases),
        "valid_cases": audit.valid_cases,
        "invalid_cases": audit.invalid_cases,
        "is_valid": audit.is_valid,
        "cases": [
            {
                "case_id": case.case_id,
                "is_valid": case.is_valid,
                "labels": list(case.labels),
                "issues": list(case.issues),
                "image": _metadata_to_dict(case.image),
                "segmentation": _metadata_to_dict(case.segmentation),
            }
            for case in audit.cases
        ],
    }


def write_audit(audit: KiTS23DatasetAudit, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit_to_dict(audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )
