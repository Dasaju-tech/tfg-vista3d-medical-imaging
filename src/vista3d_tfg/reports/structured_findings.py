from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


LEFT_KIDNEY_LABEL = 14
RIGHT_KIDNEY_LABEL = 5
LEFT_CYST_LABEL = 116
RIGHT_CYST_LABEL = 117


@dataclass(frozen=True, slots=True)
class FileEvidence:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class AutomaticRenalMeasurements:
    left_kidney_core_volume_ml: float
    right_kidney_core_volume_ml: float
    automatic_left_cyst_volume_ml: float
    automatic_right_cyst_volume_ml: float
    labels_present: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class InteractiveCystFinding:
    user_point_xyz: tuple[int, int, int]
    laterality: str
    class_label: int
    class_name: str
    laterality_margin_mm: float
    selected_kidney_distance_mm: float
    predicted_volume_ml: float
    labels_present: tuple[int, ...]
    contains_255_unprocessed: bool


@dataclass(frozen=True, slots=True)
class StructuredCaseFindings:
    schema_version: str
    case_id: str
    ground_truth_used: bool
    clinical_validation: bool
    automatic_measurements: AutomaticRenalMeasurements
    interactive_cyst: InteractiveCystFinding
    evidence: dict[str, FileEvidence]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _integer_labels(array: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise ValueError("Segmentation contains non-finite values.")

    if not np.allclose(
        array,
        np.rint(array),
        rtol=0.0,
        atol=1e-5,
    ):
        raise TypeError("Segmentation must be integer-valued.")

    return np.rint(array).astype(np.int16, copy=False)


def _volume_ml(
    labels: np.ndarray,
    *,
    label: int,
    voxel_volume_ml: float,
) -> float:
    return float(
        np.count_nonzero(labels == label) * voxel_volume_ml
    )


def measure_automatic_renal_prediction(
    prediction_path: Path,
) -> AutomaticRenalMeasurements:
    nii = nib.load(str(prediction_path))
    labels = _integer_labels(np.asanyarray(nii.dataobj))

    spacing = tuple(
        float(value)
        for value in nii.header.get_zooms()[:3]
    )
    voxel_volume_ml = float(np.prod(spacing) / 1000.0)

    return AutomaticRenalMeasurements(
        left_kidney_core_volume_ml=_volume_ml(
            labels,
            label=LEFT_KIDNEY_LABEL,
            voxel_volume_ml=voxel_volume_ml,
        ),
        right_kidney_core_volume_ml=_volume_ml(
            labels,
            label=RIGHT_KIDNEY_LABEL,
            voxel_volume_ml=voxel_volume_ml,
        ),
        automatic_left_cyst_volume_ml=_volume_ml(
            labels,
            label=LEFT_CYST_LABEL,
            voxel_volume_ml=voxel_volume_ml,
        ),
        automatic_right_cyst_volume_ml=_volume_ml(
            labels,
            label=RIGHT_CYST_LABEL,
            voxel_volume_ml=voxel_volume_ml,
        ),
        labels_present=tuple(
            sorted(int(value) for value in np.unique(labels))
        ),
    )


def load_user_guided_finding(
    workflow_path: Path,
) -> InteractiveCystFinding:
    payload = json.loads(
        workflow_path.read_text(encoding="utf-8")
    )

    if payload.get("ground_truth_used") is not False:
        raise ValueError(
            "User-guided workflow must explicitly state "
            "ground_truth_used=false."
        )

    if payload.get("clinical_validation") is not False:
        raise ValueError(
            "Research workflow must explicitly state "
            "clinical_validation=false."
        )

    raw = payload.get("result")

    if not isinstance(raw, dict):
        raise ValueError("Workflow JSON does not contain result object.")

    assignment = raw.get("semantic_assignment")

    if not isinstance(assignment, dict):
        raise ValueError(
            "Workflow result does not contain semantic_assignment."
        )

    laterality = str(assignment["laterality"])
    class_label = int(assignment["class_label"])

    expected_label = (
        LEFT_CYST_LABEL
        if laterality == "left"
        else RIGHT_CYST_LABEL
        if laterality == "right"
        else None
    )

    if expected_label is None:
        raise ValueError(
            f"Unsupported cyst laterality: {laterality!r}."
        )

    if class_label != expected_label:
        raise ValueError(
            "Workflow laterality and semantic cyst class disagree."
        )

    return InteractiveCystFinding(
        user_point_xyz=tuple(
            int(value)
            for value in raw["point_xyz"]
        ),
        laterality=laterality,
        class_label=class_label,
        class_name=str(assignment["class_name"]),
        laterality_margin_mm=float(
            assignment["laterality_margin_mm"]
        ),
        selected_kidney_distance_mm=float(
            assignment["selected_distance_mm"]
        ),
        predicted_volume_ml=float(
            raw["foreground_volume_ml"]
        ),
        labels_present=tuple(
            int(value)
            for value in raw["labels_present"]
        ),
        contains_255_unprocessed=bool(
            raw["contains_255_unprocessed"]
        ),
    )


def build_structured_case_findings(
    *,
    case_id: str,
    image_path: Path,
    automatic_prediction_path: Path,
    workflow_path: Path,
    interactive_prediction_path: Path,
) -> StructuredCaseFindings:
    required_files = {
        "input_image": image_path,
        "automatic_renal_prediction": automatic_prediction_path,
        "interactive_cyst_prediction": interactive_prediction_path,
        "interactive_workflow_manifest": workflow_path,
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required reporting evidence is missing: "
            + ", ".join(missing)
        )

    automatic = measure_automatic_renal_prediction(
        automatic_prediction_path
    )
    interactive = load_user_guided_finding(workflow_path)

    evidence = {
        name: FileEvidence(
            path=str(path.resolve()),
            sha256=sha256_file(path),
        )
        for name, path in required_files.items()
    }

    limitations = (
        "Segmentation-derived research measurements only.",
        "No ground-truth annotation is used by this reporting workflow.",
        "No diagnostic, malignancy, staging, or treatment claim is generated.",
        "Interactive cyst output depends on a user-provided spatial click.",
        "The workflow has not been clinically validated.",
    )

    return StructuredCaseFindings(
        schema_version="1.0",
        case_id=case_id,
        ground_truth_used=False,
        clinical_validation=False,
        automatic_measurements=automatic,
        interactive_cyst=interactive,
        evidence=evidence,
        limitations=limitations,
    )


def validate_structured_findings(
    findings: StructuredCaseFindings,
) -> None:
    if findings.schema_version != "1.0":
        raise ValueError("Unsupported findings schema version.")

    if findings.ground_truth_used:
        raise ValueError(
            "Structured reporting contract forbids GT-derived facts."
        )

    if findings.clinical_validation:
        raise ValueError(
            "Clinical validation must remain false for this prototype."
        )

    measurements = findings.automatic_measurements

    non_negative_values = (
        measurements.left_kidney_core_volume_ml,
        measurements.right_kidney_core_volume_ml,
        measurements.automatic_left_cyst_volume_ml,
        measurements.automatic_right_cyst_volume_ml,
        findings.interactive_cyst.predicted_volume_ml,
        findings.interactive_cyst.selected_kidney_distance_mm,
        findings.interactive_cyst.laterality_margin_mm,
    )

    if any(value < 0.0 for value in non_negative_values):
        raise ValueError(
            "Structured findings contain a negative physical quantity."
        )

    expected = (
        LEFT_CYST_LABEL
        if findings.interactive_cyst.laterality == "left"
        else RIGHT_CYST_LABEL
        if findings.interactive_cyst.laterality == "right"
        else None
    )

    if expected is None:
        raise ValueError("Invalid interactive cyst laterality.")

    if findings.interactive_cyst.class_label != expected:
        raise ValueError(
            "Interactive cyst class is inconsistent with laterality."
        )

    if not findings.evidence:
        raise ValueError("Structured findings require evidence files.")

    for evidence in findings.evidence.values():
        if len(evidence.sha256) != 64:
            raise ValueError("Invalid SHA-256 evidence digest.")


def render_deterministic_report(
    findings: StructuredCaseFindings,
) -> str:
    validate_structured_findings(findings)

    auto = findings.automatic_measurements
    cyst = findings.interactive_cyst

    lines = [
        f"Research segmentation summary — {findings.case_id}",
        "",
        "Automatic renal segmentation:",
        (
            "  Left kidney-core volume: "
            f"{auto.left_kidney_core_volume_ml:.2f} mL"
        ),
        (
            "  Right kidney-core volume: "
            f"{auto.right_kidney_core_volume_ml:.2f} mL"
        ),
        (
            "  Automatic left cyst-class volume: "
            f"{auto.automatic_left_cyst_volume_ml:.2f} mL"
        ),
        (
            "  Automatic right cyst-class volume: "
            f"{auto.automatic_right_cyst_volume_ml:.2f} mL"
        ),
        "",
        "User-guided cyst segmentation:",
        (
            f"  User point: "
            f"({cyst.user_point_xyz[0]}, "
            f"{cyst.user_point_xyz[1]}, "
            f"{cyst.user_point_xyz[2]})"
        ),
        (
            f"  Automatic semantic routing: "
            f"{cyst.laterality} / class {cyst.class_label} "
            f"({cyst.class_name})"
        ),
        (
            "  Distance to selected automatic kidney: "
            f"{cyst.selected_kidney_distance_mm:.2f} mm"
        ),
        (
            "  Laterality margin: "
            f"{cyst.laterality_margin_mm:.2f} mm"
        ),
        (
            "  Interactive predicted mask volume: "
            f"{cyst.predicted_volume_ml:.2f} mL"
        ),
        "",
        "Interpretation boundary:",
        (
            "  These are segmentation-derived research measurements. "
            "No ground-truth annotation was used by this workflow."
        ),
        (
            "  No diagnostic, malignancy, staging, or treatment "
            "interpretation is generated."
        ),
        "  Clinical validation: no.",
    ]

    return "\n".join(lines) + "\n"


def write_structured_report(
    findings: StructuredCaseFindings,
    output_directory: Path,
) -> tuple[Path, Path]:
    validate_structured_findings(findings)

    output_directory.mkdir(parents=True, exist_ok=True)

    json_path = output_directory / "findings.json"
    text_path = output_directory / "report.txt"

    json_path.write_text(
        json.dumps(
            findings.to_dict(),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    text_path.write_text(
        render_deterministic_report(findings),
        encoding="utf-8",
    )

    return json_path, text_path
