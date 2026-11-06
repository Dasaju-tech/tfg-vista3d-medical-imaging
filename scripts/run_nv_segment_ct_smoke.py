from __future__ import annotations

import argparse
import json
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.integration.nv_inference import (
    run_renal_inference,
    write_inference_result,
)
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the first supported-class NV-Segment-CT inference on one "
            "audited local KiTS23 case."
        )
    )
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Block 09 output for this case.",
    )
    return parser.parse_args()


def read_runtime_commit(project_root: Path) -> str:
    manifest_path = (
        project_root
        / "outputs"
        / "runtime"
        / "nv_segment_ct_runtime.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Block 08 runtime manifest is missing. "
            "Run prepare_nv_segment_ct_runtime.py --skip-install first."
        )

    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    probe = payload.get("probe", {})
    if probe.get("cuda_available") is not True:
        raise RuntimeError(
            "Recorded NV-Segment runtime does not have CUDA enabled."
        )

    commit = payload.get("repository_commit")
    if not isinstance(commit, str) or not commit:
        raise RuntimeError(
            "Runtime manifest does not contain a repository commit."
        )

    return commit


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()

    case_id = f"case_{args.case:05d}"
    case = discover_case(
        project_root
        / "data"
        / "raw"
        / "kits23"
        / case_id
    )

    paths = runtime_paths(project_root)
    repository_commit = read_runtime_commit(project_root)

    print("NV-Segment-CT supported renal inference")
    print(f"Case:        {case.case_id}")
    print(f"Input:       {case.imaging_path}")
    print(f"Environment: {args.environment}")
    print(f"Commit:      {repository_commit}")
    print("Labels:      5, 14, 116, 117")
    print(
        "Classes:     right kidney, left kidney, "
        "left kidney cyst, right kidney cyst"
    )
    print()
    print(
        "The first run may download the upstream model checkpoint "
        "from Hugging Face."
    )

    result = run_renal_inference(
        case_id=case.case_id,
        image_path=case.imaging_path,
        paths=paths,
        repository_commit=repository_commit,
        project_output_root=(
            project_root
            / "outputs"
            / "predictions"
            / "nv_segment_ct"
        ),
        environment_name=args.environment,
        force=args.force,
    )

    result_path = (
        project_root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
        / case.case_id
        / "inference.json"
    )
    write_inference_result(result, result_path)

    print()
    print(f"Prediction:  {result.prediction_path}")
    print(f"Wall time:   {result.wall_time_seconds:.2f} s")
    print(
        "Cold start:  "
        + (
            "no (checkpoint already present)"
            if result.checkpoint_present_before
            else "yes (checkpoint absent before run)"
        )
    )
    print(f"SHA-256:     {result.audit.sha256}")
    print(f"Shape:       {result.audit.shape}")
    print(
        f"Geometry:    shape={result.audit.shape_matches_input}, "
        f"affine={result.audit.affine_matches_input}"
    )
    print(
        "Labels:      "
        + ", ".join(
            str(value) for value in result.audit.labels_present
        )
    )
    print(
        "Named:       "
        + (
            ", ".join(result.audit.named_labels_present)
            if result.audit.named_labels_present
            else "background only"
        )
    )
    print(f"Manifest:    {result_path}")
    print("NV-Segment-CT renal smoke inference PASSED.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
