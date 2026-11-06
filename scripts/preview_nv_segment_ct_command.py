from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    build_single_case_command,
    runtime_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the exact maintained NV-Segment-CT MONAI Bundle "
            "command for one local KiTS23 CT."
        )
    )
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    paths = runtime_paths(project_root)

    case_id = f"case_{args.case:05d}"
    case = discover_case(
        project_root
        / "data"
        / "raw"
        / "kits23"
        / case_id
    )

    command = build_single_case_command(
        paths,
        image_path=case.imaging_path,
        environment_name=args.environment,
    )

    print(f"Case:       {case.case_id}")
    print(f"Bundle cwd: {paths.ct_bundle_root}")
    print("Command:")
    print(subprocess.list2cmdline(command))
    print()
    print(
        "This script only previews the command. "
        "Block 09 will execute inference and validate its NIfTI output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
