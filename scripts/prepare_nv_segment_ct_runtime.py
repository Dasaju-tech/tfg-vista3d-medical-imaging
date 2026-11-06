from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.integration.nv_segment import (
    DEFAULT_RUNTIME_ENV,
    NV_SEGMENT_REPOSITORY_URL,
    RUNTIME_PYTHON_VERSION,
    RuntimeManifest,
    ensure_repository,
    ensure_runtime_environment,
    install_runtime_requirements,
    probe_runtime,
    runtime_paths,
    validate_runtime_probe,
    write_runtime_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the isolated maintained NV-Segment-CT/VISTA3D runtime."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--environment",
        default=DEFAULT_RUNTIME_ENV,
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help=(
            "Do not pip-install the upstream requirements. Useful when the "
            "runtime has already been prepared."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else discover_project_root()
    )

    paths = runtime_paths(project_root)

    print("Preparing maintained VISTA3D/NV-Segment-CT runtime")
    print(f"Repository:  {NV_SEGMENT_REPOSITORY_URL}")
    print(f"Checkout:    {paths.checkout_root}")
    print(f"Conda env:   {args.environment}")
    print(f"Python:      {RUNTIME_PYTHON_VERSION}")

    commit = ensure_repository(paths)
    print(f"Git commit:  {commit}")

    created = ensure_runtime_environment(
        environment_name=args.environment,
    )
    print(
        "Environment: "
        + ("created" if created else "reused")
    )

    if args.skip_install:
        print("Requirements: skipped by request")
    else:
        print("Installing upstream runtime requirements ...")
        install_runtime_requirements(
            paths,
            environment_name=args.environment,
        )
        print("Requirements: installed")

    print("Probing runtime ...")
    probe = probe_runtime(
        environment_name=args.environment,
    )
    validate_runtime_probe(probe)

    print(f"Runtime Python: {probe.python_version}")
    print(f"PyTorch:        {probe.torch_version}")
    print(f"MONAI:          {probe.monai_version}")
    print(f"CUDA:           {probe.cuda_available}")
    print(f"GPU count:      {probe.cuda_device_count}")
    print(f"GPU:            {probe.cuda_device_name}")

    manifest = RuntimeManifest(
        repository_url=NV_SEGMENT_REPOSITORY_URL,
        repository_commit=commit,
        environment_name=args.environment,
        runtime_python_requested=RUNTIME_PYTHON_VERSION,
        probe=probe,
    )

    write_runtime_manifest(
        manifest,
        paths.provenance_path,
    )

    print()
    print(f"Runtime manifest: {paths.provenance_path}")
    print("NV-Segment-CT runtime PREPARATION PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
