from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.expanded100 import (
    EXPANDED100_SIZE,
    fetch_official_case_numbers,
    write_selection_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the deterministic 100-case KiTS23 validation cohort "
            "from the official metadata universe."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Selection manifest path. Defaults to "
            "configs/experiments/nv_segment_ct_expanded100_selection.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = discover_project_root()
    output_path = (
        args.output.resolve()
        if args.output is not None
        else (
            project_root
            / "configs"
            / "experiments"
            / "nv_segment_ct_expanded100_selection.json"
        )
    )

    official_case_numbers, metadata_sha256 = fetch_official_case_numbers()
    payload = write_selection_manifest(
        output_path,
        official_case_numbers=official_case_numbers,
        metadata_sha256=metadata_sha256,
    )

    selected = payload["selection"]["case_numbers"]
    additional = payload["selection"]["additional_case_numbers"]

    print("KiTS23 expanded100 selection freeze")
    print(f"Official cases: {len(official_case_numbers)}")
    print(f"Metadata SHA-256: {metadata_sha256}")
    print(f"Selected cases: {len(selected)}/{EXPANDED100_SIZE}")
    print(f"Retained pilot30: {payload['selection']['retained_pilot30_count']}")
    print(f"Additional cases: {len(additional)}")
    print()
    print("Frozen case IDs:")

    for start in range(0, len(selected), 10):
        print(
            " ".join(
                f"{value:05d}"
                for value in selected[start:start + 10]
            )
        )

    print()
    print(f"Manifest: {output_path}")
    print("No CT volumes were downloaded by this step.")
    print("Expanded100 selection freeze PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
