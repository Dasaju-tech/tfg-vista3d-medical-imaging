from __future__ import annotations

import argparse

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.interaction.manual_tumor_clicks import (
    OrthogonalClickViewer,
    load_manual_click,
    load_manual_click_cases,
    save_manual_click,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect CT-only manual tumor clicks for the frozen "
            "12-case tumor-size-stratified cohort."
        )
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Collect only one frozen case.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing manual click.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = discover_project_root()

    protocol_path = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_tumor_adaptive_relative_depth"
        / "protocol.json"
    )
    dataset_root = (
        root
        / "data"
        / "raw"
        / "kits23"
    )
    output_root = (
        root
        / "outputs"
        / "experiments"
        / "nv_segment_ct_manual_tumor_clicks"
        / "clicks"
    )

    cases = load_manual_click_cases(
        block46a_protocol_path=protocol_path,
        dataset_root=dataset_root,
    )

    if args.case_id is not None:
        cases = tuple(
            row
            for row in cases
            if row.case_id == args.case_id
        )
        if len(cases) != 1:
            raise SystemExit(
                f"Unknown frozen case: {args.case_id}"
            )

    print("NV-Segment-CT manual tumor-click collection")
    print("Frozen cohort: Block 46A, 12 cases")
    print("Viewer input: CT image only")
    print("Display orientation: canonical RAS")
    print("Saved coordinate: original NIfTI voxel xyz")
    print("Ground truth visible during selection: no")
    print("GPU inference: none")
    print()
    print(
        "Controls: scroll=navigate, left-click=point, "
        "Enter=accept, R=reset, Q=abort."
    )
    print()

    collected = 0
    reused = 0

    for index, row in enumerate(cases, start=1):
        output_path = (
            output_root
            / f"{row.case_id}.json"
        )

        if output_path.is_file() and not args.force:
            try:
                click = load_manual_click(output_path)
            except ValueError as exc:
                print(
                    f"[{index:02d}/{len(cases):02d}] "
                    f"{row.case_id}: legacy click rejected "
                    f"({exc}); recollecting."
                )
            else:
                print(
                    f"[{index:02d}/{len(cases):02d}] "
                    f"{row.case_id}: reused original={click.point_xyz}, "
                    f"display_RAS={click.display_point_ras_xyz}"
                )
                reused += 1
                continue

        print(
            f"[{index:02d}/{len(cases):02d}] "
            f"{row.case_id} ({row.quartile})"
        )

        viewer = OrthogonalClickViewer(
            image_path=row.image_path,
            case_id=row.case_id,
            quartile=row.quartile,
        )
        click = viewer.collect()

        if click is None:
            print(
                f"  collection stopped before accepting {row.case_id}."
            )
            break

        save_manual_click(
            click=click,
            output_path=output_path,
        )
        collected += 1

        print(
            f"  saved manual point: original={click.point_xyz}, "
            f"display_RAS={click.display_point_ras_xyz}"
        )

    print()
    print(f"New clicks collected: {collected}")
    print(f"Existing clicks reused: {reused}")
    print(f"Output root: {output_root}")
    print(
        "Manual tumor-click collection finished "
        "(GT hidden during selection)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
