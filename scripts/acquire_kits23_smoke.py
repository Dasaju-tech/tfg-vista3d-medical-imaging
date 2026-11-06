from __future__ import annotations

import argparse
from pathlib import Path

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.data.acquisition import download_kits23_case
from vista3d_tfg.data.audit import audit_cases, write_audit
from vista3d_tfg.data.kits23 import discover_case
from vista3d_tfg.data.provenance import write_acquisition_provenance


DEFAULT_SMOKE_CASES = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and audit a small real KiTS23 subset without "
            "fetching the complete training dataset."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=list(DEFAULT_SMOKE_CASES),
        help="KiTS23 case numbers to acquire. Default: 0 1 2.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even when a non-empty local copy exists.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Number of retries per file after the first attempt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else discover_project_root()
    )

    dataset_root = project_root / "data" / "raw" / "kits23"
    records = []

    print(f"KiTS23 local root: {dataset_root}")

    for case_number in args.cases:
        print(f"Acquiring case_{case_number:05d} ...")
        case_records = download_kits23_case(
            case_number,
            dataset_root,
            force=args.force,
            retries=args.retries,
        )
        records.extend(case_records)

        for record in case_records:
            print(
                f"  {record.kind:12s} "
                f"{record.status:10s} "
                f"{record.size_bytes / (1024 ** 2):8.2f} MiB "
                f"sha256={record.sha256[:12]}..."
            )

    case_objects = tuple(
        discover_case(dataset_root / f"case_{case_number:05d}")
        for case_number in args.cases
    )

    audit = audit_cases(case_objects)

    provenance_path = (
        project_root
        / "outputs"
        / "manifests"
        / "kits23_smoke_provenance.json"
    )
    audit_path = (
        project_root
        / "outputs"
        / "audits"
        / "kits23_smoke_audit.json"
    )

    write_acquisition_provenance(records, provenance_path)
    write_audit(audit, audit_path)

    print()
    print(
        "Audit result: "
        f"{audit.valid_cases}/{len(audit.cases)} valid cases"
    )
    print(f"Provenance: {provenance_path}")
    print(f"Audit:      {audit_path}")

    if not audit.is_valid:
        print("One or more KiTS23 cases failed the integrity audit.")
        return 2

    print("KiTS23 smoke acquisition PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
