from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.experiments.full489 import (
    EXPERIMENT,
    assert_existing_protocol_matches,
    audit_inventory,
    build_protocol,
    enforce_initial_reservation,
    fetch_metadata,
    write_protocol,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Block 57: freeze 489 KiTS23 cases and development/validation/test."
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--metadata", type=Path,
        help="Offline original kits23.json bytes matching the expanded100 SHA-256.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Verify an existing freeze and inventory without downloads or writes.",
    )
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve() if args.project_root else discover_project_root()
        baseline_path = root / "configs/experiments/nv_segment_ct_expanded100_selection.json"
        protocol_path = root / "configs/experiments/nv_segment_ct_full489_protocol.json"
        output_root = root / "outputs/protocols" / EXPERIMENT
        snapshot_path = output_root / "kits23_metadata.json"
        # Read the user's manifest before attempting any network operation.
        baseline_raw = baseline_path.read_bytes()
        if args.check and not protocol_path.is_file():
            raise ValueError("--check requires an existing full489 protocol.")
        if args.metadata:
            metadata_raw = args.metadata.read_bytes()
        elif snapshot_path.is_file():
            metadata_raw = snapshot_path.read_bytes()
        elif args.check:
            raise ValueError("--check needs the saved metadata snapshot or --metadata.")
        else:
            print("Downloading official KiTS23 metadata only...", flush=True)
            metadata_raw = fetch_metadata()

        payload = build_protocol(metadata_raw=metadata_raw, expanded100_raw=baseline_raw)
        existed = assert_existing_protocol_matches(protocol_path, payload)
        initial_audit_path = output_root / "initial_inventory_audit.json"
        if not existed and initial_audit_path.exists():
            raise ValueError(
                "An initial audit exists without its protocol. Restore the matching "
                "freeze before continuing; the audit was not overwritten."
            )
        if snapshot_path.exists() and snapshot_path.read_bytes() != metadata_raw:
            raise ValueError("Saved metadata snapshot differs; it was not overwritten.")
        audit = audit_inventory(root, payload)
        enforce_initial_reservation(audit, already_frozen=existed)
        audit["protocol_preexisted_this_run"] = existed
        audit["reserved_output_history_requires_review"] = bool(
            audit["reserved_cases_with_existing_outputs"]
        )

        if not args.check:
            output_root.mkdir(parents=True, exist_ok=True)
            if not snapshot_path.exists():
                with snapshot_path.open("xb") as handle:
                    handle.write(metadata_raw)
            created = write_protocol(protocol_path, payload)
            audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
            (output_root / "inventory_audit.json").write_text(audit_text, encoding="utf-8")
            if created:
                with initial_audit_path.open(
                    "x", encoding="utf-8", newline="\n",
                ) as handle:
                    handle.write(audit_text)

        print("KiTS23 full489 protocol - Block 57")
        print("Official cases: 489")
        print("Retained expanded100: 100")
        print("Additional cases: 389")
        for role in ("development", "validation", "test"):
            print(f"{role.capitalize()}: {payload['partition'][role]['case_count']}")
        print("Historical cases in validation/test: 0")
        print(f"Metadata SHA-256: {payload['sources']['official_metadata_sha256']}")
        print(f"Protocol SHA-256: {payload['protocol_sha256']}")
        print(f"Freeze: {'verified' if existed else 'created'}")
        print("Local nonempty files (presence only, not validated/reusable):")
        for name, count in audit["nonempty_file_counts"].items():
            print(f"  {name}: {count}/489")
        reserved_count = len(audit["reserved_cases_with_existing_outputs"])
        print(f"Reserved cases with output artifacts: {reserved_count}")
        if reserved_count:
            print("Existing allocation preserved; review when reserved results were exposed.")
        print("CT downloads / GPU inference / metric calculation: none")
        print(f"Protocol: {protocol_path}")
        if args.check:
            print("Read-only check: no downloads or files written.")
        else:
            print(f"Inventory: {output_root / 'inventory_audit.json'}")
        print("Commit the protocol before Block 58; keep test metrics and QC sealed.")
        print("Full489 protocol freeze PASSED.")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Block 57 FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
