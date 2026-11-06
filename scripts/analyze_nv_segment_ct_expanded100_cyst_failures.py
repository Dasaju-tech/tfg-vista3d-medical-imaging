from __future__ import annotations

import json
from pathlib import Path

from vista3d_tfg.analysis.cyst_failure_stratification import (
    analyze_cyst_case,
    summarize_cyst_failures,
    write_cyst_failure_outputs,
    write_cyst_failure_report,
)
from vista3d_tfg.config import discover_project_root


def _load_expanded100_case_ids(
    manifest_path: Path,
) -> tuple[str, ...]:
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    discovered: list[str] = []
    seen: set[str] = set()

    def add(value: object) -> None:
        if not isinstance(value, str):
            return

        candidate = value.strip()
        if (
            candidate.startswith("case_")
            and len(candidate) == 10
            and candidate[5:].isdigit()
            and candidate not in seen
        ):
            seen.add(candidate)
            discovered.append(candidate)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key in (
                "case_id",
                "case",
                "id",
            ):
                if key in value:
                    add(value[key])

            for nested in value.values():
                visit(nested)
            return

        if isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
            return

        add(value)

    visit(payload)

    if len(discovered) != 100:
        raise ValueError(
            f"Expected 100 unique expanded100 case IDs in "
            f"{manifest_path}, found {len(discovered)}: "
            f"{discovered[:10]}"
        )

    return tuple(discovered)


def _prediction_path(
    root: Path,
    case_id: str,
) -> Path:
    candidates = (
        root
        / "outputs"
        / "predictions"
        / "nv_segment_ct"
        / case_id
        / "prediction.nii.gz",
        root
        / "outputs"
        / "predictions"
        / "nv_segment_ct_expanded100"
        / case_id
        / "prediction.nii.gz",
    )

    for path in candidates:
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"{case_id}: frozen automatic prediction not found. "
        f"Checked: {', '.join(str(p) for p in candidates)}"
    )


def main() -> int:
    root = discover_project_root()

    manifest_path = (
        root
        / "configs"
        / "experiments"
        / "nv_segment_ct_expanded100_selection.json"
    )
    case_ids = _load_expanded100_case_ids(
        manifest_path
    )

    if len(case_ids) != 100:
        raise ValueError(
            f"Expected expanded100 N=100, got {len(case_ids)}."
        )

    rows = []

    print("NV-Segment-CT expanded100 cyst failure stratification")
    print("Frozen expanded100 cases: 100")
    print("GPU inference: none")
    print("Predictions modified: no")
    print()

    for case_id in case_ids:
        gt_path = (
            root
            / "data"
            / "raw"
            / "kits23"
            / case_id
            / "segmentation.nii.gz"
        )
        pred_path = _prediction_path(
            root,
            case_id,
        )

        try:
            result = analyze_cyst_case(
                case_id=case_id,
                ground_truth_path=gt_path,
                automatic_prediction_path=pred_path,
            )
        except ValueError as exc:
            if "not cyst-positive" in str(exc):
                continue
            raise

        rows.append(result)

        print(
            f"{case_id}: "
            f"ref={result.reference_volume_ml:.2f} mL, "
            f"Dice={result.dice:.4f}, "
            f"precision={result.precision:.4f}, "
            f"recall={result.recall:.4f}, "
            f"error={result.dominant_error}"
        )

    payload = summarize_cyst_failures(
        rows
    )

    output_root = (
        root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_expanded100_cyst_failures"
    )
    write_cyst_failure_outputs(
        payload=payload,
        output_root=output_root,
    )
    write_cyst_failure_report(
        payload=payload,
        path=output_root / "cyst_failure_report.md",
    )

    overall = payload["overall"]
    quartiles = payload["by_reference_volume_quartile"]
    assoc = payload["associations"]

    print()
    print("Cyst failure summary")
    print(
        f"  cyst-positive cases: {payload['case_count']}"
    )
    print(
        f"  Dice mean/median: "
        f"{overall['dice']['mean']:.4f} / "
        f"{overall['dice']['median']:.4f}"
    )
    print(
        f"  precision mean: "
        f"{overall['precision']['mean']:.4f}"
    )
    print(
        f"  recall mean: "
        f"{overall['recall']['mean']:.4f}"
    )
    print(
        f"  zero-Dice cases: "
        f"{overall['zero_dice_count']}/{payload['case_count']}"
    )
    print(
        "  dominant errors: "
        f"{overall['dominant_error_counts']}"
    )
    print()

    for name in (
        "smallest",
        "lower_middle",
        "upper_middle",
        "largest",
    ):
        group = quartiles[name]
        print(name)
        print(
            f"  N={group['count']}, "
            f"median ref volume="
            f"{group['reference_volume_ml']['median']:.2f} mL"
        )
        print(
            f"  Dice mean/median="
            f"{group['dice']['mean']:.4f} / "
            f"{group['dice']['median']:.4f}"
        )
        print(
            f"  precision/recall mean="
            f"{group['precision']['mean']:.4f} / "
            f"{group['recall']['mean']:.4f}"
        )
        print(
            f"  zero Dice="
            f"{group['zero_dice_count']}/{group['count']}"
        )
        print()

    print(
        "Reference volume vs Dice Pearson: "
        f"{assoc['reference_volume_vs_dice_pearson']:+.3f}"
    )
    print(
        "Reference volume vs recall Pearson: "
        f"{assoc['reference_volume_vs_recall_pearson']:+.3f}"
    )
    print()
    print(f"JSON: {output_root / 'summary.json'}")
    print(f"CSV:  {output_root / 'cases.csv'}")
    print(f"MD:   {output_root / 'cyst_failure_report.md'}")
    print("Expanded100 cyst failure stratification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
