from __future__ import annotations

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reporting.final_manuscript import (
    build_final_manuscript_package,
)


def main() -> int:
    root = discover_project_root()
    output_root = root / "outputs" / "reporting" / "final_manuscript"

    build_final_manuscript_package(
        project_root=root,
        output_root=output_root,
    )

    print("NV-Segment-CT final Results/Discussion manuscript package")
    print("Source of truth: Block 53")
    print("Figures/tables: Block 54")
    print("GPU inference: none")
    print()
    print("Sections")
    for name in (
        "results.tex",
        "discussion.tex",
        "limitations.tex",
        "conclusions.tex",
    ):
        print(f"  {output_root / 'sections' / name}")
    print()
    print(f"Preview: {output_root / 'preview_results_discussion.tex'}")
    print(f"Integration: {output_root / 'integration_snippet.tex'}")
    print(f"Manifest: {output_root / 'manifest.json'}")
    print("Final Results/Discussion manuscript package PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
