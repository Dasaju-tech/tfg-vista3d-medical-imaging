from __future__ import annotations

import json
from pathlib import Path
import shutil

from vista3d_tfg.config import discover_project_root
from vista3d_tfg.reports.dissertation_results import (
    FROZEN_RESULTS,
    build_discussion_outline,
    build_overleaf_readme,
    build_results_outline,
    render_llm_results_table,
    render_main_results_table,
    render_results_macros,
    validate_and_refresh_llm_results,
)


def _copy_if_present(
    source: Path,
    destination: Path,
) -> bool:
    if not source.is_file():
        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(source, destination)
    return True


def main() -> int:
    project_root = discover_project_root()

    synthesis_path = (
        project_root
        / "outputs"
        / "final_reporting_synthesis"
        / "final_reporting_synthesis.json"
    )

    results = validate_and_refresh_llm_results(
        FROZEN_RESULTS,
        synthesis_path,
    )

    synthesis = json.loads(
        synthesis_path.read_text(encoding="utf-8")
    )["result"]

    output_root = project_root / "outputs" / "dissertation"
    tables_root = output_root / "tables"
    latex_root = output_root / "latex"
    figures_root = output_root / "figures"

    for directory in (
        output_root,
        tables_root,
        latex_root,
        figures_root,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest_path = (
        output_root
        / "dissertation_results_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "package": "block29_dissertation_results",
                "language": "English",
                "experimental_state": "frozen",
                "clinical_validation": False,
                "deployment_valid": False,
                "interpretation_boundary": (
                    "The interactive 17-case cohort replays GT-derived "
                    "oracle clicks as controlled user-click surrogates."
                ),
                "results": [
                    result.to_dict()
                    for result in results
                ],
                "llm_synthesis_source": str(synthesis_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    main_table_path = (
        latex_root / "main_results_table.tex"
    )
    main_table_path.write_text(
        render_main_results_table(results),
        encoding="utf-8",
    )

    llm_table_path = (
        latex_root / "llm_results_table.tex"
    )
    llm_table_path.write_text(
        render_llm_results_table(
            results,
            block25_prompt_tokens=float(
                synthesis["block25"][
                    "prompt_tokens_mean"
                ]
            ),
            block27_prompt_tokens=float(
                synthesis["block27_v2"][
                    "prompt_tokens_mean"
                ]
            ),
        ),
        encoding="utf-8",
    )

    macros_path = latex_root / "results_macros.tex"
    macros_path.write_text(
        render_results_macros(results),
        encoding="utf-8",
    )

    results_outline_path = (
        output_root / "results_outline.md"
    )
    results_outline_path.write_text(
        build_results_outline(results),
        encoding="utf-8",
    )

    discussion_outline_path = (
        output_root / "discussion_outline.md"
    )
    discussion_outline_path.write_text(
        build_discussion_outline(),
        encoding="utf-8",
    )

    overleaf_readme_path = (
        output_root / "OVERLEAF_INTEGRATION.md"
    )
    overleaf_readme_path.write_text(
        build_overleaf_readme(),
        encoding="utf-8",
    )

    figure_sources = {
        "acceptance_v1_vs_v2.png": (
            project_root
            / "outputs"
            / "final_reporting_synthesis"
            / "acceptance_v1_vs_v2.png"
        ),
        "llm_latency.png": (
            project_root
            / "outputs"
            / "final_reporting_synthesis"
            / "llm_latency.png"
        ),
        "prompt_token_reduction.png": (
            project_root
            / "outputs"
            / "final_reporting_synthesis"
            / "prompt_token_reduction.png"
        ),
    }

    copied_figures = []

    for name, source in figure_sources.items():
        if _copy_if_present(
            source,
            figures_root / name,
        ):
            copied_figures.append(name)

    package_readme = output_root / "README.md"
    package_readme.write_text(
        "# Dissertation results package\n\n"
        "Frozen Block 29 package for the English Overleaf dissertation.\n\n"
        "## Generated artifacts\n\n"
        "- `dissertation_results_manifest.json`\n"
        "- `latex/results_macros.tex`\n"
        "- `latex/main_results_table.tex`\n"
        "- `latex/llm_results_table.tex`\n"
        "- `results_outline.md`\n"
        "- `discussion_outline.md`\n"
        "- `OVERLEAF_INTEGRATION.md`\n"
        "- `figures/` final reporting figures\n\n"
        "Do not manually change numerical results in the dissertation; "
        "edit the experiment outputs/catalog and regenerate this package "
        "instead.\n",
        encoding="utf-8",
    )

    print("Block 29 — dissertation results integration")
    print()
    print(f"Frozen results: {len(results)}")
    print(f"Copied figures: {len(copied_figures)}")
    print(
        "Figures: "
        + (
            ", ".join(copied_figures)
            if copied_figures
            else "none"
        )
    )
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Macros:   {macros_path}")
    print(f"Table:    {main_table_path}")
    print(f"LLM table:{llm_table_path}")
    print(f"Results:  {results_outline_path}")
    print(f"Discuss.: {discussion_outline_path}")
    print(f"Overleaf: {overleaf_readme_path}")
    print()
    print(
        "Block 29 PASSED: frozen dissertation results package generated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
