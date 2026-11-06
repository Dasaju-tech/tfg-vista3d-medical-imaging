from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class DissertationResult:
    key: str
    source_block: str
    section: str
    metric: str
    value: float | int | str
    unit: str
    n: int | None
    role: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FROZEN_RESULTS: tuple[DissertationResult, ...] = (
    DissertationResult(
        key="pilot30_renal_dice_mean",
        source_block="Block 13/14",
        section="Automatic renal segmentation",
        metric="Kidney-and-mass Dice, mean",
        value=0.9127,
        unit="",
        n=30,
        role="primary",
        interpretation=(
            "Overall renal-envelope overlap across the deterministic "
            "30-case engineering pilot."
        ),
    ),
    DissertationResult(
        key="pilot30_surface_dice_mean",
        source_block="Block 13/14",
        section="Automatic renal segmentation",
        metric="Kidney-and-mass Surface Dice, mean",
        value=0.7527,
        unit="",
        n=30,
        role="primary",
        interpretation=(
            "Boundary-aware overlap using the KiTS23 kidney-and-mass "
            "surface tolerance."
        ),
    ),
    DissertationResult(
        key="pilot30_hd95_median_mm",
        source_block="Block 13/14",
        section="Automatic renal segmentation",
        metric="Kidney-and-mass HD95, median",
        value=4.15,
        unit="mm",
        n=30,
        role="secondary",
        interpretation=(
            "Robust boundary-distance diagnostic; distribution was "
            "right-skewed by outliers."
        ),
    ),
    DissertationResult(
        key="automatic_cyst_positive_dice_mean",
        source_block="Block 12/14",
        section="Automatic cyst segmentation",
        metric="Cyst-positive Dice, mean",
        value=0.2198,
        unit="",
        n=17,
        role="primary",
        interpretation=(
            "Automatic cyst segmentation was substantially weaker than "
            "renal-envelope segmentation."
        ),
    ),
    DissertationResult(
        key="cyst_zero_overlap_cases",
        source_block="Block 15",
        section="Cyst failure decomposition",
        metric="Zero-overlap automatic cyst cases",
        value=6,
        unit="cases",
        n=17,
        role="diagnostic",
        interpretation=(
            "Failure decomposition indicated a strong localization "
            "component in automatic cyst errors."
        ),
    ),
    DissertationResult(
        key="cyst_localization_recall_30mm",
        source_block="Block 15",
        section="Cyst failure decomposition",
        metric="GT cyst-component localization recall within 30 mm",
        value=0.4310,
        unit="",
        n=116,
        role="diagnostic",
        interpretation=(
            "Only 43.1% of GT cyst components had a predicted cyst centroid "
            "within 30 mm."
        ),
    ),
    DissertationResult(
        key="single_click_oracle_dice_mean",
        source_block="Block 16",
        section="Interactive cyst segmentation",
        metric="Single-click oracle target Dice, mean",
        value=0.4417,
        unit="",
        n=17,
        role="upper_bound",
        interpretation=(
            "A single GT-derived positive click substantially improved the "
            "interactive target overlap; not deployment-valid."
        ),
    ),
    DissertationResult(
        key="single_click_oracle_improved_cases",
        source_block="Block 16",
        section="Interactive cyst segmentation",
        metric="Cases improved by one oracle click",
        value=15,
        unit="cases",
        n=17,
        role="upper_bound",
        interpretation=(
            "The interactive pathway improved 15 of 17 cyst-positive cases."
        ),
    ),
    DissertationResult(
        key="semantic_point_dice_mean",
        source_block="Block 19",
        section="Interactive cyst segmentation",
        metric="Semantic-class + point Dice, mean",
        value=0.4692,
        unit="",
        n=17,
        role="upper_bound",
        interpretation=(
            "Adding side-specific semantic class information showed a "
            "heterogeneous and non-conclusive mean improvement."
        ),
    ),
    DissertationResult(
        key="automatic_laterality_agreement",
        source_block="Block 20",
        section="Automatic semantic routing",
        metric="Automatic cyst laterality agreement",
        value=1.0,
        unit="",
        n=17,
        role="primary",
        interpretation=(
            "Automatic renal-mask routing reproduced the GT-derived "
            "laterality assignment in all evaluated interactive cases."
        ),
    ),
    DissertationResult(
        key="llm_reproducibility_acceptance",
        source_block="Block 25",
        section="Local LLM reporting",
        metric="Repeated-generation acceptance",
        value=1.0,
        unit="",
        n=5,
        role="primary",
        interpretation=(
            "Five frozen repeated generations produced identical accepted "
            "raw responses and canonical candidates."
        ),
    ),
    DissertationResult(
        key="llm_prompt_v1_acceptance",
        source_block="Block 26",
        section="Local LLM reporting",
        metric="Prompt v1 audited acceptance",
        value=16 / 17,
        unit="",
        n=17,
        role="primary",
        interpretation=(
            "One persistent semantic relation error remained after "
            "calibration of deterministic auditor false positives."
        ),
    ),
    DissertationResult(
        key="llm_prompt_v2_acceptance",
        source_block="Block 27",
        section="Local LLM reporting",
        metric="Minimal-facts prompt v2 audited acceptance",
        value=1.0,
        unit="",
        n=17,
        role="primary",
        interpretation=(
            "Prospective regeneration eliminated the sole v1 semantic "
            "failure without changing the frozen auditor."
        ),
    ),
    DissertationResult(
        key="llm_prompt_token_reduction",
        source_block="Block 27/28",
        section="Local LLM reporting",
        metric="Mean prompt-token reduction",
        value=0.479,
        unit="",
        n=17,
        role="efficiency",
        interpretation=(
            "The minimal-facts prompt reduced mean input context while "
            "improving audited factual acceptance."
        ),
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def _find_result(
    results: Iterable[DissertationResult],
    key: str,
) -> DissertationResult:
    for result in results:
        if result.key == key:
            return result

    raise KeyError(key)


def validate_and_refresh_llm_results(
    results: tuple[DissertationResult, ...],
    synthesis_path: Path,
) -> tuple[DissertationResult, ...]:
    payload = _read_json(synthesis_path)
    synthesis = payload["result"]

    refreshed = list(results)

    replacements = {
        "llm_reproducibility_acceptance": float(
            synthesis["block25"]["acceptance_fraction"]
        ),
        "llm_prompt_v1_acceptance": float(
            synthesis["block26_v1"]["acceptance_fraction"]
        ),
        "llm_prompt_v2_acceptance": float(
            synthesis["block27_v2"]["acceptance_fraction"]
        ),
        "llm_prompt_token_reduction": float(
            synthesis["prompt_token_reduction_fraction"]
        ),
    }

    for index, result in enumerate(refreshed):
        if result.key in replacements:
            refreshed[index] = replace(
                result,
                value=replacements[result.key],
            )

    if not bool(
        synthesis["block25_exact_raw_reproducibility"]
    ):
        raise ValueError(
            "Block 25 raw-response reproducibility is no longer exact."
        )

    if not bool(
        synthesis["block25_exact_candidate_reproducibility"]
    ):
        raise ValueError(
            "Block 25 candidate reproducibility is no longer exact."
        )

    if tuple(
        synthesis["block26_persistent_failure_case_ids"]
    ) != ("case_00237",):
        raise ValueError(
            "Unexpected final Block 26 persistent failure set."
        )

    if tuple(
        synthesis["block27_improved_case_ids"]
    ) != ("case_00237",):
        raise ValueError(
            "Unexpected Block 27 improved-case set."
        )

    if tuple(
        synthesis["block27_regressed_case_ids"]
    ):
        raise ValueError(
            "Block 27 contains an unexpected regression."
        )

    return tuple(refreshed)


def format_result_value(result: DissertationResult) -> str:
    value = result.value

    if isinstance(value, float):
        if result.unit == "":
            if 0.0 <= value <= 1.0:
                return f"{value:.4f}"
            return f"{value:.2f}"

        return f"{value:.2f}"

    return str(value)


def latex_escape(value: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    )

    escaped = value

    for old, new in replacements:
        escaped = escaped.replace(old, new)

    return escaped


def render_main_results_table(
    results: Iterable[DissertationResult],
) -> str:
    rows = tuple(results)

    selected_keys = (
        "pilot30_renal_dice_mean",
        "pilot30_surface_dice_mean",
        "pilot30_hd95_median_mm",
        "automatic_cyst_positive_dice_mean",
        "single_click_oracle_dice_mean",
        "semantic_point_dice_mean",
        "automatic_laterality_agreement",
        "llm_reproducibility_acceptance",
        "llm_prompt_v1_acceptance",
        "llm_prompt_v2_acceptance",
    )

    selected = tuple(
        _find_result(rows, key)
        for key in selected_keys
    )

    body = []

    for result in selected:
        n_value = "--" if result.n is None else str(result.n)
        value = format_result_value(result)

        if result.unit:
            value = f"{value} {result.unit}"

        body.append(
            " & ".join(
                (
                    latex_escape(result.source_block),
                    latex_escape(result.metric),
                    n_value,
                    latex_escape(value),
                )
            )
            + r" \\"
        )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Summary of the principal experimental results.}\n"
        "\\label{tab:main-results}\n"
        "\\small\n"
        "\\begin{tabular}{llrr}\n"
        "\\toprule\n"
        "Block & Metric & $N$ & Result \\\\\n"
        "\\midrule\n"
        + "\n".join(body)
        + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def render_llm_results_table(
    results: Iterable[DissertationResult],
    *,
    block25_prompt_tokens: float,
    block27_prompt_tokens: float,
) -> str:
    rows = tuple(results)

    reproducibility = _find_result(
        rows,
        "llm_reproducibility_acceptance",
    )
    v1 = _find_result(
        rows,
        "llm_prompt_v1_acceptance",
    )
    v2 = _find_result(
        rows,
        "llm_prompt_v2_acceptance",
    )
    reduction = _find_result(
        rows,
        "llm_prompt_token_reduction",
    )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Local-LLM reporting results under the frozen "
        "deterministic acceptance gate.}\n"
        "\\label{tab:llm-reporting}\n"
        "\\small\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Protocol & $N$ & Acceptance & Mean prompt tokens \\\\\n"
        "\\midrule\n"
        f"Repeated frozen generation & 5 & "
        f"{100.0 * float(reproducibility.value):.1f}\\% & "
        f"{block25_prompt_tokens:.1f} \\\\\n"
        f"Prompt v1 & 17 & "
        f"{100.0 * float(v1.value):.1f}\\% & -- \\\\\n"
        f"Minimal-facts prompt v2 & 17 & "
        f"{100.0 * float(v2.value):.1f}\\% & "
        f"{block27_prompt_tokens:.1f} \\\\\n"
        "\\midrule\n"
        f"Context reduction & -- & -- & "
        f"{100.0 * float(reduction.value):.1f}\\% \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def render_results_macros(
    results: Iterable[DissertationResult],
) -> str:
    rows = tuple(results)

    mapping = (
        ("PilotRenalDice", "pilot30_renal_dice_mean"),
        ("PilotSurfaceDice", "pilot30_surface_dice_mean"),
        ("PilotHDMedian", "pilot30_hd95_median_mm"),
        ("AutomaticCystDice", "automatic_cyst_positive_dice_mean"),
        ("SingleClickDice", "single_click_oracle_dice_mean"),
        ("SemanticPointDice", "semantic_point_dice_mean"),
        ("AutoLateralityAgreement", "automatic_laterality_agreement"),
        ("LLMVOneAcceptance", "llm_prompt_v1_acceptance"),
        ("LLMVTwoAcceptance", "llm_prompt_v2_acceptance"),
        ("LLMTokenReduction", "llm_prompt_token_reduction"),
    )

    rendered = [
        "% Auto-generated by Block 29.",
        "% Do not edit numerical values manually in the dissertation.",
    ]

    for macro, key in mapping:
        result = _find_result(rows, key)
        value = float(result.value)

        if key.startswith("llm_") or key == "automatic_laterality_agreement":
            if key in {
                "llm_prompt_v1_acceptance",
                "llm_prompt_v2_acceptance",
                "llm_prompt_token_reduction",
                "automatic_laterality_agreement",
            }:
                rendered_value = f"{100.0 * value:.1f}\\%"
            else:
                rendered_value = f"{value:.4f}"
        elif result.unit == "mm":
            rendered_value = f"{value:.2f}\\,mm"
        else:
            rendered_value = f"{value:.4f}"

        rendered.append(
            f"\\newcommand{{\\{macro}}}{{{rendered_value}}}"
        )

    return "\n".join(rendered) + "\n"


def build_results_outline(
    results: Iterable[DissertationResult],
) -> str:
    rows = tuple(results)

    sections: dict[str, list[DissertationResult]] = {}

    for result in rows:
        sections.setdefault(result.section, []).append(result)

    lines = [
        "# Dissertation Results outline",
        "",
        "This file is a drafting aid. Numerical values are frozen in the "
        "Block 29 manifest and LaTeX macros.",
        "",
    ]

    for section, section_results in sections.items():
        lines.extend(
            [
                f"## {section}",
                "",
            ]
        )

        for result in section_results:
            n_text = (
                ""
                if result.n is None
                else f", N={result.n}"
            )
            value = format_result_value(result)

            if result.unit:
                value = f"{value} {result.unit}"

            lines.append(
                f"- **{result.metric}:** {value}{n_text}. "
                f"{result.interpretation}"
            )

        lines.append("")

    return "\n".join(lines)


def build_discussion_outline() -> str:
    return """# Dissertation Discussion outline

## 1. Automatic renal segmentation is the strongest automatic component

The 30-case pilot supports a reasonably robust renal-envelope result, while
boundary metrics reveal a lower-performing tail that should not be hidden by a
single mean overlap score.

## 2. Automatic cyst segmentation is primarily localization-limited

The low cyst-positive Dice, zero-overlap cases, and component-level
localization analysis indicate that the automatic cyst pathway is weaker than
the renal pathway and often fails before contour refinement becomes the main
problem.

## 3. Interactive prompting exposes latent segmentation capability

A single oracle positive click improves most cyst-positive cases, supporting
the interpretation that VISTA3D contains useful interactive capability even
when automatic localization fails.

The oracle protocol is an upper-bound experiment and must not be presented as
deployment-valid.

## 4. More clicks are not automatically better

The click-response and stability experiments show that positive correction
clicks can destabilize an already useful mask, whereas negative corrections
were more consistently beneficial. This argues against naive fixed-click
interaction policies.

## 5. Automatic renal context can remove unnecessary oracle semantics

Automatic renal-mask laterality routing reproduced the GT-derived side
assignment in all 17 evaluated cases, allowing the user-guided workflow to
operate without GT once a click is supplied.

## 6. Structured facts should remain authoritative

The reporting architecture deliberately separates deterministic measurements
from language generation. The LLM is a verbalizer, not a measurement engine.

## 7. Deterministic auditing detects errors invisible to JSON constraints

Structured JSON generation was consistently successful, but the prompt-v1
cohort contained one genuine semantic overreach: click-to-kidney distance was
reinterpreted as cyst-to-kidney spatial alignment.

## 8. Minimal-facts prompting improved both reliability and efficiency

Prospective prompt v2 generation removed the sole semantic error under the
unchanged frozen auditor, with no regressions, while reducing the LLM input
context by approximately 48%.

## 9. Main limitations

- The pilot cohort is small and intended as an engineering study.
- The 17-case interactive cohort uses frozen GT-derived oracle click
  coordinates as experimental user-click surrogates.
- The work does not constitute clinical validation.
- Tumour segmentation is not supported by the maintained NV-Segment-CT class
  inventory used here, so a complete KiTS23 challenge score is not reported.
- The local LLM findings concern one frozen small model/runtime configuration
  and should not be generalized to all language models.
"""


def build_overleaf_readme() -> str:
    return """# Overleaf integration notes

Block 29 deliberately separates experimental consolidation from dissertation
visual design.

Recommended Overleaf structure:

```text
main.tex
frontmatter/
    cover.tex
    titlepage.tex
sections/
    introduction.tex
    background.tex
    methods.tex
    experiments.tex
    results.tex
    discussion.tex
    conclusions.tex
generated/
    results_macros.tex
    main_results_table.tex
    llm_results_table.tex
figures/
```

Import the generated numerical macros once:

```latex
\\input{generated/results_macros}
```

Then use commands such as:

```latex
\\PilotRenalDice
\\AutomaticCystDice
\\LLMVOneAcceptance
\\LLMVTwoAcceptance
\\LLMTokenReduction
```

This avoids manually duplicating experimental values across the dissertation.

The final cover should be created as a separate, visually designed LaTeX
front-matter component rather than being entangled with the Springer-like body
layout. The body can remain restrained and publication-like while the cover
uses a stronger graphic hierarchy appropriate for a final-degree project.
"""
