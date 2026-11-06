from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _results_section(payload: dict[str, Any]) -> str:
    baseline = payload["automatic_baseline"]
    tail = payload["hd95_tail"]
    undercoverage = payload["tumor_undercoverage"]
    interaction = payload["tumor_interaction"]
    manual = payload["manual_ct_only_localization"]
    cyst = payload["cyst_branch"]

    return rf'''\section{{Results}}
\label{{sec:results}}

\subsection{{Automatic renal segmentation on the frozen expanded100 cohort}}
\label{{sec:results-automatic}}

The final automatic evaluation comprised \TFGExpandedN{{}} frozen KiTS23
cases. The maintained NV-Segment-CT renal output achieved a mean
kidney-and-mass Dice score of \TFGKidneyMassDice{{}} $\pm$
\TFGKidneyMassDiceSD{{}}, with a mean Surface Dice of
\TFGSurfaceDice{{}}. Boundary performance was less uniform: the median HD95
was \TFGHDMedian{{}}~mm, whereas the mean increased to \TFGHDMean{{}}~mm.
This divergence motivated a dedicated analysis of the upper HD95 tail rather
than interpreting the mean boundary distance as a homogeneous cohort-wide
failure.

\input{{generated/table_final_results.tex}}

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{{figures/fig01_automatic_baseline.pdf}}
    \caption{{Automatic segmentation performance on the frozen expanded100
    cohort. Kidney-and-mass Dice and Surface Dice summarize the full 100-case
    cohort, whereas cyst Dice is reported on the
    {baseline['cyst_positive_count']} cyst-positive cases.}}
    \label{{fig:automatic-baseline}}
\end{{figure}}

\subsection{{Concentrated HD95 failure tail}}
\label{{sec:results-hd95-tail}}

A total of \TFGHDOutliers{{}}/100 cases exceeded the frozen HD95 outlier
threshold of {tail['outlier_threshold_mm']:.2f}~mm. The distribution was
highly concentrated: the five largest outliers accounted for
{100.0 * tail['top5_hd95_mean_share']:.1f}\% of the mean-HD95 burden and the
ten largest accounted for {100.0 * tail['top10_hd95_mean_share']:.1f}\%.
Residual-direction analysis separated the 22 outliers into
{tail['prediction_to_gt_dominant_count']} prediction-to-reference dominant,
{tail['gt_to_prediction_dominant_count']} reference-to-prediction dominant,
and {tail['balanced_count']} balanced case. Thus, the tail contained both
distant predicted geometry and missing reference-side coverage rather than a
single failure mechanism.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{{figures/fig02_hd95_failure_mechanisms.pdf}}
    \caption{{Dominant residual-geometry mechanism among HD95 outliers.
    Prediction-to-GT dominance indicates that distant predicted geometry was
    the larger directional residual, whereas GT-to-prediction dominance
    indicates that missing target coverage was the larger residual.}}
    \label{{fig:hd95-mechanisms}}
\end{{figure}}

A prediction-only trigger designed to identify actionable tail cases did not
generalize under leave-one-out evaluation. Its balanced accuracy was
{tail['prediction_only_trigger_balanced_accuracy']:.3f}, and it resolved
{tail['prediction_only_trigger_target_resolved_count']}/12 target cases.
Global component filtering and renal-mask expansion were likewise rejected
because they did not improve the failure tail without introducing additional
harm.

\subsection{{Tumor undercoverage and interactive capability}}
\label{{sec:results-tumor}}

The undercoverage audit identified
{undercoverage['undercoverage_case_count']} cases in which the automatic
kidney-and-mass prediction missed a substantial portion of the reference.
Tumor tissue was the dominant missing class in
{undercoverage['tumor_dominant_missing_count']}/
{undercoverage['undercoverage_case_count']} cases and represented
{100.0 * undercoverage['weighted_tumor_miss_fraction']:.1f}\% of the weighted
missing volume.

When a ground-truth-derived, well-interior positive tumor point was supplied,
VISTA3D exhibited substantial latent tumor-segmentation capability. Across the
9-case oracle cohort, mean tumor Dice reached \TFGOracleTumorDice{{}}, and the
mean reduction in kidney-and-mass HD95 was
\TFGOracleHDReduction{{}}~mm. However, subsequent oracle prompt engineering
did not yield a robust size-generalizable rule: the relative-depth strategy
produced a paired tumor-Dice change of
{interaction['relative_depth_paired_dice_delta']:+.4f}, while the dual-point
strategy produced a change of only
{interaction['dual_point_paired_dice_delta']:+.4f}. Oracle prompt engineering
was therefore closed.

\subsection{{Manual CT-only localization}}
\label{{sec:results-manual}}

To decouple lesion localization from segmentation, a separate 12-case
size-stratified experiment used manual points selected from CT images alone,
with ground truth hidden during point placement. Post-hoc geometry quality
control confirmed exact image-to-segmentation alignment and a valid coordinate
contract. Nevertheless, none of the 12 frozen manual points intersected tumor
tissue. Mean tumor Dice consequently fell to \TFGManualTumorDice{{}},
compared with \TFGDeepTumorDice{{}} for the reused deep-oracle reference on
the same 12-case cohort.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{{figures/fig03_tumor_interaction_comparison.pdf}}
    \caption{{Interactive tumor segmentation under different localization
    settings. The 9-case oracle cohort demonstrates latent interactive
    capability under GT-derived well-interior localization. The 12-case
    deep-oracle reference and manual CT-only experiment use the frozen
    size-stratified cohort. Values across the 9-case and 12-case cohorts are
    descriptive and must not be interpreted as a direct paired comparison.}}
    \label{{fig:tumor-interaction}}
\end{{figure}}

The manual-versus-deep paired tumor-Dice difference was
{manual['paired_manual_minus_deep_dice_mean']:+.4f}. Because the points were
selected by the project author rather than by a clinician, this experiment is
interpreted strictly as a simulated-user localization study and not as
clinician or radiologist validation.

\subsection{{Cyst segmentation and size-dependent detectability}}
\label{{sec:results-cyst}}

Automatic cyst segmentation constituted a distinct failure mode. Among
\TFGCystN{{}} cyst-positive cases, mean cyst Dice was \TFGCystDice{{}} and
the median was {cyst['dice_median']:.4f}. A total of
\TFGCystZeroDice{{}}/\TFGCystN{{}} cases had zero overlap, and
\TFGCystFNDominant{{}}/\TFGCystN{{}} were dominated by false-negative
undersegmentation. Reference cyst volume correlated with both Dice
($r={cyst['volume_vs_dice_pearson']:+.3f}$) and recall
($r={cyst['volume_vs_recall_pearson']:+.3f}$).

Reference volume strongly ranked cyst detectability: the AUROC for predicting
any overlap was \TFGCystOverlapAUROC{{}}, while the AUROC for predicting
Dice $\geq 0.50$ was \TFGCystUsefulAUROC{{}}. The size-response analysis
nevertheless showed overlapping success and failure volumes, so no hard size
boundary was supported.

\begin{{figure}}[t]
    \centering
    \includegraphics[width=0.92\linewidth]{{figures/fig04_cyst_volume_response.pdf}}
    \caption{{Size-dependent cyst detectability across eight bins of
    reference cyst volume. The smallest bins showed complete detection
    failure, whereas larger cysts were progressively more likely to achieve
    any overlap and Dice $\geq 0.50$. The non-monotonic high-volume endpoint
    further indicates that volume is a strong predictor rather than a
    deterministic cutoff.}}
    \label{{fig:cyst-volume-response}}
\end{{figure}}

Taken together, the final experiments identify three complementary properties
of the maintained pipeline: strong global renal overlap, a concentrated and
mechanistically heterogeneous boundary-error tail, and semantic limitations
that differ between tumor and cyst tissue.
'''


def _discussion_section(payload: dict[str, Any]) -> str:
    tail = payload["hd95_tail"]
    manual = payload["manual_ct_only_localization"]
    cyst = payload["cyst_branch"]

    return rf'''\section{{Discussion}}
\label{{sec:discussion}}

\subsection{{Strong aggregate overlap does not imply uniform geometric reliability}}

The principal automatic result is the combination of a high global
kidney-and-mass Dice score (\TFGKidneyMassDice{{}}) with a substantially
larger mean than median HD95 (\TFGHDMean{{}} versus \TFGHDMedian{{}}~mm).
This combination distinguishes average overlap quality from tail-sensitive
geometric reliability. A relatively small subset of {tail['outlier_count']}
cases was responsible for a disproportionate share of the HD95 burden, so a
single cohort-level mean would obscure distinct failure mechanisms.

The directional residual split provides a second layer of interpretation.
Prediction-to-reference dominant cases are consistent with distant or
extraneous predicted geometry, whereas reference-to-prediction dominant cases
reflect missing target coverage. The near-even split between these mechanisms
explains why a single global post-processing rule performed poorly. The
rejected post-processing experiments therefore support a conservative
conclusion: the observed HD95 tail is heterogeneous and should not be reduced
to a single morphological defect.

\subsection{{Tumor segmentation is limited by localization, not only by model capacity}}

The interactive tumor experiments reveal an important distinction between
\emph{{segmentation capability}} and \emph{{lesion localization}}.
Under GT-derived well-interior localization, mean tumor Dice reached
\TFGOracleTumorDice{{}}, and kidney-and-mass HD95 improved by
\TFGOracleHDReduction{{}}~mm on average. This demonstrates that the
underlying interactive model can delineate tumor tissue substantially better
than the maintained automatic renal label vocabulary suggests.

However, this capability cannot be interpreted as an autonomous tumor
segmentation result. The positive points used in the oracle branch were derived
from the tumor ground truth and therefore supplied information unavailable in
a deployment setting. Attempts to refine the oracle interaction through fixed
depth, relative depth, or a second interior point did not provide a consistent
size-generalizable gain. The preferred interpretation is therefore a
capability upper bound rather than a finalized prompting strategy.

The manual CT-only experiment sharpened this distinction. The coordinate audit
excluded a geometric implementation error, yet the project-author clicks
intersected tumor in {manual['any_tumor_hit_count']}/{manual['case_count']}
cases. Consequently, tumor Dice collapsed from \TFGDeepTumorDice{{}} under
deep-oracle localization to \TFGManualTumorDice{{}} under manual CT-only
localization. This negative result does not imply that expert radiologists
would fail to localize these lesions. Instead, it shows that unguided
non-expert interaction cannot be treated as an adequate proxy for expert
localization.

\subsection{{Cyst segmentation exhibits a size-dependent detectability limit}}

Cyst performance differs from the tumor interaction problem. The automatic
model already exposes cyst labels, but the failure pattern is dominated by
missed cyst tissue: {cyst['fn_dominant_count']}/{cyst['case_count']}
cyst-positive cases were false-negative dominant and
{cyst['zero_dice_count']}/{cyst['case_count']} had no overlap at all. The
strong associations between reference volume and both Dice and recall,
together with AUROCs of \TFGCystOverlapAUROC{{}} and
\TFGCystUsefulAUROC{{}}, show that lesion size is a major determinant of
recoverability.

Importantly, the observed transition is probabilistic rather than absolute.
Small cysts were frequently missed completely, while larger cysts were much
more recoverable, but the success and failure volume ranges overlapped. The
descriptive thresholds identified in the retrospective analysis are therefore
useful for characterizing the failure regime but not for defining a clinical
cutoff.

\subsection{{Overall interpretation}}

The final experimental picture is internally consistent. The maintained
NV-Segment-CT pipeline provides strong automatic renal overlap on the frozen
expanded100 cohort, but its limitations emerge when the task is examined
beyond a single global overlap metric. Boundary errors concentrate in a
heterogeneous tail; tumor tissue exposes a gap between latent interactive
capacity and practical lesion localization; and cysts expose a size-dependent
automatic detectability problem.

These findings favour a modular interpretation of future improvements.
Boundary-tail mitigation should target failure-specific geometry rather than a
single global transform. Tumor improvement requires an independent
localization mechanism or expert interaction. Cyst improvement requires better
small-lesion sensitivity rather than only post-processing.
'''


def _limitations_section(payload: dict[str, Any]) -> str:
    baseline = payload["automatic_baseline"]
    cyst = payload["cyst_branch"]

    return rf'''\section{{Limitations}}
\label{{sec:limitations}}

Several limitations constrain the scope of the conclusions.

First, the principal automatic evaluation used a frozen
{baseline['case_count']}-case KiTS23 cohort assembled for this project. The
analyses are internally reproducible but do not constitute an official KiTS23
challenge submission. In particular, an official challenge score is
unavailable under the project protocol and is not reported.

Second, the HD95 tail, tumor-interaction, and cyst-detectability analyses are
retrospective analyses of the same frozen experimental material. Subgroup
statistics, descriptive thresholds, and exploratory associations should
therefore be interpreted as explanatory evidence rather than independently
validated operating points.

Third, all positive tumor-localization experiments before the manual branch
were oracle experiments. Their points were derived from ground-truth tumor
masks and therefore cannot represent deployment-valid autonomous localization.
The strong oracle tumor Dice quantifies latent segmentation capability under
correct localization, not end-to-end automatic tumor performance.

Fourth, the CT-only localization experiment comprised 12 frozen cases and one
non-expert project author. Although ground truth was hidden during point
selection and the coordinate contract was subsequently verified, the
experiment is neither a clinician study nor a human-factors evaluation. Its
negative result must not be extrapolated to radiologists or other trained
readers.

Fifth, cyst-volume transition points were selected descriptively on the same
{cyst['case_count']} cyst-positive cases used to characterize performance.
The strong AUROC values support a size-dependent detectability relationship,
but no threshold should be interpreted as a prospective or clinically
validated cyst-size cutoff.

Finally, the maintained automatic output vocabulary constrains what can be
concluded about semantic performance. The automatic renal output provides
validated kidney and cyst classes but no validated autonomous tumor output
class. The interactive tumor branch therefore evaluates an additional
capability of VISTA3D rather than a directly comparable automatic class from
the same output vocabulary.
'''


def _conclusions_section(payload: dict[str, Any]) -> str:
    return r'''\section{Conclusions}
\label{sec:conclusions}

This work established a reproducible evaluation framework for an
NV-Segment-CT/VISTA3D renal-segmentation pipeline on KiTS23 and used it to move
from aggregate performance reporting to failure-oriented analysis.

The frozen expanded100 evaluation demonstrated strong automatic
kidney-and-mass overlap, while a concentrated HD95 tail revealed that
geometric failures were heterogeneous and could not be corrected reliably with
simple prediction-only post-processing. Tumor undercoverage then exposed
substantial latent interactive segmentation capability when a correct lesion
location was supplied. However, oracle prompt refinements did not yield a
robust generalizable rule, and unguided non-expert CT-only localization failed
to provide a viable replacement for correct localization. Cyst segmentation
constituted a separate limitation, dominated by false-negative
undersegmentation and a strong but non-deterministic dependence on lesion size.

The resulting system should therefore be viewed as a strong automatic renal
segmentation baseline with clearly characterized semantic and tail-risk
limitations, rather than as a complete autonomous kidney-tumor-cyst solution.
The most defensible future directions are failure-specific boundary modelling,
validated lesion localization for interactive tumor segmentation, and improved
small-cyst sensitivity evaluated on an independent cohort.
'''


def _preview_document() -> str:
    return r'''\documentclass[11pt,a4paper]{article}

\usepackage[margin=2.4cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{float}
\usepackage[hidelinks]{hyperref}

\input{generated/results_macros.tex}

\title{Final Experimental Results and Discussion\\
NV-Segment-CT / VISTA3D on KiTS23}
\author{}
\date{}

\begin{document}
\maketitle

\input{sections/results.tex}
\input{sections/discussion.tex}
\input{sections/limitations.tex}
\input{sections/conclusions.tex}

\end{document}
'''


def _integration_snippet() -> str:
    return r'''% Block 55 integration snippet.
% Required packages:
% \usepackage{graphicx}
% \usepackage{booktabs}
% \usepackage{amsmath}

\input{generated/results_macros.tex}
\input{sections/results.tex}
\input{sections/discussion.tex}
\input{sections/limitations.tex}
\input{sections/conclusions.tex}
'''


def _readme() -> str:
    return '''# Block 55 final manuscript package

This directory is generated from the frozen Block 53 synthesis and the
publication-ready Block 54 assets.

Compile `preview_results_discussion.tex` for a standalone Overleaf preview.

The generated prose deliberately contains no invented bibliography keys or
unsupported literature comparisons. Literature comparison can be added once
the final bibliography is available.

Scientific boundaries are preserved: no official KiTS23 challenge score,
oracle tumor points are not deployment-valid localization, manual CT-only
clicks are simulated-user rather than clinician validation, cyst thresholds
are descriptive rather than clinical cutoffs, and tested prediction-only
post-processing did not solve the HD95 tail.
'''


def build_final_manuscript_package(
    *,
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    summary_path = (
        project_root
        / "outputs"
        / "analyses"
        / "nv_segment_ct_final_experimental_synthesis"
        / "summary.json"
    )
    block54_root = (
        project_root
        / "outputs"
        / "reporting"
        / "final_results"
    )

    payload = _read_json(summary_path)
    if payload.get("analysis") != "final_experimental_synthesis":
        raise ValueError("Block 53 final synthesis is required.")

    required_assets = [
        block54_root / "latex" / "results_macros.tex",
        block54_root / "latex" / "table_final_results.tex",
        block54_root / "latex" / "table_claim_matrix.tex",
    ]

    figure_names = (
        "fig01_automatic_baseline.pdf",
        "fig02_hd95_failure_mechanisms.pdf",
        "fig03_tumor_interaction_comparison.pdf",
        "fig04_cyst_volume_response.pdf",
    )
    required_assets.extend(
        block54_root / "figures" / name
        for name in figure_names
    )

    missing = [path for path in required_assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Block 54 assets missing: "
            + ", ".join(str(path) for path in missing)
        )

    sections_root = output_root / "sections"
    generated_root = output_root / "generated"
    figures_root = output_root / "figures"

    sections_root.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    section_text = {
        "results.tex": _results_section(payload),
        "discussion.tex": _discussion_section(payload),
        "limitations.tex": _limitations_section(payload),
        "conclusions.tex": _conclusions_section(payload),
    }
    for name, text in section_text.items():
        (sections_root / name).write_text(text, encoding="utf-8")

    for name in (
        "results_macros.tex",
        "table_final_results.tex",
        "table_claim_matrix.tex",
    ):
        shutil.copy2(
            block54_root / "latex" / name,
            generated_root / name,
        )

    for name in figure_names:
        shutil.copy2(
            block54_root / "figures" / name,
            figures_root / name,
        )

    (output_root / "preview_results_discussion.tex").write_text(
        _preview_document(),
        encoding="utf-8",
    )
    (output_root / "integration_snippet.tex").write_text(
        _integration_snippet(),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        _readme(),
        encoding="utf-8",
    )

    inputs = {
        "block53_summary": {
            "path": str(summary_path.relative_to(project_root)),
            "sha256": _sha256(summary_path),
        },
        "block54_assets": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": _sha256(path),
            }
            for path in required_assets
        ],
    }

    outputs = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "sha256": _sha256(path),
                }
            )

    manifest = {
        "schema_version": "1.0",
        "package": "block55_final_results_discussion_manuscript",
        "source_of_truth": "Block 53 final experimental synthesis",
        "inputs": inputs,
        "outputs": outputs,
        "reporting_constraints": payload["reporting_constraints"],
    }

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
