# Block 29 — Dissertation Results Integration

## Objective

Block 29 freezes the completed experimental story into a dissertation-oriented
package.

It does not run:

```text
VISTA
Ollama
segmentation
prompting
auditing
training
```

The block is the interface between the research repository and the English
Overleaf dissertation.

## Design principle

The dissertation should never contain independently typed copies of important
numerical results.

Block 29 therefore generates:

```text
LaTeX macros
LaTeX tables
results outline
discussion outline
JSON provenance manifest
selected final figures
```

The later Overleaf project should import these artifacts.

## Frozen experimental narrative

The package covers the major findings from:

```text
30-case renal pilot
automatic cyst failure decomposition
single-click interactive upper bound
iterative-click stability
semantic/laterality experiments
GT-free structured reporting
local LLM reproducibility
17-case prompt-v1 cohort
prospective minimal-facts prompt-v2 cohort
```

## Output structure

```text
outputs/dissertation/
├── README.md
├── OVERLEAF_INTEGRATION.md
├── dissertation_results_manifest.json
├── results_outline.md
├── discussion_outline.md
├── latex/
│   ├── results_macros.tex
│   ├── main_results_table.tex
│   └── llm_results_table.tex
└── figures/
    ├── acceptance_v1_vs_v2.png
    ├── llm_latency.png
    └── prompt_token_reduction.png
```

## Overleaf strategy

The final dissertation should use a restrained publication-style body and a
separate graphic cover.

Recommended principles:

```text
English throughout
professional serif body typography
strong but sparse section hierarchy
booktabs tables
consistent figure captions
no decorative clutter inside technical sections
a separate visually designed cover/front matter
```

The cover should be implemented independently so that its visual language does
not compromise the readability of the technical body.

The final cover can follow the same workflow used for the previous GESST
report: establish the body first, then create a dedicated editable LaTeX cover
with university/project metadata and a stronger design treatment.

## Execution

```powershell
python .\scripts\build_dissertation_results_package.py
```

## Next phase

After Block 29 passes, the next task is not another experiment.

The next phase is to create the actual Overleaf dissertation skeleton and
write the English Results/Discussion sections from the generated package,
followed by the aesthetic cover/front matter.
