# Block 28 — Final Reporting and Experiment Synthesis

## Objective

Block 28 does not execute VISTA, Ollama, segmentation, prompting, or auditing.

It consolidates the frozen outputs of:

```text
Block 25 — repeated local-LLM reproducibility
Block 26 — 17-case prompt-v1 reporting cohort
Block 27 — prospective 17-case minimal-facts prompt-v2 cohort
```

into directly reusable dissertation artifacts.

## Inputs

The script reads:

```text
outputs/experiments/local_llm_reproducibility/
    case_00002/qwen3_4b-instruct/summary.json

outputs/experiments/local_llm_reporting_cohort/
    qwen3_4b-instruct/summary.json

outputs/experiments/local_llm_reporting_cohort/
    qwen3_4b-instruct/reaudit_summary.json

outputs/experiments/local_llm_reporting_cohort_v2/
    qwen3_4b-instruct/summary.json
```

No experimental result is recomputed.

## Outputs

```text
outputs/final_reporting_synthesis/
├── final_reporting_synthesis.json
├── final_reporting_stages.csv
├── final_reporting_synthesis.md
├── acceptance_v1_vs_v2.png
├── llm_latency.png
└── prompt_token_reduction.png
```

## Main quantities

The synthesis reports:

```text
Block 25 exact response reproducibility
Block 25 exact candidate reproducibility

Prompt-v1 final audited acceptance
Prompt-v2 prospective acceptance
acceptance delta

prompt-token reduction
generation latency

persistent prompt-v1 failure cases
prompt-v2 improved cases
prompt-v2 regressions
```

## Expected frozen findings

Based on the completed experiments:

```text
Block 25:
5/5 accepted
1 unique raw response
1 unique canonical candidate

Block 26 final:
16/17 accepted
persistent failure = case_00237

Block 27:
17/17 accepted
improved = case_00237
regressions = none
```

The generated outputs remain the authoritative numerical source for the
dissertation.

## Interpretation boundary

The 17-case cohort is not deployment-valid because the click coordinates were
originally generated from GT-derived oracle targets and replayed as
experimental user-click surrogates.

The reporting runtime itself does not read ground truth after the click
coordinate is supplied.

No result should be presented as clinical validation.

## Execution

```powershell
python .\scripts\build_final_reporting_synthesis.py
```

The resulting Markdown table and PNG figures can be inserted directly into the
Results/Discussion drafting workflow.
