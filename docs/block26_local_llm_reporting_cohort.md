# Block 26 — Cyst-Positive End-to-End Local LLM Reporting Cohort

## Objective

Block 25 demonstrated exact and factual reproducibility for five repeated local
LLM generations of `case_00002`:

```text
JSON parse success        5/5
acceptance gate           5/5
unique raw responses      1
unique candidates         1
exact raw match           5/5
exact candidate match     5/5
warm median latency       1.73 s
```

Block 26 moves from repeated generation of one case to variation across the
17 cyst-positive pilot cases.

No LLM hyperparameter, prompt, schema, or auditor is changed.

## Important provenance boundary

The 17 click coordinates were originally generated in Block 16 from the
largest GT cyst component.

Block 26 replays those frozen coordinates as experimental user-click
surrogates.

Therefore:

```text
deployment_valid = false
```

for the overall cohort.

However, after the coordinate is supplied, the runtime path itself is GT-free:

```text
CT
    ->
automatic renal prediction
    ->
frozen surrogate user click
    ->
automatic side/class routing
    ->
interactive VISTA prediction
    ->
structured findings
    ->
local LLM
    ->
deterministic audit
```

Neither the user-guided VISTA runtime, structured-reporting layer, nor LLM
generation reads the reference segmentation.

This distinction must be preserved in the dissertation.

## Frozen LLM protocol

```text
provider     = Ollama local
model        = qwen3:4b-instruct
temperature  = 0
seed         = 20260828
think        = false
stream       = false
num_predict  = 700
JSON Schema  = unchanged
prompt       = unchanged
auditor      = unchanged
```

## Why reuse the oracle clicks?

The goal of Block 26 is not to re-estimate segmentation performance.

That question was already studied in Blocks 16–20.

The goal here is to expose the reporting system to a heterogeneous set of
structured numerical findings while keeping the interaction target controlled.

Changing both interaction design and LLM reporting at the same time would make
failures harder to interpret.

## Per-case pipeline

For each of the 17 cyst-positive cases:

1. read the frozen Block 16 click coordinate,
2. run/reuse the Block 21 user-guided workflow,
3. automatically assign cyst laterality/class from renal masks,
4. build the Block 22 structured findings/report,
5. build the unchanged Block 23 LLM packet,
6. call the frozen Block 24 local model,
7. run the deterministic auditor,
8. retain every raw response and audit.

## Smoke execution

Before the full cohort:

```powershell
python .\scripts\run_local_llm_reporting_cohort.py --cases 2 480
```

This includes two already well-understood cases.

## Full execution

```powershell
python .\scripts\run_local_llm_reporting_cohort.py
```

Existing user-guided predictions are reused when the saved click is identical.

Use:

```text
--force-workflow
```

only when deliberately recomputing segmentation outputs.

## Outputs

Cohort summary:

```text
outputs/experiments/local_llm_reporting_cohort/
└── qwen3_4b-instruct/
    ├── summary.json
    └── cases.csv
```

Per-case language-model provenance:

```text
outputs/reports/local_llm_cohort/
└── case_XXXXX/
    └── qwen3_4b-instruct/
        ├── raw_response.json
        ├── raw_content.txt
        ├── candidate.json
        ├── audit.json
        └── run.json
```

Per-case structured findings remain in:

```text
outputs/reports/structured/case_XXXXX/
```

## Primary Block 26 metrics

```text
LLM JSON parse success rate
LLM deterministic acceptance rate
total audit issues
rejected case IDs
LLM latency distribution
```

The cohort also records the automatic routing context:

```text
left/right case count
workflow cyst volume distribution
selected-kidney distance
laterality margin
```

## Decision rule

### 17/17 accepted

The frozen local verbalization protocol generalizes across the full
cyst-positive pilot cohort without detected factual drift.

The next stage can focus on presenting and evaluating the resulting reports,
rather than further prompt engineering.

### Any rejected case

Do not change the model or prompt immediately.

Inspect the saved candidate/audit first and classify the failure:

```text
JSON formatting
numeric drift
laterality/class drift
semantic relation drift
forbidden clinical inference
limitations mismatch
```

Only then decide whether the issue belongs to the prompt, auditor, or model.
