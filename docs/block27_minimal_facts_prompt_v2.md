# Block 27 — Prospective Minimal-Facts Prompt v2

## Motivation

The final calibrated Block 26 cohort produced:

```text
valid structured JSON       17/17
accepted by deterministic gate 16/17
persistent semantic failure     1/17
```

The remaining rejected case was `case_00237`.

Its candidate stated:

```text
The selected kidney distance is 0.00 mm, indicating the interactive cyst is
spatially aligned with the selected kidney.
```

The authoritative quantity is the distance from the user-provided click to the
selected automatic kidney. It does not establish a cyst-to-kidney spatial
relationship.

Block 27 does not retrospectively edit that candidate and does not relax the
auditor.

Instead it prospectively generates a new 17-case cohort with a revised input
contract.

## Experimental isolation

Frozen from Block 26:

```text
structured findings
17-case cohort
qwen3:4b-instruct
temperature = 0
seed = 20260828
think = false
JSON output schema
Block 26C deterministic auditor
```

Changed:

```text
LLM prompt only
```

Not executed:

```text
VISTA
segmentation
interactive workflow
measurement extraction
```

## Prompt v1 problem

The original prompt supplied both:

1. a preferred list of narrative facts,
2. the complete `findings.json`.

That exposed information the verbalizer did not need:

```text
file paths
SHA-256 hashes
raw-precision floats
internal segmentation labels
255 unprocessed label
other implementation metadata
```

The full JSON also made the semantic role of
`selected_kidney_distance_mm` easier to reinterpret.

## Prompt v2

The model receives only:

```text
case_id

left kidney-core volume
right kidney-core volume

automatic left cyst volume
automatic right cyst volume

interactive cyst predicted volume
interactive cyst laterality
interactive cyst class label

user-click -> selected-kidney distance
laterality routing margin
user click xyz

required limitations
```

All continuous reporting values are rendered deterministically at two decimal
places before entering the prompt.

The raw `findings.json` is not included.

## Explicit semantic guard

Prompt v2 defines:

```text
selected_kidney_distance_mm =
distance(USER-PROVIDED CLICK, selected automatic kidney)
```

and explicitly states that it does not describe:

```text
cyst-to-kidney distance
cyst-to-kidney alignment
cyst-to-kidney overlap
cyst attachment
cyst location
any other cyst-to-kidney spatial relationship
```

No new auditor exception is introduced.

## Prospective evaluation

The same 17 cases are regenerated through the LLM.

The main comparison is:

```text
Block 26 final v1 acceptance: 16/17 = 94.1%
Block 27 v2 acceptance:       ?/17
```

The analysis records:

```text
improved cases
regressed cases
unchanged passes
unchanged failures
issue codes
JSON parse success
latency
```

A `17/17` result would demonstrate that the sole observed semantic overreach
was removed by a prospective input-contract refinement rather than by
retrospectively weakening the deterministic gate.

A result below `17/17` remains valid experimental evidence and must be
inspected without changing the frozen auditor.

## Smoke run

Target the former failure plus a known pass:

```powershell
python .\scripts\run_local_llm_reporting_cohort_v2.py --cases 237 2
```

## Full run

```powershell
python .\scripts\run_local_llm_reporting_cohort_v2.py
```

## Outputs

Per-case v2 generation:

```text
outputs/reports/local_llm_cohort_v2/
└── case_XXXXX/
    └── qwen3_4b-instruct/
        ├── raw_response.json
        ├── raw_content.txt
        ├── candidate.json
        ├── audit.json
        └── run.json
```

Cohort-level results:

```text
outputs/experiments/local_llm_reporting_cohort_v2/
└── qwen3_4b-instruct/
    ├── summary.json
    └── cases.csv
```

Smoke outputs are placed in a separate `smoke_*` subdirectory and therefore do
not overwrite the full-cohort summary.
