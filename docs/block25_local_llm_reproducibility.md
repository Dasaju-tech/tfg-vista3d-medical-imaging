# Block 25 — Local LLM Reproducibility Study

## Objective

Block 24 established a successful end-to-end local language-model reporting
path:

```text
findings.json
    -> frozen prompt
    -> qwen3:4b-instruct
    -> structured JSON
    -> deterministic audit
    -> accepted/rejected report
```

Block 25 asks whether that behavior is reproducible under repeated generation
with the configuration held fixed.

## Frozen protocol

Every repetition uses exactly the same:

```text
provider     = Ollama local
model        = qwen3:4b-instruct
temperature  = 0
seed         = 20260828
think        = false
stream       = false
num_predict  = 700
JSON Schema  = Block 24 candidate schema
prompt       = Block 23/24 constrained prompt
findings     = the same authoritative findings.json
auditor      = the same deterministic acceptance gate
```

The repetition index is the only experimental bookkeeping variable.

## Two definitions of reproducibility

Exact text identity is useful but is not the primary safety criterion.

### Exact reproducibility

Measured using SHA-256 fingerprints of:

1. raw generated content,
2. canonicalized parsed candidate JSON.

This detects even stylistic variation.

### Factual reproducibility

Measured using the deterministic acceptance gate.

A generation is factually reproducible when it:

- parses as the required JSON object,
- preserves numerical facts,
- preserves laterality and class identity,
- preserves semantic relations,
- includes all required limitations,
- contains no prohibited clinical claims.

Therefore two differently worded candidates may be factually reproducible even
if their hashes differ.

## Default run count

The default experiment uses five consecutive generations:

```text
N = 5
```

This is intentionally small and bounded. The aim is a reproducibility smoke
cohort, not a statistical claim about all possible sampling behavior.

The CLI permits 2 to 20 runs.

## Cold versus warm latency

The first request may include model loading or cache effects.

The summary therefore records both:

- all-run latency,
- warm latency for runs 2..N.

This distinction is important because Block 24 already showed a large
difference between the first model invocation and a subsequent warm request.

## Per-run provenance

Every repetition keeps the complete Block 24 provenance bundle:

```text
runs/run_001/
├── raw_response.json
├── raw_content.txt
├── candidate.json
├── audit.json
└── run.json
```

Rejected generations are retained rather than overwritten.

## Cohort outputs

```text
outputs/experiments/local_llm_reproducibility/
└── case_00002/
    └── qwen3_4b-instruct/
        ├── summary.json
        ├── runs.csv
        └── runs/
            ├── run_001/
            ├── run_002/
            ├── ...
            └── run_005/
```

## Reported metrics

The summary includes:

```text
JSON parse success rate
acceptance-gate pass rate

number of unique raw responses
number of unique canonical candidates

exact raw match to run 1
exact candidate match to run 1
all-responses-identical flag
all-candidates-identical flag

wall-time mean / median / min / max
warm-run wall-time mean / median

prompt-token range
output-token range
```

## Interpretation

Several outcomes are possible.

### Exact and factual reproducibility

All hashes and audits match.

This is the strongest result for the frozen local setup.

### Textual variation but factual reproducibility

Hashes differ while every generation passes the audit.

This is acceptable for the research prototype and demonstrates why factual
auditing is more important than byte-level identity.

### Audit instability

One or more generations fail the deterministic gate.

The failed outputs become direct evidence of residual generation risk. The
protocol should not be widened to more cases until the failure mode is
understood.

## Next step

If the five-run study shows stable acceptance, the next experiment should move
from repeated generation of one case to a small multi-case LLM reporting
cohort.

That cohort should use already validated structured findings rather than
changing the segmentation or prompting protocol at the same time.
