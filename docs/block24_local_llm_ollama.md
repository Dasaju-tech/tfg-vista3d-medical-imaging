# Block 24 — First Local LLM Verbalization with Ollama

## Objective

Block 23 established a deterministic acceptance gate for language-model output.

Block 24 integrates the first real language model while leaving the
authoritative measurement and audit layers unchanged.

The baseline provider is local Ollama and the frozen first model is:

```text
qwen3:4b-instruct
```

The model is used only as a text verbalizer.

## Why a local baseline?

The case data and structured findings can remain on the workstation.

The provider boundary is isolated in:

```text
src/vista3d_tfg/reports/ollama_local.py
```

Future providers can be compared without changing:

- `findings.json`,
- the LLM packet,
- the candidate schema,
- the deterministic hallucination auditor.

## Structured output

The native Ollama `/api/chat` endpoint supports a JSON Schema in the `format`
field.

Block 24 supplies the candidate-report schema both:

1. as the native structured-output `format`,
2. as text appended to the model prompt.

The model must return only:

```json
{
  "case_id": "case_XXXXX",
  "narrative": "...",
  "limitations": ["..."]
}
```

## Frozen generation settings

```text
temperature = 0
seed        = 20260828
num_predict = 700
think       = false
stream      = false
keep_alive  = 5m
```

Temperature zero and a fixed seed reduce avoidable generation variability.

## Acceptance gate

A successful model call is not automatically a successful experiment.

The response must:

1. be valid JSON,
2. satisfy the Block 23 parser,
3. pass the Block 23 deterministic audit.

Any parse or audit failure returns a non-zero process exit code.

All rejected responses are retained for analysis.

## Runtime preparation

The preparation script probes:

```text
GET /api/version
GET /api/tags
```

and optionally pulls the frozen model through:

```text
POST /api/pull
```

Example:

```powershell
python .\scripts\prepare_local_llm_runtime.py --pull
```

## Generation

```powershell
python .\scripts\run_local_llm_case_report.py --case 2
```

The experiment writes:

```text
outputs/reports/local_llm/case_00002/qwen3_4b-instruct/
├── raw_response.json
├── raw_content.txt
├── candidate.json       # when parse succeeds
├── audit.json           # when audit runs
└── run.json
```

## Provenance

`run.json` records:

- provider,
- endpoint,
- Ollama version,
- returned model identifier,
- frozen generation configuration,
- prompt SHA-256,
- response SHA-256,
- wall time,
- parsed candidate,
- deterministic audit,
- final accepted/rejected state.

Ollama response metadata also preserves prompt/output token counts and timing
information in `raw_response.json`.

## Interpretation

Block 24 is a systems experiment, not an evaluation of medical knowledge.

The relevant first metrics are:

```text
JSON parse success
deterministic audit success
numeric consistency
laterality consistency
forbidden-claim rate
runtime
```

The model should not be rewarded for producing additional clinical content.
Additional unsupported interpretation is a failure.

## Next step

After a successful single-case run, the same frozen protocol should be applied
repeatedly and/or to multiple structured cases before changing the model.

That will allow reproducibility and audit-pass rate to be measured instead of
judging one aesthetically pleasing response.
