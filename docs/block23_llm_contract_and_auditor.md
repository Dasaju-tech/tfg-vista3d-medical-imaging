# Block 23 — Constrained LLM Contract and Hallucination Auditor

## Objective

Block 22 created a deterministic, GT-free structured findings object.

Block 23 defines the contract that any future language model must obey before
the project allows generated prose to be accepted.

No LLM is called in this block.

## Principle

The reporting architecture is:

```text
segmentation outputs
        ↓
deterministic measurements
        ↓
findings.json
        ↓
LLM input packet
        ↓
future language model
        ↓
candidate JSON
        ↓
deterministic hallucination audit
        ↓
accepted / rejected report
```

The model is a verbalizer, not a source of case facts.

## Authoritative facts

The input packet extracts the numeric facts that the model is allowed to use:

- left kidney-core volume,
- right kidney-core volume,
- automatic left cyst-class volume,
- automatic right cyst-class volume,
- interactive cyst predicted volume,
- selected-kidney distance,
- laterality margin,
- cyst class,
- user click coordinates.

The packet also contains the complete `findings.json`.

## Output contract

The model must return JSON only:

```json
{
  "case_id": "case_XXXXX",
  "narrative": "...",
  "limitations": [
    "..."
  ]
}
```

No additional keys are accepted.

The limitations must reproduce the authoritative limitations exactly and in
the same order.

## Deterministic audit

The candidate report is rejected if it contains:

- a different case ID,
- missing or modified limitations,
- unsupported numeric values,
- wrong cyst laterality,
- wrong cyst semantic class,
- diagnostic claims,
- malignancy or benignity claims,
- cancer or carcinoma claims,
- staging claims,
- metastatic claims,
- treatment or therapy claims,
- recommendations,
- biopsy claims,
- unsupported suspicious/pathology language.

The audit therefore catches both factual drift and prohibited interpretation.

## Numeric policy

All numbers in the narrative must correspond to facts present in
`findings.json`.

The auditor accepts deterministic formatting variants of authoritative values,
for example:

```text
31.92
31.9
32
```

when derived from the same stored value.

This permits normal prose rounding while rejecting invented quantities such as
`45.00 mL`.

## Why this precedes model integration

A language model should not be integrated until the project has a deterministic
way to reject unsafe or inconsistent output.

Block 23 makes future model comparison measurable:

```text
valid JSON rate
audit pass rate
numeric consistency rate
laterality consistency rate
forbidden-claim rate
```

Different local or remote models can therefore be compared against the same
contract.

## Outputs

For each case:

```text
outputs/reports/llm_contract/case_XXXXX/
├── llm_input_packet.json
├── llm_prompt.txt
└── candidate_audit.json   # after candidate auditing
```

Formal candidate schema:

```text
schemas/llm_candidate_report_v1.schema.json
```

## Next step

Block 24 can integrate the first actual language model provider while keeping
all provider-specific code outside the authoritative measurement layer.

The first integration should use deterministic generation settings whenever
the provider permits them and must store:

- provider/model identifier,
- generation parameters,
- raw response,
- parsed candidate,
- audit result,
- wall time.
