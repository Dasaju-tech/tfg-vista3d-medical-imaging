# Block 26A — Cohort Auditor Calibration

Block 26 produced 17/17 valid structured JSON responses but only 9/17 passed
the deterministic acceptance gate.

Inspection of a rejected example showed:

```text
The selected kidney distance is 0.00 mm.
```

The value and field identity are correct. The previous auditor nevertheless
required the words `click` and `kidney` in the same clause.

This is an auditor false positive rather than a language-model factual error.

The calibrated rule accepts either:

```text
the user click is X mm from the selected kidney
```

or the canonical field-preserving alias:

```text
the selected kidney distance is X mm
```

It still rejects changed endpoint semantics such as:

```text
the distance between the cyst and selected kidney is X mm
the cyst mask is X mm from the selected kidney
```

The repair does not regenerate any segmentation or LLM response.

Run:

```powershell
python .\scripts\reaudit_local_llm_reporting_cohort.py
```

Original `audit.json` files are preserved. New results are written as
`audit_rechecked.json`, with cohort-level `reaudit_summary.json` and
`reaudit_cases.csv`.

If any cases remain rejected after this calibration, inspect those exact issue
codes before modifying either the prompt or model.
