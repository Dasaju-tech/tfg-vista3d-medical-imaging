# Block 26B — Authoritative Numeric and Meta-Limitation Audit Calibration

## Motivation

After Block 26A, the saved 17-case cohort re-audit produced:

```text
acceptance = 12/17
```

The remaining issues fell into three classes:

1. clinical keywords inside explicit negative reporting limitations,
2. numeric values copied from the authoritative findings JSON but absent from
   the narrower preferred narrative whitelist,
3. a genuine semantic inference in which click-to-kidney distance was
   reinterpreted as cyst-to-kidney spatial alignment.

Only the first two are auditor false positives.

## Clinical limitation handling

The auditor now distinguishes a meta-reporting limitation such as:

```text
No diagnostic, malignancy, staging, or treatment claim is generated.
```

from a patient-level clinical interpretation such as:

```text
There is no evidence of malignancy.
```

The first is permitted. The second remains prohibited.

## Numeric handling

The Block 26 prompt exposed the complete authoritative `findings.json`.

Therefore values copied exactly from that object are not hallucinations, even
when they are not desirable prose fields.

Examples include:

```text
255
185.40622380848856
0.00604263676330504
```

The auditor now recognizes every genuine numeric scalar in the authoritative
findings object while retaining the preferred rounded fact variants.

This calibration applies to the saved Block 26 protocol. A future prompt
version should expose only the intended narrative facts rather than the full
raw findings JSON.

## Preserved semantic failure

The calibration deliberately does not permit:

```text
The selected kidney distance is 0.00 mm, indicating the interactive cyst is
spatially aligned with the selected kidney.
```

The stored quantity is click-to-kidney distance. It does not justify a
cyst-to-kidney relationship.

If this remains the only rejected case after re-audit, it should be treated as
a genuine model semantic overreach rather than another auditor false positive.

## Execution

No VISTA or Ollama inference is required.

Re-run:

```powershell
python .\scripts\reaudit_local_llm_reporting_cohort.py
```

The saved candidate JSON files are re-evaluated against the calibrated gate.
