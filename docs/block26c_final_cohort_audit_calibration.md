# Block 26C — Final Cohort Audit Calibration

This repair is based on direct inspection of the remaining Block 26 rejected
candidates.

## case_00275

The candidate states:

```text
does not ... generate any diagnostic, malignancy, staging, or treatment claims
```

This is a negative meta-reporting limitation, not a patient-level clinical
claim.

The auditor now recognizes present-tense `generate` in that reporting context.

Statements such as:

```text
There is no malignancy.
```

remain prohibited because they make a patient-level interpretation.

## case_00418

The authoritative value is:

```text
automatic_left_cyst_volume_ml = 0.11587142944335938
```

The candidate reports:

```text
0.1159 mL
```

This is deterministic four-decimal rounding of the authoritative value.

The numeric validator now accepts exact deterministic rendering variants up to
four decimal places. It does not use fuzzy tolerance and does not accept nearby
invented values such as `0.1165`.

## case_00237

No relaxation is made for the remaining semantic failure:

```text
The selected kidney distance is 0.00 mm, indicating the interactive cyst is
spatially aligned with the selected kidney.
```

The authoritative quantity is distance from the user click to the automatic
kidney. It cannot justify a cyst-to-kidney spatial alignment claim.

If the saved-candidate re-audit produces 16/17 with only `case_00237`
rejected, Block 26 should be closed without further auditor relaxation.
