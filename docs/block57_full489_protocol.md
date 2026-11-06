# Block 57 - Full489 cohort and development/validation/test freeze

This incremental block follows the source snapshot supplied from commit
693f00b, tagged block55-final-experimental-manuscript. Block 56 was the
reporting/Overleaf phase in the project plan. No existing source files change.

## Purpose

Expand the study inventory from 100 to all 489 cases in the KiTS23 public
training collection. Prepare a fixed allocation for future method development.
This block downloads metadata only. It does not download CT scans, load NIfTI
volumes, compute metrics, import GPU libraries, or execute a model.

The expected official metadata SHA-256 is taken from the user's existing
expanded100 manifest, rather than silently accepting today's dataset version.
At implementation time the official download matched the supplied manifest:

    c747c720a1c8ea6476050b73eb4cc298f4030e1be0d43df1540601ffb1e9924b

Case IDs are read from that metadata; they are not a contiguous numeric range.
The old 100 IDs and their ordering are retained exactly.

## Allocation fixed before further evaluation

| Group | Historical cases | Additional cases | Total |
| --- | ---: | ---: | ---: |
| Development / future training | 100 | 189 | 289 |
| Validation / method selection | 0 | 80 | 80 |
| Final internal test | 0 | 120 | 120 |
| Entire inventory | 100 | 389 | 489 |

The additional IDs are ranked using SHA-256 of the experiment name, protocol
revision, fixed seed 20260905, and case ID. The first 120 become test, the next
80 validation, and the remaining 189 join historical development. Output lists
are sorted, except that historical order is preserved at the start of development.
Assignment never uses scores, lesion size, predictions, or inference failures.

This is a practical split, not a formal power calculation or stratification.
The metadata snapshot makes it reproducible. Future acquisition must audit
duplicate volumes and patient linkage before claiming independent observations.
Changing the population or allocation requires a documented new protocol
revision before the relevant outcomes are examined.

The 100 historical cases have already influenced research decisions. They
cannot become unseen validation or test cases simply by changing their names.

## PowerShell workflow

Place the delivered patch in the project root. Start the new research branch
from the current commit, check the patch, apply it, then run the suite:

    git switch -c research/block-57-full489
    git apply --check .\vista3d-block57-full489-protocol.patch
    git apply .\vista3d-block57-full489-protocol.patch
    python -m pytest -q
    python .\scripts\prepare_kits23_full489_protocol.py

The existing editable project installation is used, just as with earlier
scripts. This block adds no package dependencies and runs in the main project
environment; the separate vista3d-nv environment is not needed.

The first run downloads about 1.3 MB of official metadata. Subsequent runs use
the saved snapshot, verify the frozen protocol, and refresh only the inventory
audit. The original protocol and initial audit remain unchanged.

For an offline first run using original metadata bytes:

    python .\scripts\prepare_kits23_full489_protocol.py --metadata C:\path\kits23.json

For a read-only verification after preparation:

    python .\scripts\prepare_kits23_full489_protocol.py --check

The script also accepts --project-root for execution from another directory.

Expected allocation output:

    Official cases: 489
    Retained expanded100: 100
    Additional cases: 389
    Development: 289
    Validation: 80
    Test: 120
    Historical cases in validation/test: 0
    Full489 protocol freeze PASSED.

Inventory counts depend on files actually present on the machine.

## Outputs and persistence

- configs/experiments/nv_segment_ct_full489_protocol.json: immutable allocation,
  input digests, reporting boundaries, and canonical SHA-256 certificate.
- outputs/protocols/nv_segment_ct_full489/kits23_metadata.json: original bytes
  needed to repeat the freeze offline; preserve this file when moving machines.
- outputs/protocols/nv_segment_ct_full489/initial_inventory_audit.json: initial
  file inventory, saved once when the protocol is first created.
- outputs/protocols/nv_segment_ct_full489/inventory_audit.json: current inventory.

Once the tests and preparation pass, commit only the new block files and the
generated protocol. Preserve the metadata snapshot and audit with experiment
outputs. Explicit paths avoid staging the external runtime checkout:

    git add docs/block57_full489_protocol.md scripts/prepare_kits23_full489_protocol.py src/vista3d_tfg/experiments/full489.py tests/test_full489.py configs/experiments/nv_segment_ct_full489_protocol.json
    git commit -m "feat: freeze full489 cohort and protected evaluation splits"

## Inventory and failure behavior

The inventory distinguishes missing, empty, and nonempty files at the established
renal data/output paths. A nonempty file is not yet a verified input or reusable
prediction. Block 58 must validate input identity, geometry, checkpoint, runtime,
prompts, precision, and evaluation settings before accepting cached results.

Before the first freeze, nonempty output files with a reserved case ID in their
path cause an explicit failure. The scan covers outputs recursively, including
other experiment branches, and reads file names/sizes only. It cannot establish
whether somebody viewed a CT, used another directory, or inspected aggregate
reports without case IDs in their paths. Such history still requires review.
Do not delete evidence or reseed repeatedly to manufacture an untouched test.

After a valid freeze exists, reruns preserve its allocation and report any
reserved-case outputs for review; they cannot determine when those results were
seen. A changed source manifest, metadata hash, partition, or certificate fails
without replacing the frozen protocol. The baseline manifest is never written.

## Implementation verification

The 26 new tests cover allocation, leakage, metadata drift, malformed identifiers,
certificate tampering, immutable reruns, offline execution, read-only checks, and
previously generated reserved-case artifacts. Real official metadata was also
checked against the supplied expanded100 manifest and produced the allocation above.
The implementation uses Python 3.11-compatible syntax and standard-library imports.

Validation in Linux also reproduces two failures in the unchanged original suite:
test_list_conda_environments_extracts_names and test_existing_environment_is_reused
in test_nv_segment_integration.py. Those tests pass Windows-formatted paths into
host-native Path parsing. They fail without this patch as well. No existing runtime
or test files were changed to conceal these platform-dependent baseline failures.
Run the complete suite in the existing Windows project environment as shown above.
No GPU inference was validated in this block.

## Boundary for the next blocks

Block 58 should profile representative historical cases on the desktop GPU and
provide a resumable runner that understands these partitions. Do not pass this
manifest to run_nv_segment_ct_expanded100.py: that runner deliberately requires
exactly 100 cases and its existing reporting behavior is unsuitable for a sealed
test split. Do not change its size constant to bypass that contract.

Development is available for training and hypothesis development; validation
selects models and thresholds. Keep test metrics, overlays and failure reviews
sealed until all compared methods and analysis rules are fixed. Acquisition of
all 489 does not require immediate inspection of all 489 outcomes.

Report historical100, additional389, and all489 descriptively at the appropriate
stage. Report final method comparisons on the reserved test cases separately;
all489 is not an independent test set, and the nested cohorts are not independent
samples. Document failures and exclusions rather than silently replacing cases.

The split is internal to the public training collection, not the challenge's
110-case test set. It does not establish absence of overlap with foundation-model
pretraining or external clinical generalization. Keep automatic, ground-truth
oracle and CT-only interactive results separately identified.

Sources: https://kits-challenge.org/kits23/ and
https://github.com/neheller/kits23 .
