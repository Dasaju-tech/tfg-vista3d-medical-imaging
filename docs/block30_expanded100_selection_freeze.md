# Block 30A — Expanded100 KiTS23 Selection Freeze

## Objective

The previous renal experiment used a frozen 30-case engineering cohort. The
next phase increases the automatic renal validation cohort to 100 cases before
changing the model, prompt, runtime, or supported evaluation definitions.

Block 30A freezes **selection only**. It deliberately performs no CT download
and no model inference.

## Why selection is a separate step

The earlier pilot exposed an important dataset detail: KiTS23 contains 489
cases, but the numeric identifiers are not a contiguous `00000..00488` range.
Therefore Block 30A does not generate candidate IDs from a numeric interval.
It reads the official `kits23.json` metadata and samples only identifiers that
actually occur in that metadata.

The official metadata source is:

```text
https://raw.githubusercontent.com/neheller/kits23/main/dataset/kits23.json
```

The downloaded bytes are SHA-256 hashed and the digest is stored with the
selection so the dataset universe used for freezing can be audited later.

## Nested design

The 100-case cohort contains:

```text
30 frozen pilot30 cases
+ 70 new official KiTS23 cases
= 100 total cases
```

The pilot30 cases are retained exactly and in their existing order. This gives
three advantages:

1. existing predictions/evaluations can be reused;
2. the old and new estimates remain directly comparable;
3. no previously observed model result influences which 70 new cases enter.

The additional 70 cases are sampled uniformly without replacement from the
remaining official IDs using:

```text
seed = 20260901
```

## Registered selection

Run:

```powershell
python .\scripts\prepare_kits23_expanded100_selection.py
```

This creates:

```text
configs/experiments/nv_segment_ct_expanded100_selection.json
```

The file records the complete 100 IDs, the 70 additions, the metadata URL,
metadata SHA-256, seed, and protocol boundary.

**Commit this JSON before running Block 30B inference.** Once committed, the
100-case list is frozen. A future change to the official metadata must not
silently change the cohort.

## Acceptance criteria

Block 30A passes only if:

- official metadata contains exactly 489 unique case IDs;
- every frozen pilot30 ID exists in that universe;
- the final selection has exactly 100 unique cases;
- its first 30 entries equal the frozen pilot30 list exactly;
- all 70 additional cases come from the official metadata universe;
- selection is deterministic under the fixed seed;
- the manifest records the metadata SHA-256.

## Interpretation boundary

Increasing the renal cohort from 30 to 100 should narrow uncertainty and expose
more failure modes, but it does not by itself establish clinical validity or
population generalisability. Block 30B will keep the segmentation runtime and
evaluation contract frozen so that the effect being measured is increased
sample size rather than a changed method.
