from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Callable, Iterable
from urllib.request import Request, urlopen

from vista3d_tfg.experiments.pilot30 import PILOT30_CASES


EXPANDED100_EXPERIMENT = "nv_segment_ct_expanded100"
EXPANDED100_SEED = 20260901
EXPANDED100_SIZE = 100
OFFICIAL_KITS23_CASE_COUNT = 489
OFFICIAL_KITS23_METADATA_URL = (
    "https://raw.githubusercontent.com/"
    "neheller/kits23/main/dataset/kits23.json"
)

_CASE_ID_PATTERN = re.compile(r"^case_(\d{5})$")
URLReader = Callable[..., object]


def parse_official_case_numbers(payload: object) -> tuple[int, ...]:
    """Extract sorted, unique KiTS23 numeric IDs from official metadata."""

    if not isinstance(payload, list):
        raise ValueError("Official KiTS23 metadata must be a JSON list.")

    case_numbers: list[int] = []

    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(
                f"KiTS23 metadata record {index} is not an object."
            )

        case_id = record.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError(
                f"KiTS23 metadata record {index} has no string case_id."
            )

        match = _CASE_ID_PATTERN.fullmatch(case_id)
        if match is None:
            raise ValueError(f"Invalid KiTS23 case_id: {case_id!r}.")

        case_numbers.append(int(match.group(1)))

    if len(case_numbers) != len(set(case_numbers)):
        raise ValueError("Official KiTS23 metadata contains duplicate case IDs.")

    return tuple(sorted(case_numbers))


def validate_official_universe(
    case_numbers: Iterable[int],
) -> tuple[int, ...]:
    """Validate the universe used to freeze the expanded cohort."""

    numbers = tuple(sorted({int(value) for value in case_numbers}))

    if len(numbers) != OFFICIAL_KITS23_CASE_COUNT:
        raise ValueError(
            "Expected exactly "
            f"{OFFICIAL_KITS23_CASE_COUNT} official KiTS23 cases, "
            f"found {len(numbers)}."
        )

    missing_pilot = sorted(set(PILOT30_CASES) - set(numbers))
    if missing_pilot:
        raise ValueError(
            "Official metadata does not contain frozen pilot30 cases: "
            + ", ".join(f"{value:05d}" for value in missing_pilot)
        )

    return numbers


def fetch_official_case_numbers(
    *,
    opener: URLReader = urlopen,
) -> tuple[tuple[int, ...], str]:
    """Download official metadata and return IDs plus its SHA-256 digest."""

    request = Request(
        OFFICIAL_KITS23_METADATA_URL,
        headers={"User-Agent": "vista3d-tfg-expanded100/1.0"},
    )
    response = opener(request, timeout=60)

    try:
        raw = response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()

    if not isinstance(raw, bytes) or not raw:
        raise RuntimeError("Official KiTS23 metadata download was empty.")

    payload = json.loads(raw.decode("utf-8"))
    case_numbers = validate_official_universe(
        parse_official_case_numbers(payload)
    )

    return case_numbers, sha256(raw).hexdigest()


def deterministic_expanded100_cases(
    official_case_numbers: Iterable[int],
) -> tuple[int, ...]:
    """
    Return a deterministic 100-case cohort nested around frozen pilot30.

    Pilot30 is retained exactly. Seventy additional cases are sampled from
    the remaining official KiTS23 IDs using a new fixed seed. Selection uses
    dataset identity only and never model outputs.
    """

    universe = validate_official_universe(official_case_numbers)
    pilot = tuple(PILOT30_CASES)
    pilot_set = set(pilot)
    candidates = [value for value in universe if value not in pilot_set]

    rng = random.Random(EXPANDED100_SEED)
    additional = tuple(
        sorted(rng.sample(candidates, EXPANDED100_SIZE - len(pilot)))
    )

    selection = tuple((*pilot, *additional))

    if len(selection) != EXPANDED100_SIZE:
        raise AssertionError("Expanded100 selection has the wrong size.")
    if len(set(selection)) != EXPANDED100_SIZE:
        raise AssertionError("Expanded100 selection contains duplicates.")

    return selection


def selection_manifest_payload(
    *,
    official_case_numbers: Iterable[int],
    metadata_sha256: str,
) -> dict[str, object]:
    universe = validate_official_universe(official_case_numbers)
    selected = deterministic_expanded100_cases(universe)
    additional = selected[len(PILOT30_CASES):]

    return {
        "schema_version": "1.0",
        "dataset": "KiTS23",
        "experiment": EXPANDED100_EXPERIMENT,
        "selection": {
            "revision": 1,
            "seed": EXPANDED100_SEED,
            "requested_case_count": EXPANDED100_SIZE,
            "retained_pilot30_count": len(PILOT30_CASES),
            "additional_case_count": len(additional),
            "method": (
                "retain the frozen pilot30 cohort exactly, then sample 70 "
                "additional IDs uniformly without replacement from the "
                "official KiTS23 metadata universe"
            ),
            "case_numbers": list(selected),
            "case_ids": [f"case_{value:05d}" for value in selected],
            "additional_case_numbers": list(additional),
        },
        "official_universe": {
            "source_url": OFFICIAL_KITS23_METADATA_URL,
            "case_count": len(universe),
            "metadata_sha256": metadata_sha256,
            "minimum_case_number": min(universe),
            "maximum_case_number": max(universe),
        },
        "protocol_freeze": {
            "selection_uses_model_outputs": False,
            "pilot30_is_nested": True,
            "selection_must_be_committed_before_inference": True,
        },
        "interpretation_boundary": (
            "Expanded engineering validation cohort. Increasing N improves "
            "precision and failure-mode coverage but does not by itself "
            "establish clinical generalisability."
        ),
    }


def write_selection_manifest(
    output_path: Path,
    *,
    official_case_numbers: Iterable[int],
    metadata_sha256: str,
) -> dict[str, object]:
    payload = selection_manifest_payload(
        official_case_numbers=official_case_numbers,
        metadata_sha256=metadata_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_frozen_selection(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if payload.get("experiment") != EXPANDED100_EXPERIMENT:
        raise ValueError("Selection manifest is not expanded100.")

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Selection manifest has no selection object.")

    values = selection.get("case_numbers")
    if not isinstance(values, list):
        raise ValueError("Selection manifest has no case_numbers list.")

    case_numbers = tuple(int(value) for value in values)

    if len(case_numbers) != EXPANDED100_SIZE:
        raise ValueError("Expanded100 manifest must contain exactly 100 cases.")
    if len(set(case_numbers)) != EXPANDED100_SIZE:
        raise ValueError("Expanded100 manifest contains duplicate cases.")
    if case_numbers[:len(PILOT30_CASES)] != tuple(PILOT30_CASES):
        raise ValueError("Expanded100 manifest does not retain pilot30 exactly.")

    return case_numbers
