from __future__ import annotations

import json
from pathlib import Path

import pytest

from vista3d_tfg.experiments.expanded100 import (
    EXPANDED100_EXPERIMENT,
    EXPANDED100_SEED,
    EXPANDED100_SIZE,
    OFFICIAL_KITS23_CASE_COUNT,
    deterministic_expanded100_cases,
    load_frozen_selection,
    parse_official_case_numbers,
    selection_manifest_payload,
    validate_official_universe,
    write_selection_manifest,
)
from vista3d_tfg.experiments.pilot30 import PILOT30_CASES


def official_fixture() -> tuple[int, ...]:
    # A non-contiguous 489-ID fixture that still contains every pilot30 case.
    removed = {324, 345, 396, 398}
    values = [value for value in range(493) if value not in removed]
    assert len(values) == OFFICIAL_KITS23_CASE_COUNT
    return tuple(values)


def test_metadata_parser_preserves_noncontiguous_ids() -> None:
    payload = [
        {"case_id": "case_00000"},
        {"case_id": "case_00002"},
        {"case_id": "case_00482"},
    ]

    assert parse_official_case_numbers(payload) == (0, 2, 482)


def test_metadata_parser_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_official_case_numbers(
            [
                {"case_id": "case_00001"},
                {"case_id": "case_00001"},
            ]
        )


def test_official_universe_requires_489_cases() -> None:
    with pytest.raises(ValueError, match="489"):
        validate_official_universe(range(100))


def test_expanded100_is_deterministic_nested_and_unique() -> None:
    universe = official_fixture()

    first = deterministic_expanded100_cases(universe)
    second = deterministic_expanded100_cases(reversed(universe))

    assert first == second
    assert len(first) == EXPANDED100_SIZE
    assert len(set(first)) == EXPANDED100_SIZE
    assert first[:len(PILOT30_CASES)] == PILOT30_CASES
    assert set(first).issubset(set(universe))


def test_manifest_records_selection_contract() -> None:
    payload = selection_manifest_payload(
        official_case_numbers=official_fixture(),
        metadata_sha256="a" * 64,
    )

    assert payload["experiment"] == EXPANDED100_EXPERIMENT
    assert payload["selection"]["seed"] == EXPANDED100_SEED
    assert payload["selection"]["requested_case_count"] == 100
    assert payload["selection"]["retained_pilot30_count"] == 30
    assert payload["selection"]["additional_case_count"] == 70
    assert payload["official_universe"]["case_count"] == 489
    assert payload["protocol_freeze"]["selection_uses_model_outputs"] is False
    assert payload["protocol_freeze"]["pilot30_is_nested"] is True


def test_selection_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"

    write_selection_manifest(
        path,
        official_case_numbers=official_fixture(),
        metadata_sha256="b" * 64,
    )

    frozen = load_frozen_selection(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(frozen) == 100
    assert frozen[:30] == PILOT30_CASES
    assert payload["official_universe"]["metadata_sha256"] == "b" * 64
