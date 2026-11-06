from __future__ import annotations

import json
from pathlib import Path

from vista3d_tfg.experiments.expanded100 import (
    EXPANDED100_EXPERIMENT,
    EXPANDED100_SIZE,
    load_frozen_selection,
)
from vista3d_tfg.experiments.pilot30 import PILOT30_CASES, pilot_case_paths


def frozen_selection_payload() -> dict[str, object]:
    additions = tuple(range(1000, 1070))
    values = tuple(PILOT30_CASES) + additions
    assert len(values) == EXPANDED100_SIZE
    return {
        "experiment": EXPANDED100_EXPERIMENT,
        "selection": {"case_numbers": list(values)},
    }


def test_expanded100_frozen_selection_keeps_pilot30_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(frozen_selection_payload()),
        encoding="utf-8",
    )

    case_numbers = load_frozen_selection(path)

    assert len(case_numbers) == 100
    assert case_numbers[:30] == PILOT30_CASES
    assert len(set(case_numbers)) == 100


def test_expanded100_reuses_case_artifact_namespace(tmp_path: Path) -> None:
    pilot_paths = pilot_case_paths(tmp_path, PILOT30_CASES[0])
    expanded_paths = pilot_case_paths(tmp_path, PILOT30_CASES[0])

    assert expanded_paths.prediction_path == pilot_paths.prediction_path
    assert expanded_paths.inference_json == pilot_paths.inference_json
    assert expanded_paths.evaluation_json == pilot_paths.evaluation_json


def test_expanded100_manifest_rejects_wrong_size(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "experiment": EXPANDED100_EXPERIMENT,
                "selection": {"case_numbers": list(PILOT30_CASES)},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_frozen_selection(path)
    except ValueError as exc:
        assert "exactly 100" in str(exc)
    else:
        raise AssertionError("Expected invalid expanded100 selection to fail.")
