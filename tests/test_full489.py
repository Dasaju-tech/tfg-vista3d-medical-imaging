from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path

import pytest

from vista3d_tfg.experiments.expanded100 import selection_manifest_payload
from vista3d_tfg.experiments.full489 import (
    assert_existing_protocol_matches,
    audit_inventory,
    build_protocol,
    canonical_sha256,
    enforce_initial_reservation,
    fetch_metadata,
    load_protocol,
    write_protocol,
)


@pytest.fixture
def inputs():
    # Synthetic, non-contiguous metadata; never presented as real KiTS23 data.
    numbers = [x for x in range(493) if x not in {324, 345, 396, 398}]
    metadata = json.dumps([{"case_id": f"case_{x:05d}"} for x in numbers]).encode()
    baseline = selection_manifest_payload(
        official_case_numbers=numbers, metadata_sha256=sha256(metadata).hexdigest(),
    )
    return metadata, baseline


def build(inputs):
    metadata, baseline = inputs
    return build_protocol(metadata_raw=metadata, expanded100_raw=json.dumps(baseline).encode())


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def cli():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_kits23_full489_protocol.py"
    spec = importlib.util.spec_from_file_location("block57_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_cli(tmp_path, inputs):
    metadata, baseline = inputs
    root = tmp_path / "project"
    save_json(
        root / "configs/experiments/nv_segment_ct_expanded100_selection.json", baseline,
    )
    metadata_path = tmp_path / "official.json"
    metadata_path.write_bytes(metadata)
    return root, metadata_path


def test_complete_nested_allocation_has_no_split_leakage(inputs):
    payload = build(inputs)
    cohorts = payload["cohorts"]
    splits = payload["partition"]
    historical = inputs[1]["selection"]["case_ids"]
    assert cohorts["all489"]["case_ids"][:100] == historical
    assert cohorts["additional389"]["case_count"] == 389
    assert "case_00324" not in cohorts["all489"]["case_ids"]
    groups = [set(splits[key]["case_ids"]) for key in ("development", "validation", "test")]
    assert [len(group) for group in groups] == [289, 80, 120]
    assert len(set.union(*groups)) == 489
    assert not any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2)))
    assert set(historical).issubset(groups[0])
    assert set.union(*groups) == set(cohorts["all489"]["case_ids"])
    assert payload == build(inputs)


def test_baseline_bom_and_line_endings_do_not_change_allocation_or_certificate(inputs):
    metadata, baseline = inputs
    windows = ("\ufeff" + json.dumps(baseline, indent=2).replace("\n", "\r\n")).encode()
    assert build_protocol(metadata_raw=metadata, expanded100_raw=windows) == build(inputs)


def test_metadata_byte_drift_is_rejected_even_with_same_ids(inputs):
    metadata, baseline = inputs
    with pytest.raises(ValueError, match="SHA-256 differs"):
        build((metadata + b"\n", baseline))


@pytest.mark.parametrize("bad", [True, 0.0, "0", -1, 100000])
def test_identifier_coercion_is_rejected(inputs, bad):
    metadata, baseline = inputs
    baseline["selection"]["case_numbers"][0] = bad
    with pytest.raises(ValueError, match="integer identifiers"):
        build((metadata, baseline))


def test_disagreeing_id_representations_are_rejected(inputs):
    inputs[1]["selection"]["case_ids"][0] = "case_99999"
    with pytest.raises(ValueError, match="disagree"):
        build(inputs)


def test_duplicate_historical_case_is_rejected(inputs):
    selection = inputs[1]["selection"]
    selection["case_ids"][31] = selection["case_ids"][30]
    selection["case_numbers"][31] = selection["case_numbers"][30]
    with pytest.raises(ValueError, match="unique"):
        build(inputs)


def test_missing_historical_case_in_metadata_is_rejected(inputs):
    metadata, baseline = inputs
    records = json.loads(metadata)
    missing = baseline["selection"]["case_ids"][31]
    next(record for record in records if record["case_id"] == missing)["case_id"] = "case_99999"
    metadata = json.dumps(records).encode()
    baseline["official_universe"]["metadata_sha256"] = sha256(metadata).hexdigest()
    with pytest.raises(ValueError, match="absent from official"):
        build((metadata, baseline))


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "malformed"])
def test_invalid_official_universe_is_rejected(inputs, mutation):
    metadata, baseline = inputs
    records = json.loads(metadata)
    if mutation == "duplicate":
        records[-1] = records[-2]
    elif mutation == "missing":
        records.pop()
    else:
        records[-1]["case_id"] = "case_492"
    metadata = json.dumps(records).encode()
    baseline["official_universe"]["metadata_sha256"] = sha256(metadata).hexdigest()
    with pytest.raises(ValueError):
        build((metadata, baseline))


def test_freeze_is_idempotent_and_preserves_windows_file_bytes(inputs, tmp_path):
    payload = build(inputs)
    path = tmp_path / "protocol.json"
    assert write_protocol(path, payload)
    windows = b"\xef\xbb\xbf" + path.read_bytes().replace(b"\n", b"\r\n")
    path.write_bytes(windows)
    timestamp = path.stat().st_mtime_ns
    assert not write_protocol(path, payload)
    assert path.read_bytes() == windows
    assert path.stat().st_mtime_ns == timestamp
    assert load_protocol(path) == payload


def test_changed_historical_manifest_cannot_overwrite_freeze(inputs, tmp_path):
    path = tmp_path / "protocol.json"
    write_protocol(path, build(inputs))
    before = path.read_bytes()
    inputs[1]["interpretation_boundary"] += " Changed source."
    with pytest.raises(ValueError, match="incompatible"):
        write_protocol(path, build(inputs))
    assert path.read_bytes() == before


@pytest.mark.parametrize("resign", [False, True])
def test_tampered_split_is_rejected_even_if_certificate_is_recomputed(inputs, tmp_path, resign):
    payload = deepcopy(build(inputs))
    payload["partition"]["test"]["case_ids"][0] = payload["cohorts"]["historical100"]["case_ids"][0]
    if resign:
        payload["protocol_sha256"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "protocol_sha256"}
        )
    path = tmp_path / "protocol.json"
    save_json(path, payload)
    with pytest.raises(ValueError, match="certificate|allocation"):
        load_protocol(path)


def test_inventory_does_not_open_volume_or_metric_contents(inputs, tmp_path):
    payload = build(inputs)
    cid = payload["partition"]["test"]["case_ids"][0]
    image = tmp_path / "data/raw/kits23" / cid / "imaging.nii.gz"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"deliberately not a NIfTI")
    (image.parent / "segmentation.nii.gz").write_bytes(b"")
    result = tmp_path / "outputs/experiments/custom_branch" / f"{cid}_result.json"
    result.parent.mkdir(parents=True)
    result.write_bytes(b"deliberately not JSON")
    audit = audit_inventory(tmp_path, payload)
    row = next(row for row in audit["cases"] if row["case_id"] == cid)
    assert row["files"]["imaging"] == "present"
    assert row["files"]["segmentation"] == "empty"
    assert row["files"]["prediction"] == "missing"
    assert audit["nonempty_file_counts"]["imaging"] == 1
    assert audit["reserved_cases_with_existing_outputs"] == [cid]
    with pytest.raises(ValueError, match="already have output artifacts"):
        enforce_initial_reservation(audit, already_frozen=False)
    enforce_initial_reservation(audit, already_frozen=True)


def test_historical_outputs_do_not_block_initial_freeze(inputs, tmp_path):
    payload = build(inputs)
    result = tmp_path / "outputs/evaluations/nv_segment_ct/case_00000/result.json"
    result.parent.mkdir(parents=True)
    result.write_text("{}", encoding="utf-8")
    audit = audit_inventory(tmp_path, payload)
    assert audit["reserved_cases_with_existing_outputs"] == []
    enforce_initial_reservation(audit, already_frozen=False)


def test_fetch_is_bounded_closes_response_and_uses_official_url(inputs):
    raw = inputs[0]
    response = io.BytesIO(raw)
    def opener(request, timeout):
        assert request.full_url.endswith("/neheller/kits23/main/dataset/kits23.json")
        assert timeout == 30
        return response
    assert fetch_metadata(opener=opener) == raw
    assert response.closed
    with pytest.raises(ValueError, match="empty"):
        fetch_metadata(opener=lambda *args, **kwargs: io.BytesIO(b""))


def test_cli_offline_then_cached_rerun_then_read_only_check(inputs, tmp_path, monkeypatch):
    root, metadata_path = setup_cli(tmp_path, inputs)
    module = cli()
    monkeypatch.setattr(module, "fetch_metadata", lambda: pytest.fail("Unexpected network access"))
    args = ["--project-root", str(root)]
    assert module.main(args + ["--metadata", str(metadata_path)]) == 0
    protocol = root / "configs/experiments/nv_segment_ct_full489_protocol.json"
    initial = root / "outputs/protocols/nv_segment_ct_full489/initial_inventory_audit.json"
    before = (protocol.read_bytes(), protocol.stat().st_mtime_ns, initial.read_bytes())
    assert module.main(args) == 0
    assert before == (protocol.read_bytes(), protocol.stat().st_mtime_ns, initial.read_bytes())
    files = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
    assert module.main(args + ["--check"]) == 0
    assert files == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_cli_refuses_reserved_outputs_before_creating_protocol_or_snapshot(inputs, tmp_path):
    root, metadata_path = setup_cli(tmp_path, inputs)
    cid = build(inputs)["partition"]["validation"]["case_ids"][0]
    evidence = root / "outputs/another_study" / cid / "metrics.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("{}", encoding="utf-8")
    assert cli().main(["--project-root", str(root), "--metadata", str(metadata_path)]) == 1
    assert not (root / "configs/experiments/nv_segment_ct_full489_protocol.json").exists()
    assert not (root / "outputs/protocols/nv_segment_ct_full489").exists()


def test_cli_check_without_freeze_has_no_side_effects(inputs, tmp_path, monkeypatch):
    root, _ = setup_cli(tmp_path, inputs)
    module = cli()
    monkeypatch.setattr(module, "fetch_metadata", lambda: pytest.fail("Unexpected network access"))
    assert module.main(["--project-root", str(root), "--check"]) == 1
    assert not (root / "outputs").exists()


def test_cli_preserves_orphan_initial_audit_without_publishing_a_new_freeze(inputs, tmp_path):
    root, metadata_path = setup_cli(tmp_path, inputs)
    initial = root / "outputs/protocols/nv_segment_ct_full489/initial_inventory_audit.json"
    save_json(initial, {"old_audit": True})
    before = initial.read_bytes()
    assert cli().main(["--project-root", str(root), "--metadata", str(metadata_path)]) == 1
    assert initial.read_bytes() == before
    assert not (root / "configs/experiments/nv_segment_ct_full489_protocol.json").exists()
    assert not (initial.parent / "kits23_metadata.json").exists()


def test_checked_loader_detects_missing_file(inputs, tmp_path):
    assert not assert_existing_protocol_matches(tmp_path / "absent.json", build(inputs))
