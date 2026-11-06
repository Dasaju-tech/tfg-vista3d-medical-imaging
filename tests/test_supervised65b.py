from copy import deepcopy
from pathlib import Path
import shutil
import zipfile

import pytest

from vista3d_tfg.training import contract65 as c, contract65b as b, amp65b as amp
from vista3d_tfg.training import engine65 as old_engine, engine65b as engine
from vista3d_tfg.experiments import supervised65b as experiment
from test_supervised65 import prepared, small_training
from test_supervised64 import frozen
from test_full489_execution import script


def setup_step():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(1.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.)
    x, y = torch.ones(1, 1), torch.zeros(1, 1)
    return torch, model, optimizer, scaler, torch.nn.MSELoss(), x, y


def test_overflow_retries_identical_patch_and_updates_once():
    torch, model, optimizer, scaler, loss, x, y = setup_step()
    before = model.weight.detach().clone(); calls = []; events = []; inputs = []
    model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
    model.weight.register_hook(lambda g: g*float("inf") if scaler.get_scale() > 64 else g)
    original = optimizer.step
    def step(*a, **kw): calls.append(1); return original(*a, **kw)
    optimizer.step = step
    def audit(event):
        assert torch.equal(model.weight, before) and not optimizer.state and not calls
        events.append(event)
    value, retries = amp.optimizer_step(model, optimizer, scaler, loss, x, y, 12., "cpu", on_retry=audit)
    assert value == 1. and retries == 1 and len(calls) == 1
    assert len(inputs) == 2 and torch.equal(inputs[0], inputs[1])
    assert events[0]["scale_before"] == 128 and events[0]["scale_after"] == 64
    assert not torch.equal(before, model.weight) and scaler.get_scale() == 64
    assert int(optimizer.state[model.weight]["step"]) == 1


def test_finite_step_matches_original_algorithm_exactly():
    torch, model, optimizer, scaler, loss, x, y = setup_step()
    reference = deepcopy(model)
    opt = torch.optim.AdamW(reference.parameters(), lr=.01)
    scale = torch.amp.GradScaler("cpu", init_scale=128.)
    optimizer_value, retries = amp.optimizer_step(model, optimizer, scaler, loss, x, y, 12., "cpu")
    opt.zero_grad(set_to_none=True)
    expected_loss = loss(reference(x).float(), y)
    scale.scale(expected_loss).backward(); scale.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 12., error_if_nonfinite=True)
    scale.step(opt); scale.update()
    assert optimizer_value == float(expected_loss.detach()) and retries == 0
    assert torch.equal(model.weight, reference.weight) and scaler.state_dict() == scale.state_dict()
    for key, value in optimizer.state[model.weight].items():
        assert torch.equal(value, opt.state[reference.weight][key])


@pytest.mark.parametrize("mode", ["persistent", "fp32", "loss", "zero", "norm_overflow"])
def test_invalid_update_never_reaches_optimizer(mode):
    torch, model, optimizer, scaler, loss, x, y = setup_step()
    before = model.weight.detach().clone(); events = []
    if mode == "fp32": scaler = torch.amp.GradScaler("cpu", enabled=False)
    if mode in ("persistent", "fp32"):
        model.weight.register_hook(lambda g: g*float("nan"))
    if mode == "loss": loss = lambda prediction, target: prediction.sum()*float("nan")
    if mode == "zero": model.weight.register_hook(lambda g: g*0)
    if mode == "norm_overflow":
        # Two finite elements have an L2 norm whose FP32 reduction overflows.
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        x = torch.ones(1, 2); before = model.weight.detach().clone()
        model.weight.register_hook(lambda g: torch.full_like(g, 1e30))
    with pytest.raises((ValueError, RuntimeError)):
        amp.optimizer_step(model, optimizer, scaler, loss, x, y, 12., "cpu", on_retry=events.append)
    assert torch.equal(model.weight, before) and not optimizer.state
    assert len(events) == (amp.MAX_RETRIES + 1 if mode == "persistent" else 0)


@pytest.fixture
def parent_run(small_training, monkeypatch):
    torch = pytest.importorskip("torch"); pytest.importorskip("monai")
    root, protocol, output, parent = small_training
    for name in b.SOURCES:
        dest = root/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).resolve().parents[1]/name, dest)
    # Internal integration fixture only: use the existing real-engine small
    # tensors while retaining all production contract checks and source hashes.
    expected = deepcopy(parent["settings"])
    expected["epochs"] = 3
    expected["validation_epochs"] = [1, 2, 3]
    parent["settings"] = deepcopy(expected)
    original = c.settings_for_grids
    monkeypatch.setattr(c, "settings_for_grids", lambda audit, ids: {
        **original(audit, ids), **expected})
    parent["experiment_sha256"] = c.canonical_sha256({k: v for k, v in parent.items() if k != "experiment_sha256"})
    # Regenerate only synthetic fixture caches under this valid small plan hash.
    from vista3d_tfg.training.cache65 import prepare_cache
    for partition in ("development", "validation"): prepare_cache(root, parent, partition)
    output = c.output_root(root, parent)
    c.atomic_json(root/c.CONFIG, parent); c.atomic_json(output/"protocol.json", parent)
    threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        old_engine.train(root, parent, {"test": "cpu"}, epochs=1, device="cpu")
        yield root, protocol, output, parent
    finally: torch.set_num_threads(threads)


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): c.sha256_file(p) for p in folder.rglob("*") if p.is_file()}


def test_versioned_resume_matches_parent_continuation_and_preserves_source(parent_run, tmp_path):
    torch = pytest.importorskip("torch")
    root, protocol, source, parent = parent_run
    uninterrupted = tmp_path/"reference"; shutil.copytree(root, uninterrupted)
    before = snapshot(source); cache_before = snapshot(c.cache_root(root, parent))
    test_path = root/"data/raw/kits23/case_00011/imaging.nii.gz"
    test_path.parent.mkdir(parents=True, exist_ok=True); test_path.write_bytes(b"sealed")
    child = b.freeze(root, protocol)
    assert b.freeze(root, protocol) == child and child["experiment_sha256"] != parent["experiment_sha256"]
    assert child["settings"] == parent["settings"] and child["parent_last_epoch"] == 1
    recovered = engine.train_until(root, child, {"test": "cpu"}, until_epoch=2, device="cpu")
    reference = old_engine.train(uninterrupted, parent, {"test": "cpu"}, epochs=1, device="cpu")
    def checkpoint(base, plan):
        out = c.output_root(base, plan); receipt = old_engine.epoch_receipt(out, plan, 2)
        return torch.load(out/"epochs/epoch_002"/receipt["file"], map_location="cpu", weights_only=True)
    a, ref = checkpoint(root, child), checkpoint(uninterrupted, parent)
    assert a["model_state_sha256"] == ref["model_state_sha256"]
    assert a["optimizer"]["param_groups"] == ref["optimizer"]["param_groups"]
    assert a["scaler"] == ref["scaler"] and torch.equal(a["torch_rng"], ref["torch_rng"])
    for key, state in a["optimizer"]["state"].items():
        for field, tensor in state.items(): assert torch.equal(tensor, ref["optimizer"]["state"][key][field])
    assert recovered["optimizer_steps"] == reference["optimizer_steps"] == 12
    assert recovered["selection"]["best_epoch"] == reference["selection"]["best_epoch"]
    assert snapshot(source) == before and snapshot(c.cache_root(root, parent)) == cache_before
    assert test_path.read_bytes() == b"sealed"
    # Repeating an absolute target does not train an extra epoch.
    assert engine.train_until(root, child, {"test": "cpu"}, until_epoch=2, device="cpu") == recovered
    archive = experiment.export(root, child)
    with zipfile.ZipFile(archive) as z:
        assert "parent/epochs/epoch_001/complete.json" in z.namelist()
        assert "parent/validation/epoch_001/summary.json" in z.namelist()
        assert not any(n.endswith((".pt", ".npy", ".nii.gz")) for n in z.namelist())
        manifest = __import__("json").loads(z.read("archive_manifest.json"))
        assert set(manifest) == set(z.namelist()) - {"archive_manifest.json"}
    b.check_plan(root, child, protocol)


def test_interrupted_child_epoch_repeats_from_parent(parent_run, monkeypatch):
    root, protocol, source, parent = parent_run
    child = b.freeze(root, protocol); before = snapshot(source)
    original = engine.optimizer_step; calls = []
    def failure(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise ValueError("synthetic failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, "optimizer_step", failure)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        engine.train_until(root, child, {"test": "cpu"}, until_epoch=2, device="cpu")
    assert old_engine.epoch_receipt(c.output_root(root, child), child, 2) is None
    monkeypatch.setattr(engine, "optimizer_step", original)
    assert engine.train_until(root, child, {"test": "cpu"}, until_epoch=2, device="cpu")["completed_epochs"] == 2
    assert snapshot(source) == before


def test_resume_from_child_checkpoint_finishes_pending_validation(parent_run, monkeypatch):
    root, protocol, source, parent = parent_run
    child = b.freeze(root, protocol)
    original = engine.validate_epoch
    def interrupt(*args, **kwargs):
        raise RuntimeError("synthetic validation interruption")
    monkeypatch.setattr(engine, "validate_epoch", interrupt)
    with pytest.raises(RuntimeError, match="validation interruption"):
        engine.train_until(root, child, {"test": "cpu"}, until_epoch=2, device="cpu")
    output = c.output_root(root, child)
    saved = old_engine.epoch_receipt(output, child, 2)
    digest = c.sha256_file(output/"epochs/epoch_002"/saved["file"])
    monkeypatch.setattr(engine, "validate_epoch", original)
    # Finish the validation at the existing absolute target without another
    # optimizer step, then continue training from the child checkpoint.
    state = engine.train_until(root, child, {"test": "cpu"}, until_epoch=2, device="cpu")
    assert state["optimizer_steps"] == 12 and state["selection"]["evaluated_epochs"] == [1, 2]
    state = engine.train_until(root, child, {"test": "cpu"}, until_epoch=3, device="cpu")
    assert state["optimizer_steps"] == 18 and state["status"] == "complete"
    assert state["selection"]["evaluated_epochs"] == [1, 2, 3]
    assert c.sha256_file(output/"epochs/epoch_002"/saved["file"]) == digest


@pytest.mark.parametrize("change", ["code", "parent", "checkpoint", "settings", "lineage"])
def test_continuation_rejects_drift(parent_run, change):
    root, protocol, source, parent = parent_run
    plan = b.freeze(root, protocol)
    if change == "code": (root/b.SOURCES[0]).write_text("changed")
    if change == "parent":
        altered = deepcopy(parent); altered["settings"]["epochs"] = 51; c.atomic_json(root/c.CONFIG, altered)
    if change == "checkpoint":
        r = old_engine.epoch_receipt(source, parent, 1); (source/"epochs/epoch_001"/r["file"]).write_bytes(b"changed")
    if change == "settings": plan["settings"]["optimizer"]["lr"] *= 2
    if change == "lineage": plan["parent_artifacts"].pop("training_identity.json")
    plan["experiment_sha256"] = c.canonical_sha256({k: v for k, v in plan.items() if k != "experiment_sha256"})
    with pytest.raises(ValueError): b.check_plan(root, plan, protocol)


@pytest.mark.parametrize("args", [["--stage", "train", "--until-epoch", "51"],
                                  ["--stage", "train", "--until-epoch", "0"],
                                  ["--stage", "train", "--partition", "test"]])
def test_cli_rejects_invalid_targets_and_test(args):
    with pytest.raises(SystemExit) as exc: script("run_kits23_supervised65b").main(args)
    assert exc.value.code == 2
