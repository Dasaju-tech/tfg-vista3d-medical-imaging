"""Freeze, dispatch and export the separate AMP recovery continuation."""
from pathlib import Path
from uuid import uuid4
import zipfile

from vista3d_tfg.training import contract65 as c, contract65b as b
from vista3d_tfg.experiments.full489_execution import invoke_worker
from vista3d_tfg.integration.nv_segment import resolve_conda_executable


def export(root: Path, plan: dict) -> Path:
    output = c.output_root(root, plan)
    parent = c.read_json(root/c.CONFIG)
    source = c.output_root(root, parent)
    files = {"protocol.json": output/"protocol.json"}
    for name in ("training_progress.json", "training_identity.json", "selection.json"):
        if (output/name).exists(): files[name] = output/name
    for sub in ("epochs", "validation", "attempts", "amp_events"):
        for path in sorted((output/sub).rglob("*.json")):
            files[path.relative_to(output).as_posix()] = path
    # Include only the explicitly frozen parent JSON evidence, never test or
    # mutable parent progress. Large cache, predictions and checkpoints stay local.
    for name in plan["parent_artifacts"]:
        if name.endswith(".json"): files["parent/"+name] = source/name
    manifest = {name: c.sha256_file(path) for name, path in files.items()}
    c.atomic_json(output/"archive_manifest.json", manifest)
    files["archive_manifest.json"] = output/"archive_manifest.json"
    archive = output/"block65b_supervised_training_report.zip"
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items(): z.write(path, name)
    temporary.replace(archive)
    return archive


def dispatch(root, protocol, *, stage, environment, until_epoch):
    plan = c.read_json(root/b.CONFIG); b.check_plan(root, plan, protocol)
    output = c.output_root(root, plan)
    if stage == "report":
        return c.read_json(output/"training_progress.json"), export(root, plan)
    if stage != "train" or type(until_epoch) is not int or not 1 <= until_epoch <= plan["settings"]["epochs"]:
        raise ValueError("Train requires an absolute target epoch in the frozen range.")
    attempt = output/"attempts"/("attempt_"+uuid4().hex[:12]); attempt.mkdir(parents=True)
    c.atomic_json(attempt/"job.json", {"root": str(root), "plan": plan, "until_epoch": until_epoch})
    command = [resolve_conda_executable(), "run", "--no-capture-output", "-n", environment,
               "python", str(root/b.WORKER), "--job", str(attempt/"job.json"),
               "--report", str(attempt/"worker.json")]
    try:
        code = invoke_worker(command, root, timeout=7*24*3600)
        state = c.read_json(attempt/"worker.json")
        if code or state.get("status") not in ("complete", "partial"):
            raise RuntimeError(state.get("error", f"Worker exit code {code}"))
        b.check_plan(root, plan, protocol)
    except (Exception, KeyboardInterrupt) as exc:
        state = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "test_reads": False}
        c.atomic_json(attempt/"failure.json", state)
    return state, export(root, plan)
