"""Main-environment preparation and dispatch; original inference remains frozen."""
from __future__ import annotations

from pathlib import Path
import math
import re
import shutil
from uuid import uuid4
import zipfile

import nibabel as nib

from vista3d_tfg.experiments import supervised64 as previous
from vista3d_tfg.experiments.full489_execution import invoke_worker
from vista3d_tfg.integration.nv_segment import resolve_conda_executable
from vista3d_tfg.training import contract65 as c


def prepare(root: Path, protocol: dict) -> tuple[Path, dict]:
    old=c.read_json(root/previous.CONFIG)
    previous.check_plan(root,old,protocol)
    previous.verify_inputs(root,old)
    source=root/"outputs/reporting/full489_block64"/old["experiment_sha256"][:16]
    receipt=c.read_json(source/"complete.json");c.check_digest(receipt,"receipt_sha256")
    if receipt["experiment_sha256"]!=old["experiment_sha256"] or not re.fullmatch(r"attempt_[0-9a-f]{12}",receipt["attempt"]):
        raise ValueError("Block 64 completion identity changed.")
    if set(receipt["artifacts"])!={"job.json","worker.json","trace.json","checkpoint.pt","runtime.json"}:
        raise ValueError("Incomplete Block 64 artifact list.")
    attempt=source/receipt["attempt"]
    for name,digest in receipt["artifacts"].items():
        if c.sha256_file(attempt/name)!=digest:raise ValueError(f"Block 64 artifact changed: {name}")
    worker=c.read_json(attempt/"worker.json");previous.validate_result(worker,old)
    runtime=c.read_json(attempt/"runtime.json")
    if runtime["identity"]!=worker["runtime_identity"] or runtime["fp32_fp16_conv3d_smoke"] is not True:
        raise ValueError("Block 64 GPU runtime evidence changed.")
    inputs,seen,cache_bytes,grid_audit={}, {}, 0, {}
    original_run=root/"outputs/full489"/old["source_contract_sha256"][:16]
    for partition in ("development","validation"):
        ids=protocol["partition"][partition]["case_ids"]
        for number,cid in enumerate(ids):
            if number%25==0:print(f"Freeze {partition} inputs {number+1}/{len(ids)}",flush=True)
            identity=c.input_identity(root,cid)
            if partition=="development" and identity!=c.read_json(original_run/"cases"/cid/"inputs.json"):
                raise ValueError(f"Historical development input changed: {cid}")
            if identity["image_sha256"] in seen:
                raise ValueError(f"Byte-identical CT across case IDs: {cid}, {seen[identity['image_sha256']]}")
            seen[identity["image_sha256"]]=cid;inputs[cid]=identity
            image=nib.load(root/"data/raw/kits23"/cid/"imaging.nii.gz")
            # Header-only estimate. Orientation permutations preserve voxel count.
            spacing=nib.affines.voxel_sizes(image.affine)
            if (len(image.shape)!=3 or any(n<1 for n in image.shape)
                    or any(not math.isfinite(float(s)) or s<=0 for s in spacing)):
                raise ValueError(f"Invalid CT shape/spacing in header: {cid}")
            shape=[int(math.ceil((n-1)*float(s)/1.5))+1 for n,s in zip(image.shape,spacing)]
            grid_audit[cid]={"original_shape":[int(n) for n in image.shape],"resampled_shape":shape}
            cache_bytes+=math.prod(shape)*(5 if partition=="development" else 4)
    included=protocol["partition"]["development"]["case_ids"]+protocol["partition"]["validation"]["case_ids"]
    allocation_settings=c.settings_for_grids(grid_audit,included)
    plan={"block":65,"protocol_sha256":protocol["protocol_sha256"],"partition":protocol["partition"],
          "settings":allocation_settings,"grid_audit":grid_audit,"inputs":inputs,"implementation":c.code_identity(root),
          "block64_experiment_sha256":old["experiment_sha256"],"block64_receipt_sha256":receipt["receipt_sha256"],
          "runtime_identity":worker["runtime_identity"],"estimated_cache_bytes":cache_bytes,
          "duplicate_audit":"compressed CT byte equality only; patient/voxel duplicates remain unresolved",
          "test_outputs_opened":False}
    plan["experiment_sha256"]=c.canonical_sha256(plan)
    c.check_plan(root,plan,protocol)
    c.frozen_json(root/c.CONFIG,plan)
    output=c.output_root(root,plan);c.frozen_json(output/"protocol.json",plan)
    print(f"Estimated new array cache: {cache_bytes/2**30:.2f} GiB; disk free: {shutil.disk_usage(root).free/2**30:.2f} GiB",flush=True)
    for key in ("original_shape","resampled_shape"):
        largest=max(included,key=lambda cid:math.prod(grid_audit[cid][key]))
        print(f"Frozen {key} maximum: {math.prod(grid_audit[largest][key]):,} voxels; {largest}; {grid_audit[largest][key]}",flush=True)
    return output,plan


def export(root: Path, plan: dict) -> Path:
    output=c.output_root(root,plan)
    files=[output/"protocol.json"]
    for name in ("training_progress.json","selection.json","training_identity.json",
                 "cache_development_progress.json","cache_validation_progress.json"):
        if (output/name).is_file():files.append(output/name)
    # Explicit Block 65 output root only; never traverse original/test output trees.
    for sub in ("epochs","validation","attempts"):
        files.extend(sorted((output/sub).rglob("*.json")))
    manifest={p.relative_to(output).as_posix():c.sha256_file(p) for p in files}
    c.atomic_json(output/"archive_manifest.json",manifest)
    archive=output/"block65_supervised_training_report.zip"
    temporary=archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as bundle:
        for path in files+[output/"archive_manifest.json"]:
            bundle.write(path,path.relative_to(output).as_posix())
    temporary.replace(archive)
    return archive


def dispatch(root: Path, protocol: dict, *, stage: str, partition: str, environment: str,
             epochs: int=1, max_cases: int | None=None) -> tuple[dict,Path]:
    plan=c.read_json(root/c.CONFIG);c.check_plan(root,plan,protocol)
    if stage not in ("cache","train","report"):raise ValueError("Unknown Block 65 stage.")
    c.partition_ids(plan,partition)
    if stage=="report":
        return c.read_json(c.output_root(root,plan)/"training_progress.json"),export(root,plan)
    attempt=c.output_root(root,plan)/"attempts"/("attempt_"+uuid4().hex[:12])
    attempt.mkdir(parents=True)
    job={"root":str(root),"stage":stage,"partition":partition,"epochs":epochs,"max_cases":max_cases,"plan":plan}
    c.atomic_json(attempt/"job.json",job)
    command=[resolve_conda_executable(),"run","--no-capture-output","-n",environment,"python",str(root/c.WORKER),
             "--job",str(attempt/"job.json"),"--report",str(attempt/"worker.json")]
    try:
        # Long training is deliberately allowed; Ctrl+C terminates the child tree.
        code=invoke_worker(command,root,timeout=7*24*3600)
        state=c.read_json(attempt/"worker.json")
        if code or state.get("status") not in ("complete","partial"):
            raise RuntimeError(state.get("error",f"Worker exit code {code}"))
        c.check_plan(root,plan,protocol)
    except (Exception,KeyboardInterrupt) as exc:
        state={"status":"failed","stage":stage,"error":f"{type(exc).__name__}: {exc}","test_reads":False}
        c.atomic_json(attempt/"failure.json",state)
    return state,export(root,plan)
