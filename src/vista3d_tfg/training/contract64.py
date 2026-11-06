"""Shared Block 64 contract and IO; standard library only in the GPU worker."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4


def canonical_sha256(payload: object) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex[:8] + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def frozen_json(path: Path, payload: dict) -> None:
    if path.exists():
        if read_json(path) != payload:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        atomic_json(path, payload)


def input_identity(root: Path, cid: str) -> dict:
    folder = root / "data/raw/kits23" / cid
    return {"case_id": cid, "image_sha256": sha256_file(folder / "imaging.nii.gz"),
            "ground_truth_sha256": sha256_file(folder / "segmentation.nii.gz")}


CONFIG = "configs/experiments/kits23_supervised_block64.json"

WORKER = "scripts/kits23_supervised64_worker.py"

SOURCES = (
    "src/vista3d_tfg/experiments/supervised64.py",
    "src/vista3d_tfg/training/supervised64.py",
    "src/vista3d_tfg/training/__init__.py",
    "src/vista3d_tfg/training/contract64.py",
    "scripts/run_kits23_supervised64.py", WORKER,
    "scripts/nv_segment_ct_full489_worker.py",
)


def settings() -> dict:
    return {
        "seed": 6401, "smoke_cases": 3, "steps_per_case": 8,
        "labels": {"0": "background", "1": "kidney", "2": "tumor", "3": "cyst"},
        "model": {"architecture": "monai.UNet", "spatial_dims": 3, "in_channels": 1,
                  "out_channels": 4, "channels": [16, 32, 64, 128], "strides": [2, 2, 2],
                  "num_res_units": 2, "norm": "INSTANCE", "dropout": 0.0},
        "initialization": "random; no NV checkpoint or pretrained weights",
        "patch_size": [64, 64, 64], "batch_size": 1, "spacing_mm": [1.5, 1.5, 1.5],
        "orientation": "closest RAS; preserve native obliquity",
        "ct_window_hu": [-200.0, 300.0], "normalized_range": [-1.0, 1.0],
        "interpolation": {"image": "linear", "label": "nearest"},
        "padding": {"image": -1.0, "label": 0},
        "sampling": "cycle centers 0,1,2,3 twice per case; uniform within selected class",
        "augmentation": "none in feasibility smoke", "num_workers": 0,
        "max_original_voxels": 256000000, "max_resampled_voxels": 64000000,
        "loss": "DiceCELoss: foreground Dice + four-class CE; equal weights",
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.00001},
        "precision": "CUDA FP16 autocast; FP32 loss; GradScaler initial scale 128",
        "max_gradient_norm": 12.0, "cudnn_benchmark": False,
        "determinism": "seeded; deterministic algorithms requested with warn_only=True",
        "future_full_training": "requires separate frozen schedule, augmentation, selection and evaluation contract",
        "validation_role": "future model selection on validation80; no reads in Block 64",
        "test_role": "test120 sealed until final method and evaluation are frozen",
        "checkpoint_role": "technical smoke only; future full training starts afresh",
    }


def code_identity(root: Path) -> dict:
    return {name: sha256_file(root / name) for name in SOURCES}


def check_digest(payload: dict, key: str) -> None:
    if payload.get(key) != canonical_sha256({k: v for k, v in payload.items() if k != key}):
        raise ValueError(f"Changed {key}.")


def check_plan(root: Path, plan: dict, protocol: dict) -> None:
    check_digest(protocol, "protocol_sha256")
    check_digest(plan, "experiment_sha256")
    ids = plan["smoke_case_ids"]
    development = set(protocol["partition"]["development"]["case_ids"])
    reserved = set(protocol["partition"]["validation"]["case_ids"]) | set(protocol["partition"]["test"]["case_ids"])
    if (plan["block"] != 64 or plan["protocol_sha256"] != protocol["protocol_sha256"]
            or plan["partition"] != protocol["partition"] or len(ids) != 3 or len(set(ids)) != 3
            or not set(ids) <= development or development & reserved or set(plan["inputs"]) != set(ids)
            or set(plan["block63_receipts"]) != development
            or plan["validation_test_reads"] is not False or plan["settings"] != settings()):
        raise ValueError("Changed development-only feasibility scope or settings.")
    if plan["implementation"] != code_identity(root):
        raise ValueError("Block 64 implementation changed; preserve this freeze and version a new experiment.")


def verify_inputs(root: Path, plan: dict) -> None:
    # Explicit selected IDs only; never enumerate or open reserved case directories.
    for cid in plan["smoke_case_ids"]:
        if input_identity(root, cid) != plan["inputs"][cid]:
            raise ValueError(f"Frozen training input changed: {cid}")
