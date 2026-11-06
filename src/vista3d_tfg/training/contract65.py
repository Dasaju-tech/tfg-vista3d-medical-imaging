"""Frozen supervised development/validation contract; no test execution route."""
from __future__ import annotations

from pathlib import Path
import math
import re

from vista3d_tfg.training.contract64 import (
    atomic_json, canonical_sha256, check_digest, frozen_json, input_identity,
    read_json, sha256_file, settings as smoke_settings,
)

CONFIG = "configs/experiments/kits23_supervised_block65.json"
WORKER = "scripts/kits23_supervised65_worker.py"
SOURCES = (
    WORKER, "scripts/run_kits23_supervised65.py",
    "src/vista3d_tfg/experiments/supervised65.py",
    "src/vista3d_tfg/training/contract65.py", "src/vista3d_tfg/training/cache65.py",
    "src/vista3d_tfg/training/engine65.py", "src/vista3d_tfg/training/validation65.py",
    "src/vista3d_tfg/training/contract64.py", "src/vista3d_tfg/training/supervised64.py",
    "src/vista3d_tfg/training/__init__.py", "scripts/nv_segment_ct_full489_worker.py",
)


def settings() -> dict:
    old = smoke_settings()
    return {
        "seed": 6501, "model": old["model"], "initialization": "random; never load Block 64 or NV weights",
        "labels": old["labels"], "patch_size": [96, 96, 96], "batch_size": 1,
        "spacing_mm": [1.5, 1.5, 1.5], "ct_window_hu": [-200.0, 300.0],
        "max_original_voxels": 512000000, "max_resampled_voxels": 64000000,
        "preprocessing": "closest RAS, native obliquity; linear CT, nearest labels; normalize [-1,1]",
        "cache_dtype": "image float32, training label uint8; no held-out label cache",
        "epochs": 50, "patches_per_case": 8,
        "sampling": "uniform case permutation each epoch; uniform class among present classes then uniform voxel",
        "augmentation": {"flip_probability_per_axis": 0.5, "intensity_scale": [0.9, 1.1], "intensity_shift": [-0.1, 0.1]},
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.00001},
        "schedule": "lr(epoch)=base_lr*0.5*(1+cos(pi*(epoch-1)/50)); no early stopping",
        "loss": "foreground soft Dice + four-class cross entropy; both weight 1; FP32 reductions",
        "max_gradient_norm": 12.0, "amp": "CUDA FP16; GradScaler initial scale 128",
        "determinism": "seeded; deterministic algorithms warn_only=True; not bitwise CUDA reproducibility",
        "resume": "epoch boundaries; model, optimizer, scaler and torch CPU/CUDA RNG; interrupted epoch repeats",
        "validation_epochs": [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "inference": {"roi_size": [96, 96, 96], "overlap": 0.25, "sw_batch_size": 2,
                      "blend": "gaussian logits", "accumulator": "CPU FP32", "tta": False,
                      "restore": "argmax on resampled grid, then nearest to original CT grid", "postprocessing": "none"},
        "selection": "maximum mean over validation cases of mean Dice for labels 1,2,3; earliest epoch breaks ties",
        "empty_class_rule": "both masks empty: Dice=1; exactly one empty: Dice=0",
        "secondary": "reference-positive class Dice, reference-negative predicted mL, renal-envelope Dice",
        "test_inference_enabled": False,
    }


def code_identity(root: Path) -> dict:
    return {name: sha256_file(root / name) for name in SOURCES}


def settings_for_grids(audit: dict, expected_ids: list[str]) -> dict:
    """Freeze finite allocation bounds from all included CT headers, not a pilot."""
    if not expected_ids or set(audit) != set(expected_ids):
        raise ValueError("Grid audit must cover every development/validation case.")
    maxima = {"original_shape": 0, "resampled_shape": 0}
    for cid, entry in audit.items():
        if set(entry) != set(maxima):
            raise ValueError(f"Invalid grid audit fields: {cid}")
        for key in maxima:
            shape = entry[key]
            if (not isinstance(shape, list) or len(shape) != 3
                    or any(type(n) is not int or n < 1 for n in shape)):
                raise ValueError(f"Invalid positive 3D grid: {cid}/{key}")
            count = math.prod(shape)
            if count > (2**63 - 1) // 40:
                raise ValueError(f"Grid exceeds addressable allocation size: {cid}/{key}")
            maxima[key] = max(maxima[key], count)
    result = settings()
    result["max_original_voxels"] = maxima["original_shape"]
    result["max_resampled_voxels"] = maxima["resampled_shape"]
    return result


def partition_ids(plan: dict, partition: str) -> list[str]:
    if partition not in ("development", "validation"):
        raise ValueError("Only development/validation are enabled; test is sealed.")
    return plan["partition"][partition]["case_ids"]


def check_plan(root: Path, plan: dict, protocol: dict) -> None:
    check_digest(protocol, "protocol_sha256")
    check_digest(plan, "experiment_sha256")
    groups = [protocol["partition"][k]["case_ids"] for k in ("development", "validation", "test")]
    expected_settings = settings_for_grids(plan.get("grid_audit", {}), groups[0] + groups[1])
    if (plan.get("block") != 65 or plan["protocol_sha256"] != protocol["protocol_sha256"]
            or plan["partition"] != protocol["partition"] or plan["settings"] != expected_settings
            or any(not g or len(set(g)) != len(g) for g in groups)
            or any(not re.fullmatch(r"case_[0-9]{5}", c) for g in groups for c in g)
            or sum(map(len, groups)) != len(set().union(*map(set, groups)))
            or set(plan["inputs"]) != set(groups[0] + groups[1])):
        raise ValueError("Changed Block 65 scope, split or settings.")
    if plan["implementation"] != code_identity(root):
        raise ValueError("Block 65 source changed; keep the freeze and version a new experiment.")


def output_root(root: Path, plan: dict) -> Path:
    return root / "outputs/reporting/full489_block65" / plan["experiment_sha256"][:16]


def cache_root(root: Path, plan: dict) -> Path:
    return root / "data/processed/kits23_supervised65" / plan["experiment_sha256"][:16]


def case_paths(root: Path, plan: dict, partition: str, cid: str) -> tuple[Path, Path]:
    if cid not in partition_ids(plan, partition):
        raise ValueError("Case is outside the requested frozen partition.")
    folder = root / "data/raw/kits23" / cid
    return folder / "imaging.nii.gz", folder / "segmentation.nii.gz"


def verify_input(root: Path, plan: dict, partition: str, cid: str) -> None:
    case_paths(root, plan, partition, cid)
    if input_identity(root, cid) != plan["inputs"][cid]:
        raise ValueError(f"Frozen source input changed: {cid}")
