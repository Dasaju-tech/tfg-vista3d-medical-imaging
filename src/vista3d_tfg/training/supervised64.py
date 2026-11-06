"""Memory-bounded patch training; torch and MONAI are imported only by training."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import random
import time

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from vista3d_tfg.training.contract64 import atomic_json, sha256_file


def validate_pair(image: nib.spatialimages.SpatialImage, label: nib.spatialimages.SpatialImage):
    if (len(image.shape) != 3 or image.shape != label.shape or any(n < 1 for n in image.shape)
            or not np.isfinite(image.affine).all() or not np.isfinite(label.affine).all()
            or not np.allclose(image.affine, label.affine, atol=1e-5, rtol=0)
            or abs(np.linalg.det(image.affine[:3, :3])) < 1e-8):
        raise ValueError("CT/reference geometry mismatch or invalid affine.")
    # Read each compressed volume once, then validate scaled values in RAM slabs.
    image_data = image.get_fdata(dtype=np.float32, caching="unchanged")
    label_data = np.asanyarray(label.dataobj)
    for z in range(0, image.shape[2], 16):
        ct = image_data[:, :, z:z+16]
        gt = label_data[:, :, z:z+16]
        if not np.isfinite(ct).all() or not np.isfinite(gt).all() or not np.isin(gt, (0, 1, 2, 3)).all():
            raise ValueError("Non-finite CT or invalid KiTS23 reference labels.")
    return image_data, label_data.astype(np.uint8)


def preprocess(image_path: Path, label_path: Path, config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    image, label = nib.load(image_path), nib.load(label_path)
    if int(np.prod(image.shape, dtype=np.int64)) > config["max_original_voxels"]:
        raise ValueError("Original grid exceeds frozen RAM guard.")
    image_data, label_data = validate_pair(image, label)
    original_shape = list(image.shape)
    image = nib.as_closest_canonical(nib.Nifti1Image(image_data, image.affine))
    label = nib.as_closest_canonical(nib.Nifti1Image(label_data, label.affine))
    spacing = np.asarray(config["spacing_mm"], dtype=float)
    old_spacing = nib.affines.voxel_sizes(image.affine)
    shape = tuple(int(v) for v in np.ceil((np.asarray(image.shape) - 1) * old_spacing / spacing) + 1)
    if int(np.prod(shape, dtype=np.int64)) > config["max_resampled_voxels"]:
        raise ValueError(f"Resampled grid exceeds frozen RAM guard: {shape}")
    affine = image.affine.copy()
    affine[:3, :3] *= spacing / old_spacing
    target = (shape, affine)
    # Force floating CT interpolation even when the source storage dtype is integer.
    image_float = nib.Nifti1Image(image.get_fdata(dtype=np.float32, caching="unchanged"), image.affine)
    ct = np.asarray(resample_from_to(image_float, target, order=1, cval=-200.0).dataobj, dtype=np.float32)
    gt = np.asarray(resample_from_to(label, target, order=0, cval=0).dataobj)
    if not np.isfinite(ct).all() or not np.isin(gt, (0, 1, 2, 3)).all():
        raise ValueError("Invalid resampled CT or reference.")
    gt = gt.astype(np.uint8)
    low, high = config["ct_window_hu"]
    np.clip(ct, low, high, out=ct)
    ct -= low
    ct *= 2.0 / (high - low)
    ct -= 1.0
    labels = np.unique(gt).tolist()
    if labels != [0, 1, 2, 3]:
        raise ValueError(f"Selected smoke case lost a reference class after resampling: {labels}. No case is substituted.")
    info = {"original_shape": original_shape, "resampled_shape": list(shape),
            "resampled_affine": affine.tolist(), "labels": labels,
            "class_voxels": {str(k): int(np.count_nonzero(gt == k)) for k in range(4)}}
    return ct, gt, info


def sample_patch(ct: np.ndarray, gt: np.ndarray, size, class_id: int, rng: np.random.Generator):
    # Select a uniform voxel of the class using slice counts; avoid a full-volume
    # int64 background index array, which can exceed the CT's own memory usage.
    counts = np.array([np.count_nonzero(gt[:, :, z] == class_id) for z in range(gt.shape[2])], dtype=np.int64)
    total = int(counts.sum())
    if not total:
        raise ValueError(f"Requested sampling class absent: {class_id}")
    rank = int(rng.integers(total))
    cumulative = counts.cumsum()
    z = int(np.searchsorted(cumulative, rank, side="right"))
    offset = rank - (int(cumulative[z-1]) if z else 0)
    flat = np.flatnonzero(gt[:, :, z] == class_id)[offset]
    center = (*np.unravel_index(flat, gt.shape[:2]), z)
    starts = [int(c) - int(n)//2 for c, n in zip(center, size)]
    src, dst = [], []
    for start, n, dim in zip(starts, size, gt.shape):
        lower, upper = max(start, 0), min(start + n, dim)
        src.append(slice(lower, upper))
        dst.append(slice(lower - start, upper - start))
    x = np.full(size, -1.0, dtype=np.float32)
    y = np.zeros(size, dtype=np.int64)
    x[tuple(dst)] = ct[tuple(src)]
    y[tuple(dst)] = gt[tuple(src)]
    if y[tuple(int(n)//2 for n in size)] != class_id:
        raise ValueError("Sampling center label changed.")
    return x, y, [int(c) for c in center]


def build_model(config: dict):
    from monai.networks.nets import UNet
    kwargs = {k: v for k, v in config["model"].items() if k != "architecture"}
    return UNet(**kwargs)


def state_digest(model) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_smoke(plan: dict, pairs: list[tuple[str, Path, Path]], output: Path, *, device: str = "cuda") -> dict:
    import torch
    from monai.losses import DiceCELoss

    config = plan["settings"]
    if [cid for cid, _, _ in pairs] != plan["smoke_case_ids"]:
        raise ValueError("Training pairs differ from frozen smoke selection.")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    cuda = device == "cuda"
    if cuda:
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model = build_model(config).to(device).train()
    initial_digest = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"],
                                  weight_decay=config["optimizer"]["weight_decay"])
    loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True,
                         lambda_dice=1.0, lambda_ce=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda, init_scale=128.0)
    losses, trace, volumes, observed = [], [], [], set()
    output.mkdir(parents=True, exist_ok=True)
    for case_index, (cid, image_path, label_path) in enumerate(pairs):
        print(f"[{case_index+1:02d}/{len(pairs):02d}] prepare training patches {cid}", flush=True)
        prep_start = time.perf_counter()
        ct, gt, info = preprocess(image_path, label_path, config)
        volumes.append({"case_id": cid, "preprocessing_seconds": time.perf_counter() - prep_start, **info})
        for local_step in range(config["steps_per_case"]):
            target_class = local_step % 4
            x_np, y_np, center = sample_patch(ct, gt, config["patch_size"], target_class, rng)
            observed.update(int(v) for v in np.unique(y_np))
            x = torch.from_numpy(x_np)[None, None].to(device)
            y = torch.from_numpy(y_np)[None, None].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cuda):
                logits = model(x)
            if logits.shape != (1, 4, *config["patch_size"]):
                raise ValueError("Unexpected four-class output shape.")
            # Surface/loss reductions stay in FP32, outside autocast.
            loss = loss_fn(logits.float(), y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite training loss.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_gradient_norm"], error_if_nonfinite=True)
            if float(norm) <= 0:
                raise ValueError("Zero gradient: optimizer update not demonstrated.")
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            trace.append({"step": len(losses), "case_id": cid, "center_class": target_class,
                          "center_resampled_voxel": center, "loss": losses[-1], "gradient_norm": float(norm)})
            atomic_json(output / "trace.json", {"steps": trace, "volumes": volumes})
            print(f"  step {len(losses):02d}/{len(pairs)*config['steps_per_case']}; class={target_class}; loss={losses[-1]:.6f}", flush=True)
            del x, y, logits, loss
        del ct, gt
    final_digest = state_digest(model)
    if initial_digest == final_digest or observed != {0, 1, 2, 3}:
        raise ValueError("No parameter change or incomplete observed target classes.")
    # This smoke checkpoint is intentionally not a resumable research training run.
    payload = {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "experiment_sha256": plan["experiment_sha256"], "optimizer_steps": len(losses),
               "role": "technical_smoke_only", "settings": config}
    checkpoint = output / "checkpoint.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = build_model(config)
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    if state_digest(restored) != final_digest or loaded["experiment_sha256"] != plan["experiment_sha256"]:
        raise ValueError("Checkpoint reload parity failed.")
    if cuda:
        torch.cuda.synchronize()
    return {"status": "complete", "experiment_sha256": plan["experiment_sha256"],
            "case_ids": plan["smoke_case_ids"], "optimizer_steps": len(losses), "losses": losses,
            "parameters_changed": True, "initial_state_sha256": initial_digest, "final_state_sha256": final_digest,
            "checkpoint_reload_exact": True, "checkpoint_sha256": sha256_file(checkpoint),
            "parameter_count": sum(p.numel() for p in model.parameters()), "observed_target_labels": sorted(observed),
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
            "memory_scope": "PyTorch allocator; not physical VRAM residency", "device": device,
            "validation_test_reads": False, "quality_evaluation_performed": False}
