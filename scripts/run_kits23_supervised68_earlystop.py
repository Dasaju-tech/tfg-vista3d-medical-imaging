#!/usr/bin/env python
r"""
KiTS23 supervised training: max 100 epochs + validation early stopping.

This is a NEW, separate experiment. It reuses the verified Block 65 cache,
but starts the model from scratch with the same seed and does not touch Test120.

Default early-stopping policy:
  - max_epochs = 100
  - validate after every epoch
  - patience = 10 validation epochs
  - min_delta = 1e-4
  - monitored metric = validation selection_score already used by Block 65:
      mean of kidney/tumor/cyst mean Dice over Validation80
  - best checkpoint = highest validation selection_score (earliest epoch breaks ties)

The script is resumable at committed epoch boundaries. If interrupted during
validation, rerunning the same command resumes the pending validation receipts.

Run from the repository root, preferably without activating another env:
    conda run --no-capture-output -n vista3d-nv python .\scripts\run_kits23_supervised68_earlystop.py
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vista3d_tfg.training import contract65 as c
from vista3d_tfg.training.cache65 import CacheIndex
from vista3d_tfg.training.supervised64 import build_model, state_digest
from vista3d_tfg.training.validation65 import validate_epoch
from vista3d_tfg.training.engine65 import (
    augmented_patch,
    commit_checkpoint,
    epoch_receipt,
)
from vista3d_tfg.training.amp65b import optimizer_step
from nv_segment_ct_full489_worker import runtime_snapshot


CONFIG = "configs/experiments/kits23_supervised_block68_earlystop.json"
OUTPUT_DIRNAME = "full489_block68"
REVISION = "block68-supervised100-earlystop-v1"


def output_root(root: Path, plan: dict) -> Path:
    return root / "outputs" / "reporting" / OUTPUT_DIRNAME / plan["experiment_sha256"][:16]


# validation65 uses contract65.output_root internally. Redirect only the output
# location; cache_root/case_paths/input validation remain Block-65 compatible.
c.output_root = output_root


def _script_sha256() -> str:
    return c.sha256_file(Path(__file__).resolve())


def _protocol(root: Path) -> dict:
    path = root / "configs/experiments/nv_segment_ct_full489_protocol.json"
    protocol = c.read_json(path)
    c.check_digest(protocol, "protocol_sha256")
    return protocol


def _parent(root: Path, protocol: dict) -> dict:
    """Read the frozen Block-65 cache/input contract without modifying it."""
    parent = c.read_json(root / c.CONFIG)
    c.check_digest(parent, "experiment_sha256")

    if parent["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("Block 65 and full489 protocol hashes differ.")
    if parent["partition"] != protocol["partition"]:
        raise ValueError("Block 65 partitions differ from the current frozen full489 protocol.")
    if parent.get("test_outputs_opened") is not False:
        raise ValueError("Unexpected Block 65 test state.")
    if len(parent["partition"]["development"]["case_ids"]) != 289:
        raise ValueError("Expected Development289.")
    if len(parent["partition"]["validation"]["case_ids"]) != 80:
        raise ValueError("Expected Validation80.")
    if len(parent["partition"]["test"]["case_ids"]) != 120:
        raise ValueError("Expected sealed Test120.")
    return parent


def _new_plan(root: Path, protocol: dict, parent: dict, *, max_epochs: int,
              patience: int, min_delta: float) -> dict:
    settings = deepcopy(parent["settings"])
    settings["epochs"] = max_epochs
    settings["validation_epochs"] = list(range(1, max_epochs + 1))
    settings["schedule"] = (
        f"lr(epoch)=base_lr*0.5*(1+cos(pi*(epoch-1)/{max_epochs})); "
        "validation every epoch; early stopping enabled"
    )
    settings["early_stopping"] = {
        "monitor": "validation selection_score",
        "mode": "max",
        "patience": patience,
        "min_delta": min_delta,
        "max_epochs": max_epochs,
        "restore_policy": "select best committed validation checkpoint; training history remains intact",
    }

    plan = {
        "block": 68,
        "revision": REVISION,
        "protocol_sha256": protocol["protocol_sha256"],
        "partition": deepcopy(parent["partition"]),
        "inputs": deepcopy(parent["inputs"]),
        "settings": settings,
        "runtime_identity": deepcopy(parent["runtime_identity"]),
        "cache_parent_experiment_sha256": parent["experiment_sha256"],
        "cache_parent_config": c.CONFIG,
        "initialization": "fresh random initialization with inherited seed; no Block65/65b weights loaded",
        "script_sha256": _script_sha256(),
        "test_outputs_opened": False,
    }
    plan["experiment_sha256"] = c.canonical_sha256(plan)
    return plan


def _check_plan(root: Path, plan: dict, protocol: dict, parent: dict,
                *, max_epochs: int, patience: int, min_delta: float) -> None:
    c.check_digest(plan, "experiment_sha256")
    expected = _new_plan(
        root, protocol, parent,
        max_epochs=max_epochs, patience=patience, min_delta=min_delta,
    )
    if plan != expected:
        raise ValueError(
            "Early-stop experiment configuration/code changed. "
            "Do not silently alter a frozen run; use a new config/output."
        )
    if plan["test_outputs_opened"] is not False:
        raise ValueError("Test120 must remain sealed.")


def freeze_or_load(root: Path, *, max_epochs: int, patience: int, min_delta: float):
    protocol = _protocol(root)
    parent = _parent(root, protocol)
    config_path = root / CONFIG

    if config_path.exists():
        plan = c.read_json(config_path)
        _check_plan(
            root, plan, protocol, parent,
            max_epochs=max_epochs, patience=patience, min_delta=min_delta,
        )
    else:
        plan = _new_plan(
            root, protocol, parent,
            max_epochs=max_epochs, patience=patience, min_delta=min_delta,
        )
        _check_plan(
            root, plan, protocol, parent,
            max_epochs=max_epochs, patience=patience, min_delta=min_delta,
        )
        c.frozen_json(config_path, plan)

    out = output_root(root, plan)
    c.frozen_json(out / "protocol.json", plan)
    return protocol, parent, plan, out


def _validation_summary_path(out: Path, epoch: int) -> Path:
    return out / "validation" / f"epoch_{epoch:03d}" / "summary.json"


def _completed_epochs(out: Path, plan: dict) -> list[dict]:
    records = []
    # epoch 0 is the fresh initialization checkpoint.
    for epoch in range(0, plan["settings"]["epochs"] + 1):
        r = epoch_receipt(out, plan, epoch)
        if r is None:
            if any(
                (out / "epochs" / f"epoch_{later:03d}" / "complete.json").exists()
                for later in range(epoch + 1, plan["settings"]["epochs"] + 1)
            ):
                raise ValueError("Gap in committed early-stop checkpoint history.")
            break
        records.append(r)
    return records


def _read_validation(out: Path, plan: dict, record: dict) -> dict | None:
    epoch = record["epoch"]
    if epoch == 0:
        return None
    path = _validation_summary_path(out, epoch)
    if not path.exists():
        return None
    summary = c.read_json(path)
    c.check_digest(summary, "summary_sha256")
    if (
        summary["experiment_sha256"] != plan["experiment_sha256"]
        or summary["epoch"] != epoch
        or summary["checkpoint_sha256"] != record["sha256"]
        or summary["status"] != "complete"
        or summary["n_cases"] != 80
        or not math.isfinite(summary["selection_score"])
    ):
        raise ValueError(f"Invalid validation summary at epoch {epoch}.")
    return summary


def _early_stop_state(summaries: list[dict], patience: int, min_delta: float) -> dict:
    absolute_best = None
    reference_best = None
    bad_epochs = 0
    stopped_epoch = None

    for s in summaries:
        score = float(s["selection_score"])

        if absolute_best is None or score > float(absolute_best["selection_score"]):
            absolute_best = s

        if reference_best is None or score > float(reference_best["selection_score"]) + min_delta:
            reference_best = s
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            stopped_epoch = int(s["epoch"])
            break

    return {
        "best_epoch": None if absolute_best is None else int(absolute_best["epoch"]),
        "best_score": None if absolute_best is None else float(absolute_best["selection_score"]),
        "patience_reference_epoch": None if reference_best is None else int(reference_best["epoch"]),
        "patience_reference_score": None if reference_best is None else float(reference_best["selection_score"]),
        "epochs_without_min_delta_improvement": bad_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "stopped_epoch": stopped_epoch,
        "should_stop": stopped_epoch is not None,
    }


def _write_state(out: Path, plan: dict, records: list[dict], summaries: list[dict],
                 early: dict, *, device: str, status: str) -> dict:
    best_record = None
    if early["best_epoch"] is not None:
        best_record = next(r for r in records if r["epoch"] == early["best_epoch"])

    selection = {
        "experiment_sha256": plan["experiment_sha256"],
        "best_epoch": early["best_epoch"],
        "best_score": early["best_score"],
        "best_checkpoint_sha256": None if best_record is None else best_record["sha256"],
        "evaluated_epochs": [int(s["epoch"]) for s in summaries],
        "selection_rule": "maximum validation selection_score; earliest epoch breaks exact ties",
        "test_reads": False,
    }
    c.atomic_json(out / "selection.json", selection)
    c.atomic_json(out / "early_stopping.json", {
        **early,
        "monitor": "validation selection_score",
        "mode": "max",
        "max_epochs": plan["settings"]["epochs"],
        "test_reads": False,
    })

    last_epoch = records[-1]["epoch"] if records else -1
    steps_per_epoch = (
        len(plan["partition"]["development"]["case_ids"])
        * plan["settings"]["patches_per_case"]
    )
    result = {
        "status": status,
        "completed_epochs": last_epoch,
        "planned_max_epochs": plan["settings"]["epochs"],
        "optimizer_steps": max(0, last_epoch) * steps_per_epoch,
        "selection": selection,
        "early_stopping": early,
        "device": device,
        "test_reads": False,
    }
    c.atomic_json(out / "training_progress.json", result)
    return result


def run(root: Path, *, max_epochs: int, patience: int, min_delta: float,
        device: str = "cuda") -> dict:
    import torch
    from monai.losses import DiceCELoss

    if device != "cuda":
        raise ValueError("This production runner is intended for CUDA training.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    protocol, parent, plan, out = freeze_or_load(
        root, max_epochs=max_epochs, patience=patience, min_delta=min_delta
    )
    del protocol  # Test partition is never opened.

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime = runtime_snapshot(test_kernel=True)["identity"]
    if runtime != plan["runtime_identity"]:
        raise ValueError("CUDA/runtime identity differs from the frozen Block 65 environment.")

    index = CacheIndex(root, parent)
    cache_identity = c.canonical_sha256(index.receipts)
    c.frozen_json(out / "training_identity.json", {
        "runtime": runtime,
        "device": device,
        "cache_sha256": cache_identity,
        "cache_parent_experiment_sha256": parent["experiment_sha256"],
        "experiment_sha256": plan["experiment_sha256"],
    })

    config = plan["settings"]

    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.manual_seed_all(config["seed"])

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=128.0)
    loss_fn = DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    records = _completed_epochs(out, plan)

    if not records:
        init_stats = {
            "epoch": 0,
            "optimizer_steps": 0,
            "role": "fresh_initialization",
            "seed": config["seed"],
            "revision": REVISION,
        }
        records.append(
            commit_checkpoint(
                out, plan, runtime, model, optimizer, scaler,
                0, init_stats, device
            )
        )
    else:
        latest = records[-1]
        checkpoint = torch.load(
            out / "epochs" / f"epoch_{latest['epoch']:03d}" / latest["file"],
            map_location="cpu",
            weights_only=True,
        )
        if (
            checkpoint["experiment_sha256"] != plan["experiment_sha256"]
            or checkpoint["runtime"] != runtime
            or checkpoint["epoch"] != latest["epoch"]
            or checkpoint["stats"] != latest["stats"]
        ):
            raise ValueError("Resume checkpoint identity mismatch.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if state_digest(model) != checkpoint["model_state_sha256"]:
            raise ValueError("Resume model state mismatch.")
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint

    # Reconstruct validation/early-stop state. All epochs except possibly the
    # last committed one must already have a complete validation summary.
    summaries = []
    positive_records = [r for r in records if r["epoch"] > 0]
    for r in positive_records[:-1]:
        s = _read_validation(out, plan, r)
        if s is None:
            raise ValueError(
                f"Historical epoch {r['epoch']} lacks validation; "
                "only the latest committed epoch may have pending validation."
            )
        summaries.append(s)

    if positive_records:
        latest = positive_records[-1]
        s = _read_validation(out, plan, latest)
        if s is None:
            print(
                f"Completing pending validation for epoch {latest['epoch']:03d}...",
                flush=True,
            )
            s = validate_epoch(
                root, plan, index, model,
                latest["epoch"], latest["sha256"], device
            )
        summaries.append(s)

    early = _early_stop_state(summaries, patience, min_delta)
    if early["should_stop"]:
        result = _write_state(
            out, plan, records, summaries, early,
            device=device, status="early_stopped"
        )
        print(
            f"Early stopping already satisfied at epoch {early['stopped_epoch']}; "
            f"best epoch={early['best_epoch']} score={early['best_score']:.6f}",
            flush=True,
        )
        return result

    ids = c.partition_ids(plan, "development")
    start_epoch = records[-1]["epoch"] + 1

    for epoch in range(start_epoch, max_epochs + 1):
        started = time.perf_counter()
        model.train()
        torch.cuda.reset_peak_memory_stats()

        rng = np.random.default_rng(config["seed"] + epoch)
        ordered = [ids[int(i)] for i in rng.permutation(len(ids))]

        lr = (
            config["optimizer"]["lr"]
            * 0.5
            * (1.0 + math.cos(math.pi * (epoch - 1) / max_epochs))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        losses = []
        case_losses = []
        class_centers = {str(k): 0 for k in range(4)}
        amp_retries = 0
        scale_start = float(scaler.get_scale())

        for number, cid in enumerate(ordered):
            image, label, counts, _ = index.load("development", cid)
            case_values = []

            for patch_index in range(config["patches_per_case"]):
                x_np, y_np, k = augmented_patch(image, label, counts, config, rng)
                x = torch.from_numpy(x_np)[None, None].to(device)
                y = torch.from_numpy(y_np)[None, None].to(device)

                def record_retry(event):
                    payload = {
                        **event,
                        "epoch": epoch,
                        "case_id": cid,
                        "patch_index": patch_index,
                        "experiment_sha256": plan["experiment_sha256"],
                        "revision": REVISION,
                    }
                    c.atomic_json(
                        out / "amp_events" / ("event_" + uuid4().hex + ".json"),
                        payload,
                    )
                    print(
                        f"  AMP retry {event['retry']}: {cid}; "
                        f"scale {event['scale_before']:g} -> {event['scale_after']:g}; "
                        "same patch",
                        flush=True,
                    )

                try:
                    value, retries = optimizer_step(
                        model, optimizer, scaler, loss_fn, x, y,
                        config["max_gradient_norm"], device,
                        on_retry=record_retry,
                    )
                except (ValueError, RuntimeError) as exc:
                    raise RuntimeError(
                        f"Epoch {epoch}, case {cid}, patch {patch_index + 1}: {exc}"
                    ) from exc

                amp_retries += retries
                losses.append(value)
                case_values.append(value)
                class_centers[str(k)] += 1
                del x, y

            case_mean = float(np.mean(case_values))
            case_losses.append({"case_id": cid, "mean_loss": case_mean})
            print(
                f"Epoch {epoch:03d}/{max_epochs:03d} "
                f"[{number + 1:03d}/{len(ids):03d}] {cid}; "
                f"loss={case_mean:.6f}",
                flush=True,
            )
            c.atomic_json(out / "training_progress.json", {
                "status": "running",
                "epoch_in_progress": epoch,
                "last_committed_epoch": epoch - 1,
                "processed_cases_this_epoch": number + 1,
                "max_epochs": max_epochs,
                "test_reads": False,
            })

        torch.cuda.synchronize()
        stats = {
            "epoch": epoch,
            "optimizer_steps": epoch * len(ids) * config["patches_per_case"],
            "steps_this_epoch": len(losses),
            "effective_optimizer_steps_this_epoch": len(losses),
            "amp_retries": amp_retries,
            "amp_scale_start": scale_start,
            "amp_scale_end": float(scaler.get_scale()),
            "mean_loss": float(np.mean(losses)),
            "lr": lr,
            "case_order": ordered,
            "case_losses": case_losses,
            "class_centers": class_centers,
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "revision": REVISION,
        }

        latest = commit_checkpoint(
            out, plan, runtime, model, optimizer, scaler,
            epoch, stats, device
        )
        records.append(latest)

        summary = validate_epoch(
            root, plan, index, model,
            epoch, latest["sha256"], device
        )
        summaries.append(summary)

        early = _early_stop_state(summaries, patience, min_delta)
        print(
            f"Epoch {epoch:03d}: validation score={summary['selection_score']:.6f}; "
            f"best={early['best_score']:.6f} @ {early['best_epoch']}; "
            f"no-improve={early['epochs_without_min_delta_improvement']}/{patience}",
            flush=True,
        )

        if early["should_stop"]:
            result = _write_state(
                out, plan, records, summaries, early,
                device=device, status="early_stopped"
            )
            print(
                f"EARLY STOP: epoch {epoch}; "
                f"best epoch={early['best_epoch']} score={early['best_score']:.6f}",
                flush=True,
            )
            print(f"Output: {out}", flush=True)
            print("Test120 remains sealed.", flush=True)
            return result

        _write_state(
            out, plan, records, summaries, early,
            device=device, status="running"
        )

    early = _early_stop_state(summaries, patience, min_delta)
    result = _write_state(
        out, plan, records, summaries, early,
        device=device, status="max_epochs_reached"
    )
    print(
        f"MAX 100 EPOCHS REACHED; best epoch={early['best_epoch']} "
        f"score={early['best_score']:.6f}",
        flush=True,
    )
    print(f"Output: {out}", flush=True)
    print("Test120 remains sealed.", flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    if args.max_epochs != 100:
        parser.error("This protocol is intentionally fixed to --max-epochs 100.")
    if args.patience < 1:
        parser.error("--patience must be >= 1.")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error("--min-delta must be finite and >= 0.")

    root = args.project_root.resolve()
    if root != ROOT:
        parser.error(
            f"This script belongs to checkout {ROOT}; "
            "copy it into the target repository's scripts directory."
        )

    try:
        run(
            root,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            device="cuda",
        )
        return 0
    except KeyboardInterrupt:
        print(
            "Interrupted. Re-run the same command; the last complete epoch is preserved.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Block 68 early-stop FAILED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
