"""Revisión de tres TAC de Validation80 con el checkpoint ct de la época 19."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import run_kits23_supervised72_metrics as block72


ANALYSIS_SHA256 = "b74a9dd0f29291e237a34d7809f86e4bc8282e07cbfeff84071fa1d01dc624b6"
ARM = "ct"
CASE_PURPOSES = {
    "case_00036": "Dice alto y HD95 alto",
    "case_00223": "HD95 extremo",
    "case_00581": "Control con HD95 bajo",
}
DIRECTIONS = {
    "prediction_to_reference": "Predicción → referencia",
    "reference_to_prediction": "Referencia → predicción",
}


def read_block72_report(path, evidence, *, validation80=False):
    with ZipFile(path) as archive:
        protocol = json.loads(archive.read("protocol.json"))
        evidence.check(protocol, "analysis_sha256")
        if (
            protocol["analysis_sha256"] != ANALYSIS_SHA256
            or protocol["test_reads"] is not False
            or protocol["definition"] != block72.DEFINITION
            or not set(CASE_PURPOSES).issubset(protocol["validation_case_ids"])
        ):
            raise ValueError("El ZIP debe corresponder al análisis b74a9dd0f29291e2 del Bloque 72B.")
        receipts = {}
        case_ids = protocol["validation_case_ids"] if validation80 else CASE_PURPOSES
        for case_id in case_ids:
            receipt = json.loads(archive.read(f"{ARM}/{case_id}/metrics.json"))
            evidence.check(receipt, "report_sha256")
            if (
                receipt["case_id"] != case_id
                or receipt["identity"]["analysis_sha256"] != ANALYSIS_SHA256
                or receipt["identity"]["arm"] != ARM
                or receipt["test_reads"] is not False
                or not receipt["matches_source_prediction"]
                or receipt["prediction_array_sha256"] != receipt["source_prediction_array_sha256"]
            ):
                raise ValueError(f"El registro de {case_id} no corresponde a la máscara verificada del Bloque 72B.")
            receipts[case_id] = receipt
    return protocol, receipts


def directed_p95(source, target):
    distances = block72.nearest_distances(source, target, p=2)
    p95 = float(np.percentile(distances, 95, method="linear"))
    sample_index = int(np.argmin(np.abs(distances - p95)))
    sample = source[sample_index]
    nearest = target[int(np.argmin(np.linalg.norm(target - sample, axis=1)))]
    return {
        "p95_mm": p95,
        "plotted_distance_mm": float(distances[sample_index]),
        "source_ras_mm": sample.tolist(),
        "nearest_ras_mm": nearest.tolist(),
    }


def measure_directions(reference, prediction, affine):
    ref_surface = block72.surface_points(reference, affine)
    pred_surface = block72.surface_points(prediction, affine)
    return {
        "prediction_to_reference": directed_p95(pred_surface, ref_surface),
        "reference_to_prediction": directed_p95(ref_surface, pred_surface),
    }


def render_overlay(image, reference, prediction, directions, row, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

    orientation = io_orientation(image.affine)
    to_native = inv_ornt_aff(orientation, image.shape)
    display_affine = image.affine @ to_native
    spacing = np.linalg.norm(display_affine[:3, :3], axis=0)
    ct, ref, pred = (
        apply_orientation(array, orientation)
        for array in (np.asanyarray(image.dataobj), reference, prediction)
    )
    colors = ListedColormap(["#101720", "#65b8f2", "#ff9e45", "#63c69e"])
    planes = ("Axial", "Coronal", "Sagital")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor="#101720")
    try:
        for panels, direction in zip(axes, DIRECTIONS):
            item = directions[direction]
            world = np.asarray(item["source_ras_mm"])
            center = np.rint(np.linalg.solve(display_affine[:3, :3], world - display_affine[:3, 3])).astype(int)
            for ax, name, axis in zip(panels, planes, (2, 1, 0)):
                horizontal, vertical = [dim for dim in range(3) if dim != axis]
                aspect = spacing[vertical] / spacing[horizontal]
                ct_slice, ref_slice, pred_slice = (
                    np.take(array, center[axis], axis=axis).T for array in (ct, ref, pred)
                )
                overlay = ref_slice.astype(np.uint8) + 2 * pred_slice.astype(np.uint8)
                ax.imshow(ct_slice, cmap="gray", vmin=-200, vmax=300, origin="lower", aspect=aspect)
                ax.imshow(np.ma.masked_equal(overlay, 0), cmap=colors, vmin=0, vmax=3, alpha=.62,
                          origin="lower", aspect=aspect, interpolation="nearest")
                ax.plot(center[horizontal], center[vertical], "+", color="white", ms=12, mew=1.5)
                ax.set_title(name, color="white", fontsize=10)
                ax.set_axis_off()
        for y, direction in zip((.878, .475), DIRECTIONS):
            item = directions[direction]
            fig.text(.5, y, f"{DIRECTIONS[direction]}   ·   P95 {item['p95_mm']:.1f} mm",
                     ha="center", color="white", fontsize=12)
        fig.suptitle(
            f"{row['case_id']}  ·  ct, época {row['epoch']}\n"
            f"Dice {row['dice']:.3f}   ·   HD95 {row['hd95_mm']:.1f} mm",
            color="white", fontsize=15, y=.98,
        )
        handles = [
            Patch(color=colors(3), label="Coincidencia"),
            Patch(color=colors(2), label="Exceso en predicción"),
            Patch(color=colors(1), label="Tejido de referencia omitido"),
            Line2D([], [], marker="+", color="white", linestyle="none", label="Punto próximo al P95"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, labelcolor="white")
        fig.subplots_adjust(left=.025, right=.975, top=.838, bottom=.075, wspace=.04, hspace=.42)
        fig.savefig(path, dpi=125, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)


def replay_case(replay, case_id):
    import nibabel as nib
    from vista3d_tfg.training.cache65 import verified_cache

    root, runner, parent = replay.root, replay.runner, replay.parent
    plan, checkpoint = replay.plan, replay.checkpoint
    receipt = replay.receipts[case_id]
    runner.c.verify_input(root, parent, "validation", case_id)
    cache = verified_cache(root, parent, "validation", case_id)
    if cache is None:
        raise FileNotFoundError(f"Falta la caché de validación de {case_id}.")
    index = SimpleNamespace(receipts={case_id: cache["receipt_sha256"]})
    identity = runner.validation_identity(plan, index, ARM, checkpoint, case_id)
    epoch = checkpoint["epoch"]
    old = runner.case_metrics(replay.study / "validation" / f"epoch_{epoch:03d}" / case_id, identity)
    expected_sha = receipt["prediction_array_sha256"]
    if (
        identity != receipt["identity"]["source_identity"]
        or old is None
        or old[0]["prediction_array_sha256"] != expected_sha
    ):
        raise ValueError(f"La procedencia de {case_id} difiere entre los bloques 71 y 72B.")

    cache_folder = runner.c.cache_root(root, parent) / "validation" / case_id
    meta = runner.b.read(cache_folder / "meta.json")
    cached_ct = np.load(cache_folder / "image.npy", mmap_mode="r", allow_pickle=False)
    if cached_ct.dtype != np.float32 or list(cached_ct.shape) != meta["shape"]:
        raise ValueError(f"Geometría o tipo inesperados en la caché de {case_id}.")
    image_path, ref_path = runner.c.case_paths(root, parent, "validation", case_id)
    image, reference = nib.load(image_path), nib.load(ref_path)
    geometry = block72.native_geometry_mm(image, reference)
    affine = np.asarray(meta["original_affine"])
    if (
        geometry != replay.protocol["validation_native_geometry"][case_id]
        or geometry["shape"] != meta["original_shape"]
        or not np.allclose(reference.affine, affine, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"La geometría original de {case_id} difiere del Bloque 72B.")

    prediction = runner.predict_original(replay.model, cached_ct, meta, plan["settings"], "cuda")
    prediction_sha = runner.array_sha(prediction)
    if prediction_sha != expected_sha:
        raise ValueError(f"La predicción de {case_id} tiene hash {prediction_sha}. Se esperaba {expected_sha}.")
    return image, reference, prediction, affine, geometry


def review_case(replay, case_id, folder):
    receipt = replay.receipts[case_id]
    image, reference, prediction, affine, geometry = replay_case(replay, case_id)
    labels = block72.LABELS["renal_envelope"]
    ref_mask = np.isin(np.asanyarray(reference.dataobj), labels)
    pred_mask = np.isin(prediction, labels)
    del prediction
    directions = measure_directions(ref_mask, pred_mask, affine)
    hd95_mm = max(item["p95_mm"] for item in directions.values())
    metrics = receipt["metrics"]["renal_envelope"]
    if not math.isclose(hd95_mm, metrics["hd95_euclidean_mm"], rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"El HD95 recalculado de {case_id} difiere del Bloque 72B.")
    row = {
        "case_id": case_id,
        "purpose": CASE_PURPOSES[case_id],
        "epoch": replay.checkpoint["epoch"],
        "dice": metrics["dice"],
        "hd95_mm": hd95_mm,
        "prediction_to_reference_p95_mm": directions["prediction_to_reference"]["p95_mm"],
        "reference_to_prediction_p95_mm": directions["reference_to_prediction"]["p95_mm"],
        "false_positive_ml": metrics["false_positive_ml"],
        "false_negative_ml": metrics["false_negative_ml"],
    }
    render_overlay(image, ref_mask, pred_mask, directions, row, folder / f"{case_id}.png")
    replay.runner.c.verify_input(replay.root, replay.parent, "validation", case_id)
    return row, {"prediction_array_sha256": receipt["prediction_array_sha256"],
                 "directions": directions, "geometry": geometry}


@contextmanager
def open_replay(root, report_path, *, validation80=False):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Ejecuta el script en el entorno vista3d-nv con CUDA.")
    with block72.checked_source(root) as (runner, parent, plan, _), runner.execution_lock(root):
        protocol, receipts = read_block72_report(report_path, runner.b, validation80=validation80)
        if (
            protocol["source_experiment_sha256"] != plan["experiment_sha256"]
            or protocol["validation_case_ids"] != plan["partition"]["validation"]["case_ids"]
        ):
            raise ValueError("El ZIP y el experimento local del Bloque 71 tienen distinto origen.")
        study = runner.arm_root(root, plan, ARM)
        selected = protocol["selected"][ARM]
        epoch = selected["best_epoch"]
        checkpoint = runner.checkpoint_receipt(study, plan, ARM, epoch)
        if checkpoint is None or checkpoint["sha256"] != selected["checkpoint_sha256"]:
            raise ValueError(f"Falta el checkpoint verificado de ct en la época {epoch}.")
        checkpoint_path = study / "epochs" / f"epoch_{epoch:03d}" / checkpoint["file"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        identity = {"experiment_sha256": plan["experiment_sha256"], "arm": ARM, "epoch": epoch,
                    "stats": checkpoint["stats"], "runtime": plan["runtime"]}
        if any(payload[key] != value for key, value in identity.items()):
            raise ValueError("La identidad del checkpoint difiere del Bloque 71.")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        model = runner.initialized_model(plan["settings"]["model"], ARM, plan["settings"]["seed"]).to("cuda")
        model.load_state_dict(payload["model"], strict=True)
        if runner.b.state_digest(model) != payload["model_state_sha256"]:
            raise ValueError("Los pesos cargados no coinciden con el checkpoint.")
        del payload
        yield SimpleNamespace(root=root, runner=runner, parent=parent, plan=plan, model=model,
                              study=study, checkpoint=checkpoint, protocol=protocol, receipts=receipts)


def run(root, report_path):
    with open_replay(root, report_path) as replay:
        output = root / "outputs/reporting/full489_block73_hd95_qc" / ANALYSIS_SHA256[:16]
        output.mkdir(parents=True, exist_ok=True)
        target = output / "block73_hd95_qc_report.zip"
        rows, details = [], {}
        with TemporaryDirectory(dir=output) as tmp:
            folder = Path(tmp)
            for number, case_id in enumerate(replay.receipts, 1):
                print(f"[{number}/{len(replay.receipts)}] {case_id} · {CASE_PURPOSES[case_id]}", flush=True)
                row, info = review_case(replay, case_id, folder)
                rows.append(row)
                details[case_id] = info
                print(f"  Hash verificado · HD95 {row['hd95_mm']:.1f} mm · "
                      f"predicción→referencia {row['prediction_to_reference_p95_mm']:.1f} mm · "
                      f"referencia→predicción {row['reference_to_prediction_p95_mm']:.1f} mm", flush=True)
            with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            replay.runner.b.write(folder / "qc_report.json", {
                "source_analysis_sha256": ANALYSIS_SHA256,
                "checkpoint_sha256": replay.checkpoint["sha256"],
                "partition": "Validation80",
                "test_reads": False,
                "sampling": "One surface point with a distance closest to each directional P95",
                "cases": details,
            })
            pending = folder / target.name
            with ZipFile(pending, "w", ZIP_DEFLATED) as archive:
                for artifact in sorted(folder.iterdir()):
                    if artifact != pending:
                        archive.write(artifact, artifact.name)
            os.replace(pending, target)
        print(f"Share this ZIP: {target}", flush=True)


def main():
    root = Path(__file__).resolve().parents[1]
    default_report = (root / "outputs/reporting/full489_block72_metrics" / ANALYSIS_SHA256[:16]
                      / "block72_validation_metrics_report.zip")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=default_report, help="Ruta al ZIP del Bloque 72B.")
    args = parser.parse_args()
    if not args.report.is_file():
        raise FileNotFoundError(f"No encuentro el ZIP del Bloque 72B en {args.report}. Indica su ruta con --report.")
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    run(root, args.report)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Block 73 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
