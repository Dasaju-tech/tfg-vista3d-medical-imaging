#!/usr/bin/env python
r"""Block 69: development-only feasibility of CT + soft 3D edges and rotations.

Copy this NEW file to scripts/. Run in vista3d-nv from the repository root:
  python .\scripts\run_kits23_supervised69_edges_smoke.py --stage self-test
  python .\scripts\run_kits23_supervised69_edges_smoke.py --stage smoke

The smoke is 3 class-complete development cases, 8 updates each, patch 96^3,
new 2-channel UNet, DiceCE loss. It is NOT a validation score or a comparison
with Block 68. No validation/test volumes or prior model weights are loaded.
Completed runs are verified/reused; interrupted attempts restart all 24 updates
from the same seed. Previous attempts remain available for inspection.

Image pipeline: cached CT [-1,1] -> paired physical rotation (CT linear, labels
nearest) on a larger patch -> Gaussian derivatives -> center crop -> [CT,edge].
Sigma=1.5 mm, truncate=4; edge=clip(sigma*|grad_world CT|,0,1). Fixed scaling,
no GT-derived edges/normalization, no binary edge threshold. The magnitude is
nonnegative; the preview also shows signed dCT/dR, which is NOT a model input.
The Gaussian halo avoids patch-crop-induced edges. Orthogonal oblique grids
are supported; a sheared grid is rejected rather than claiming isotropic
physical smoothing. All rotation angles are pull-sampling RAS angles.

This smoke deliberately uses class cycling and no intensity/flipping augments;
its loss curve cannot be compared with full-cohort training. Full ablations,
new losses, and 100-epoch early-stop training belong to the next block.

References:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
https://github.com/Project-MONAI/MONAI/blob/1.5.0/monai/networks/nets/unet.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4
import zipfile

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
CONFIG65 = "configs/experiments/kits23_supervised_block65.json"
CONFIG68 = "configs/experiments/kits23_supervised_block68_earlystop.json"
CONFIG69 = "configs/experiments/kits23_supervised_block69_edges_smoke.json"
REVISION = "block69-soft-edges-smoke-v1"
EXPECTED_CACHE = {"image.npy", "label.npy", "slice_counts.npy", "meta.json"}
POLICY = {
    "seed": 6901, "cases": 3, "steps_per_case": 8, "patch_size": [96, 96, 96],
    "sigma_mm": 1.5, "truncate": 4.0, "rotation_degrees_per_world_axis": 10.0,
    "rotation_probability": 1.0, "rotation_order": "pull Rz @ Ry @ Rx in RAS",
    "edge_scale": "clip(sigma_mm * world_gradient_magnitude, 0, 1)",
    "padding": {"CT": -1.0, "label": 0}, "loss": "DiceCE: foreground Dice + four-class CE, weights 1/1",
    "sampling": "deterministic SHA256-ranked development cases with all 4 classes; cycle 0,1,2,3 twice",
    "extra_augmentation": "none; no intensity jitter or flips in this feasibility smoke",
    "checkpoint": "fresh two-channel model; feasibility only; never resume full research training",
}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): d.update(b)
    return d.hexdigest()


def read(path):
    with Path(path).open(encoding="utf-8-sig") as f: obj = json.load(f)
    if not isinstance(obj, dict): raise ValueError(f"Expected JSON object: {path}")
    return obj


def check(obj, field):
    body = {k: v for k, v in obj.items() if k != field}
    if obj.get(field) != digest(body): raise ValueError(f"Invalid {field}")


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
        os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)


def freeze(path, obj):
    if path.exists():
        if read(path) != obj: raise ValueError(f"Frozen experiment changed: {path}. Preserve it; do not overwrite.")
    else: write(path, obj)


@contextmanager
def run_lock(root):
    """OS lock: released on crash too; the file's presence is not a stale lock."""
    path = root / "outputs/reporting/full489_block69/run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0: f.write(b"0"); f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e: raise RuntimeError("Another Block 69 process is running.") from e
        try: yield
        finally:
            f.seek(0)
            if os.name == "nt": msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f, fcntl.LOCK_UN)


def geometry(affine):
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0,0,0,1]):
        raise ValueError("Invalid cached affine.")
    A = a[:3,:3]; spacing = np.linalg.norm(A, axis=0)
    if np.any(spacing <= 0) or abs(np.linalg.det(A)) < 1e-8: raise ValueError("Singular cached grid.")
    directions = A / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError("Sheared cached grid: isotropic Gaussian needs a separately versioned resampling policy.")
    return A, spacing


def rotation(degrees):
    x, y, z = np.deg2rad(degrees)
    cx,sx,cy,sy,cz,sz = np.cos(x),np.sin(x),np.cos(y),np.sin(y),np.cos(z),np.sin(z)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])


def sample_rotation(image, label, affine, center, shape, degrees):
    """Pull resampling on a bounded source crop, same coordinates for CT/GT."""
    A, _ = geometry(affine)
    shape = np.asarray(shape, dtype=int); center = np.asarray(center, dtype=float)
    matrix = np.linalg.solve(A, rotation(degrees) @ A)
    offset = center - matrix @ (shape // 2)
    corners = np.array(list(itertools.product(*[(0,n-1) for n in shape])))
    coords = corners @ matrix.T + offset
    lower = np.maximum(0, np.floor(coords.min(axis=0)).astype(int)-2)
    upper = np.minimum(image.shape, np.ceil(coords.max(axis=0)).astype(int)+3)
    if np.any(upper <= lower): raise ValueError("Rotation misses image grid.")
    sl = tuple(slice(int(l), int(u)) for l,u in zip(lower,upper))
    ct = np.ascontiguousarray(image[sl], dtype=np.float32)
    gt = np.ascontiguousarray(label[sl], dtype=np.uint8)
    if not np.isfinite(ct).all() or float(ct.min()) < -1.0001 or float(ct.max()) > 1.0001:
        raise ValueError("Cached CT is not finite and normalized to [-1,1].")
    if not np.isin(gt,[0,1,2,3]).all(): raise ValueError("Invalid cached labels.")
    kw = dict(matrix=matrix, offset=offset-lower, output_shape=tuple(shape), mode="constant", prefilter=False)
    x = affine_transform(ct, order=1, cval=-1.0, **kw)
    y = affine_transform(gt, order=0, cval=0, **kw)
    valid = affine_transform(np.ones(ct.shape,dtype=np.uint8), order=0,cval=0,**kw)
    return x, y, valid


def soft_edges(ct, affine, sigma_mm, truncate=4.0, *, signed=False):
    """Image-only function. World gradient = inv(A).T @ index gradient."""
    if not math.isfinite(sigma_mm) or sigma_mm <= 0: raise ValueError("Positive finite sigma required.")
    A, spacing = geometry(affine)
    x = np.asarray(ct, dtype=np.float32)
    if x.ndim != 3 or not np.isfinite(x).all(): raise ValueError("Finite 3D CT required.")
    gradients = []
    for axis in range(3):
        order = [0,0,0]; order[axis] = 1
        gradients.append(gaussian_filter(x, sigma=sigma_mm/spacing, order=order,
                                        mode="reflect", truncate=truncate, output=np.float32))
    invt = np.linalg.inv(A).T
    magnitude = np.zeros(x.shape, dtype=np.float32); signed_r = None
    for i in range(3):
        component = sum(np.float32(invt[i,j])*gradients[j] for j in range(3))
        magnitude += component*component
        if i == 0 and signed: signed_r = component
    np.sqrt(magnitude, out=magnitude)
    magnitude *= np.float32(sigma_mm)
    np.clip(magnitude, 0., 1., out=magnitude)
    return magnitude, signed_r


def make_patch(image, label, affine, center, degrees, policy=POLICY, *, with_signed=False):
    _, spacing = geometry(affine)
    halo = np.ceil(policy["truncate"]*policy["sigma_mm"]/spacing).astype(int)+2
    size = np.asarray(policy["patch_size"],dtype=int)
    x, y, valid = sample_rotation(image,label,affine,center,size+2*halo,degrees)
    edge, signed = soft_edges(x,affine,policy["sigma_mm"],policy["truncate"],signed=with_signed)
    sl = tuple(slice(int(h),int(h+n)) for h,n in zip(halo,size))
    inputs = np.ascontiguousarray(np.stack([x[sl],edge[sl]]),dtype=np.float32)
    target = np.ascontiguousarray(y[sl],dtype=np.int64)
    detail = {"center": [int(v) for v in center], "angles_degrees": [float(v) for v in degrees],
              "halo_voxels": halo.tolist(), "padding_fraction": float(1-valid[sl].mean()),
              "class_voxels": {str(k):int(np.count_nonzero(target==k)) for k in range(4)},
              "edge_min":float(inputs[1].min()), "edge_max":float(inputs[1].max()),
              "edge_mean":float(inputs[1].mean()), "edge_nonzero_voxels":int(np.count_nonzero(inputs[1]))}
    return inputs, target, detail, None if signed is None else signed[sl].copy()


def center_for_class(label, counts, k, rng):
    c = counts[k].cumsum(); total = int(c[-1])
    if total <= 0: raise ValueError(f"Selected smoke class absent: {k}")
    rank = int(rng.integers(total)); z = int(np.searchsorted(c,rank,side="right"))
    offset = rank - (int(c[z-1]) if z else 0)
    candidates = np.flatnonzero(label[:,:,z]==k)
    if len(candidates) != int(counts[k,z]): raise ValueError("Cached class counts disagree with labels.")
    return [int(v) for v in (*np.unravel_index(candidates[offset],label.shape[:2]),z)]


def context(root):
    protocol = read(root/"configs/experiments/nv_segment_ct_full489_protocol.json"); check(protocol,"protocol_sha256")
    parent = read(root/CONFIG65); check(parent,"experiment_sha256")
    source = read(root/CONFIG68); check(source,"experiment_sha256")
    for plan in (parent,source):
        if plan["partition"] != protocol["partition"] or plan["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("Source/protocol partition mismatch.")
        if plan.get("test_outputs_opened") is not False: raise ValueError("Test must remain sealed.")
    groups = [protocol["partition"][k]["case_ids"] for k in ("development","validation","test")]
    if list(map(len,groups)) != [289,80,120] or len(set(sum(groups,[]))) != 489:
        raise ValueError("Expected disjoint Development289 / Validation80 / Test120.")
    if any(not re.fullmatch(r"case_[0-9]{5}",cid) for cid in sum(groups,[])): raise ValueError("Invalid case ID.")
    if source["cache_parent_experiment_sha256"] != parent["experiment_sha256"] or source["inputs"] != parent["inputs"]:
        raise ValueError("Block 68 cache parent/input mismatch.")
    if source.get("script_sha256") != sha(root/"scripts/run_kits23_supervised68_earlystop.py"):
        raise ValueError("Block 68 runner changed since its freeze.")
    cfg = source["settings"]
    if cfg["model"]["in_channels"] != 1 or cfg["model"]["out_channels"] != 4 or cfg["model"]["spatial_dims"] != 3:
        raise ValueError("Expected one-channel, four-class 3D parent UNet.")
    if cfg["model"].get("dropout",0) != 0 or cfg["model"].get("norm") != "INSTANCE":
        raise ValueError("AMP replay requires frozen InstanceNorm without dropout.")
    if cfg["patch_size"] != POLICY["patch_size"] or cfg["ct_window_hu"] != [-200.,300.]:
        raise ValueError("Unexpected parent patch/intensity policy.")
    out = root/"outputs/reporting/full489_block68"/source["experiment_sha256"][:16]
    progress, selection = read(out/"training_progress.json"), read(out/"selection.json")
    if progress["status"] not in ("early_stopped","max_epochs_reached") or progress.get("test_reads") is not False:
        raise ValueError("Block 68 is not completed with test sealed.")
    if progress["selection"] != selection or selection["experiment_sha256"] != source["experiment_sha256"]:
        raise ValueError("Block 68 selection/progress mismatch.")
    epoch = selection["best_epoch"]
    if type(epoch) is not int or epoch < 1 or epoch > progress["completed_epochs"]: raise ValueError("Invalid selected epoch.")
    if not math.isfinite(selection["best_score"]): raise ValueError("Invalid selected score.")
    receipt = read(out/"epochs"/f"epoch_{epoch:03d}"/"complete.json"); check(receipt,"receipt_sha256")
    if (receipt["epoch"] != epoch or receipt["experiment_sha256"] != source["experiment_sha256"]
        or receipt["sha256"] != selection["best_checkpoint_sha256"]
        or not re.fullmatch(r"checkpoint_[0-9a-f]{12}\.pt",receipt["file"])):
        raise ValueError("Selected checkpoint receipt mismatch.")
    # Check weights' bytes for provenance, never load them into the new model.
    if sha(out/"epochs"/f"epoch_{epoch:03d}"/receipt["file"]) != receipt["sha256"]:
        raise ValueError("Selected Block 68 checkpoint bytes changed.")
    summary = read(out/"validation"/f"epoch_{epoch:03d}"/"summary.json"); check(summary,"summary_sha256")
    if (summary["status"] != "complete" or summary["n_cases"] != 80 or summary["epoch"] != epoch
        or summary["checkpoint_sha256"] != receipt["sha256"]
        or summary["experiment_sha256"] != source["experiment_sha256"]
        or summary["selection_score"] != selection["best_score"]):
        raise ValueError("Selected validation summary mismatch.")
    lineage = {"protocol_sha256":protocol["protocol_sha256"], "cache_parent_sha256":parent["experiment_sha256"],
               "block68_sha256":source["experiment_sha256"], "selection":selection,
               "source_summary_sha256":summary["summary_sha256"], "progress_sha256":sha(out/"training_progress.json")}
    return parent,source,lineage


def cache_folder(root,parent,cid):
    if cid not in parent["partition"]["development"]["case_ids"]: raise ValueError("Development-only cache access.")
    return root/"data/processed/kits23_supervised65"/parent["experiment_sha256"][:16]/"development"/cid


def cache_receipt(root,parent,cid,*,full=False):
    folder = cache_folder(root,parent,cid); r = read(folder/"complete.json"); check(r,"receipt_sha256")
    if (r["case_id"] != cid or r["partition"] != "development" or r["experiment_sha256"] != parent["experiment_sha256"]
        or r["input_identity"] != parent["inputs"][cid] or set(r["artifacts"]) != EXPECTED_CACHE):
        raise ValueError(f"Cache identity mismatch: {cid}")
    names = EXPECTED_CACHE if full else {"meta.json","slice_counts.npy"}
    for name in names:
        if sha(folder/name) != r["artifacts"][name]: raise ValueError(f"Cache changed: {cid}/{name}")
    meta=read(folder/"meta.json")
    if meta["case_id"] != cid or meta["partition"] != "development": raise ValueError("Metadata scope mismatch.")
    counts=np.load(folder/"slice_counts.npy",allow_pickle=False)
    if counts.dtype != np.int64 or counts.shape != (4,meta["shape"][2]) or (counts < 0).any():
        raise ValueError("Invalid cached slice counts.")
    if counts.sum() != math.prod(meta["shape"]): raise ValueError("Slice counts do not cover the grid.")
    if full:
        for filename,key in [("imaging.nii.gz","image_sha256"),("segmentation.nii.gz","ground_truth_sha256")]:
            if sha(root/"data/raw/kits23"/cid/filename) != parent["inputs"][cid][key]:
                raise ValueError(f"Frozen raw input changed: {cid}/{filename}")
    return r,meta,counts


def select_cases(root,parent,policy=POLICY):
    ids=parent["partition"]["development"]["case_ids"]
    ranked=sorted(ids,key=lambda cid:digest({"seed":policy["seed"],"case_id":cid}))
    selected=[]; checked=[]
    for cid in ranked:
        r,meta,counts=cache_receipt(root,parent,cid)
        present=np.flatnonzero(counts.sum(axis=1)).tolist()
        checked.append({"case_id":cid,"receipt_sha256":r["receipt_sha256"],"present":present})
        if present == [0,1,2,3]: selected.append(cid)
        if len(selected)==policy["cases"]: break
    if len(selected)!=policy["cases"]: raise ValueError("Not enough class-complete development cases for smoke.")
    return selected,checked


def runtime():
    import torch
    return {"python":sys.version.split()[0], "packages":{k:importlib.metadata.version(k) for k in
            ("numpy","scipy","torch","monai","matplotlib")}, "torch_cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_total_bytes":torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}


def prepare(root,run_identity):
    parent,source,lineage=context(root)
    ids,screen=select_cases(root,parent)
    receipts={}
    for cid in ids:
        print(f"Verify selected development cache/input: {cid}",flush=True)
        r,meta,_=cache_receipt(root,parent,cid,full=True); geometry(meta["affine"])
        if not np.allclose(np.linalg.norm(np.array(meta["affine"])[:3,:3],axis=0),source["settings"]["spacing_mm"],atol=1e-4):
            raise ValueError("Cached spacing differs from frozen parent.")
        receipts[cid]=r["receipt_sha256"]
    model=deepcopy(source["settings"]["model"]); model["in_channels"]=2
    plan={"block":69,"revision":REVISION,"policy":POLICY,"model":model,
          "optimizer":source["settings"]["optimizer"],"max_gradient_norm":source["settings"]["max_gradient_norm"],
          "source":lineage,"case_ids":ids,"selection_screen":screen,"cache_receipts":receipts,
          "runtime":run_identity,"script_sha256":sha(Path(__file__)),
          "validation_test_volume_reads":False,"validation_summary_reads":True,"test_outputs_opened":False}
    plan["experiment_sha256"]=digest(plan)
    out=root/"outputs/reporting/full489_block69"/plan["experiment_sha256"][:16]
    freeze(root/CONFIG69,plan); freeze(out/"protocol.json",plan)
    return parent,plan,out


def arrays(root,parent,cid):
    folder=cache_folder(root,parent,cid)
    image=np.load(folder/"image.npy",mmap_mode="r",allow_pickle=False)
    label=np.load(folder/"label.npy",mmap_mode="r",allow_pickle=False)
    r,meta,counts=cache_receipt(root,parent,cid)
    if image.dtype!=np.float32 or label.dtype!=np.uint8 or label.shape!=image.shape or list(image.shape)!=meta["shape"]:
        raise ValueError("Invalid cached CT/GT array.")
    return image,label,counts,meta


def preview(path,cid,base,rotated):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cm=ListedColormap(["#00b5d8","#ffa500","#ff40bf"])
    fig,axes=plt.subplots(2,4,figsize=(12,6),constrained_layout=True)
    for row,(inputs,target,detail,signed) in enumerate((base,rotated)):
        z=target.shape[2]//2;ct=inputs[0,:,:,z].T
        axes[row,0].imshow(ct,cmap="gray",vmin=-1,vmax=1,origin="lower")
        im=axes[row,1].imshow(inputs[1,:,:,z].T,cmap="magma",vmin=0,vmax=1,origin="lower")
        sigmax=0.8/POLICY["sigma_mm"]
        sg=axes[row,2].imshow(signed[:,:,z].T,cmap="coolwarm",vmin=-sigmax,vmax=sigmax,origin="lower")
        axes[row,3].imshow(ct,cmap="gray",vmin=-1,vmax=1,origin="lower")
        axes[row,3].imshow(np.ma.masked_where(target[:,:,z].T==0,target[:,:,z].T),cmap=cm,vmin=1,vmax=3,alpha=.65,origin="lower",interpolation="nearest")
        for col in range(4): axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
        axes[row,0].set_ylabel("Original" if row==0 else "Rotated",fontsize=11)
    for ax,title in zip(axes[0],["CT [-1,1]","Soft edge [0,1]","Signed dCT/dR (per mm)","Reference overlay"]):ax.set_title(title,fontsize=10)
    fig.colorbar(im,ax=axes[:,1],shrink=.7);fig.colorbar(sg,ax=axes[:,2],shrink=.7)
    ang=rotated[2]["angles_degrees"]
    fig.suptitle(f"{cid} | pull RAS angles = {ang[0]:.1f}, {ang[1]:.1f}, {ang[2]:.1f} deg\n"
                 "Patch plane (not radiological convention). Cyan: kidney; orange: tumor; magenta: cyst.",fontsize=11)
    fig.savefig(path,dpi=130);plt.close(fig)


def state_digest(model):
    d=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        d.update(name.encode());d.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()


def model_from_config(config):
    from monai.networks.nets import UNet
    return UNet(**{k:v for k,v in config.items() if k!="architecture"})


def update(model,optimizer,scaler,loss_fn,x,y,max_norm,device,events):
    """Bounded AMP backoff on the exact same tensors; no skipped update counted."""
    import torch
    params=list(model.parameters())
    first=next(m for m in model.modules() if isinstance(m,torch.nn.Conv3d))
    if first.in_channels!=2: raise ValueError("The first convolution must receive CT + edge.")
    for attempt in range(9):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=="cuda"):
            logits=model(x)
        if logits.shape != (x.shape[0],4,*x.shape[2:]): raise ValueError("Unexpected logits shape.")
        loss=loss_fn(logits.float(),y)
        if not torch.isfinite(loss).item(): raise ValueError("Non-finite loss; no optimizer update.")
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        grads=[q.grad for q in params if q.grad is not None]
        if not grads: raise ValueError("Missing parameter gradients.")
        if not all(torch.isfinite(g).all().item() for g in grads):
            if not scaler.is_enabled(): raise ValueError("Non-finite FP32 gradients.")
            before=float(scaler.get_scale());scaler.step(optimizer);scaler.update();after=float(scaler.get_scale())
            events.append({"attempt":attempt+1,"scale_before":before,"scale_after":after,"optimizer_update":False})
            print(f"  AMP backoff {before:g} -> {after:g}; retry same patch",flush=True)
            if not 0<after<before or attempt==8: raise ValueError("AMP overflow persists after bounded retries.")
            continue
        channel_norms=[float(first.weight.grad[:,i].float().norm()) for i in range(2)]
        if any(not math.isfinite(v) or v<0 for v in channel_norms): raise ValueError("Invalid CT/edge channel gradient.")
        norm=torch.nn.utils.clip_grad_norm_(params,max_norm,error_if_nonfinite=True)
        if not math.isfinite(float(norm)) or float(norm)<=0: raise ValueError("Invalid total gradient norm.")
        scaler.step(optimizer);scaler.update()
        if not all(torch.isfinite(q).all().item() for q in params): raise ValueError("Non-finite parameters after update.")
        return {"loss":float(loss.detach()),"gradient_norm":float(norm),"input_weight_gradient_norms":channel_norms,"amp_retries":attempt}
    raise AssertionError("Unreachable")


def export(out,attempt):
    # Explicit whitelist; the local model weights and patient volumes stay out.
    names=[out/"protocol.json",attempt/"summary.json",attempt/"trace.json",attempt/"report.html"]
    names.extend(sorted(attempt.glob("preview_*.png")))
    if (out/"complete.json").exists(): names.append(out/"complete.json")
    names=[p for p in names if p.is_file()]
    manifest={p.relative_to(out).as_posix():sha(p) for p in names}
    target=out/"block69_edges_smoke_report.zip";tmp=target.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in names:z.write(p,p.relative_to(out).as_posix())
        z.writestr("archive_manifest.json",json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,target)
    return target


def report_html(attempt,summary):
    import html
    text=("Prueba de viabilidad: CT y bordes suaves como dos canales; 24 pasos sobre tres casos de desarrollo. "
          "No es una comparacion de precision ni un entrenamiento completo. Se usan rotaciones fisicas "
          "de hasta 10 grados y Dice + entropia cruzada. Los bordes se calculan desde el CT rotado, "
          "sin usar las etiquetas. El gradiente firmado mostrado es diagnostico; el segundo canal es su magnitud.")
    pictures="".join(f'<h2>{html.escape(p.stem)}</h2><img src="{p.name}" alt="CT, bordes y referencia">' for p in sorted(attempt.glob("preview_*.png")))
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>Bloque 69</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#153448}img{max-width:100%}pre{white-space:pre-wrap;background:#eef4f6;padding:16px}</style><h1>Bloque 69: bordes suaves 3D</h1><p>'+text+'</p><pre>'+html.escape(json.dumps(summary,indent=2,ensure_ascii=False))+'</pre>'+pictures+'</html>'
    (attempt/"report.html").write_text(page,encoding="utf-8")


def verify_completed(out,plan):
    if not (out/"complete.json").exists():return None
    r=read(out/"complete.json");check(r,"receipt_sha256")
    if r["experiment_sha256"]!=plan["experiment_sha256"] or not re.fullmatch(r"attempt_[0-9a-f]{12}",r["attempt"]):
        raise ValueError("Completed smoke identity mismatch.")
    folder=out/r["attempt"]
    expected={"checkpoint.pt","summary.json","trace.json","report.html"}|{f"preview_{cid}.png" for cid in plan["case_ids"]}
    if set(r["artifacts"])!=expected:raise ValueError("Completed artifact set mismatch.")
    for n,h in r["artifacts"].items():
        if sha(folder/n)!=h:raise ValueError(f"Completed smoke artifact changed: {n}")
    summary=read(folder/"summary.json")
    if summary["status"]!="complete" or summary["optimizer_steps"]!=24 or not summary["checkpoint_reload_passed"]:
        raise ValueError("Incomplete committed smoke.")
    return folder,summary


def training(root,parent,plan,out,device="cuda"):
    import torch
    from monai.losses import DiceCELoss
    done=verify_completed(out,plan)
    if done:
        path=export(out,done[0]);print("Verified completed smoke; no new training.",flush=True)
        return done[1],path
    attempt=out/("attempt_"+uuid4().hex[:12]);attempt.mkdir()
    policy=plan["policy"];seed=policy["seed"];start=time.perf_counter();trace=[];events=[];case_data=[]
    torch.manual_seed(seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    if device=="cuda":torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    rng=np.random.default_rng(seed)
    try:
        model=model_from_config(plan["model"]).to(device).train();initial=state_digest(model)
        opt=torch.optim.AdamW(model.parameters(),lr=plan["optimizer"]["lr"],weight_decay=plan["optimizer"]["weight_decay"])
        scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda",init_scale=128.)
        loss_fn=DiceCELoss(include_background=False,to_onehot_y=True,softmax=True,lambda_dice=1.,lambda_ce=1.)
        probe=None
        for number,cid in enumerate(plan["case_ids"],1):
            print(f"[{number:02d}/03] CT + soft edges: {cid}",flush=True)
            image,label,counts,meta=arrays(root,parent,cid)
            preview_center=center_for_class(label,counts,3,np.random.default_rng(seed+number))
            preview_angles=[-7.,5.,10.]
            base=make_patch(image,label,meta["affine"],preview_center,[0.,0.,0.],policy,with_signed=True)
            rotated=make_patch(image,label,meta["affine"],preview_center,preview_angles,policy,with_signed=True)
            preview(attempt/f"preview_{cid}.png",cid,base,rotated)
            case_data.append({"case_id":cid,"preview_original":base[2],"preview_rotated":rotated[2],"cache_affine":meta["affine"]})
            del base,rotated
            for local_step in range(policy["steps_per_case"]):
                k=local_step%4;center=center_for_class(label,counts,k,rng)
                angles=rng.uniform(-policy["rotation_degrees_per_world_axis"],policy["rotation_degrees_per_world_axis"],size=3)
                x_np,y_np,detail,_=make_patch(image,label,meta["affine"],center,angles,policy)
                if y_np[tuple(n//2 for n in policy["patch_size"])]!=k:raise ValueError("Rotated sampling center changed class.")
                x=torch.from_numpy(x_np)[None].to(device);y=torch.from_numpy(y_np)[None,None].to(device)
                step_events=[]
                result=update(model,opt,scaler,loss_fn,x,y,plan["max_gradient_norm"],device,step_events)
                events.extend({**ev,"step":len(trace)+1,"case_id":cid} for ev in step_events)
                trace.append({"step":len(trace)+1,"case_id":cid,"center_class":k,**detail,**result})
                write(attempt/"trace.json",{"steps":trace,"amp_events":events,"cases":case_data})
                print(f"  step {len(trace):02d}/24; class={k}; loss={result['loss']:.6f}; edge weight grad={result['input_weight_gradient_norms'][1]:.5g}",flush=True)
                if probe is None:probe=x.detach().clone()
                del x,y
            del image,label,counts
        if len(trace)!=24 or {r["center_class"] for r in trace}!={0,1,2,3}:raise ValueError("Incomplete smoke updates/classes.")
        if any(not any(r["input_weight_gradient_norms"][i]>0 for r in trace) for i in range(2)):
            raise ValueError("An input channel never received a nonzero gradient.")
        final=state_digest(model)
        if final==initial:raise ValueError("Model parameters did not change.")
        model.eval()
        with torch.no_grad():pred=model(probe).float().cpu()
        if not torch.isfinite(pred).all():raise ValueError("Non-finite smoke prediction.")
        torch.save({"experiment_sha256":plan["experiment_sha256"],"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "model_config":plan["model"],"optimizer_steps":24,"role":"feasibility_only","state_sha256":final},attempt/"checkpoint.pt")
        payload=torch.load(attempt/"checkpoint.pt",map_location="cpu",weights_only=True)
        restored=model_from_config(plan["model"]).to(device).eval();restored.load_state_dict(payload["model"],strict=True)
        if state_digest(restored)!=final:raise ValueError("Checkpoint state round-trip failed.")
        with torch.no_grad():second=restored(probe).float().cpu()
        if not torch.allclose(pred,second,atol=1e-6,rtol=1e-6):raise ValueError("Checkpoint forward parity failed.")
        for cid in plan["case_ids"]:
            r,_,_=cache_receipt(root,parent,cid,full=True)
            if r["receipt_sha256"]!=plan["cache_receipts"][cid]:raise ValueError("Input/cache changed during smoke.")
        if context(root)[2]!=plan["source"]:raise ValueError("Block 68 source changed during smoke.")
        if device=="cuda":torch.cuda.synchronize()
        summary={"status":"complete","block":69,"experiment_sha256":plan["experiment_sha256"],"case_ids":plan["case_ids"],
                 "optimizer_steps":len(trace),"input_channels":2,"checkpoint_reload_passed":True,
                 "both_input_channels_receive_gradients":True,"loss_first":trace[0]["loss"],"loss_last":trace[-1]["loss"],
                 "amp_retries":len(events),"seed":seed,"initial_state_sha256":initial,"final_state_sha256":final,
                 "wall_seconds":time.perf_counter()-start,
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None,
                 "peak_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else None,
                 "device":device,"validation_test_volume_reads":False,"test_outputs_opened":False,
                 "interpretation":"Feasibility only. No validation score, loss comparison or evidence of accuracy improvement."}
        write(attempt/"summary.json",summary);report_html(attempt,summary)
        artifacts={p.name:sha(p) for p in attempt.iterdir() if p.suffix in (".json",".pt",".png",".html")}
        r={"experiment_sha256":plan["experiment_sha256"],"attempt":attempt.name,"artifacts":artifacts};r["receipt_sha256"]=digest(r)
        write(out/"complete.json",r)
        verify_completed(out,plan)
        return summary,export(out,attempt)
    except BaseException as exc:
        if (out/"complete.json").exists():
            print("Completed artifacts preserved; report export/verification failed.",flush=True)
            raise
        # Keep failed/interrupted artifacts; no partial result is marked complete.
        summary={"status":"interrupted" if isinstance(exc,KeyboardInterrupt) else "failed","error":f"{type(exc).__name__}: {exc}",
                 "optimizer_steps":len(trace),"experiment_sha256":plan["experiment_sha256"],"test_outputs_opened":False}
        write(attempt/"summary.json",summary);write(attempt/"trace.json",{"steps":trace,"amp_events":events,"cases":case_data})
        report_html(attempt,summary)
        print(f"Share failure report: {export(out,attempt)}",flush=True)
        raise


def self_test():
    """Synthetic numerical + real CPU MONAI backward tests. No project data IO."""
    import torch
    from monai.losses import DiceCELoss
    torch.set_num_threads(2)
    a=np.diag([1.,2.,3.,1.]);shape=(49,49,49)
    constant=np.ones(shape,dtype=np.float32)
    e,_=soft_edges(constant,a,3.);assert float(e.max())<1e-7
    print("PASS: constant CT has zero soft edges")
    grid=np.indices(shape,dtype=np.float32)
    ramp=grid[0]+2*grid[1]+3*grid[2]
    e,signed=soft_edges(ramp*.01,a,3.,signed=True)
    assert np.allclose(e[15:-15,15:-15,15:-15],.03*np.sqrt(3),rtol=.01)
    assert np.allclose(signed[15:-15,15:-15,15:-15],.01,rtol=.01)
    e_neg,signed_neg=soft_edges(-ramp*.01,a,3.,signed=True)
    assert np.allclose(e,e_neg) and np.allclose(signed,-signed_neg)
    print("PASS: physical spacing, gradient units and signed derivatives")
    image=np.full((49,49,49),-1.,np.float32);label=np.zeros(image.shape,np.uint8)
    image[18:31,20:29,15:34]=1.;label[18:31,20:29,15:34]=3
    policy={**POLICY,"patch_size":[24,24,24],"sigma_mm":3.}
    original=make_patch(image,label,a,[24,24,24],[0,0,0],policy)
    assert np.array_equal(original[0][0],image[12:36,12:36,12:36])
    assert np.array_equal(original[1],label[12:36,12:36,12:36])
    rotated=make_patch(image,label,a,[24,24,24],[0,0,90],policy)
    assert rotated[1][12,12,12]==3 and set(np.unique(rotated[1]))<={0,3}
    assert float(rotated[0][0][rotated[1]==3].mean())>.8
    print("PASS: paired physical rotation and discrete target labels")
    altered=make_patch(image,np.zeros_like(label),a,[24,24,24],[0,0,90],policy)
    assert np.array_equal(rotated[0],altered[0])
    full_edge,_=soft_edges(image,a,3.)
    assert np.allclose(original[0][1],full_edge[12:36,12:36,12:36],atol=1e-6)
    print("PASS: GT-independent inputs and halo/crop edge parity")
    shear=a.copy();shear[0,1]=.4
    try: geometry(shear)
    except ValueError: pass
    else:raise AssertionError("Sheared grid not rejected")
    print("PASS: unsupported sheared geometry rejected explicitly")
    config={"spatial_dims":3,"in_channels":2,"out_channels":4,"channels":[4,8,16],"strides":[2,2],"num_res_units":1,"norm":"INSTANCE"}
    torch.manual_seed(6901);model=model_from_config(config)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    x=torch.from_numpy(rotated[0])[None];y=torch.from_numpy(rotated[1])[None,None]
    before=state_digest(model)
    result=update(model,opt,scaler,DiceCELoss(include_background=False,to_onehot_y=True,softmax=True),x,y,12.,"cpu",[])
    assert state_digest(model)!=before and all(v>0 for v in result["input_weight_gradient_norms"])
    print("PASS: real two-channel UNet forward/backward; finite updates through CT and edge")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"test.pt";torch.save(model.state_dict(),path)
        other=model_from_config(config);other.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"))
        assert state_digest(other)==state_digest(model)
    print("PASS: real checkpoint reload")
    print("Block 69 synthetic self-test PASSED. No patient data read.")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage",choices=["self-test","smoke"],default="smoke")
    parser.add_argument("--project-root",type=Path,default=ROOT)
    args=parser.parse_args(argv)
    try:
        if args.stage=="self-test":self_test();return 0
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
        import torch
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable. Run this script with conda run -n vista3d-nv.")
        root=args.project_root.resolve()
        if not (root/CONFIG68).is_file():raise FileNotFoundError("Run from the project checkout with the completed Block 68 config.")
        with run_lock(root):
            parent,plan,out=prepare(root,runtime())
            summary,report=training(root,parent,plan,out)
        print(f"Status: {summary['status']}")
        print(f"Optimizer steps: {summary['optimizer_steps']}/24; two-channel checkpoint reload PASSED")
        print(f"Peak allocated: {summary['peak_allocated_bytes']/2**30:.2f} GiB; reserved: {summary['peak_reserved_bytes']/2**30:.2f} GiB")
        print(f"Share this ZIP: {report}")
        print("Training feasibility only. No validation/test volume reads. Test120 remains sealed.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command. Incomplete smoke repeats from the seed.",file=sys.stderr);return 130
    except Exception as exc:
        print(f"Block 69 FAILED: {type(exc).__name__}: {exc}",file=sys.stderr);return 1


if __name__=="__main__":raise SystemExit(main())
