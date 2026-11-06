"""Versioned CPU preprocessing, absent classes allowed, memory-mapped cache."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import shutil
import time
from uuid import uuid4

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from vista3d_tfg.training import contract65 as c
from vista3d_tfg.training.supervised64 import validate_pair


def runtime_identity() -> dict:
    return {k: version(k) for k in ("numpy", "nibabel", "scipy")}


def grid(image, config):
    shape = tuple(int(n) for n in np.ceil((np.asarray(image.shape)-1)*nib.affines.voxel_sizes(image.affine)
                                         /np.asarray(config["spacing_mm"]))+1)
    if int(np.prod(shape, dtype=np.int64)) > config["max_resampled_voxels"]:
        raise ValueError(f"Resampled grid exceeds frozen RAM guard: {shape}")
    affine = image.affine.copy()
    affine[:3,:3] *= np.asarray(config["spacing_mm"])/nib.affines.voxel_sizes(image.affine)
    return shape, affine


def preprocess(image_path: Path, label_path: Path, config: dict, *, training: bool):
    image, label = nib.load(image_path), nib.load(label_path)
    if int(np.prod(image.shape, dtype=np.int64)) > config["max_original_voxels"]:
        raise ValueError("Original grid exceeds frozen RAM guard.")
    # Validation reference is checked for integrity/geometry, but never controls
    # the image grid, normalization, crop, window coordinates or model prediction.
    image_data, label_data = validate_pair(image, label)
    info = {"original_shape": list(image.shape), "original_affine": image.affine.tolist()}
    canonical = nib.as_closest_canonical(nib.Nifti1Image(image_data, image.affine))
    target = grid(canonical, config)
    ct = np.asarray(resample_from_to(canonical, target, order=1, cval=-200.0).dataobj, dtype=np.float32)
    np.clip(ct, -200., 300., out=ct); ct += 200.; ct *= 2./500.; ct -= 1.
    if not np.isfinite(ct).all():
        raise ValueError("Non-finite preprocessed CT.")
    info.update(shape=list(ct.shape), affine=target[1].tolist())
    arrays = {"image.npy": ct}
    if training:
        reference = nib.Nifti1Image(label_data, label.affine)
        gt = np.asarray(resample_from_to(reference, target, order=0, cval=0).dataobj, dtype=np.uint8)
        counts = np.asarray([[np.count_nonzero(gt[:,:,z] == k) for z in range(gt.shape[2])] for k in range(4)], dtype=np.int64)
        present = np.flatnonzero(counts.sum(axis=1)).tolist()
        if not any(k in present for k in (1,2,3)):
            raise ValueError("All foreground disappeared after resampling; case is not silently excluded.")
        info.update(present_classes=present, original_classes=np.unique(label_data).tolist(),
                    lost_classes=sorted(set(np.unique(label_data).tolist())-set(present)))
        arrays.update({"label.npy": gt, "slice_counts.npy": counts})
    return arrays, info


def verified_cache(root: Path, plan: dict, partition: str, cid: str) -> dict | None:
    c.case_paths(root, plan, partition, cid)
    folder = c.cache_root(root, plan) / partition / cid
    if not (folder/"complete.json").exists():
        return None
    receipt = c.read_json(folder/"complete.json")
    c.check_digest(receipt,"receipt_sha256")
    expected = {"image.npy", "meta.json"} | ({"label.npy", "slice_counts.npy"} if partition=="development" else set())
    if (receipt["experiment_sha256"] != plan["experiment_sha256"] or receipt["case_id"] != cid
            or receipt["partition"] != partition or receipt["input_identity"] != plan["inputs"][cid]
            or receipt["runtime"] != runtime_identity() or set(receipt["artifacts"]) != expected):
        raise ValueError(f"Cache identity changed: {cid}")
    for name,digest in receipt["artifacts"].items():
        if c.sha256_file(folder/name) != digest:
            raise ValueError(f"Cache artifact changed: {cid}/{name}")
    return receipt


def prepare_cache(root: Path, plan: dict, partition: str, *, max_cases: int | None = None) -> dict:
    ids = c.partition_ids(plan, partition)
    done, attempted = [], 0
    for index,cid in enumerate(ids):
        c.verify_input(root, plan, partition, cid)
        receipt = verified_cache(root, plan, partition, cid)
        if receipt:
            done.append(cid); continue
        if max_cases is not None and attempted >= max_cases:
            continue
        attempted += 1
        print(f"[{index+1:03d}/{len(ids):03d}] cache {partition} {cid}", flush=True)
        start = time.perf_counter()
        folder = c.cache_root(root, plan)/partition/cid
        folder.parent.mkdir(parents=True, exist_ok=True)
        arrays, info = preprocess(*c.case_paths(root,plan,partition,cid), plan["settings"], training=partition=="development")
        required = sum(a.nbytes for a in arrays.values()) + 2*2**30
        if shutil.disk_usage(folder.parent).free < required:
            raise OSError(f"Insufficient cache disk space: need {required/2**30:.2f} GiB including reserve.")
        temporary = folder.with_name(cid+"_tmp_"+uuid4().hex[:8])
        temporary.mkdir()
        try:
            for name,array in arrays.items():
                np.save(temporary/name,array,allow_pickle=False)
            c.atomic_json(temporary/"meta.json", {**info,"case_id":cid,"partition":partition})
            c.verify_input(root,plan,partition,cid)
            receipt = {"experiment_sha256":plan["experiment_sha256"],"case_id":cid,"partition":partition,
                       "input_identity":plan["inputs"][cid],"runtime":runtime_identity(),
                       "seconds":time.perf_counter()-start,
                       "artifacts":{p.name:c.sha256_file(p) for p in temporary.iterdir()}}
            receipt["receipt_sha256"] = c.canonical_sha256(receipt)
            c.atomic_json(temporary/"complete.json",receipt)
            if folder.exists():
                raise ValueError("Uncommitted cache directory exists; preserve it for inspection.")
            temporary.replace(folder)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
        done.append(cid)
        print(f"  cached; {sum(a.nbytes for a in arrays.values())/2**20:.1f} MiB; {receipt['seconds']:.2f} s",flush=True)
    audit=[]
    for cid in done:
        folder=c.cache_root(root,plan)/partition/cid
        meta=c.read_json(folder/"meta.json");receipt=c.read_json(folder/"complete.json")
        audit.append({"case_id":cid,"receipt_sha256":receipt["receipt_sha256"],"shape":meta["shape"],
                      "lost_classes":meta.get("lost_classes",[]),
                      "present_classes":meta.get("present_classes"),
                      "bytes":sum((folder/name).stat().st_size for name in receipt["artifacts"])})
    result={"status":"complete" if done==ids else "partial", "partition":partition,
            "completed_case_ids":done,"requested_case_ids":ids,"test_reads":False,"cache_audit":audit}
    c.atomic_json(c.output_root(root,plan)/f"cache_{partition}_progress.json",result)
    return result


class CacheIndex:
    """Hash all files at worker start; detect size/mtime drift before each reuse."""
    def __init__(self, root: Path, plan: dict):
        self.root,self.plan,self.stamps,self.receipts = root,plan,{},{}
        for partition in ("development","validation"):
            for index,cid in enumerate(c.partition_ids(plan,partition)):
                if index%25==0:print(f"Verify {partition} cache {index+1}/{len(c.partition_ids(plan,partition))}",flush=True)
                c.verify_input(root,plan,partition,cid)
                receipt=verified_cache(root,plan,partition,cid)
                if receipt is None:raise ValueError(f"Missing {partition} cache: {cid}")
                self.receipts[cid]=receipt["receipt_sha256"]
                folder=c.cache_root(root,plan)/partition/cid
                for name in [*receipt["artifacts"],"complete.json"]:
                    p=folder/name; st=p.stat();self.stamps[p]=(st.st_size,st.st_mtime_ns)

    def load(self, partition: str, cid: str):
        c.case_paths(self.root,self.plan,partition,cid)
        folder=c.cache_root(self.root,self.plan)/partition/cid
        for p,stamp in self.stamps.items():
            if p.parent==folder:
                st=p.stat()
                if (st.st_size,st.st_mtime_ns)!=stamp:raise ValueError(f"Cache changed during run: {p}")
        meta=c.read_json(folder/"meta.json")
        image=np.load(folder/"image.npy",mmap_mode="r",allow_pickle=False)
        if image.dtype!=np.float32 or list(image.shape)!=meta["shape"]:
            raise ValueError("Invalid cached image dtype/shape.")
        if partition=="validation":return image,meta
        label=np.load(folder/"label.npy",mmap_mode="r",allow_pickle=False)
        counts=np.load(folder/"slice_counts.npy",allow_pickle=False)
        if label.dtype!=np.uint8 or label.shape!=image.shape or counts.shape!=(4,image.shape[2]):
            raise ValueError("Invalid cached training labels/counts.")
        return image,label,counts,meta
